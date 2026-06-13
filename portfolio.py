"""Portfolio state: positions, equity, trade log. Persists to state.json for the dashboard."""
import json
import os
import datetime as dt

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


class Portfolio:
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.initial = initial_capital
        self.positions = {}     # symbol -> {side, qty, entry, stop, target, opened_at}
        self.trades = []        # closed trades
        self.equity_curve = []  # [ts, equity]
        self.signals = {}       # last signal per symbol

    def equity(self, prices: dict) -> float:
        eq = self.cash
        for sym, p in self.positions.items():
            px = prices.get(sym, p["entry"])
            if p["side"] == "long":
                eq += p["qty"] * px
            else:
                eq += p["qty"] * (2 * p["entry"] - px)  # short PnL
        return eq

    def open(self, symbol, side, qty, price, stop, target, ts, fee_rate=0.001):
        cost = qty * price
        fee = cost * fee_rate
        self.cash -= (cost + fee) if side == "long" else fee
        if side == "short":
            self.cash -= cost  # reserve margin 1x
        self.positions[symbol] = {"side": side, "qty": qty, "entry": price, "stop": stop,
                                  "target": target, "opened_at": str(ts)}

    def close(self, symbol, price, ts, reason, fee_rate=0.001):
        p = self.positions.pop(symbol, None)
        if not p:
            return None
        notional = p["qty"] * price
        fee = notional * fee_rate
        if p["side"] == "long":
            pnl = p["qty"] * (price - p["entry"]) - fee
            self.cash += notional - fee
        else:
            pnl = p["qty"] * (p["entry"] - price) - fee
            self.cash += p["qty"] * p["entry"] + pnl
        trade = {"symbol": symbol, "side": p["side"], "qty": round(p["qty"], 8),
                 "entry": p["entry"], "exit": price, "pnl": round(pnl, 2),
                 "opened_at": p["opened_at"], "closed_at": str(ts), "reason": reason}
        self.trades.append(trade)
        return trade

    def apply_funding(self, prices: dict, funding_rate: float):
        """v2 ด้าน 3.2: futures มี funding ทุก 8 ชม. หักจากเงินสดของโพซิชันที่ถือข้ามรอบ.
        long จ่ายเมื่อ funding บวก, short ได้รับ (และกลับกัน)."""
        for sym, p in self.positions.items():
            px = prices.get(sym, p["entry"])
            notional = p["qty"] * px
            cost = notional * funding_rate
            self.cash -= cost if p["side"] == "long" else -cost

    def record_equity(self, ts, prices):
        self.equity_curve.append([str(ts), round(self.equity(prices), 2)])
        self.equity_curve = self.equity_curve[-5000:]

    def snapshot(self, prices: dict, status: str = "running"):
        eq = self.equity(prices)
        wins = [t for t in self.trades if t["pnl"] > 0]
        return {
            "updated_at": str(dt.datetime.now(dt.timezone.utc)),
            "status": status,
            "equity": round(eq, 2),
            "cash": round(self.cash, 2),
            "pnl_total": round(eq - self.initial, 2),
            "pnl_pct": round((eq / self.initial - 1) * 100, 2),
            "n_trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1) if self.trades else None,
            "positions": [{"symbol": s, **p, "mark": prices.get(s)} for s, p in self.positions.items()],
            "signals": self.signals,
            "trades": self.trades[-50:],
            "equity_curve": self.equity_curve[-1500:],
        }

    def save(self, prices: dict, status="running"):
        with open(STATE_FILE, "w") as f:
            json.dump(self.snapshot(prices, status), f, indent=1)
