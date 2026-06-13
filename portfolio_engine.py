"""Live PORTFOLIO bot (paper) — validated config: 5% x 10 coins + BTC market filter.

Strategy (daily, long-only spot, net Bitkub 0.25% fee):
- universe of established THB coins
- enter when close > EMA200 (per coin), pick strongest-trend coins first
- hold up to max_positions, each at alloc_pct of equity
- exit when close < EMA200 (trend break) or trailing stop (trail_atr x ATR)
- MARKET FILTER: only hold when BTC > its EMA200; if BTC turns bear -> sell all to cash

Writes state.json (same schema the dashboard reads).
Backtested (Bitkub real 6y): ~+56%/yr, Sharpe 1.14, MaxDD -34% at 5% x 10 (see docs)."""
import datetime as dt
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from data_feed import DataFeed

log = logging.getLogger("quantbot.portfolio")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def _indicators(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    d = df.copy()
    d["ema_trend"] = d["close"].ewm(span=ema_period, adjust=False).mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def _slip(atr: float, px: float, cfg: dict) -> float:
    if px <= 0:
        return cfg["base_slippage_bps"] / 10000.0
    bps = cfg["base_slippage_bps"] + (atr / px) * 10000 * cfg["slippage_vol_coef"]
    return min(bps, cfg["slippage_cap_bps"]) / 10000.0


class PortfolioEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.p = cfg["portfolio"]
        self.exec_cfg = cfg["execution_v2"]
        self.fee = self.exec_cfg["fee_rate"]
        self.feed = DataFeed(cfg)
        self.universe = self.p["universe"]
        self.market_sym = self.p.get("market_symbol", "BTC/THB")
        self.cash = float(cfg["trading"]["initial_capital"])
        self.initial = self.cash
        self.positions = {}        # sym -> {qty, entry, stop, opened_at}
        self.trades = []
        self.equity_curve = []
        self.signals = {}
        self.peak = self.initial
        self.killed = False

    # ---------- data ----------
    def _load(self):
        out = {}
        n = int(self.p.get("warmup_bars", 320))
        for sym in set(self.universe + [self.market_sym]):
            try:
                df = self.feed.ohlcv(sym, limit=n)
                if len(df) >= self.p.get("ema_period", 200):
                    out[sym] = _indicators(df, self.p.get("ema_period", 200))
            except Exception as e:
                log.warning(f"load {sym} failed: {type(e).__name__}: {str(e)[:60]}")
        return out

    def equity(self, px: dict) -> float:
        return self.cash + sum(p["qty"] * px.get(s, p["entry"]) for s, p in self.positions.items())

    # ---------- one cycle ----------
    def step(self):
        data = self._load()
        if self.market_sym not in data:
            log.warning("market symbol data missing; skipping cycle")
            return
        now = dt.datetime.now(dt.timezone.utc)
        last = {s: d.iloc[-1] for s, d in data.items()}
        px = {s: float(r["close"]) for s, r in last.items()}
        atrv = {s: float(r["atr"]) for s, r in last.items()}
        ema = {s: float(r["ema_trend"]) for s, r in last.items()}

        market_on = px[self.market_sym] > ema[self.market_sym]
        trail = float(self.p.get("trail_atr", 6.0))

        # 1) exits / trailing
        for s in list(self.positions):
            if s not in px:
                continue
            P = self.positions[s]
            P["stop"] = max(P["stop"], px[s] - trail * atrv[s])
            if px[s] <= P["stop"] or px[s] < ema[s] or not market_on:
                fill = px[s] * (1 - _slip(atrv[s], px[s], self.exec_cfg))
                proceeds = P["qty"] * fill
                fee = proceeds * self.fee
                pnl = proceeds - fee - P["cost"]
                self.cash += proceeds - fee
                reason = ("market_off" if not market_on else
                          "trend_break" if px[s] < ema[s] else "trail_stop")
                self.trades.append({"symbol": s, "side": "long", "qty": round(P["qty"], 8),
                                    "entry": round(P["entry"], 6), "exit": round(fill, 6),
                                    "pnl": round(pnl, 2), "opened_at": P["opened_at"],
                                    "closed_at": str(now), "reason": reason})
                log.info(f"SELL {s} @ {fill:.4f} pnl={pnl:.2f} ({reason})")
                del self.positions[s]

        equity = self.equity(px)

        # 2) drawdown kill switch (portfolio level)
        self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak else 0
        if dd >= self.cfg["risk"]["max_drawdown_pct"]:
            self.killed = True

        # 3) entries (only if market_on and not killed)
        cand = []
        for s in self.universe:
            if s in self.positions or s not in px:
                continue
            if px[s] > ema[s] and atrv[s] > 0 and np.isfinite(ema[s]):
                cand.append((px[s] / ema[s], s))   # momentum rank
        cand.sort(reverse=True)
        if market_on and not self.killed:
            for _, s in cand:
                if len(self.positions) >= int(self.p["max_positions"]):
                    break
                cost = float(self.p["alloc_pct"]) * equity
                if self.cash >= cost and cost > 15:
                    fill = px[s] * (1 + _slip(atrv[s], px[s], self.exec_cfg))
                    qty = cost / (fill * (1 + self.fee))
                    self.cash -= qty * fill * (1 + self.fee)
                    self.positions[s] = {"qty": qty, "entry": fill, "stop": fill - trail * atrv[s],
                                         "cost": qty * fill * (1 + self.fee), "opened_at": str(now)}
                    log.info(f"BUY {s} @ {fill:.4f} cost={cost:.0f}")

        # 4) signals snapshot
        for s in self.universe:
            if s in px:
                self.signals[s] = {"price": round(px[s], 6), "above_ema200": bool(px[s] > ema[s]),
                                   "held": s in self.positions, "ts": str(now)}
        self.signals["_market"] = {"symbol": self.market_sym, "on": bool(market_on),
                                   "btc": round(px[self.market_sym], 2), "ema200": round(ema[self.market_sym], 2)}

        self.equity_curve.append([str(now), round(equity, 2)])
        self.equity_curve = self.equity_curve[-2000:]
        self._save(px, equity, market_on)

    def _save(self, px, equity, market_on):
        wins = [t for t in self.trades if t["pnl"] > 0]
        snap = {
            "updated_at": str(dt.datetime.now(dt.timezone.utc)),
            "status": "killed" if self.killed else ("running" if market_on else "risk_off_cash"),
            "equity": round(equity, 2), "cash": round(self.cash, 2),
            "pnl_total": round(equity - self.initial, 2),
            "pnl_pct": round((equity / self.initial - 1) * 100, 2),
            "n_trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1) if self.trades else None,
            "market_on": bool(market_on),
            "positions": [{"symbol": s, "side": "long", "qty": round(p["qty"], 8),
                           "entry": round(p["entry"], 6), "mark": px.get(s),
                           "stop": round(p["stop"], 6), "target": None} for s, p in self.positions.items()],
            "signals": self.signals, "trades": self.trades[-50:],
            "equity_curve": self.equity_curve[-1500:],
        }
        with open(STATE_FILE, "w") as f:
            json.dump(snap, f, indent=1)

    def run(self):
        poll = int(self.cfg["trading"].get("poll_seconds", 300))
        log.info(f"Portfolio bot started | {self.p['alloc_pct']:.0%} x max {self.p['max_positions']} "
                 f"| market_filter={self.p.get('market_filter')} | universe={len(self.universe)} coins")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception(f"step error: {e}")
            time.sleep(poll)
