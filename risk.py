"""Risk management: position sizing, stops, kill switches."""
import datetime as dt


class RiskManager:
    def __init__(self, cfg: dict):
        self.r = cfg["risk"]
        self.day = None
        self.day_start_equity = None
        self.peak_equity = None
        self.killed = False

    # ---- sizing ----
    def position_size(self, equity: float, price: float, atr_value: float) -> float:
        """Volatility-based sizing: risk a fixed % of equity per trade.
        qty = (equity * risk_per_trade) / stop_distance"""
        stop_dist = atr_value * self.r["atr_stop_multiplier"]
        if stop_dist <= 0:
            return 0.0
        qty = (equity * self.r["risk_per_trade"]) / stop_dist
        max_qty = (equity * self.r["max_position_pct"]) / price
        return max(0.0, min(qty, max_qty))

    def stop_and_target(self, side: str, price: float, atr_value: float):
        s, t = self.r["atr_stop_multiplier"] * atr_value, self.r["atr_target_multiplier"] * atr_value
        if side == "long":
            return price - s, price + t
        return price + s, price - t

    # ---- circuit breakers ----
    def check_circuit_breakers(self, equity: float, now: dt.datetime) -> dict:
        """Returns {'trade_allowed': bool, 'close_all': bool, 'reason': str}"""
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        today = now.date()
        if self.day != today:
            self.day = today
            self.day_start_equity = equity

        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0
        if dd >= self.r["max_drawdown_pct"]:
            self.killed = True
            return {"trade_allowed": False, "close_all": True,
                    "reason": f"KILL SWITCH: drawdown {dd:.1%} >= {self.r['max_drawdown_pct']:.0%}"}
        if self.killed:
            return {"trade_allowed": False, "close_all": False, "reason": "kill switch active"}

        day_loss = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity else 0
        if day_loss >= self.r["max_daily_loss_pct"]:
            return {"trade_allowed": False, "close_all": False,
                    "reason": f"daily loss limit {day_loss:.1%} hit - paused until tomorrow"}
        return {"trade_allowed": True, "close_all": False, "reason": ""}

    def update_trailing_stop(self, pos: dict, price: float, atr_value: float):
        if not self.r.get("trailing_stop"):
            return
        dist = self.r["atr_stop_multiplier"] * atr_value
        if pos["side"] == "long":
            pos["stop"] = max(pos["stop"], price - dist)
        else:
            pos["stop"] = min(pos["stop"], price + dist)
