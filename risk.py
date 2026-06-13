"""Risk management: position sizing, stops, kill switches.
v2: conviction-scaled sizing, time gate, consecutive-loss breaker, vol-spike halt."""
import datetime as dt


class RiskManager:
    def __init__(self, cfg: dict):
        self.r = cfg["risk"]
        self.ex = cfg["execution_v2"]
        self.bk = cfg["breakers_v2"]
        self.day = None
        self.day_start_equity = None
        self.peak_equity = None
        self.killed = False
        self.consec_losses = 0
        self.pause_until = None

    def position_size(self, equity: float, price: float, atr_value: float, size_mult: float = 1.0) -> float:
        """Volatility-based sizing scaled by signal conviction (size_mult 0..1)."""
        stop_dist = atr_value * self.r["atr_stop_multiplier"]
        if stop_dist <= 0:
            return 0.0
        qty = (equity * self.r["risk_per_trade"] * max(0.0, min(size_mult, 1.0))) / stop_dist
        max_qty = (equity * self.r["max_position_pct"]) / price
        return max(0.0, min(qty, max_qty))

    def stop_and_target(self, side: str, price: float, atr_value: float):
        s = self.r["atr_stop_multiplier"] * atr_value
        t = self.r["atr_target_multiplier"] * atr_value
        if side == "long":
            return price - s, price + t
        return price + s, price - t

    def entry_allowed(self, now: dt.datetime, atr_value: float, atr_avg_24h: float) -> dict:
        """ตรวจเงื่อนไขก่อนเปิดไม้ใหม่ (ไม่กระทบไม้ที่เปิดอยู่). v2 ด้าน 3.3/3.4"""
        lo, hi = self.ex.get("time_gate_utc", [None, None])
        if lo is not None and lo <= now.hour < hi:
            return {"ok": False, "reason": f"time gate {lo:02d}-{hi:02d} UTC (low liquidity)"}
        if self.pause_until and now < self.pause_until:
            return {"ok": False, "reason": "consec-loss pause"}
        if atr_avg_24h > 0 and atr_value > self.bk["vol_spike_multiple"] * atr_avg_24h:
            return {"ok": False, "reason": f"vol spike halt ({atr_value/atr_avg_24h:.1f}x 24h avg)"}
        return {"ok": True, "reason": ""}

    def record_trade_result(self, pnl: float, now: dt.datetime):
        """นับแพ้ติดกัน -> ตั้ง pause ถ้าถึงเกณฑ์."""
        if pnl <= 0:
            self.consec_losses += 1
            if self.consec_losses >= self.bk["consecutive_losses"]:
                self.pause_until = now + dt.timedelta(hours=self.bk["consecutive_pause_hours"])
                self.consec_losses = 0
        else:
            self.consec_losses = 0

    def check_circuit_breakers(self, equity: float, now: dt.datetime) -> dict:
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
                    "reason": f"KILL SWITCH: drawdown {dd:.1%}"}
        if self.killed:
            return {"trade_allowed": False, "close_all": False, "reason": "kill switch active"}
        day_loss = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity else 0
        if day_loss >= self.r["max_daily_loss_pct"]:
            return {"trade_allowed": False, "close_all": False,
                    "reason": f"daily loss limit {day_loss:.1%}"}
        return {"trade_allowed": True, "close_all": False, "reason": ""}

    def update_trailing_stop(self, pos: dict, price: float, atr_value: float):
        if not self.r.get("trailing_stop"):
            return
        dist = self.r["atr_stop_multiplier"] * atr_value
        if pos["side"] == "long":
            pos["stop"] = max(pos["stop"], price - dist)
        else:
            pos["stop"] = min(pos["stop"], price + dist)
