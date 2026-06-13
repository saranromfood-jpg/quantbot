"""Order execution. paper = simulated fills, live = real orders via ccxt.
v2: dynamic slippage (scales with volatility), realistic fees."""
import logging

log = logging.getLogger("quantbot.exec")


def estimate_slippage(atr: float, price: float, cfg: dict) -> float:
    """v2 ด้าน 3.1: slippage จริงพุ่งตามความผันผวน. คืนค่าเป็นสัดส่วน (0.0008 = 8bps)."""
    if price <= 0:
        return cfg["base_slippage_bps"] / 10000.0
    vol_ratio = atr / price
    dynamic_bps = cfg["base_slippage_bps"] + (vol_ratio * 10000 * cfg["slippage_vol_coef"])
    return min(dynamic_bps, cfg["slippage_cap_bps"]) / 10000.0


class Executor:
    def __init__(self, cfg: dict, feed):
        self.mode = cfg["trading"]["mode"]
        self.feed = feed
        self.ex_cfg = cfg["execution_v2"]

    def market_order(self, symbol: str, side: str, qty: float, price_hint: float, atr: float = 0.0) -> float:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if self.mode == "paper":
            slip = estimate_slippage(atr, price_hint, self.ex_cfg)
            fill = price_hint * (1 + slip) if side == "buy" else price_hint * (1 - slip)
            log.info(f"[PAPER] {side} {qty:.6f} {symbol} @ {fill:.4f} (slip {slip*10000:.1f}bps)")
            return fill
        order = self.feed.exchange.create_order(symbol, "market", side, qty)
        fill = float(order.get("average") or order.get("price") or price_hint)
        log.info(f"[LIVE] {side} {qty:.6f} {symbol} @ {fill:.4f} id={order.get('id')}")
        return fill
