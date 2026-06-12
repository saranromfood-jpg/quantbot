"""Order execution. paper = simulated fills, live = real orders via ccxt."""
import logging

log = logging.getLogger("quantbot.exec")


class Executor:
    def __init__(self, cfg: dict, feed):
        self.mode = cfg["trading"]["mode"]
        self.feed = feed

    def market_order(self, symbol: str, side: str, qty: float, price_hint: float) -> float:
        """Returns fill price. side: 'buy' | 'sell'"""
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if self.mode == "paper":
            # simulate slippage 0.05%
            slip = 1.0005 if side == "buy" else 0.9995
            fill = price_hint * slip
            log.info(f"[PAPER] {side} {qty:.6f} {symbol} @ {fill:.2f}")
            return fill
        order = self.feed.exchange.create_order(symbol, "market", side, qty)
        fill = float(order.get("average") or order.get("price") or price_hint)
        log.info(f"[LIVE] {side} {qty:.6f} {symbol} @ {fill:.2f} id={order.get('id')}")
        return fill
