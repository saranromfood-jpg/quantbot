"""Main trading loop: data -> indicators -> regime ensemble -> risk -> execution.
v2: conviction sizing, dynamic slippage, time/vol/consec-loss gates, funding (futures)."""
import datetime as dt
import logging
import time

from data_feed import DataFeed
from execution import Executor
from indicators import enrich
from portfolio import Portfolio
from risk import RiskManager
from strategies import Ensemble

log = logging.getLogger("quantbot")


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.feed = DataFeed(cfg)
        self.executor = Executor(cfg, self.feed)
        self.risk = RiskManager(cfg)
        self.portfolio = Portfolio(cfg["trading"]["initial_capital"])
        self.ensembles = {s: Ensemble(cfg) for s in cfg["trading"]["symbols"]}
        self.fee_rate = cfg["execution_v2"]["fee_rate"]
        self.is_futures = cfg.get("market", {}).get("type") == "futures"
        self.funding = cfg["execution_v2"]["funding_rate_per_8h"]
        self._last_funding_hour = None

    def step(self):
        cfg_t = self.cfg["trading"]
        now = dt.datetime.now(dt.timezone.utc)
        prices = {}

        # funding for futures (every 8h at 0/8/16 UTC)
        if self.is_futures and now.hour in (0, 8, 16) and now.hour != self._last_funding_hour:
            # need marks; gather after price loop. mark here with last known entries as fallback
            self._last_funding_hour = now.hour
            self._apply_funding_pending = True
        elif now.hour not in (0, 8, 16):
            self._last_funding_hour = None

        for symbol in cfg_t["symbols"]:
            df = enrich(self.feed.ohlcv(symbol, limit=400), self.cfg)
            r = df.iloc[-1]
            price = float(r["close"])
            atr_v = float(r["atr"])
            atr_avg = float(r["atr_avg_24h"]) if not (r["atr_avg_24h"] != r["atr_avg_24h"]) else atr_v
            prices[symbol] = price
            if atr_v <= 0:
                continue

            # 1) manage open position
            pos = self.portfolio.positions.get(symbol)
            if pos:
                self.risk.update_trailing_stop(pos, price, atr_v)
                hit_stop = price <= pos["stop"] if pos["side"] == "long" else price >= pos["stop"]
                hit_tgt = price >= pos["target"] if pos["side"] == "long" else price <= pos["target"]
                if hit_stop or hit_tgt:
                    side = "sell" if pos["side"] == "long" else "buy"
                    fill = self.executor.market_order(symbol, side, pos["qty"], price, atr_v)
                    t = self.portfolio.close(symbol, fill, now, "stop" if hit_stop else "target", fee_rate=self.fee_rate)
                    if t:
                        self.risk.record_trade_result(t["pnl"], now)
                        log.info(f"closed {symbol}: pnl={t['pnl']}")

            # 2) circuit breakers
            cb = self.risk.check_circuit_breakers(self.portfolio.equity(prices), now)
            if cb["close_all"]:
                for sym in list(self.portfolio.positions):
                    p = self.portfolio.positions[sym]
                    side = "sell" if p["side"] == "long" else "buy"
                    fill = self.executor.market_order(sym, side, p["qty"], prices.get(sym, p["entry"]), atr_v)
                    self.portfolio.close(sym, fill, now, "kill_switch", fee_rate=self.fee_rate)
                log.error(cb["reason"])
            if not cb["trade_allowed"]:
                if cb["reason"]:
                    log.warning(cb["reason"])
                continue

            # 3) regime-aware signal
            decision = self.ensembles[symbol].decide(df)
            self.portfolio.signals[symbol] = {**decision, "price": price, "ts": str(now)}

            pos = self.portfolio.positions.get(symbol)
            if pos and decision["action"] != pos["side"] and decision["action"] != "flat":
                side = "sell" if pos["side"] == "long" else "buy"
                fill = self.executor.market_order(symbol, side, pos["qty"], price, atr_v)
                t = self.portfolio.close(symbol, fill, now, "signal_flip", fee_rate=self.fee_rate)
                if t:
                    self.risk.record_trade_result(t["pnl"], now)
                pos = None

            if (not pos and decision["action"] in ("long", "short")
                    and len(self.portfolio.positions) < self.cfg["risk"]["max_open_positions"]):
                gate = self.risk.entry_allowed(now, atr_v, atr_avg)
                if not gate["ok"]:
                    log.info(f"{symbol} entry blocked: {gate['reason']}")
                else:
                    equity = self.portfolio.equity(prices)
                    qty = self.risk.position_size(equity, price, atr_v, decision["size_mult"])
                    if qty * price >= self.cfg["trading"].get("min_notional", 10):
                        stop, target = self.risk.stop_and_target(decision["action"], price, atr_v)
                        side = "buy" if decision["action"] == "long" else "sell"
                        fill = self.executor.market_order(symbol, side, qty, price, atr_v)
                        self.portfolio.open(symbol, decision["action"], qty, fill, stop, target, now, fee_rate=self.fee_rate)
                        log.info(f"opened {decision['action']} {symbol} [{decision['regime']}] "
                                 f"qty={qty:.6f} @ {fill:.4f} score={decision['score']} size_mult={decision['size_mult']}")

        # apply funding now that we have marks
        if self.is_futures and getattr(self, "_apply_funding_pending", False):
            self.portfolio.apply_funding(prices, self.funding)
            self._apply_funding_pending = False

        self.portfolio.record_equity(now, prices)
        self.portfolio.save(prices, status="killed" if self.risk.killed else "running")

    def run(self):
        poll = self.cfg["trading"]["poll_seconds"]
        mode = self.cfg["trading"]["mode"]
        log.info(f"QuantBot v2 started | mode={mode} | market={self.cfg['market']['type']} "
                 f"| symbols={self.cfg['trading']['symbols']}")
        if mode == "live":
            log.warning("*** LIVE MODE - real money at risk ***")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                log.info("stopped by user")
                break
            except Exception as e:
                log.exception(f"step error: {e}")
            time.sleep(poll)
