"""Main trading loop: data -> indicators -> ensemble -> risk -> execution."""
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

    def step(self):
        cfg_t = self.cfg["trading"]
        now = dt.datetime.now(dt.timezone.utc)
        prices = {}

        for symbol in cfg_t["symbols"]:
            df = enrich(self.feed.ohlcv(symbol, limit=400), self.cfg)
            price = float(df["close"].iloc[-1])
            atr_v = float(df["atr"].iloc[-1])
            prices[symbol] = price

            # 1) manage open position (stop / target / trailing)
            pos = self.portfolio.positions.get(symbol)
            if pos:
                self.risk.update_trailing_stop(pos, price, atr_v)
                hit_stop = price <= pos["stop"] if pos["side"] == "long" else price >= pos["stop"]
                hit_tgt = price >= pos["target"] if pos["side"] == "long" else price <= pos["target"]
                if hit_stop or hit_tgt:
                    side = "sell" if pos["side"] == "long" else "buy"
                    fill = self.executor.market_order(symbol, side, pos["qty"], price)
                    t = self.portfolio.close(symbol, fill, now, "stop" if hit_stop else "target")
                    log.info(f"closed {symbol}: {t}")

            # 2) circuit breakers
            cb = self.risk.check_circuit_breakers(self.portfolio.equity(prices), now)
            if cb["close_all"]:
                for sym in list(self.portfolio.positions):
                    p = self.portfolio.positions[sym]
                    side = "sell" if p["side"] == "long" else "buy"
                    fill = self.executor.market_order(sym, side, p["qty"], prices.get(sym, p["entry"]))
                    self.portfolio.close(sym, fill, now, "kill_switch")
                log.error(cb["reason"])
            if not cb["trade_allowed"]:
                if cb["reason"]:
                    log.warning(cb["reason"])
                continue

            # 3) new signal
            decision = self.ensembles[symbol].decide(df)
            self.portfolio.signals[symbol] = {**decision, "price": price, "ts": str(now)}

            pos = self.portfolio.positions.get(symbol)
            if pos and decision["action"] != pos["side"] and decision["action"] != "flat":
                # signal flipped -> close
                side = "sell" if pos["side"] == "long" else "buy"
                fill = self.executor.market_order(symbol, side, pos["qty"], price)
                self.portfolio.close(symbol, fill, now, "signal_flip")
                pos = None

            if (not pos and decision["action"] in ("long", "short")
                    and len(self.portfolio.positions) < self.cfg["risk"]["max_open_positions"]):
                equity = self.portfolio.equity(prices)
                qty = self.risk.position_size(equity, price, atr_v)
                if qty * price >= 10:  # exchange min notional
                    stop, target = self.risk.stop_and_target(decision["action"], price, atr_v)
                    side = "buy" if decision["action"] == "long" else "sell"
                    fill = self.executor.market_order(symbol, side, qty, price)
                    self.portfolio.open(symbol, decision["action"], qty, fill, stop, target, now)
                    log.info(f"opened {decision['action']} {symbol} qty={qty:.6f} @ {fill:.2f} "
                             f"stop={stop:.2f} target={target:.2f} score={decision['score']}")

        self.portfolio.record_equity(now, prices)
        self.portfolio.save(prices, status="killed" if self.risk.killed else "running")

    def run(self):
        poll = self.cfg["trading"]["poll_seconds"]
        mode = self.cfg["trading"]["mode"]
        log.info(f"QuantBot started | mode={mode} | symbols={self.cfg['trading']['symbols']}")
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
