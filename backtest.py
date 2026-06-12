"""Walk-forward backtester.
Usage:
  python backtest.py                     # fetch real data from exchange
  python backtest.py --synthetic        # use synthetic data (no internet)
"""
import argparse
import logging

import numpy as np
import pandas as pd
import yaml

from indicators import enrich
from portfolio import Portfolio
from risk import RiskManager
from strategies import Ensemble

logging.basicConfig(level=logging.WARNING)


def synthetic_ohlcv(bars=4000, seed=42, p0=50000.0):
    rng = np.random.default_rng(seed)
    # regime-switching returns: trends + ranges
    rets = []
    i = 0
    while len(rets) < bars:
        regime = rng.choice(["up", "down", "range"], p=[0.35, 0.25, 0.40])
        n = int(rng.integers(50, 300))
        mu = {"up": 0.0004, "down": -0.0004, "range": 0.0}[regime]
        rets.extend(rng.normal(mu, 0.004, n))
        i += 1
    rets = np.array(rets[:bars])
    close = p0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, bars)))
    open_ = np.roll(close, 1); open_[0] = p0
    vol = rng.lognormal(10, 0.5, bars)
    idx = pd.date_range("2025-01-01", periods=bars, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def run_backtest(cfg, data: dict):
    pf = Portfolio(cfg["trading"]["initial_capital"])
    risk = RiskManager(cfg)
    ens = {s: Ensemble(cfg) for s in data}
    warmup = 200
    n = min(len(df) for df in data.values())
    enriched = {s: enrich(df, cfg) for s, df in data.items()}

    for i in range(warmup, n):
        prices, ts = {}, None
        for sym, df in enriched.items():
            win = df.iloc[: i + 1]
            r = win.iloc[-1]
            ts = win.index[-1]
            price, atr_v = float(r["close"]), float(r["atr"])
            prices[sym] = price

            pos = pf.positions.get(sym)
            if pos:
                risk.update_trailing_stop(pos, price, atr_v)
                hit_stop = price <= pos["stop"] if pos["side"] == "long" else price >= pos["stop"]
                hit_tgt = price >= pos["target"] if pos["side"] == "long" else price <= pos["target"]
                if hit_stop or hit_tgt:
                    pf.close(sym, price, ts, "stop" if hit_stop else "target")

            cb = risk.check_circuit_breakers(pf.equity(prices), ts.to_pydatetime())
            if cb["close_all"]:
                for s2 in list(pf.positions):
                    pf.close(s2, prices.get(s2, pf.positions[s2]["entry"]), ts, "kill_switch")
            if not cb["trade_allowed"]:
                continue

            d = ens[sym].decide(win)
            pos = pf.positions.get(sym)
            if pos and d["action"] not in ("flat", pos["side"]):
                pf.close(sym, price, ts, "signal_flip")
                pos = None
            if not pos and d["action"] in ("long", "short") and len(pf.positions) < cfg["risk"]["max_open_positions"]:
                qty = risk.position_size(pf.equity(prices), price, atr_v)
                if qty * price >= 10:
                    stop, target = risk.stop_and_target(d["action"], price, atr_v)
                    pf.open(sym, d["action"], qty, price, stop, target, ts)
        pf.record_equity(ts, prices)

    # close remaining
    for s in list(pf.positions):
        pf.close(s, prices[s], ts, "end_of_test")
    return pf


def report(pf: Portfolio):
    eq = pd.DataFrame(pf.equity_curve, columns=["ts", "equity"])
    eq["ts"] = pd.to_datetime(eq["ts"])
    rets = eq["equity"].pct_change().dropna()
    total = eq["equity"].iloc[-1] / pf.initial - 1
    bars_per_year = 365 * 24 * 4  # 15m bars
    sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0
    peak = eq["equity"].cummax()
    mdd = ((eq["equity"] - peak) / peak).min()
    wins = [t["pnl"] for t in pf.trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in pf.trades if t["pnl"] <= 0]
    pfactor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    print("\n========== BACKTEST REPORT ==========")
    print(f"Final equity   : {eq['equity'].iloc[-1]:,.2f} USDT")
    print(f"Total return   : {total:+.2%}")
    print(f"Sharpe (ann.)  : {sharpe:.2f}")
    print(f"Max drawdown   : {mdd:.2%}")
    print(f"Trades         : {len(pf.trades)}  (win rate {len(wins)/len(pf.trades)*100:.1f}%)" if pf.trades else "Trades         : 0")
    print(f"Profit factor  : {pfactor:.2f}" if pf.trades else "")
    print("=====================================\n")
    pf.save({}, status="backtest_done")
    print("saved state.json -> เปิด dashboard ดูผลได้: python dashboard/app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--bars", type=int, default=4000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    if args.synthetic:
        data = {s: synthetic_ohlcv(args.bars, seed=i, p0=p)
                for i, (s, p) in enumerate({"BTC/USDT": 50000, "ETH/USDT": 3000, "SOL/USDT": 150}.items())}
        print("using SYNTHETIC data (สำหรับทดสอบระบบเท่านั้น ไม่ใช่ผลตอบแทนจริง)")
    else:
        from data_feed import DataFeed
        feed = DataFeed(cfg)
        data = {s: feed.ohlcv_history(s, args.bars) for s in cfg["trading"]["symbols"]}
        print(f"fetched {args.bars} bars per symbol from {cfg['exchange']['id']}")
    pf = run_backtest(cfg, data)
    report(pf)
