"""ด้าน 4: Walk-Forward Backtest Framework + full metrics + OOS guard.

ใช้ strategy/risk/execution ตัวเดียวกับ live เป๊ะ (same code path).
ML เทรนบนข้อมูลอดีตเท่านั้น (causal) จึงไม่มี lookahead.

วิธีใช้:
  python backtest_wf.py --synth                 # ทดสอบ framework ด้วยข้อมูลจำลอง
  python backtest_wf.py                          # ใช้ parquet จริงใน data/ (ดึงด้วย data_history.py)
  python backtest_wf.py --assets BTC/USDT --bars 20000 --fast
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from indicators import enrich
from portfolio import Portfolio
from risk import RiskManager
from strategies import Ensemble

DATA = Path(__file__).parent / "data"


# ---------------- metrics ----------------
def compute_metrics(eq: pd.Series, trades: list, bars_per_year: int, exposure_bars: int, total_bars: int):
    rets = eq.pct_change().dropna()
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    ann = np.sqrt(bars_per_year)
    sharpe = rets.mean() / rets.std() * ann if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = rets.mean() / downside.std() * ann if len(downside) > 1 and downside.std() > 0 else 0.0
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min() if len(eq) else 0.0
    calmar = (total_ret / abs(mdd)) if mdd < 0 else 0.0
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    # max consecutive losses
    mcl = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        mcl = max(mcl, cur)
    fees = sum(t.get("fee_paid", 0) for t in trades)
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "n_trades": len(pnls),
        "max_consec_losses": mcl,
        "exposure_pct": round(exposure_bars / total_bars * 100, 1) if total_bars else 0.0,
        "fee_drag": round(fees, 2),
    }


# ---------------- single causal pass over one asset ----------------
def run_asset(cfg, df: pd.DataFrame):
    pf = Portfolio(cfg["trading"]["initial_capital"])
    risk = RiskManager(cfg)
    ens = Ensemble(cfg)
    fee_rate = cfg["execution_v2"]["fee_rate"]
    is_futures = cfg.get("market", {}).get("type") == "futures"
    funding = cfg["execution_v2"]["funding_rate_per_8h"]
    from execution import estimate_slippage
    ex_cfg = cfg["execution_v2"]

    d = enrich(df, cfg)
    warmup = 250
    sym = "ASSET"
    exposure = 0
    last_funding_hour = None

    for i in range(warmup, len(d)):
        win = d.iloc[: i + 1]
        r = win.iloc[-1]
        ts = win.index[-1]
        price, atr_v = float(r["close"]), float(r["atr"])
        atr_avg = float(r["atr_avg_24h"]) if not np.isnan(r["atr_avg_24h"]) else atr_v
        prices = {sym: price}
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        # funding (futures only, every 8h at 0/8/16 UTC)
        if is_futures and ts.hour in (0, 8, 16) and ts.hour != last_funding_hour:
            pf.apply_funding(prices, funding)
            last_funding_hour = ts.hour
        elif ts.hour not in (0, 8, 16):
            last_funding_hour = None

        # manage open position
        pos = pf.positions.get(sym)
        if pos:
            exposure += 1
            risk.update_trailing_stop(pos, price, atr_v)
            hit_stop = price <= pos["stop"] if pos["side"] == "long" else price >= pos["stop"]
            hit_tgt = price >= pos["target"] if pos["side"] == "long" else price <= pos["target"]
            if hit_stop or hit_tgt:
                slip = estimate_slippage(atr_v, price, ex_cfg)
                fill = price * (1 - slip) if pos["side"] == "long" else price * (1 + slip)
                t = pf.close(sym, fill, ts, "stop" if hit_stop else "target", fee_rate=fee_rate)
                if t:
                    t["fee_paid"] = pos["qty"] * fill * fee_rate
                    risk.record_trade_result(t["pnl"], ts.to_pydatetime())

        # portfolio breakers
        cb = risk.check_circuit_breakers(pf.equity(prices), ts.to_pydatetime())
        if cb["close_all"]:
            for s2 in list(pf.positions):
                pf.close(s2, prices.get(s2, pf.positions[s2]["entry"]), ts, "kill_switch", fee_rate=fee_rate)
        if not cb["trade_allowed"]:
            pf.record_equity(ts, prices)
            continue

        dec = ens.decide(win)
        pos = pf.positions.get(sym)
        if pos and dec["action"] not in ("flat", pos["side"]):
            slip = estimate_slippage(atr_v, price, ex_cfg)
            fill = price * (1 - slip) if pos["side"] == "long" else price * (1 + slip)
            t = pf.close(sym, fill, ts, "signal_flip", fee_rate=fee_rate)
            if t:
                t["fee_paid"] = pos["qty"] * fill * fee_rate
                risk.record_trade_result(t["pnl"], ts.to_pydatetime())
            pos = None

        if not pos and dec["action"] in ("long", "short"):
            gate = risk.entry_allowed(ts.to_pydatetime(), atr_v, atr_avg)
            if gate["ok"]:
                qty = risk.position_size(pf.equity(prices), price, atr_v, dec["size_mult"])
                if qty * price >= cfg["trading"].get("min_notional", 10):
                    stop, target = risk.stop_and_target(dec["action"], price, atr_v)
                    slip = estimate_slippage(atr_v, price, ex_cfg)
                    fill = price * (1 + slip) if dec["action"] == "long" else price * (1 - slip)
                    pf.open(sym, dec["action"], qty, fill, stop, target, ts, fee_rate=fee_rate)
        pf.record_equity(ts, prices)

    eq = pd.DataFrame(pf.equity_curve, columns=["ts", "equity"])
    eq["ts"] = pd.to_datetime(eq["ts"]); eq = eq.set_index("ts")["equity"]
    return pf, eq, exposure, len(d) - warmup


def split_oos(eq: pd.Series, trades: list, holdout_pct: float):
    """แยก in-sample / out-of-sample ตามเวลา. OOS = ส่วนท้ายที่ 'ห้ามแตะ' จน finalize."""
    if len(eq) < 10:
        return eq, eq, trades, trades
    cut = eq.index[int(len(eq) * (1 - holdout_pct))]
    is_eq, oos_eq = eq[eq.index < cut], eq[eq.index >= cut]
    is_tr = [t for t in trades if pd.to_datetime(t["closed_at"]) < cut]
    oos_tr = [t for t in trades if pd.to_datetime(t["closed_at"]) >= cut]
    return is_eq, oos_eq, is_tr, oos_tr


def fmt(m: dict) -> str:
    pf = m["profit_factor"]
    return (f"ret {m['total_return_pct']:+.1f}% | Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} "
            f"| Calmar {m['calmar']:.2f} | MaxDD {m['max_drawdown_pct']:.1f}% "
            f"| WR {m['win_rate_pct']:.0f}% | PF {pf} | trades {m['n_trades']} "
            f"| maxConsecL {m['max_consec_losses']} | expo {m['exposure_pct']:.0f}% | fees {m['fee_drag']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--bars", type=int, default=0, help="cap bars per asset (0=all)")
    ap.add_argument("--fast", action="store_true", help="lighter ML for quick framework test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    if args.fast:
        cfg["ml_v2"]["calibrate"] = False
        cfg["ml_v2"]["retrain_every_bars"] = 192
        cfg["ml_v2"]["train_window"] = 2500

    pattern = "*_SYNTH.parquet" if args.synth else "*.parquet"
    files = sorted(glob.glob(str(DATA / pattern)))
    if args.synth:
        files = [f for f in files if "SYNTH" in f]
    else:
        files = [f for f in files if "SYNTH" not in f]
    if not files:
        print(f"ไม่พบไฟล์ข้อมูลใน {DATA}/ — รัน data_history.py (จริง) หรือ synth_data.py (จำลอง) ก่อน")
        return

    bars_per_year = 365 * 24 * 4
    holdout = cfg["backtest"]["oos_holdout_pct"]
    print("=" * 78)
    print(f"WALK-FORWARD / OOS BACKTEST  ({'SYNTH (จำลอง — ไม่ใช่ผลจริง)' if args.synth else 'REAL data'})")
    print(f"market={cfg['market']['type']} | holdout OOS={holdout:.0%} | fast={args.fast}")
    print("=" * 78)

    all_is, all_oos = [], []
    for f in files:
        name = Path(f).stem.replace("_SYNTH", "")
        if args.assets and not any(a.replace("/", "_") in name for a in args.assets):
            continue
        df = pd.read_parquet(f)
        if args.bars:
            df = df.tail(args.bars)
        pf, eq, expo, nbars = run_asset(cfg, df)
        is_eq, oos_eq, is_tr, oos_tr = split_oos(eq, pf.trades, holdout)
        # exposure approximated proportional to bars in each split
        is_n = int(nbars * (1 - holdout)); oos_n = nbars - is_n
        m_is = compute_metrics(is_eq, is_tr, bars_per_year, int(expo * (1 - holdout)), max(is_n, 1))
        m_oos = compute_metrics(oos_eq, oos_tr, bars_per_year, int(expo * holdout), max(oos_n, 1))
        all_is.append(m_is); all_oos.append(m_oos)
        print(f"\n[{name}]")
        print(f"  IN-SAMPLE : {fmt(m_is)}")
        print(f"  OUT-SAMPLE: {fmt(m_oos)}")

    def agg(ms, key):
        vals = [m[key] for m in ms if m[key] is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    if all_oos:
        print("\n" + "=" * 78)
        print("สรุปเฉลี่ยทุกเหรียญ (ดู OUT-OF-SAMPLE เป็นหลัก — นี่คือผลที่เชื่อได้):")
        for label, ms in [("IN-SAMPLE ", all_is), ("OUT-SAMPLE", all_oos)]:
            print(f"  {label}: Sharpe {agg(ms,'sharpe')} | Sortino {agg(ms,'sortino')} | "
                  f"Calmar {agg(ms,'calmar')} | ret {agg(ms,'total_return_pct')}% | "
                  f"MaxDD {agg(ms,'max_drawdown_pct')}% | WR {agg(ms,'win_rate_pct')}%")
        print("=" * 78)


if __name__ == "__main__":
    main()
