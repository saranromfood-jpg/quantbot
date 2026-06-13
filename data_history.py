"""ด้าน 5: Pipeline ดึงข้อมูลย้อนหลังหลายปี หลายเหรียญ เก็บเป็น parquet
รันบนเครื่องที่เข้าถึง exchange ได้ (เช่น เครื่องคุณ / Render):
    python data_history.py                 # ดึงตาม config
    python data_history.py --years 5 --tf 15m
ข้อมูล public ไม่ต้องใช้ API key. parquet เร็วและเล็กกว่า CSV มาก.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

DATA_DIR = Path(__file__).parent / "data"


def fetch_history(ex, symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    since = ex.parse8601((pd.Timestamp.utcnow() - pd.DateOffset(years=years)).isoformat())
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    rows, limit = [], 1000
    now = ex.milliseconds()
    while since < now:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        time.sleep(ex.rateLimit / 1000)
        if len(batch) < limit:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def quality_report(df: pd.DataFrame, symbol: str, timeframe: str):
    """เช็คคุณภาพ: gap + outlier (กับดักข้อ 4 ในแผน)"""
    if df.empty:
        print(f"  {symbol}: ⚠️ ไม่มีข้อมูล")
        return
    expected = pd.date_range(df.index[0], df.index[-1], freq=timeframe.replace("m", "min"))
    missing = len(expected) - len(df)
    pct = missing / len(expected) * 100 if len(expected) else 0
    # outlier: return เกิน 40% ในแท่งเดียว = น่าสงสัย
    ret = df["close"].pct_change().abs()
    spikes = int((ret > 0.40).sum())
    flag = "⚠️" if (pct > 1 or spikes > 0) else "✓"
    print(f"  {symbol}: {flag} {len(df):,} แท่ง | gap {missing} ({pct:.2f}%) | spike>40% {spikes} ไม้ "
          f"| {df.index[0].date()} → {df.index[-1].date()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--exchange", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    if args.exchange:
        cfg["exchange"]["id"] = args.exchange
    from data_feed import _make_exchange
    ex = _make_exchange(cfg)
    ex_id = cfg["exchange"]["id"]
    tf = args.tf or cfg["trading"]["timeframe"]
    # ใช้ universe กว้างกว่า symbols ที่เทรด เพื่อทดสอบ generalization (ข้าม survivorship ไม่ได้บน live exchange
    # แต่ครอบคลุมหลายเหรียญช่วยลด overfit รายเหรียญ)
    universe = cfg.get("backtest", {}).get("universe", cfg["trading"]["symbols"])

    DATA_DIR.mkdir(exist_ok=True)
    print(f"ดึงข้อมูล {args.years} ปี | tf={tf} | exchange={ex_id}")
    print(f"เหรียญ: {universe}\n")
    for sym in universe:
        try:
            df = fetch_history(ex, sym, tf, args.years)
            quality_report(df, sym, tf)
            out = DATA_DIR / f"{sym.replace('/', '_')}_{tf}.parquet"
            df.to_parquet(out)
        except Exception as e:
            print(f"  {sym}: ❌ {type(e).__name__}: {str(e)[:80]}")
    print(f"\nบันทึกที่ {DATA_DIR}/ — ใช้กับ backtest_wf.py ได้เลย")


if __name__ == "__main__":
    main()
