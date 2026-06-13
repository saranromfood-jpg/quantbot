"""ตัวสร้างข้อมูลจำลองหลาย regime ~5 ปี สำหรับ 'ทดสอบ framework' เท่านั้น
*** ไม่ใช่ข้อมูลจริง — ตัวเลขผลตอบแทนที่ได้ใช้พิสูจน์ว่าโค้ดทำงานถูก ไม่ใช่ผลงานจริง ***
สร้าง regime จริงจัง: bull 2021, bear 2022 (มี flash crash), sideways 2023, recovery 2024-25
"""
import numpy as np
import pandas as pd

BARS_PER_YEAR = 365 * 24 * 4  # 15m


def synth_asset(seed: int, p0: float, years: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(BARS_PER_YEAR * years)
    # regime timeline เป็นสัดส่วนของ 5 ปี
    segments = [
        ("bull", 0.20, 0.00045, 0.004),
        ("bear", 0.20, -0.00040, 0.006),
        ("sideways", 0.25, 0.00000, 0.0035),
        ("recovery", 0.20, 0.00030, 0.005),
        ("bull2", 0.15, 0.00040, 0.0045),
    ]
    rets = []
    for name, frac, mu, sigma in segments:
        m = int(n * frac)
        seg = rng.normal(mu, sigma, m)
        # แทรก flash crash ในช่วง bear
        if name == "bear":
            crash_at = rng.integers(0, m - 50)
            seg[crash_at:crash_at + 5] += rng.normal(-0.05, 0.01, 5)
        rets.extend(seg)
    rets = np.array(rets[:n])
    # volatility clustering (GARCH-ish)
    vol = np.ones(n)
    for i in range(1, n):
        vol[i] = 0.94 * vol[i - 1] + 0.06 * (abs(rets[i - 1]) / 0.004)
    rets = rets * np.clip(vol, 0.5, 3.0)
    close = p0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0015, n)))
    open_ = np.roll(close, 1); open_[0] = p0
    vol_series = rng.lognormal(10, 0.6, n) * np.clip(vol, 0.5, 3.0)
    idx = pd.date_range("2021-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol_series}, index=idx)


def build_universe(years: float = 5.0) -> dict:
    base = {"BTC/THB": 1000000, "ETH/THB": 70000, "USDT/THB": 35,
            "ADA/THB": 15, "XRP/THB": 18}
    return {s: synth_asset(i, p, years) for i, (s, p) in enumerate(base.items())}


if __name__ == "__main__":
    from pathlib import Path
    d = build_universe()
    out = Path("data"); out.mkdir(exist_ok=True)
    for s, df in d.items():
        df.to_parquet(out / f"{s.replace('/','_')}_15m_SYNTH.parquet")
        print(f"{s}: {len(df):,} แท่ง {df.index[0].date()}→{df.index[-1].date()}")
