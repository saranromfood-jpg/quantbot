"""Technical indicators (pure pandas/numpy, no TA-Lib dependency)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def bollinger(close: pd.Series, period=20, n_std=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + n_std * std, mid, mid - n_std * std


def zscore(close: pd.Series, period=20) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ((close - mean) / std.replace(0, np.nan)).fillna(0)


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach all indicator columns used by strategies."""
    tm = cfg["strategies"]["trend_momentum"]
    mr = cfg["strategies"]["mean_reversion"]
    out = df.copy()
    out["ema_fast"] = ema(out["close"], tm["ema_fast"])
    out["ema_slow"] = ema(out["close"], tm["ema_slow"])
    out["rsi"] = rsi(out["close"], tm["rsi_period"])
    out["macd"], out["macd_sig"], out["macd_hist"] = macd(out["close"])
    out["atr"] = atr(out)
    out["adx"] = adx(out)
    out["bb_up"], out["bb_mid"], out["bb_lo"] = bollinger(out["close"], mr["bb_period"], mr["bb_std"])
    out["zscore"] = zscore(out["close"], mr["bb_period"])
    out["ret_1"] = out["close"].pct_change()
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_16"] = out["close"].pct_change(16)
    out["vol_16"] = out["ret_1"].rolling(16).std()
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    return out
