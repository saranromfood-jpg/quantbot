"""Strategy modules. Each returns a score in [-1, +1] per bar.
+1 = strong long, -1 = strong short, 0 = no view.
The ensemble combines weighted scores into one decision."""
import numpy as np
import pandas as pd


class TrendMomentum:
    """EMA trend + RSI/MACD momentum, filtered by ADX (trade only in trends)."""

    def __init__(self, p: dict):
        self.p = p

    def score(self, df: pd.DataFrame) -> float:
        r = df.iloc[-1]
        if r["adx"] < self.p["adx_threshold"]:
            return 0.0
        s = 0.0
        s += 0.40 if r["ema_fast"] > r["ema_slow"] else -0.40
        s += 0.30 if r["macd_hist"] > 0 else -0.30
        if r["rsi"] > 55:
            s += 0.30 * min((r["rsi"] - 55) / 20, 1)
        elif r["rsi"] < 45:
            s -= 0.30 * min((45 - r["rsi"]) / 20, 1)
        return float(np.clip(s, -1, 1))


class MeanReversion:
    """Bollinger/z-score reversal, only in low-ADX (range-bound) regimes."""

    def __init__(self, p: dict):
        self.p = p

    def score(self, df: pd.DataFrame) -> float:
        r = df.iloc[-1]
        if r["adx"] > self.p["adx_max"]:
            return 0.0  # trending market -> don't fade
        z = r["zscore"]
        if z <= -self.p["zscore_entry"]:
            return float(np.clip(-z / 3.0, 0, 1))      # oversold -> long
        if z >= self.p["zscore_entry"]:
            return float(np.clip(-z / 3.0, -1, 0))     # overbought -> short
        return 0.0


class MLFactor:
    """Gradient boosting on multi-factor features, predicts next-bar direction.
    Retrains periodically on a rolling window (walk-forward)."""

    FEATURES = ["rsi", "macd_hist", "adx", "zscore", "ret_1", "ret_4", "ret_16", "vol_16", "vol_ratio"]

    def __init__(self, p: dict):
        self.p = p
        self.model = None
        self.bars_since_train = 10**9

    def _fit(self, df: pd.DataFrame):
        from sklearn.ensemble import GradientBoostingClassifier
        d = df.dropna(subset=self.FEATURES).copy()
        d["y"] = (d["close"].shift(-1) > d["close"]).astype(int)
        d = d.iloc[:-1].tail(self.p["lookback_bars"])
        if len(d) < 300:
            return
        m = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8)
        m.fit(d[self.FEATURES], d["y"])
        self.model = m
        self.bars_since_train = 0

    def score(self, df: pd.DataFrame) -> float:
        if self.model is None or self.bars_since_train >= self.p["retrain_every_bars"]:
            self._fit(df)
        self.bars_since_train += 1
        if self.model is None:
            return 0.0
        x = df[self.FEATURES].iloc[[-1]]
        if x.isna().any().any():
            return 0.0
        p_up = float(self.model.predict_proba(x)[0][1])
        conf = self.p["min_confidence"]
        if p_up >= conf:
            return (p_up - 0.5) * 2
        if p_up <= 1 - conf:
            return (p_up - 0.5) * 2
        return 0.0


class Ensemble:
    """Weighted vote across strategies -> final decision."""

    def __init__(self, cfg: dict):
        sc = cfg["strategies"]
        self.cfg = cfg["ensemble"]
        self.members = []
        if sc["trend_momentum"]["enabled"]:
            self.members.append((TrendMomentum(sc["trend_momentum"]), sc["trend_momentum"]["weight"]))
        if sc["mean_reversion"]["enabled"]:
            self.members.append((MeanReversion(sc["mean_reversion"]), sc["mean_reversion"]["weight"]))
        if sc["ml_factor"]["enabled"]:
            self.members.append((MLFactor(sc["ml_factor"]), sc["ml_factor"]["weight"]))

    def decide(self, df: pd.DataFrame) -> dict:
        total_w = sum(w for _, w in self.members) or 1.0
        parts = {type(s).__name__: s.score(df) for s, w in self.members}
        score = sum(parts[type(s).__name__] * w for s, w in self.members) / total_w
        if score >= self.cfg["long_threshold"]:
            action = "long"
        elif score <= self.cfg["short_threshold"] and self.cfg["allow_short"]:
            action = "short"
        else:
            action = "flat"
        return {"action": action, "score": round(float(score), 4), "parts": {k: round(v, 4) for k, v in parts.items()}}
