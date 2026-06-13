"""Strategy modules. Each returns a score in [-1, +1] per bar.
+1 = strong long, -1 = strong short, 0 = no view.

v2: regime-aware ensemble (ADX-switched weights) + confidence scaling,
    robust ML (calibration, sample weight, larger window, health checks)."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("quantbot.strategy")


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
    """Gradient boosting + isotonic calibration on multi-factor features.
    Predicts next-bar direction. Walk-forward retrain with recency weighting.

    v2 safety:
    - train_window ~4000 (~40d on 15m) to see more regimes
    - sample_weight: recent bars weighted higher (crypto regime shifts fast)
    - CalibratedClassifierCV: predicted prob actually means what it says
    - conf_threshold 0.62: trade only when genuinely confident (after fees)
    - health checks: class balance + feature importance logged each retrain
    - NO lookahead: label uses shift(-1), current open bar never used as feature
    """

    FEATURES = ["rsi", "macd_hist", "adx", "zscore", "ret_1", "ret_4", "ret_16", "vol_16", "vol_ratio"]

    def __init__(self, p: dict):
        self.p = p
        self.model = None
        self.bars_since_train = 10**9
        self.last_health = {}

    def _fit(self, df: pd.DataFrame):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier

        d = df.dropna(subset=self.FEATURES).copy()
        d["y"] = (d["close"].shift(-1) > d["close"]).astype(int)  # next bar up? no lookahead
        d = d.iloc[:-1].tail(self.p["train_window"])
        if len(d) < 500:
            return
        X, y = d[self.FEATURES], d["y"]
        up_frac = float(y.mean())                       # health: class balance
        weights = np.linspace(0.5, 1.0, len(y))         # recency weighting

        base = GradientBoostingClassifier(
            n_estimators=self.p.get("n_estimators", 200),
            max_depth=3, learning_rate=0.05, subsample=0.8)

        imp = None
        if self.p.get("calibrate", True) and 0.2 < up_frac < 0.8 and len(d) >= 1500:
            model = CalibratedClassifierCV(base, method="isotonic", cv=3)
            model.fit(X, y, sample_weight=weights)
            try:
                imp = model.calibrated_classifiers_[0].estimator.feature_importances_
            except Exception:
                imp = None
        else:
            base.fit(X, y, sample_weight=weights)
            model = base
            imp = base.feature_importances_

        self.model = model
        self.bars_since_train = 0
        top = None
        if imp is not None:
            pairs = sorted(zip(self.FEATURES, [round(float(v), 3) for v in imp]), key=lambda t: -t[1])
            top = pairs[:3]
        self.last_health = {"n": len(d), "up_frac": round(up_frac, 3),
                            "top_features": top, "warn_imbalance": not (0.35 < up_frac < 0.65)}
        if self.last_health["warn_imbalance"]:
            log.warning(f"ML class imbalance: up_frac={up_frac:.2f} (model may be biased)")

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
        conf = self.p["conf_threshold"]
        if p_up >= conf or p_up <= 1 - conf:
            return float(np.clip((p_up - 0.5) * 2, -1, 1))
        return 0.0


def regime_weights(adx: float, rcfg: dict) -> dict:
    """v2 ด้าน 1: สลับน้ำหนัก ensemble ตาม market regime (วัดด้วย ADX)."""
    if adx >= rcfg["adx_trend"]:
        return rcfg["weights_trending"]
    if adx < rcfg["adx_range"]:
        return rcfg["weights_ranging"]
    return rcfg["weights_transition"]


class Ensemble:
    """Regime-aware weighted vote -> decision + conviction-based size multiplier."""

    def __init__(self, cfg: dict):
        sc = cfg["strategies"]
        self.ecfg = cfg["ensemble"]
        self.rcfg = cfg["regime"]
        self.allow_short = (cfg.get("market", {}).get("type") == "futures") and self.ecfg.get("allow_short", False)
        self.trend = TrendMomentum(sc["trend_momentum"]) if sc["trend_momentum"]["enabled"] else None
        self.meanrev = MeanReversion(sc["mean_reversion"]) if sc["mean_reversion"]["enabled"] else None
        self.ml = MLFactor(cfg["ml_v2"]) if sc["ml_factor"]["enabled"] else None

    def decide(self, df: pd.DataFrame) -> dict:
        adx = float(df["adx"].iloc[-1])
        w = regime_weights(adx, self.rcfg)
        parts = {
            "trend": self.trend.score(df) if self.trend else 0.0,
            "meanrev": self.meanrev.score(df) if self.meanrev else 0.0,
            "ml": self.ml.score(df) if self.ml else 0.0,
        }
        total_w = sum(w.values()) or 1.0
        score = (w["trend"] * parts["trend"] + w["meanrev"] * parts["meanrev"]
                 + w["ml"] * parts["ml"]) / total_w

        if score >= self.ecfg["long_threshold"]:
            action = "long"
        elif score <= self.ecfg["short_threshold"] and self.allow_short:
            action = "short"
        else:
            action = "flat"

        thr = self.ecfg["long_threshold"]
        conviction = min(abs(score), 1.0)
        if self.rcfg.get("confidence_scaling") and conviction > thr:
            size_mult = (conviction - thr) / (1.0 - thr)
        else:
            size_mult = 1.0 if action != "flat" else 0.0
        size_mult = float(np.clip(size_mult, 0.0, 1.0))

        regime = ("trending" if adx >= self.rcfg["adx_trend"]
                  else "ranging" if adx < self.rcfg["adx_range"] else "transition")
        return {"action": action, "score": round(float(score), 4),
                "size_mult": round(size_mult, 3), "regime": regime,
                "parts": {k: round(v, 4) for k, v in parts.items()}}
