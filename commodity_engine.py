"""Commodity/Index trend portfolio bot (paper) — runs alongside the crypto bot.
Self-contained: Yahoo ETF data + EMA200 trend + SPY market filter. Writes state_comm.json.
Validated config: 12.5% x 6 + SPY filter (Sharpe 0.36, DD ~23% on 20y backtest)."""
import datetime as dt
import json
import logging
import os
import time
import urllib.request

import numpy as np
import pandas as pd

log = logging.getLogger("quantbot.commodity")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_comm.json")

UNIVERSE = {  # display -> Yahoo ticker (ETFs = clean daily data)
    "GOLD": "GLD", "SILVER": "SLV", "OIL": "USO", "NATGAS": "UNG",
    "COMMOD": "DBC", "AGRI": "DBA", "GOLD_MINERS": "GDX",
    "SP500": "SPY", "NASDAQ100": "QQQ", "DOW": "DIA", "RUSSELL2000": "IWM",
    "US20Y_BOND": "TLT",
}
ALLOC = 0.125
MAXPOS = 6
TRAIL = 6.0
FEE = 0.0003
SLIP = 0.0002
MARKET_SYM = "SP500"   # regime filter: hold only when SPY > its EMA200


def _yahoo(ticker, years=3):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={years}y&interval=1d"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read()
    r = json.loads(raw)["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"high": q["high"], "low": q["low"], "close": q["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True)).dropna()
    df = df[df["close"] > 0]
    ret = df["close"].pct_change().abs()
    return df[(ret < 0.40) | ret.isna()]


def _ind(df):
    d = df.copy()
    d["ema"] = d["close"].ewm(span=200, adjust=False).mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def _slip(a, p):
    return min(0.005, SLIP + (a / p) * 0.15) if p > 0 else SLIP


class CommodityEngine:
    def __init__(self, cfg):
        self.cap = float(cfg["trading"].get("initial_capital", 100000))
        self.cash = self.cap
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signals = {}
        self.peak = self.cap
        self.poll = 3600

    def _load(self):
        out = {}
        for name, tk in UNIVERSE.items():
            try:
                df = _yahoo(tk)
                if len(df) > 210:
                    out[name] = _ind(df)
            except Exception as e:
                log.warning(f"comm load {name} fail: {type(e).__name__}")
        return out

    def equity(self, px):
        return self.cash + sum(p["qty"] * px.get(s, p["entry"]) for s, p in self.positions.items())

    def step(self):
        data = self._load()
        if MARKET_SYM not in data:
            log.warning("commodity: SPY data missing, skip"); return
        now = dt.datetime.now(dt.timezone.utc)
        last = {s: d.iloc[-1] for s, d in data.items()}
        px = {s: float(r["close"]) for s, r in last.items()}
        atrv = {s: float(r["atr"]) for s, r in last.items()}
        ema = {s: float(r["ema"]) for s, r in last.items()}
        on = px[MARKET_SYM] > ema[MARKET_SYM]

        for s in list(self.positions):
            if s not in px: continue
            P = self.positions[s]
            P["stop"] = max(P["stop"], px[s] - TRAIL * atrv[s])
            if px[s] <= P["stop"] or px[s] < ema[s] or not on:
                fill = px[s] * (1 - _slip(atrv[s], px[s]))
                proceeds = P["qty"] * fill; fee = proceeds * FEE
                pnl = proceeds - fee - P["cost"]; self.cash += proceeds - fee
                reason = "market_off" if not on else ("trend_break" if px[s] < ema[s] else "trail_stop")
                self.trades.append({"symbol": s, "side": "long", "qty": round(P["qty"], 6),
                                    "entry": round(P["entry"], 4), "exit": round(fill, 4),
                                    "pnl": round(pnl, 2), "opened_at": P["opened_at"],
                                    "closed_at": str(now), "reason": reason})
                del self.positions[s]

        equity = self.equity(px)
        self.peak = max(self.peak, equity)

        if on:
            cand = []
            for s in UNIVERSE:
                if s in self.positions or s not in px: continue
                if px[s] > ema[s] and atrv[s] > 0:
                    cand.append((px[s] / ema[s], s))
            cand.sort(reverse=True)
            for _, s in cand:
                if len(self.positions) >= MAXPOS: break
                cost = ALLOC * equity
                if self.cash >= cost and cost > 1:
                    fill = px[s] * (1 + _slip(atrv[s], px[s]))
                    qty = cost / (fill * (1 + FEE)); self.cash -= qty * fill * (1 + FEE)
                    self.positions[s] = {"qty": qty, "entry": fill, "stop": fill - TRAIL * atrv[s],
                                         "cost": qty * fill * (1 + FEE), "opened_at": str(now)}

        for s in UNIVERSE:
            if s in px:
                self.signals[s] = {"price": round(px[s], 4), "above_ema200": bool(px[s] > ema[s]),
                                   "held": s in self.positions, "ts": str(now)}
        self.signals["_market"] = {"symbol": MARKET_SYM, "on": bool(on),
                                   "px": round(px[MARKET_SYM], 2), "ema200": round(ema[MARKET_SYM], 2)}
        self.equity_curve.append([str(now), round(equity, 2)])
        self.equity_curve = self.equity_curve[-2000:]
        self._save(px, equity, on)

    def _save(self, px, equity, on):
        wins = [t for t in self.trades if t["pnl"] > 0]
        snap = {"updated_at": str(dt.datetime.now(dt.timezone.utc)),
                "status": "running" if on else "risk_off_cash",
                "equity": round(equity, 2), "cash": round(self.cash, 2),
                "pnl_total": round(equity - self.cap, 2), "pnl_pct": round((equity / self.cap - 1) * 100, 2),
                "n_trades": len(self.trades), "win_rate": round(len(wins) / len(self.trades) * 100, 1) if self.trades else None,
                "market_on": bool(on),
                "positions": [{"symbol": s, "side": "long", "qty": round(p["qty"], 6), "entry": round(p["entry"], 4),
                               "mark": px.get(s), "stop": round(p["stop"], 4), "target": None} for s, p in self.positions.items()],
                "signals": self.signals, "trades": self.trades[-50:], "equity_curve": self.equity_curve[-1500:]}
        with open(STATE_FILE, "w") as f:
            json.dump(snap, f, indent=1)

    def run(self):
        log.info(f"Commodity bot started | {ALLOC:.0%} x max {MAXPOS} + SPY filter | {len(UNIVERSE)} ETFs")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception(f"commodity step error: {e}")
            time.sleep(self.poll)
