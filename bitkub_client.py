"""Bitkub adapter — ccxt ไม่รองรับ Bitkub จึงต้องเขียนเอง.
เปิดเผยเมธอดชุดเดียวกับ object ของ ccxt เพื่อให้ data_feed.py / execution.py ใช้ได้โดยไม่ต้องแก้มาก.

Public market data (ไม่ต้อง key): OHLCV ผ่าน TradingView endpoint, ticker.
Trading (ต้อง key): Bitkub API v3 signed (HMAC-SHA256).

หมายเหตุสำคัญ: ส่วน live order เขียนตามสเปก Bitkub v3 แต่ผมทดสอบกับ API จริงไม่ได้ในที่นี้
(เครือข่ายถูกบล็อก) ก่อนใช้เงินจริง โปรดตรวจกับเอกสาร https://github.com/bitkub/bitkub-official-api-docs
ส่วน paper/backtest ใช้ได้เต็มที่เพราะใช้แค่ public data."""
import hashlib
import hmac
import json as _json
import time
import urllib.parse
import urllib.request

BASE = "https://api.bitkub.com"

# timeframe -> Bitkub TradingView resolution (นาที, หรือ 'D')
_TF_RES = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
           "1h": "60", "4h": "240", "1d": "1D"}
_TF_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
           "1h": 3600, "4h": 14400, "1d": 86400}


def _to_tv_symbol(symbol: str) -> str:
    """'BTC/THB' -> 'BTC_THB' (รูปแบบ TradingView/v3)."""
    return symbol.replace("/", "_").upper()


def _to_ticker_symbol(symbol: str) -> str:
    """'BTC/THB' -> 'THB_BTC' (รูปแบบ ticker เก่า กลับด้าน)."""
    b, q = symbol.upper().split("/")
    return f"{q}_{b}"


def _http_get(path: str, params: dict = None, timeout=15):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _json.loads(r.read())


class BitkubClient:
    """ccxt-compatible surface สำหรับ Bitkub."""

    rateLimit = 250  # ms ระหว่าง request (Bitkub จำกัด ~ 200-250/10s)

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key or ""
        self.api_secret = (api_secret or "").encode()

    # ---- ccxt-like helpers ----
    def set_sandbox_mode(self, on: bool):
        pass  # Bitkub ไม่มี testnet public — paper mode จัดการที่ชั้น Executor แทน

    def parse_timeframe(self, tf: str) -> int:
        return _TF_SEC[tf]

    def milliseconds(self) -> int:
        return int(time.time() * 1000)

    # ---- public market data ----
    def fetch_ohlcv(self, symbol, timeframe="15m", since=None, limit=500):
        """คืน list ของ [ts_ms, open, high, low, close, volume] เหมือน ccxt."""
        res = _TF_RES.get(timeframe, "15")
        sec = _TF_SEC[timeframe]
        to_ts = int(time.time())
        if since is not None:
            frm = int(since / 1000)
            to_ts = min(to_ts, frm + sec * (limit + 2))
        else:
            frm = to_ts - sec * (limit + 2)
        data = _http_get("/tradingview/history",
                         {"symbol": _to_tv_symbol(symbol), "resolution": res,
                          "from": frm, "to": to_ts})
        if not data or data.get("s") != "ok":
            return []
        out = []
        for i in range(len(data["t"])):
            out.append([int(data["t"][i]) * 1000, float(data["o"][i]), float(data["h"][i]),
                        float(data["l"][i]), float(data["c"][i]), float(data["v"][i])])
        return out[-limit:] if limit else out

    def fetch_ticker(self, symbol):
        sym = _to_ticker_symbol(symbol)
        data = _http_get("/api/market/ticker", {"sym": sym})
        row = data.get(sym, {}) if isinstance(data, dict) else {}
        return {"last": float(row.get("last", 0)), "info": row}

    # ---- private (signed v3) ----
    def _signed_post(self, path: str, payload: dict):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Bitkub API key/secret ไม่ได้ตั้งค่า (ตั้งใน env API_KEY/API_SECRET)")
        ts = str(self.milliseconds())
        body = _json.dumps(payload, separators=(",", ":"))
        # v3 signature: HMAC_SHA256(secret, timestamp + METHOD + path + body)
        sig_payload = ts + "POST" + path + body
        sign = hmac.new(self.api_secret, sig_payload.encode(), hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            BASE + path, data=body.encode(), method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json",
                     "X-BTK-APIKEY": self.api_key, "X-BTK-TIMESTAMP": ts, "X-BTK-SIGN": sign})
        with urllib.request.urlopen(req, timeout=15) as r:
            return _json.loads(r.read())

    def fetch_balance(self):
        res = self._signed_post("/api/v3/market/balances", {})
        out = {}
        for cur, info in (res.get("result", {}) or {}).items():
            out[cur] = {"free": float(info.get("available", 0)),
                        "used": float(info.get("reserved", 0))}
        return out

    def create_order(self, symbol, type_, side, qty, price=None):
        """market order. side: 'buy'|'sell'. คืน dict มี average/price/id เหมือน ccxt (พอใช้)."""
        sym = _to_tv_symbol(symbol)  # v3 ใช้ BTC_THB
        path = "/api/v3/market/place-bid" if side == "buy" else "/api/v3/market/place-ask"
        # Bitkub: bid amt = จำนวน THB ที่จะใช้ซื้อ, ask amt = จำนวนเหรียญที่จะขาย
        payload = {"sym": sym, "typ": "market"}
        if side == "buy":
            payload["amt"] = qty * (price or self.fetch_ticker(symbol)["last"])  # THB spend
        else:
            payload["amt"] = qty                                                 # coin amount
        res = self._signed_post(path, payload)
        r = res.get("result", {}) or {}
        return {"id": r.get("id"), "average": float(r.get("rat", price or 0)) or None,
                "price": price, "info": res}
