"""Real-time & historical market data. ccxt for most exchanges; Bitkub via custom adapter."""
import time
import pandas as pd


def _make_exchange(cfg: dict):
    ex_cfg = cfg["exchange"]
    ex_id = ex_cfg["id"].lower()
    key = ex_cfg.get("api_key", "")
    secret = ex_cfg.get("api_secret", "")
    has_key = key and "YOUR_" not in key
    if ex_id == "bitkub":
        from bitkub_client import BitkubClient
        return BitkubClient(key if has_key else "", secret if has_key else "")
    import ccxt
    klass = getattr(ccxt, ex_id)
    params = {"enableRateLimit": True}
    if has_key:
        params.update({"apiKey": key, "secret": secret})
    ex = klass(params)
    if ex_cfg.get("testnet"):
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass
    return ex


class DataFeed:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.exchange = _make_exchange(cfg)
        self.timeframe = cfg["trading"]["timeframe"]
        self.quote = cfg["trading"].get("quote", "USDT")

    def ohlcv(self, symbol: str, limit: int = 300, since=None) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def ohlcv_history(self, symbol: str, total_bars: int = 3000) -> pd.DataFrame:
        """Paginate backwards to fetch long history for backtest/ML."""
        tf_ms = self.exchange.parse_timeframe(self.timeframe) * 1000
        since = self.exchange.milliseconds() - total_bars * tf_ms
        frames, fetched = [], 0
        while fetched < total_bars:
            batch = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=1000)
            if not batch:
                break
            frames.extend(batch)
            fetched += len(batch)
            since = batch[-1][0] + tf_ms
            time.sleep(self.exchange.rateLimit / 1000)
            if len(batch) < 1000:
                break
        df = pd.DataFrame(frames, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def ticker(self, symbol: str) -> float:
        return float(self.exchange.fetch_ticker(symbol)["last"])

    def balance_quote(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get(self.quote, {}).get("free", 0))
