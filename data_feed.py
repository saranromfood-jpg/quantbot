"""Real-time & historical market data via ccxt."""
import time
import ccxt
import pandas as pd


class DataFeed:
    def __init__(self, cfg: dict):
        ex_cfg = cfg["exchange"]
        klass = getattr(ccxt, ex_cfg["id"])
        params = {"enableRateLimit": True}
        if ex_cfg.get("api_key") and "YOUR_" not in ex_cfg["api_key"]:
            params.update({"apiKey": ex_cfg["api_key"], "secret": ex_cfg["api_secret"]})
        self.exchange = klass(params)
        if ex_cfg.get("testnet"):
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass
        self.timeframe = cfg["trading"]["timeframe"]

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
            batch = self.exchange.fetch_ohlcv(self.exchange.symbol(symbol) if hasattr(self.exchange, "symbol") else symbol,
                                              self.timeframe, since=since, limit=1000)
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

    def balance_usdt(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))
