"""Historical + live candles. Sources: ccxt exchanges (crypto), yfinance (NSE/US stocks), synthetic (for testing the pipeline)."""
import os, time
import numpy as np, pandas as pd
from config import tf_minutes

COLS = ["open", "high", "low", "close", "volume"]


def synthetic(n, seed=7, tf="15m"):
    """Fake market with volatility clustering and a WEAK regime-switching momentum edge.
    Only for testing the plumbing: a good result here proves nothing about real markets."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    logv = np.log(0.002)
    phi, left = 0.12, 0
    for t in range(1, n):
        if left <= 0:
            phi, left = rng.choice([0.14, -0.08, 0.0]), int(rng.integers(300, 900))
        left -= 1
        logv += 0.02 * (np.log(0.002) - logv) + 0.05 * rng.standard_normal()
        r[t] = phi * r[t - 1] + np.exp(logv) * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(r))
    opn = np.r_[close[0], close[:-1]]
    spread = np.abs(r) * 0.5 + 0.0005
    high = np.maximum(opn, close) * (1 + spread * rng.random(n))
    low = np.minimum(opn, close) * (1 - spread * rng.random(n))
    vol = np.exp(rng.normal(8, 0.4, n) + 40 * np.abs(r))
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{tf_minutes(tf)}min", tz="UTC", name="ts")
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _ccxt(cfg):
    import ccxt
    return getattr(ccxt, cfg.exchange)({"enableRateLimit": True})


def fetch_ccxt(cfg, since_ms, log=print):
    """page through exchange candles from since_ms; returns CLOSED candles only"""
    ex = _ccxt(cfg)
    ms = tf_minutes(cfg.timeframe) * 60_000
    since, rows, now = int(since_ms), [], ex.milliseconds()
    while since < now:
        for attempt in range(4):
            try:
                chunk = ex.fetch_ohlcv(cfg.symbol, cfg.timeframe, since=since, limit=1000)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f"  retry ({type(e).__name__})")
                time.sleep(2 * (attempt + 1))
        if not chunk:
            break
        rows += chunk
        if chunk[-1][0] + ms <= since:
            break
        since = chunk[-1][0] + ms
        if len(rows) % 10000 < 1000:
            log(f"  fetched {len(rows)} candles up to {pd.to_datetime(chunk[-1][0], unit='ms')}")
    df = pd.DataFrame(rows, columns=["ts"] + COLS)
    df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df[df.index + pd.Timedelta(milliseconds=ms) <= pd.Timestamp.now(tz="UTC")]


def fetch_yf(cfg):
    import yfinance as yf
    period = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "1d": "max"}.get(cfg.timeframe, "60d")
    d = yf.download(cfg.symbol, period=period, interval=cfg.timeframe, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[COLS].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    d.index.name = "ts"
    return d


def load_history(cfg, log=print):
    """cached + incremental: the first call downloads `years` of candles, later calls only fetch what is new"""
    if cfg.source == "synthetic":
        return synthetic(cfg.bars, cfg.seed, cfg.timeframe)
    os.makedirs("data", exist_ok=True)
    tag = cfg.exchange if cfg.source == "ccxt" else cfg.source
    path = f"data/{tag}_{cfg.symbol.replace('/', '-')}_{cfg.timeframe}.csv"
    if cfg.source == "yfinance":
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        df = fetch_yf(cfg)
    else:
        ms = tf_minutes(cfg.timeframe) * 60_000
        old = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else None
        start = int(old.index[-1].timestamp() * 1000) + ms if old is not None else int(time.time() * 1000 - cfg.years * 365 * 86400_000)
        log(f"loading {cfg.symbol} {cfg.timeframe} from {cfg.exchange}" + (" (cache + new candles)" if old is not None else f" ({cfg.years:g} years)"))
        new = fetch_ccxt(cfg, start, log)
        df = pd.concat([old, new]) if old is not None else new
        df = df[~df.index.duplicated()].sort_index()
        if len(df) < 500:
            raise RuntimeError(f"only {len(df)} candles for {cfg.symbol} on {cfg.exchange}; check symbol/exchange/network")
    df.to_csv(path)
    if cfg.source == "ccxt":
        df = df[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cfg.years * 365)]
    return df


class LiveFeed:
    """poll() returns newly CLOSED candles. Synthetic mode replays fresh fake data one bar per call."""

    def __init__(self, cfg):
        self.cfg, self.last_ts, self.i = cfg, None, 0
        if cfg.source == "synthetic":
            full = synthetic(cfg.warm_bars + cfg.paper_bars, cfg.seed + 100, cfg.timeframe)
            self.hist, self.replay = full.iloc[:cfg.warm_bars], full.iloc[cfg.warm_bars:]
        else:
            self.ex = _ccxt(cfg) if cfg.source != "yfinance" else None
            if cfg.source == "yfinance":
                self.hist = fetch_yf(cfg).tail(cfg.warm_bars)
            else:
                self.hist = load_history(cfg).tail(cfg.warm_bars)
        self.last_ts = self.hist.index[-1]

    def last_price(self):
        """latest traded price (for live mark-to-market between candle closes)"""
        try:
            if self.cfg.source == "synthetic":
                return None
            if self.cfg.source == "yfinance":
                import yfinance as yf
                return float(yf.Ticker(self.cfg.symbol).fast_info["last_price"])
            return float(self.ex.fetch_ticker(self.cfg.symbol)["last"])
        except Exception:
            return None

    @property
    def exhausted(self):
        return self.cfg.source == "synthetic" and self.i >= len(self.replay)

    def poll(self):
        if self.cfg.source == "synthetic":
            if self.exhausted:
                return self.hist.iloc[:0]
            bar = self.replay.iloc[self.i:self.i + 1]
            self.i += 1
            self.last_ts = bar.index[-1]
            return bar
        try:
            if self.cfg.source == "yfinance":
                d = fetch_yf(self.cfg)
            else:
                ms = tf_minutes(self.cfg.timeframe) * 60_000
                rows = self.ex.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, limit=10)
                d = pd.DataFrame(rows, columns=["ts"] + COLS)
                d["ts"] = pd.to_datetime(d.ts, unit="ms", utc=True)
                d = d.set_index("ts")
                d = d[d.index + pd.Timedelta(milliseconds=ms) <= pd.Timestamp.now(tz="UTC")]
        except Exception as e:
            print(f"  feed error: {type(e).__name__}: {str(e)[:120]}")
            return self.hist.iloc[:0]
        new = d[d.index > self.last_ts]
        if len(new):
            self.last_ts = new.index[-1]
        return new
