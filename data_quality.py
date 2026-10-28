"""Candle integrity checks. Candle timestamps are OPEN times (ccxt convention): a candle is complete only at ts + timeframe.

validate_ohlcv(df, tf)  -> report dict (never modifies anything)
sanitize(df, tf)        -> (clean df, report): sorts, drops duplicate timestamps (keeps the LAST = most recently fetched copy), drops the
                           still-forming latest candle and rows with NaN / non-positive prices / impossible OHLC. Gaps are REPORTED, not
                           invented: downstream code must not treat rows across a gap as consecutive bars (see contiguous_runs).
"""
import numpy as np, pandas as pd
from config import tf_minutes


def to_ns(index):
    """DatetimeIndex -> int64 nanoseconds since the epoch (unit-safe: Index.asi8 follows the index's own unit, which differs by pandas version)"""
    i = index.tz_convert("UTC") if index.tz is not None else index
    return np.asarray(i.tz_localize(None).values.astype("datetime64[ns]").astype("int64"))


def _bad_ohlc(df):
    o, h, l, c = df.open, df.high, df.low, df.close
    nan = df[["open", "high", "low", "close"]].isna().any(axis=1)
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    impossible = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    return nan | nonpos | impossible


def validate_ohlcv(df, tf, now=None):
    d = pd.Timedelta(minutes=tf_minutes(tf))
    now = now or pd.Timestamp.now(tz="UTC")
    idx = df.index
    rep0_tz = idx.tz is not None
    if not rep0_tz:                                    # naive timestamps are reported, then treated as UTC for the remaining checks
        idx = idx.tz_localize("UTC")
    rep = dict(rows=int(len(df)), start=str(idx[0]) if len(df) else None, end=str(idx[-1]) if len(df) else None, timeframe=tf)
    rep["tz_aware"] = rep0_tz
    rep["duplicates"] = int(idx.duplicated().sum())
    rep["non_monotonic"] = bool(not idx.is_monotonic_increasing)
    u = idx[~idx.duplicated()].sort_values()
    step = u.to_series().diff().dropna()
    gaps = step[step > d]
    rep["gap_count"] = int(len(gaps))
    rep["missing_bars"] = int(((gaps / d) - 1).round().sum()) if len(gaps) else 0
    rep["largest_gap_bars"] = int(round(gaps.max() / d)) - 1 if len(gaps) else 0
    rep["gaps"] = [(str(t - g), str(t), int(round(g / d)) - 1) for t, g in list(gaps.items())[:20]]
    rep["off_grid"] = int((((to_ns(u) // 10 ** 9) % int(d.total_seconds())) != 0).sum()) if len(u) else 0
    rep["bad_ohlc_rows"] = int(_bad_ohlc(df).sum())
    rep["zero_volume_rows"] = int((df.volume <= 0).sum()) if "volume" in df else 0
    rep["incomplete_last_candle"] = bool(len(df) and idx.max() + d > now)
    rep["coverage"] = float(len(u) / (len(u) + rep["missing_bars"])) if len(u) else 0.0
    rep["ok"] = not (rep["duplicates"] or rep["non_monotonic"] or rep["bad_ohlc_rows"] or rep["incomplete_last_candle"] or rep["off_grid"])
    return rep


def sanitize(df, tf, now=None, log=print):
    before = len(df)
    now = now or pd.Timestamp.now(tz="UTC")
    d = pd.Timedelta(minutes=tf_minutes(tf))
    if df.index.tz is None:
        log("  WARNING: naive timestamps, assuming UTC")
        df = df.tz_localize("UTC")
    out = df.sort_index(kind="stable")               # stable, so "keep=last" really means the most recently appended copy
    out = out[~out.index.duplicated(keep="last")]
    n_dup = before - len(out)
    bad = _bad_ohlc(out)
    out = out[~bad]
    n_forming = int((out.index + d > now).sum())
    out = out[out.index + d <= now]                    # the candle that is still forming is never data
    rep = validate_ohlcv(out, tf, now)
    rep.update(dropped_duplicates=int(n_dup), dropped_bad_rows=int(bad.sum()), dropped_incomplete=n_forming)
    if n_dup or bad.sum() or n_forming:
        log(f"  data cleaning: {n_dup} duplicate / {int(bad.sum())} malformed / {n_forming} still-forming candles removed")
    if rep["gap_count"]:
        log(f"  WARNING: {rep['gap_count']} gaps ({rep['missing_bars']} missing bars, largest {rep['largest_gap_bars']}); rows across a gap are not consecutive")
    return out, rep


def contiguous_runs(index, tf):
    """[(start_pos, end_pos), ...] maximal runs of consecutive candles (positions into `index`, end exclusive)"""
    d = pd.Timedelta(minutes=tf_minutes(tf))
    step = np.diff(to_ns(index))
    cuts = np.flatnonzero(step != int(d.total_seconds()) * 10 ** 9) + 1
    b = np.r_[0, cuts, len(index)]
    return list(zip(b[:-1], b[1:]))
