"""validate_ohlcv / sanitize must detect and handle every data problem we know can occur."""
import numpy as np, pandas as pd
from data_quality import validate_ohlcv, sanitize, contiguous_runs

NOW = pd.Timestamp("2026-01-10 12:30", tz="UTC")
n_ok = 0


def ok(cond, msg):
    global n_ok
    assert cond, "FAILED: " + msg
    n_ok += 1


def frame(n=200, start="2026-01-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="ts")
    px = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px + 0.2, "volume": 10.0}, index=idx)


clean = frame()
r = validate_ohlcv(clean, "1h", NOW)
ok(r["ok"] and r["gap_count"] == 0 and r["coverage"] == 1.0, "a clean frame validates")

# duplicates: the LAST copy wins (the later fetch is the more recent one)
d = pd.concat([clean, clean.iloc[[50]].assign(close=100.9)])
c, rep = sanitize(d, "1h", NOW, log=lambda *_: None)
ok(rep["dropped_duplicates"] == 1 and c.loc[clean.index[50], "close"] == 100.9 and len(c) == len(clean), "duplicate timestamp: keep last")

# unsorted
c, rep = sanitize(clean.iloc[::-1], "1h", NOW, log=lambda *_: None)
ok(c.index.is_monotonic_increasing and len(c) == len(clean), "unsorted input is sorted")

# missing candles are REPORTED (never invented)
g = clean.drop(clean.index[100:106])
r = validate_ohlcv(g, "1h", NOW)
ok(r["gap_count"] == 1 and r["missing_bars"] == 6 and r["largest_gap_bars"] == 6 and r["coverage"] < 1, "gap detected with exact size")
c, rep = sanitize(g, "1h", NOW, log=lambda *_: None)
ok(len(c) == len(g), "sanitize does not fabricate candles")
runs = contiguous_runs(c.index, "1h")
ok(runs == [(0, 100), (100, 194)], f"contiguous runs split at the gap: {runs}")

# the candle still forming is dropped (NOW = 12:30, the 12:00 candle is incomplete)
f = frame(start="2026-01-02 00:00", n=205)
ok(validate_ohlcv(f, "1h", NOW)["incomplete_last_candle"], "incomplete latest candle detected")
c, rep = sanitize(f, "1h", NOW, log=lambda *_: None)
ok(c.index[-1] == pd.Timestamp("2026-01-10 11:00", tz="UTC") and rep["dropped_incomplete"] >= 1, "forming candle dropped, last complete candle kept")

# malformed rows
b = clean.copy()
b.iloc[10, b.columns.get_loc("high")] = b.iloc[10].low - 5           # high below low
b.iloc[20, b.columns.get_loc("close")] = np.nan
b.iloc[30, b.columns.get_loc("open")] = -1.0
ok(validate_ohlcv(b, "1h", NOW)["bad_ohlc_rows"] == 3, "three malformed rows found")
c, rep = sanitize(b, "1h", NOW, log=lambda *_: None)
ok(rep["dropped_bad_rows"] == 3 and len(c) == len(clean) - 3, "malformed rows removed")

# off-grid timestamps (e.g. a feed that stamps 00:00:30)
o = clean.copy()
o.index = o.index + pd.Timedelta(seconds=30)
ok(validate_ohlcv(o, "1h", NOW)["off_grid"] == len(o), "off-grid timestamps flagged")

# zero volume is reported, not silently removed
z = clean.copy()
z.iloc[5, z.columns.get_loc("volume")] = 0.0
ok(validate_ohlcv(z, "1h", NOW)["zero_volume_rows"] == 1, "zero-volume candle reported")
ok(validate_ohlcv(clean, "1h", NOW)["tz_aware"] and not validate_ohlcv(clean.tz_localize(None), "1h", NOW)["tz_aware"], "naive timestamps flagged")

print(f"{n_ok} data-quality checks passed")
print("ALL OK")
