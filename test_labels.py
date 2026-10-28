"""labels.triple_barrier: entry at the NEXT open, stop wins ties, ATR known at the signal close, no future leakage, NaN tail."""
import numpy as np, pandas as pd
from labels import triple_barrier
from data import synthetic

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


def frame(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


# flat then a clean rally: bar 20's label must be 1 (upper barrier first), entry = open of bar 21
rows = [(100, 101, 99, 100)] * 20 + [(100, 100.5, 99.5, 100)] + [(100, 103, 99.8, 102.5)] + [(102.5, 106, 102, 105)] * 10
df = frame(rows)
y, r = triple_barrier(df, 6, mult=1.0)
ok(y.iloc[20] == 1.0 and r.iloc[20] == 1.0, "rally after the signal bar -> upper barrier first")

# both barriers inside ONE bar -> the stop (lower barrier) is assumed first
rows = [(100, 101, 99, 100)] * 20 + [(100, 100.5, 99.5, 100)] + [(100, 110, 90, 100)] + [(100, 100.5, 99.5, 100)] * 10
y, r = triple_barrier(frame(rows), 6, mult=1.0)
ok(y.iloc[20] == 0.0 and r.iloc[20] == -1.0, "same-bar double touch counts as the stop")

# the label of bar t must not depend on anything before t+1 except ATR_t: changing bar t's own HIGH changes ATR (known at its close) but a change
# AFTER the horizon must change nothing
df = synthetic(1500, 9, "1h")
y0, _ = triple_barrier(df, 8)
d2 = df.copy()
d2.iloc[900:, d2.columns.get_loc("close")] *= 1.3
d2.iloc[900:, d2.columns.get_loc("high")] *= 1.3
d2.iloc[900:, d2.columns.get_loc("low")] *= 1.3
d2.iloc[900:, d2.columns.get_loc("open")] *= 1.3
y1, _ = triple_barrier(d2, 8)
ok(np.array_equal(y0.iloc[:891].to_numpy(), y1.iloc[:891].to_numpy(), equal_nan=True), "labels more than horizon+1 bars before a change are untouched")

# tail: the last `horizon` rows have no complete future -> NaN, never a guess
ok(y0.iloc[-8:].isna().all() and y0.iloc[:-8].notna().all(), "unknown-future tail is NaN")
print(f"{n_ok} label checks passed")
print("ALL OK")
