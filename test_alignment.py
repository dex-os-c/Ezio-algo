"""(1) Prove the no-lookahead test is capable of failing: a deliberately leaky feature must be caught.
(2) Higher-timeframe candles must be invisible until their LAST base bar has closed (no peeking at an unfinished 4h / 1d candle).
(3) Signal timing: nothing that happens after a setup's signal bar can change the setup itself."""
import numpy as np, pandas as pd
from data import synthetic
from smc import _htf
from setups import find_setups

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


def leaks(fn, df, cuts=(1200, 1800)):
    """same invariance procedure as test_features: rows computed on a prefix must equal the same rows computed on the full data"""
    full = fn(df)
    for cut in cuts:
        part = fn(df.iloc[:cut])
        a, b = np.asarray(full.iloc[:cut], float), np.asarray(part, float)
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True).all():
            return True
    return False


df = synthetic(2400, 5, "1h")

# (1) mutation test of the leak detector
ok(leaks(lambda d: d.close.rolling(5, center=True).mean(), df), "centered rolling mean (uses 2 future bars) IS detected as a leak")
ok(leaks(lambda d: d.close.shift(-1), df), "next-bar close IS detected as a leak")
ok(leaks(lambda d: d.close.rolling(24).max().shift(-3), df), "shifted-back rolling max IS detected as a leak")
ok(not leaks(lambda d: d.close.rolling(5).mean(), df), "a causal rolling mean is NOT flagged")
ok(not leaks(lambda d: d.close.ewm(span=20, adjust=False).mean(), df), "a causal EMA is NOT flagged")

# (2) HTF visibility: changing the LAST base bar of a 4h candle may change that bar's own row, never any earlier row
base = _htf(df, 60)
for mins, per, name in ((240, 4, "4h"), (1440, 24, "1d")):
    hr = df.index.hour
    ends = np.where((hr % (mins // 60)) == (mins // 60 - 1))[0]
    k = int(ends[len(ends) // 2])                       # last base bar of some HTF candle
    d2 = df.copy()
    d2.iloc[k, d2.columns.get_loc("close")] = d2.close.iloc[k] * 1.5
    d2.iloc[k, d2.columns.get_loc("high")] = d2.high.iloc[k] * 1.6
    f2 = _htf(d2, 60)
    cols = [c for c in base.columns if name in c]
    same_before = np.isclose(base[cols].iloc[:k].values, f2[cols].iloc[:k].values, equal_nan=True).all()
    ok(same_before, f"{name}: no row before the candle's last bar changes when that bar changes (no peeking inside the candle)")
    # and a change in the FIRST bar of the NEXT htf candle must not alter anything up to and including bar k
    d3 = df.copy()
    d3.iloc[k + 1, d3.columns.get_loc("close")] = d3.close.iloc[k + 1] * 1.5
    d3.iloc[k + 1, d3.columns.get_loc("high")] = d3.high.iloc[k + 1] * 1.6
    f3 = _htf(d3, 60)
    ok(np.isclose(base[cols].iloc[:k + 1].values, f3[cols].iloc[:k + 1].values, equal_nan=True).all(), f"{name}: next candle cannot leak backwards")

# (3) signal timing: truncating everything after (signal bar + 0) must keep the setup's ARMING fields unchanged
ev = find_setups(df.iloc[:1800])
full = find_setups(df)
armed = ["sig", "dir", "s_sweep_depth", "s_sweep_age", "s_mss_atr", "s_body_atr", "s_zone_atr", "s_risk_atr", "s_tgt_R", "s_mode"]
m = ev.merge(full, on=["sig", "dir"], suffixes=("_a", "_b"))
ok(len(m) > 20, f"enough overlapping setups to compare ({len(m)})")
ok(all(np.allclose(m[c + "_a"], m[c + "_b"], equal_nan=True) for c in armed[2:]), "arming features identical whether or not the future exists")
ok((ev.fill > ev.sig).all() and (ev.exit >= ev.fill).all(), "entry always strictly after the signal bar")
ok((ev.fill_ts > ev.ts).all(), "timestamps: fill strictly later than the signal candle")
print(f"{n_ok} alignment checks passed")
print("ALL OK")
