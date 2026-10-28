import numpy as np, pandas as pd
from baselines import breakout_flags, ma_flags, run_weights, _weights_from_flags, all_baselines, beta_alpha, compare, scaled, perf

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


idx = pd.date_range("2022-01-01", periods=600, freq="D", tz="UTC")
rng = np.random.default_rng(2)
close = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, 600))) for c in "ABC"}, index=idx)

# no look-ahead: flags at day t must not change when later data is removed
for fn in (ma_flags, breakout_flags):
    full, part = fn(close), fn(close.iloc[:400])
    ok(part.equals(full.iloc[:400]), f"{fn.__name__}: signals identical on truncated data")

# a breakout can only trigger on a close ABOVE the PRIOR 55-day high (the current bar is excluded from the channel)
x = pd.DataFrame({"A": np.r_[np.full(60, 100.0), 101.0, 101.0, np.full(10, 100.0), 90.0, 90.0]}, index=pd.date_range("2022-01-01", periods=74, freq="D", tz="UTC"))
f = breakout_flags(x)["A"]
ok(not f.iloc[59] and f.iloc[60] and f.iloc[61], "enter on the first close above the prior high")
ok(f.iloc[72] is np.bool_(False) or not f.iloc[72], "exit after a close below the prior 20-day low")

# buy&hold on a flat market loses exactly the entry cost; cash is zero
flat = pd.DataFrame(100.0, index=idx[:100], columns=["A", "B"])
bh = all_baselines(flat)["buy_hold"]
ok(abs(bh.sum() + 7e-4) < 1e-12, "flat market: only the entry cost is paid")
ok(all_baselines(close)["cash"].abs().sum() == 0, "cash baseline is exactly zero")

# weights are equal across live coins and never exceed 1 (unlevered)
w = _weights_from_flags(close.notna())
ok(np.allclose(w.sum(axis=1), 1.0) and (w.abs().sum(axis=1) <= 1 + 1e-12).all(), "unlevered equal weight")

# alpha/beta: y = 0.5 * market + constant daily alpha -> recovered, t-stats large
m = rng.normal(0.001, 0.03, 3000)
y = 0.5 * m + 0.0004 + rng.normal(0, 0.002, 3000)
ba = beta_alpha(y, m)
ok(abs(ba["beta"] - 0.5) < 0.02 and abs(ba["alpha_ann"] - 0.0004 * 365) < 0.06 and ba["t_alpha"] > 5, "beta/alpha recovered")
# pure beta has no alpha
ba = beta_alpha(0.7 * m + rng.normal(0, 0.002, 3000), m)
ok(abs(ba["t_alpha"]) < 3, "a pure-beta strategy shows no significant alpha")

# vol scaling really equalises risk; compare() runs on common dates
s = pd.Series(rng.normal(0, 0.01, 500), index=idx[:500])
ok(abs(perf(scaled(s, 0.2))["vol"] - 0.2) < 1e-9, "vol scaling")
tbl = compare({"buy_hold": pd.Series(m[:500], index=idx[:500]), "x": s})
ok(len(tbl) == 2 and "alpha_ann" in tbl.columns, "comparison table")

print(f"{n_ok} baseline checks passed")
print("ALL OK")
