"""The iid standard error is wrong for clustered trades; the cluster bootstrap must be calibrated. Simulated, so the truth is known."""
import numpy as np, pandas as pd
from research_stats import week_clusters, cluster_boot_mean, iid_mean, design_effect, summarize, block_boot_mean

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


# week clusters follow the calendar
t = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-11 23:00", "2026-01-12", "2026-01-19"], utc=True, format="mixed")
ok(list(week_clusters(t)) == [0, 0, 0, 1, 2], "Monday-start weeks")

# truth: mean 0 (NO edge). Each week has a shared shock (burst of correlated trades) -> trades are not independent.
rng = np.random.default_rng(11)
SIMS, W, PER = 300, 100, 20
fp_iid = fp_cl = 0
des = []
for s in range(SIMS):
    week = np.repeat(np.arange(W), PER)
    R = rng.normal(0, 1, W * PER) + np.repeat(rng.normal(0, 0.6, W), PER)
    fp_iid += iid_mean(R)["p"] < 0.05
    fp_cl += cluster_boot_mean(R, week, B=400, seed=s)["p"] < 0.05
    des.append(design_effect(R, week))
fp_iid, fp_cl = fp_iid / SIMS, fp_cl / SIMS
print(f"false-positive rate at nominal 5%: iid {fp_iid:.1%} vs cluster bootstrap {fp_cl:.1%}; mean design effect {np.mean(des):.1f}")
ok(fp_iid > 0.15, "iid method is badly overconfident when trades are clustered")
ok(fp_cl < 0.10, "cluster bootstrap stays close to the nominal 5%")
ok(5.0 < np.mean(des) < 7.0, "design effect matches theory: 1 + (20-1) * 0.36/1.36 = 6.0")

# a REAL edge is still detected (power): true mean +0.15R
hits = 0
for s in range(100):
    week = np.repeat(np.arange(W), PER)
    R = rng.normal(0.15, 1, W * PER) + np.repeat(rng.normal(0, 0.3, W), PER)
    hits += cluster_boot_mean(R, week, B=400, seed=s)["p"] < 0.05
ok(hits >= 60, f"power with a real +0.15R edge ({hits}/100)")

# independent data: design effect ~ 1, both methods agree
R = rng.normal(0, 1, 4000)
ok(0.7 < design_effect(R, np.arange(4000) // 20) < 1.4, "no clustering -> design effect near 1")
s = summarize(R, pd.date_range("2024-01-01", periods=4000, freq="3h", tz="UTC"))
ok(s["lo"] < 0 < s["hi"], "zero-mean data: CI contains 0")

# block bootstrap for a single series keeps autocorrelation: wider than iid for a persistent series
x = np.zeros(3000)
e = rng.normal(0, 1, 3000)
for i in range(1, 3000):
    x[i] = 0.8 * x[i - 1] + e[i]
bb = block_boot_mean(x, block=50, B=1000)
ok((bb["hi"] - bb["lo"]) > 2.5 * (3.92 * x.std() / np.sqrt(3000)) / 1.0 * 0.5, "block bootstrap CI reflects autocorrelation")

print(f"{n_ok} statistics checks passed")
print("ALL OK")
