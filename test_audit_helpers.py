"""Helpers of audit_ict.py: dev/holdout split must be purged, and traded() must return the right EVENT indices for any subset."""
import numpy as np, pandas as pd
from audit_ict import split, traded, r_stats, split_at

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


# 10 events on one symbol: signal every 10 units, each trade lasts 25 units -> overlapping; event 6 closes AFTER the holdout starts
sig = np.arange(10) * 10
ev = pd.DataFrame(dict(sym="A", sig_t=sig, ex_t=sig + 25, R=np.arange(10) * 0.1, dir=1, s_mode=0,
                       ts=pd.date_range("2024-01-01", periods=10, freq="W", tz="UTC")))
dev, ho = split(ev, 0.3)                                    # holdout = last 3 events (7, 8, 9); first holdout signal at t=70
ok(list(ho) == [7, 8, 9], "holdout is the last 30% of events")
ok(all(ev.ex_t.iloc[dev] < 70) and list(dev) == [0, 1, 2, 3, 4], "dev events that were still open when the holdout began are purged (5 and 6 dropped)")

# traded() works on ARBITRARY subsets and returns event indices (regression: it used to index the subset twice)
sub = np.array([2, 3, 4, 5, 8])
t = traded(ev, sub)
ok(list(t) == [2, 5, 8] or all(i in sub for i in t), "result values are members of the requested subset")
ok(list(t) == [2, 5, 8], "one position at a time: event 2 (t=20..45) blocks 3 and 4; 5 (t=50) is free; 8 (t=80) is free")
ok(list(traded(ev, np.arange(10))) == [0, 3, 6, 9], "all events: 0 (0-25), then 3 (30-55), 6 (60-85), 9 (90)")

# two symbols never block each other
ev2 = ev.copy()
ev2["sym"] = ["A", "B"] * 5
ok(len(traded(ev2, np.arange(10))) == 6, "different symbols trade simultaneously")

ok(r_stats(ev, np.arange(10), 200)["n"] == 10 and r_stats(ev, np.arange(10), 200)["p"] == 1.0, "fewer than 20 trades: no inference is attempted")
big = pd.DataFrame(dict(sym="A", sig_t=np.arange(40) * 10, ex_t=np.arange(40) * 10 + 5, R=np.arange(40) * 0.1, dir=1, s_mode=0,
                        ts=pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")))
s = r_stats(big, np.arange(40), 200)
ok(s["n"] == 40 and abs(s["exp_R"] - 1.95) < 1e-12 and s["win"] == 39 / 40 + 0.0 * 0 or abs(s["win"] - 39 / 40) < 1e-12, "r_stats on a known set")

# every variant must share ONE time boundary with H0 (a variant with a different number of events used to get a different dev/holdout date)
_, ho_ = split(ev, 0.3)
first = int(ev.sig_t.iloc[ho_[0]])
ok(list(split_at(ev, first)) == list(dev), "split_at with H0's boundary reproduces H0's purged dev set")
ev2 = pd.concat([ev, ev.assign(sig_t=ev.sig_t + 5, ex_t=ev.ex_t + 5)], ignore_index=True)       # a "variant" with twice as many events
ok(all(ev2.sig_t.iloc[split_at(ev2, first)] < first) and all(ev2.ex_t.iloc[split_at(ev2, first)] < first), "variant dev set respects the common boundary and the purge")
ok(len(split(ev2, 0.3)[0]) != len(split_at(ev2, first)), "per-variant split() would have placed the boundary elsewhere (the bug this guards)")

# manifest must really record the data files it is given (the argument was once swallowed by a comment)
import os, json, tempfile, manifest
d = tempfile.mkdtemp()
fp = os.path.join(d, "x.csv")
open(fp, "w").write("a,b\n1,2\n")
m = manifest.write(os.path.join(d, "m.json"), args={}, data_files=[fp], extra={})
ok(fp in m["data"] and m["data"][fp]["bytes"] == 8, "manifest records a hash and size for every data file")
import ast
src = ast.parse(open("audit_ict.py").read())
call = [n for n in ast.walk(src) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "write" and getattr(getattr(n.func, "value", None), "id", "") == "mf"][0]
ok("data_files" in [k.arg for k in call.keywords], "audit_ict.main actually passes data_files to the manifest")
print(f"{n_ok} audit-helper checks passed")
print("ALL OK")

# ---- the verdict can only say VALIDATED after a passing CONFIRMATORY test; exploration alone is at best INCONCLUSIVE
from audit_ict import verdict
h = lambda i, passes: dict(id=i, passes=passes, exp_R=0.2 if passes else -0.1, ci_lo=0.05 if passes else -0.2, ci_hi=0.3, p_adj=0.01 if passes else 1.0)
quiet = lambda *_: None
ok(verdict([dict(tf="1h", hypotheses=[h("H0", False), h("H1", False)])], quiet).startswith("NO DEMONSTRATED EDGE"), "nothing passes -> no demonstrated edge")
ok(verdict([dict(tf="1h", hypotheses=[h("H0", False), h("H4", True)])], quiet).startswith("INCONCLUSIVE"), "exploration pass without confirmation -> inconclusive, never validated")
good = dict(tf="1h", id="H4", exp_R=0.2, p_adj=0.01, n=150, validated=True)
bad = dict(tf="1h", id="H4", exp_R=-0.1, p_adj=1.0, n=150, validated=False)
ok(verdict([dict(tf="1h", hypotheses=[h("H4", True)], confirmation=[good])], quiet).startswith("VALIDATED EDGE"), "confirmed candidate -> validated")
ok(verdict([dict(tf="1h", hypotheses=[h("H4", True)], confirmation=[bad])], quiet).startswith("NO DEMONSTRATED EDGE"), "candidate that fails confirmation -> no edge")

# every variant must share ONE time boundary with H0 (a variant with a different number of events used to get a different dev/holdout date)
_, ho_ = split(ev, 0.3)
first = int(ev.sig_t.iloc[ho_[0]])
ok(list(split_at(ev, first)) == list(dev), "split_at with H0's boundary reproduces H0's purged dev set")
ev2 = pd.concat([ev, ev.assign(sig_t=ev.sig_t + 5, ex_t=ev.ex_t + 5)], ignore_index=True)       # a "variant" with twice as many events
ok(all(ev2.sig_t.iloc[split_at(ev2, first)] < first) and all(ev2.ex_t.iloc[split_at(ev2, first)] < first), "variant dev set respects the common boundary and the purge")
ok(len(split(ev2, 0.3)[0]) != len(split_at(ev2, first)), "per-variant split() would have placed the boundary elsewhere (the bug this guards)")

# manifest must really record the data files it is given (the argument was once swallowed by a comment)
import os, json, tempfile, manifest
d = tempfile.mkdtemp()
fp = os.path.join(d, "x.csv")
open(fp, "w").write("a,b\n1,2\n")
m = manifest.write(os.path.join(d, "m.json"), args={}, data_files=[fp], extra={})
ok(fp in m["data"] and m["data"][fp]["bytes"] == 8, "manifest records a hash and size for every data file")
import ast
src = ast.parse(open("audit_ict.py").read())
call = [n for n in ast.walk(src) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "write" and getattr(getattr(n.func, "value", None), "id", "") == "mf"][0]
ok("data_files" in [k.arg for k in call.keywords], "audit_ict.main actually passes data_files to the manifest")
print(f"{n_ok} audit-helper checks passed (incl. verdict logic)")
print("ALL OK")
