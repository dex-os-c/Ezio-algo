"""Slice analysis must find a planted slice edge and must not flag pure noise."""
import numpy as np, pandas as pd
from analyze_setups import evaluate


def fake(n, planted, seed):
    rng = np.random.default_rng(seed)
    sig = np.arange(n) * 6
    ev = pd.DataFrame({"dir": rng.choice([1, -1], n), "s_mode": rng.integers(0, 3, n), "sym": rng.choice(["A", "B"], n),
                       "sig_t": sig, "ex_t": sig + 3, "ts": pd.date_range("2020", periods=n, freq="7h", tz="UTC")})
    X = pd.DataFrame({"s_risk_atr": rng.uniform(0.3, 3, n), "s_sweep_depth": rng.uniform(0, 2, n), "s_zone_atr": rng.uniform(0, 2, n),
                      "s_tgt_R": rng.uniform(0, 8, n), "s_sweep_age": rng.integers(1, 20, n),
                      "ctx_smc_htf4h_trend": rng.choice([-1, 0, 1], n), "ctx_smc_trend": rng.choice([-1, 1], n),
                      "ctx_smc_pd": rng.normal(0, .3, n), "ctx_sweep_with": rng.choice([0., 1.], n), "ctx_vol_ratio": rng.uniform(.5, 1.5, n)})
    good = (X.ctx_smc_htf4h_trend > 0) & (X.ctx_sweep_with > 0.5) if planted else np.zeros(n, bool)
    win = rng.random(n) < np.where(good, 0.62, 0.34)
    ev["R"] = np.where(win, 2.0, -1.0) - 0.1
    return ev, X


ev, X = fake(9000, True, 1)
o = evaluate(ev, X, log=lambda *_: None)
assert o["survives"] and o["holdout"]["exp_R"] > 0 and o["candidate"], o["best"]
print("planted slice found and confirmed on holdout:", o["best"]["name"])
flags = 0
for s in range(8):
    ev, X = fake(9000, False, 100 + s)
    flags += evaluate(ev, X, log=lambda *_: None)["candidate"]
print(f"noise: {flags}/8 runs produced a false candidate")
assert flags <= 1
print("ALL OK")
