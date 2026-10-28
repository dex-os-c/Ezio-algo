"""The take/skip gate must (a) find a planted edge and (b) abstain on pure noise."""
import numpy as np, pandas as pd
from train_setups import train_block, seq_filter, stats


def make(n, planted, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f_good": rng.normal(size=n), "f_noise1": rng.normal(size=n), "f_noise2": rng.normal(size=n)})
    p_win = 0.30 + (0.35 * (X.f_good > 0.5) if planted else 0.0)          # 30% base win rate at 2R, 65% when f_good is high
    win = rng.random(n) < p_win
    R = np.where(win, 2.0, -1.0) - 0.1
    sig = np.arange(n) * 5
    return X, R, (R > 0).astype(int), sig, sig + 3


def run(planted):
    X, R, y, sig, ex = make(1600, planted)
    dev, hold = 1200, 400
    m, thr = train_block(X, R, y, sig, ex, dev, 30, 7)
    if m is None or not np.isfinite(thr):
        return None
    p = m.predict(X.iloc[dev:], num_iteration=m.best_iteration)
    pick = seq_filter(sig, ex, np.arange(dev, 1600)[p >= thr])
    return stats(R[pick]), stats(R[dev:])


a = run(True)
assert a is not None and a[0]["exp_R"] > 0.3 and a[0]["exp_R"] > a[1]["exp_R"] + 0.3, a
print(f"planted edge found: filtered {a[0]['exp_R']:+.2f}R over {a[0]['n']} trades vs unfiltered {a[1]['exp_R']:+.2f}R")
bad = 0
for s in range(6):
    X, R, y, sig, ex = make(1600, False, seed=s + 10)
    m, thr = train_block(X, R, y, sig, ex, 1200, 30, 7)
    if m is not None and np.isfinite(thr):
        p = m.predict(X.iloc[1200:], num_iteration=m.best_iteration)
        r = R[1200:][p >= thr]
        bad += stats(r)["exp_R"] > 0.1
print(f"noise: {bad}/6 runs traded with a spuriously positive result")
assert bad <= 1
print("ALL OK")
