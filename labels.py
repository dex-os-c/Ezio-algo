"""Triple-barrier labels (drop-in for features.make_labels).

For every bar t (signal known at the close of t):
  entry  = open of bar t+1            (you cannot fill at the close you just saw)
  upper  = entry + mult * ATR_t       lower = entry - mult * ATR_t
  y = 1 if the upper barrier is touched before the lower one, else 0
  if both are touched inside the same bar -> assume the stop hit first (conservative)
  if neither is touched within `horizon` bars -> label by the sign of the move at the horizon
Barriers are symmetric so p > 0.5 still means "long" and p < 0.5 means "short".
Also returns r = realised move in ATR units (use it for expectancy / R-multiple reports).
"""
import numpy as np, pandas as pd


def triple_barrier(df, horizon, mult=1.0, atr_n=14):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / atr_n, adjust=False).mean().to_numpy()

    entry = np.r_[o[1:], np.nan]
    up, dn = entry + mult * atr, entry - mult * atr
    y, r = np.full(n, np.nan), np.full(n, np.nan)
    done = np.zeros(n, bool)

    def ahead(a, j):                       # a[t + j], NaN past the end
        return np.r_[a[j:], np.full(j, np.nan)]

    for j in range(1, horizon + 1):
        hit_up, hit_dn = ahead(h, j) >= up, ahead(l, j) <= dn
        first = ~done & (hit_up | hit_dn)
        won = hit_up & ~hit_dn             # same-bar double touch counts as a loss
        y[first], r[first] = won[first].astype(float), np.where(won, mult, -mult)[first]
        done |= first

    end = ahead(c, horizon)                # timeouts: sign of the move at the horizon
    t = ~done & np.isfinite(end) & np.isfinite(entry)
    y[t] = (end[t] > entry[t]).astype(float)
    r[t] = (end[t] - entry[t]) / atr[t]
    bad = ~np.isfinite(end) | ~np.isfinite(entry) | ~np.isfinite(atr)   # tail bars: future unknown
    y[bad] = r[bad] = np.nan
    return pd.Series(y, df.index), pd.Series(r, df.index)
