"""Simple baselines on DAILY closes under the same cost model, so ICT can be judged against them.

  buy_hold   equal weight across the coins that exist that day, long all the time
  ma_trend   per coin: long while SMA(50) > SMA(200), else cash          (classic golden-cross trend following)
  breakout   per coin: long after a close above the prior 55-day high, flat after a close below the prior 20-day low   (Donchian / Turtle)
  cash       nothing
All are long-only, unlevered, equal weight across the coins that exist, signals use only data up to the close and are traded at that
close (position change pays `cost_bps` per unit traded). Risk differs between strategies, so comparisons are also shown scaled to a common
volatility, and `beta_alpha` asks whether a strategy adds return beyond the market (buy_hold) with Newey-West standard errors."""
import math
import numpy as np, pandas as pd

BPY = 365


def _weights_from_flags(flags):
    """boolean DataFrame (coin in position?) -> equal-weight weights over the coins that exist; each live coin gets 1/N_live"""
    live = flags.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    return flags.fillna(False).astype(float).div(n, axis=0).fillna(0.0)


def ma_flags(close, fast=50, slow=200):
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    return (f > s).where(s.notna() & close.notna())


def breakout_flags(close, entry=55, exit_=20):
    hi = close.rolling(entry).max().shift(1)              # prior N days only
    lo = close.rolling(exit_).min().shift(1)
    out = pd.DataFrame(False, index=close.index, columns=close.columns)
    for c in close.columns:
        x, h, l_, on, arr = close[c].to_numpy(), hi[c].to_numpy(), lo[c].to_numpy(), False, np.zeros(len(close), bool)
        for i in range(len(x)):
            if x[i] == x[i]:
                if not on and h[i] == h[i] and x[i] > h[i]:
                    on = True
                elif on and l_[i] == l_[i] and x[i] < l_[i]:
                    on = False
            arr[i] = on
        out[c] = arr
    return out.where(close.notna())


def run_weights(w, close, cost_bps=7.0):
    """daily net returns of target weights decided at each close and held over the next day"""
    r = close / close.shift(1) - 1
    pos = w.shift(1).fillna(0.0)
    gross = (pos * r.fillna(0.0)).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1)
    turn.iloc[0] = pos.iloc[0].abs().sum()
    net = gross - turn * cost_bps / 1e4
    live = (w.abs().sum(axis=1) > 0) | (net != 0)
    return net.loc[live.idxmax():] if live.any() else net.iloc[:0]


def all_baselines(close, cost_bps=7.0):
    n_live = close.notna()
    always = n_live.copy()
    out = {"buy_hold": run_weights(_weights_from_flags(always.where(n_live)), close, cost_bps),
           "ma_trend(50/200)": run_weights(_weights_from_flags(ma_flags(close)), close, cost_bps),
           "breakout(55/20)": run_weights(_weights_from_flags(breakout_flags(close)), close, cost_bps)}
    first = max(s.index[0] for s in out.values() if len(s))
    out["cash"] = pd.Series(0.0, index=close.loc[first:].index)
    return out


def perf(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 30 or x.std() == 0:
        return dict(n=n, cagr=0.0, vol=0.0, sharpe=0.0, max_dd=0.0)
    eq = np.cumprod(1 + x)
    return dict(n=n, cagr=float(eq[-1] ** (BPY / n) - 1), vol=float(x.std(ddof=1) * math.sqrt(BPY)),
                sharpe=float(x.mean() / x.std(ddof=1) * math.sqrt(BPY)), max_dd=float((eq / np.maximum.accumulate(eq) - 1).min()))


def scaled(x, target_vol=0.20):
    """rescale a return series to a common annual volatility so strategies with different risk can be compared"""
    x = pd.Series(x)
    v = x.std() * math.sqrt(BPY)
    return x * (target_vol / v) if v > 0 else x


def beta_alpha(y, bench, lags=5):
    """OLS y = a + b * bench with Newey-West standard errors -> annual alpha, beta, t(alpha), t(beta)"""
    y, x = np.asarray(y, float), np.asarray(bench, float)
    T = min(len(y), len(x))
    y, x = y[-T:], x[-T:]
    X = np.c_[np.ones(T), x]
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    return dict(alpha_ann=float(beta[0] * BPY), beta=float(beta[1]), t_alpha=float(beta[0] / se[0]), t_beta=float(beta[1] / se[1]))


def compare(rets, bench="buy_hold", common_vol=0.20):
    """table of strategies on their COMMON dates: raw and vol-scaled performance, correlation and alpha/beta vs the benchmark"""
    idx = None
    for s in rets.values():
        if len(s):
            idx = s.index if idx is None else idx.intersection(s.index)
    rows = []
    for k, s in rets.items():
        s = s.reindex(idx).fillna(0.0)
        p, ps = perf(s), perf(scaled(s, common_vol))
        row = dict(strategy=k, days=p["n"], cagr=p["cagr"], vol=p["vol"], sharpe=p["sharpe"], max_dd=p["max_dd"],
                   cagr_at_20pct_vol=ps["cagr"], max_dd_at_20pct_vol=ps["max_dd"])
        if k != bench and s.std() > 0 and bench in rets:
            b = rets[bench].reindex(idx).fillna(0.0)
            row.update(corr_to_market=float(np.corrcoef(s, b)[0, 1]), **{k2: v for k2, v in beta_alpha(s, b).items()})
        rows.append(row)
    return pd.DataFrame(rows)
