"""Uncertainty for DEPENDENT trades. Setups on 8 correlated coins arrive in bursts (same weeks, same volatility regimes), so treating
every trade as an independent draw overstates precision. Here trades are grouped into clusters (default: calendar week of the signal) and
whole clusters are resampled. `design_effect` > 1 means the iid standard error was too small by sqrt(design_effect)."""
import math
import numpy as np, pandas as pd


def week_clusters(ts):
    """integer cluster id per timestamp = calendar week (UTC, Monday start); ts is a DatetimeIndex/Series of signal times"""
    t = pd.DatetimeIndex(ts)
    t = t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t
    return pd.factorize(t.to_period("W").astype(str))[0]


def _sums(R, cl):
    R, cl = np.asarray(R, float), np.asarray(cl)
    k = cl.max() + 1
    S, N = np.bincount(cl, weights=R, minlength=k), np.bincount(cl, minlength=k).astype(float)
    keep = N > 0
    return S[keep], N[keep]


def cluster_boot_mean(R, cl, B=2000, seed=1):
    """mean R with a cluster-bootstrap 95% CI and one-sided p-value for 'true mean <= 0'"""
    R = np.asarray(R, float)
    if len(R) < 20:
        return dict(n=len(R), mean=float(R.mean()) if len(R) else 0.0, lo=0.0, hi=0.0, p=1.0, clusters=0)
    S, N = _sums(R, cl)
    K = len(S)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, K, size=(B, K))
    m = S[pick].sum(1) / N[pick].sum(1)
    obs = float(R.mean())
    p = float(((m - m.mean()) >= obs).sum() + 1) / (B + 1)
    return dict(n=int(len(R)), mean=obs, lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)), p=p, clusters=int(K))


def iid_mean(R):
    R = np.asarray(R, float)
    n = len(R)
    if n < 20:
        return dict(n=n, mean=float(R.mean()) if n else 0.0, lo=0.0, hi=0.0, p=1.0)
    se = R.std(ddof=1) / math.sqrt(n)
    z = R.mean() / se if se > 0 else 0.0
    return dict(n=n, mean=float(R.mean()), lo=float(R.mean() - 1.96 * se), hi=float(R.mean() + 1.96 * se), p=float(0.5 * math.erfc(z / math.sqrt(2))))


def design_effect(R, cl):
    """variance of the mean under clustering / variance under independence. effective n = n / design_effect"""
    R = np.asarray(R, float)
    n = len(R)
    S, N = _sums(R, cl)
    K = len(S)
    if n < 20 or K < 3:
        return 1.0
    mu = R.mean()
    v_cluster = (K / (K - 1)) * np.sum((S - mu * N) ** 2) / n ** 2
    v_iid = R.var(ddof=1) / n
    return float(max(v_cluster / v_iid, 1e-9)) if v_iid > 0 else 1.0


def summarize(R, ts, B=2000, seed=1):
    cl = week_clusters(ts)
    c, i = cluster_boot_mean(R, cl, B, seed), iid_mean(R)
    de = design_effect(R, cl)
    return dict(n=c["n"], mean_R=c["mean"], iid_lo=i["lo"], iid_hi=i["hi"], p_iid=i["p"], lo=c["lo"], hi=c["hi"], p=c["p"],
                clusters=c["clusters"], design_effect=de, ess=float(c["n"] / de) if de > 0 else float(c["n"]))


def block_boot_mean(x, block=20, B=2000, seed=1):
    """moving-block bootstrap for ONE time-ordered series (e.g. daily strategy returns)"""
    x = np.asarray(x, float)
    T = len(x)
    if T < 2 * block:
        return dict(mean=float(x.mean()) if T else 0.0, lo=0.0, hi=0.0, p=1.0)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(B, nb))[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T]
    m = x[idx].mean(1)
    obs = float(x.mean())
    return dict(mean=obs, lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)), p=float(((m - m.mean()) >= obs).sum() + 1) / (B + 1))


def profit_factor(R):
    R = np.asarray(R, float)
    gl = -R[R < 0].sum()
    return float(R[R > 0].sum() / gl) if gl > 0 else float("inf")
