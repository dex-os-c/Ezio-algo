"""Auction Market Theory (Market Profile style) features from OHLCV candles. Causal: row t only uses bars <= t.

Honest note: real Market Profile uses TPO letters / tick data. Here the volume of every candle is spread evenly across its
high-low range to build a volume-at-price profile, which is the standard approximation when you only have candles.
Outputs: POC (point of control), VAH / VAL (70% value area), balance vs imbalance, value migration, rejection tails,
failed auctions at the value-area edges, and the PREVIOUS DAY's value area (the levels auction traders actually watch).
"""
import numpy as np, pandas as pd
from smc import atr


def _profile(lo, hi, vol, edges):
    top = np.minimum(hi[:, None], edges[None, 1:])
    bot = np.maximum(lo[:, None], edges[None, :-1])
    ov = np.clip(top - bot, 0, None)
    return (ov / np.maximum(hi - lo, 1e-12)[:, None] * vol[:, None]).sum(0)


def _value_area(p, pct=0.7):
    poc = int(p.argmax())
    lo = hi = poc
    acc, tot = p[poc], p.sum()
    while acc < pct * tot and (lo > 0 or hi < len(p) - 1):
        up = p[hi + 1] if hi < len(p) - 1 else -1.0
        dn = p[lo - 1] if lo > 0 else -1.0
        if up >= dn:
            hi += 1
            acc += p[hi]
        else:
            lo -= 1
            acc += p[lo]
    return poc, lo, hi


def levels(lo, hi, vol, nb=40):
    """(poc, val, vah) for a block of candles"""
    mn, mx = lo.min(), hi.max()
    if not mx > mn:
        return np.nan, np.nan, np.nan
    edges = np.linspace(mn, mx, nb + 1)
    p = _profile(lo, hi, vol, edges)
    ip, il, ih = _value_area(p)
    return (edges[ip] + edges[ip + 1]) / 2, edges[il], edges[ih + 1]


def rolling_levels(df, win, nb=40):
    l, h, v, n = df.low.values, df.high.values, df.volume.values, len(df)
    poc, val, vah = (np.full(n, np.nan) for _ in range(3))
    for t in range(win - 1, n):
        s = slice(t - win + 1, t + 1)
        poc[t], val[t], vah[t] = levels(l[s], h[s], v[s], nb)
    return poc, val, vah


def previous_day_levels(df):
    day = df.index.normalize()
    rows = {}
    for d, g in df.groupby(day):
        rows[d] = levels(g.low.values, g.high.values, g.volume.values, 30)
    lv = pd.DataFrame.from_dict(rows, orient="index", columns=["poc", "val", "vah"])
    return lv.shift(1, freq="D").reindex(day)


def amt_features(df, tf_min):
    """returns (features DataFrame, overlays dict with the latest profile levels per window)"""
    n = len(df)
    a = atr(df).values
    a = np.where(a > 0, a, np.nan)
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    bars_day = max(round(1440 / tf_min), 1)
    wins = {"s": max(bars_day, 10), "l": max(7 * bars_day, 40)}
    f = pd.DataFrame(index=df.index)
    lvl = {}
    for name, w in wins.items():
        poc, val, vah = rolling_levels(df, w)
        lvl[name] = (poc, val, vah)
        f[f"amt_{name}_d_poc"] = (c - poc) / a
        f[f"amt_{name}_d_vah"] = (c - vah) / a
        f[f"amt_{name}_d_val"] = (c - val) / a
        f[f"amt_{name}_in_value"] = ((c >= val) & (c <= vah)).astype(float)
        f[f"amt_{name}_va_width"] = (vah - val) / a
        k = max(w // 4, 1)
        f[f"amt_{name}_poc_shift"] = (poc - pd.Series(poc).shift(k).values) / a
        pv, pl = pd.Series(vah).shift(1).values, pd.Series(val).shift(1).values   # previous bar's value area
        f[f"amt_{name}_reject_vah"] = ((h > pv) & (c < pv)).astype(float)        # failed auction above value
        f[f"amt_{name}_reject_val"] = ((l < pl) & (c > pl)).astype(float)        # failed auction below value
    w = wins["s"]
    cs = pd.Series(c)
    f["amt_eff_ratio"] = ((cs - cs.shift(w)).abs() / cs.diff().abs().rolling(w).sum()).values   # 1 = trending, 0 = balanced
    f["amt_range_atr"] = ((pd.Series(h).rolling(w).max() - pd.Series(l).rolling(w).min()) / a / np.sqrt(w)).values
    delta = pd.Series((c - o) / np.maximum(h - l, 1e-12) * v)
    f["amt_delta"] = (delta.rolling(w).sum() / pd.Series(v).rolling(w).sum()).values             # volume-delta proxy
    f["amt_upper_tail"] = (h - np.maximum(o, c)) / a
    f["amt_lower_tail"] = (np.minimum(o, c) - l) / a
    if tf_min <= 240:
        pdl = previous_day_levels(df)
        for k in ("poc", "val", "vah"):
            f[f"amt_pd_d_{k}"] = (c - pdl[k].values) / a
    ov = {"series": {k: {"poc": x[0], "val": x[1], "vah": x[2]} for k, x in lvl.items()}}   # numpy arrays; the API cleans NaNs
    return f.replace([np.inf, -np.inf], np.nan), ov
