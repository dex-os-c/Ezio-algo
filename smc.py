"""Smart-Money-Concepts / ICT features, computed bar by bar with NO lookahead.

Key rule: a swing high/low is only *known* L bars after it happened (it needs L lower highs after it), so every swing
used at bar t was confirmed at or before t. The same single pass also records the structures (FVGs, order blocks, BOS/CHoCH,
sweeps, swings) so the frontend can draw exactly what the model saw.
"""
import numpy as np, pandas as pd

LEN_INT, LEN_EXT = 3, 10      # internal / external swing length (bars on each side)
DISP_ATR = 1.5                # displacement candle: body > 1.5 ATR
MAX_ZONES = 24


def atr(df, n=14):
    c = df.close
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def confirmed_swings(df, L):
    """last confirmed swing high/low price + bar position, as of each bar (known only after L bars)"""
    h, l, n = df.high, df.low, len(df)
    pos = pd.Series(np.arange(n, dtype=float), index=df.index)
    is_sh = h == h.rolling(2 * L + 1, center=True).max()
    is_sl = l == l.rolling(2 * L + 1, center=True).min()
    sh = h.where(is_sh).shift(L).ffill()
    sl = l.where(is_sl).shift(L).ffill()
    shp = pos.where(is_sh).shift(L).ffill()
    slp = pos.where(is_sl).shift(L).ffill()
    return sh, sl, shp, slp


def _decay(events, k=12):
    """event flags -> 'how recent' feature in (0,1]: 1 on the bar, exp decay after"""
    out, v = np.zeros(len(events)), 0.0
    for i, e in enumerate(events):
        v = 1.0 if e else v * np.exp(-1.0 / k)
        out[i] = v
    return out


def structure_trend(df, L=LEN_EXT):
    """+1 / -1 market-structure trend (BOS / CHoCH logic) per bar. Used for higher-timeframe bias."""
    sh, sl, _, _ = confirmed_swings(df, L)
    c, trend, t, used_h, used_l = df.close.values, np.zeros(len(df)), 0, np.nan, np.nan
    shv, slv = sh.values, sl.values
    for i in range(len(df)):
        if shv[i] == shv[i] and c[i] > shv[i] and shv[i] != used_h:
            t, used_h = 1, shv[i]
        elif slv[i] == slv[i] and c[i] < slv[i] and slv[i] != used_l:
            t, used_l = -1, slv[i]
        trend[i] = t
    return pd.Series(trend, index=df.index)


def smc_features(df, tf_min):
    """returns (features DataFrame, overlays dict)"""
    n = len(df)
    a = atr(df).values
    a = np.where(a > 0, a, np.nan)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    idx = df.index
    sh, sl, shp, slp = confirmed_swings(df, LEN_EXT)
    ish, isl, _, _ = confirmed_swings(df, LEN_INT)
    shv, slv, shpv, slpv = sh.values, sl.values, shp.values, slp.values

    trend = np.zeros(n)
    ev = {k: np.zeros(n, bool) for k in ("bos_up", "bos_dn", "choch_up", "choch_dn", "sweep_hi", "sweep_lo", "disp_up", "disp_dn")}
    t, used_h, used_l = 0, np.nan, np.nan
    bull_fvg, bear_fvg, bull_ob, bear_ob = [], [], [], []        # active zones: [t0, bottom, top]
    overlay = {"bos": [], "fvg": [], "ob": [], "sweeps": [], "swings": []}
    cols = {k: np.full(n, np.nan) for k in (
        "d_bull_fvg", "d_bear_fvg", "n_bull_fvg", "n_bear_fvg", "in_bull_fvg", "in_bear_fvg",
        "d_bull_ob", "d_bear_ob", "n_bull_ob", "n_bear_ob", "in_bull_ob", "in_bear_ob")}

    for i in range(n):
        ai = a[i]
        body = c[i] - o[i]
        if ai == ai:
            if body > DISP_ATR * ai: ev["disp_up"][i] = True
            if -body > DISP_ATR * ai: ev["disp_dn"][i] = True
        # ---- liquidity sweeps: wick beyond the last confirmed swing, close back inside
        if shv[i] == shv[i] and h[i] > shv[i] and c[i] < shv[i]:
            ev["sweep_hi"][i] = True
            overlay["sweeps"].append((i, "hi", float(shv[i])))
        if slv[i] == slv[i] and l[i] < slv[i] and c[i] > slv[i]:
            ev["sweep_lo"][i] = True
            overlay["sweeps"].append((i, "lo", float(slv[i])))
        # ---- structure: close beyond the last confirmed swing = BOS (with trend) or CHoCH (against it)
        broke = 0
        if shv[i] == shv[i] and c[i] > shv[i] and shv[i] != used_h:
            used_h, broke = shv[i], 1
            ev["choch_up" if t <= 0 else "bos_up"][i] = True
            overlay["bos"].append((i, "choch" if t <= 0 else "bos", 1, float(shv[i])))
            t = 1
        elif slv[i] == slv[i] and c[i] < slv[i] and slv[i] != used_l:
            used_l, broke = slv[i], -1
            ev["choch_dn" if t >= 0 else "bos_dn"][i] = True
            overlay["bos"].append((i, "choch" if t >= 0 else "bos", -1, float(slv[i])))
            t = -1
        trend[i] = t
        # ---- order block: last opposite candle before the move that broke structure
        if broke == 1 and slpv[i] == slpv[i]:
            for j in range(i - 1, max(int(slpv[i]) - 1, i - 30, 0), -1):
                if c[j] < o[j]:
                    bull_ob.append([j, l[j], h[j]]); overlay["ob"].append([j, 1, float(l[j]), float(h[j]), None, i]); break
        if broke == -1 and shpv[i] == shpv[i]:
            for j in range(i - 1, max(int(shpv[i]) - 1, i - 30, 0), -1):
                if c[j] > o[j]:
                    bear_ob.append([j, l[j], h[j]]); overlay["ob"].append([j, -1, float(l[j]), float(h[j]), None, i]); break
        # ---- fair value gaps (3-candle imbalance), recorded when the 3rd candle closes
        if i >= 2:
            if l[i] > h[i - 2]:
                bull_fvg.append([i - 1, h[i - 2], l[i]]); overlay["fvg"].append([i - 1, 1, float(h[i - 2]), float(l[i]), None, i])
            if h[i] < l[i - 2]:
                bear_fvg.append([i - 1, h[i], l[i - 2]]); overlay["fvg"].append([i - 1, -1, float(h[i]), float(l[i - 2]), None, i])
        # ---- mitigation: bullish zones die when price closes below them, bearish when it closes above
        for zones, kind, up in ((bull_fvg, "fvg", True), (bear_fvg, "fvg", False), (bull_ob, "ob", True), (bear_ob, "ob", False)):
            for z in zones[:]:
                dead = c[i] < z[1] if up else c[i] > z[2]
                if dead or i - z[0] > 600:
                    zones.remove(z)
                    for ov in reversed(overlay[kind]):
                        if ov[0] == z[0] and ov[4] is None and ov[2] == float(z[1]):
                            ov[4] = i
                            break
            del zones[:max(0, len(zones) - MAX_ZONES)]
        if ai == ai:
            px = c[i]
            below = [z for z in bull_fvg if z[2] <= px + 1e-12]
            above = [z for z in bear_fvg if z[1] >= px - 1e-12]
            cols["d_bull_fvg"][i] = min((px - z[2]) / ai for z in below) if below else 10.0
            cols["d_bear_fvg"][i] = min((z[1] - px) / ai for z in above) if above else 10.0
            cols["n_bull_fvg"][i], cols["n_bear_fvg"][i] = len(bull_fvg), len(bear_fvg)
            cols["in_bull_fvg"][i] = float(any(z[1] <= px <= z[2] for z in bull_fvg))
            cols["in_bear_fvg"][i] = float(any(z[1] <= px <= z[2] for z in bear_fvg))
            ob_b = [z for z in bull_ob if z[2] <= px + 1e-12]
            ob_s = [z for z in bear_ob if z[1] >= px - 1e-12]
            cols["d_bull_ob"][i] = min(((px - z[2]) / ai for z in ob_b)) if ob_b else 10.0
            cols["d_bear_ob"][i] = min(((z[1] - px) / ai for z in ob_s)) if ob_s else 10.0
            cols["n_bull_ob"][i], cols["n_bear_ob"][i] = len(bull_ob), len(bear_ob)
            cols["in_bull_ob"][i] = float(any(z[1] <= px <= z[2] for z in bull_ob))
            cols["in_bear_ob"][i] = float(any(z[1] <= px <= z[2] for z in bear_ob))

    f = pd.DataFrame(index=idx)
    f["smc_trend"] = trend
    for k, v in ev.items():
        f[f"smc_{k}"] = _decay(v)
    last, bsc = -1, np.zeros(n)
    for i in range(n):
        if ev["choch_up"][i] or ev["choch_dn"][i]:
            last = i
        bsc[i] = np.log1p(i - last) if last >= 0 else np.log1p(500)
    f["smc_bars_since_choch"] = bsc
    for k, v in cols.items():
        f[f"smc_{k}"] = v
    # distance to the liquidity pools (last confirmed swings), in ATR
    f["smc_d_swing_hi"] = (shv - c) / a
    f["smc_d_swing_lo"] = (c - slv) / a
    # premium / discount inside the current dealing range, and OTE (62-79% retrace) zones
    rng = shv - slv
    pd_ = np.where(rng > 0, (c - slv) / rng, np.nan)
    f["smc_pd"] = np.clip(pd_, -0.5, 1.5) - 0.5
    f["smc_ote_long"] = ((trend > 0) & (pd_ >= 0.21) & (pd_ <= 0.38)).astype(float)
    f["smc_ote_short"] = ((trend < 0) & (pd_ >= 0.62) & (pd_ <= 0.79)).astype(float)
    # internal-structure swing proximity
    f["smc_d_int_hi"] = (ish.values - c) / a
    f["smc_d_int_lo"] = (c - isl.values) / a
    f["smc_body_atr"] = (c - o) / a

    if tf_min <= 240:
        f = pd.concat([f, _killzones(df, a), _htf(df, tf_min)], axis=1)

    # confirmed swings for the chart (each is known LEN_EXT bars after it happened)
    is_sh = (df.high == df.high.rolling(2 * LEN_EXT + 1, center=True).max()).values
    is_sl = (df.low == df.low.rolling(2 * LEN_EXT + 1, center=True).min()).values
    overlay["swings"] = [(i, "hi", float(h[i])) for i in np.where(is_sh)[0] if i + LEN_EXT < n] + \
                        [(i, "lo", float(l[i])) for i in np.where(is_sl)[0] if i + LEN_EXT < n]
    return f.replace([np.inf, -np.inf], np.nan), overlay


def _killzones(df, a):
    """ICT killzones (New York time) + Asian-range liquidity (00:00-08:00 UTC, usable after it completes)"""
    ny = df.index.tz_convert("America/New_York")
    hr = ny.hour + ny.minute / 60
    f = pd.DataFrame(index=df.index)
    f["smc_kz_asia"] = ((hr >= 20) | (hr < 0)).astype(float)
    f["smc_kz_london"] = ((hr >= 2) & (hr < 5)).astype(float)
    f["smc_kz_ny_am"] = ((hr >= 7) & (hr < 10)).astype(float)
    f["smc_kz_ny_pm"] = ((hr >= 13.5) & (hr < 16)).astype(float)
    utc_h = df.index.hour
    day = df.index.normalize()
    asia = df[utc_h < 8].groupby(day[utc_h < 8]).agg(hi=("high", "max"), lo=("low", "min"))
    same = asia.reindex(day)
    prv = asia.shift(1, freq="D").reindex(day)
    use_same = utc_h >= 8
    hi = np.where(use_same, same.hi.values, prv.hi.values)
    lo = np.where(use_same, same.lo.values, prv.lo.values)
    # prior-day high / low (liquidity every ICT trader watches)
    dly = df.groupby(day).agg(hi=("high", "max"), lo=("low", "min")).shift(1, freq="D").reindex(day)
    c = df.close.values
    f["smc_d_asia_hi"] = (hi - c) / a
    f["smc_d_asia_lo"] = (c - lo) / a
    f["smc_d_pdh"] = (dly.hi.values - c) / a
    f["smc_d_pdl"] = (c - dly.lo.values) / a
    f["smc_sweep_pdh"] = ((df.high.values > dly.hi.values) & (c < dly.hi.values)).astype(float)
    f["smc_sweep_pdl"] = ((df.low.values < dly.lo.values) & (c > dly.lo.values)).astype(float)
    return f


def _htf(df, tf_min):
    """higher-timeframe bias (4h and 1d market structure). HTF bars only become visible once they have fully closed."""
    f = pd.DataFrame(index=df.index)
    for name, mins in (("4h", 240), ("1d", 1440)):
        if mins <= tf_min:
            continue
        r = df.resample(f"{mins}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        tr = structure_trend(r)
        sh, sl, _, _ = confirmed_swings(r, LEN_INT)
        pdv = ((r.close - sl) / (sh - sl)).clip(-0.5, 1.5) - 0.5
        vis = pd.DataFrame({"t": tr, "pd": pdv})
        vis.index = vis.index + pd.Timedelta(minutes=mins - tf_min)       # visible when its last base bar closes
        vis = vis[~vis.index.duplicated()].reindex(df.index, method="ffill")
        f[f"smc_htf{name}_trend"] = vis.t
        f[f"smc_htf{name}_pd"] = vis.pd
    return f
