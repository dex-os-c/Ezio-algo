"""ICT setup generator + trade simulator.

A LONG setup (shorts are the exact mirror, computed on the price-negated series):
  1. SWEEP   a bar wicks below the last confirmed swing low and closes back above it (sell-side liquidity taken)
  2. MSS     within `win` bars a close above the last internal swing high / the sweep bar's high (market-structure shift)
  3. ZONE    where to buy the pullback, first valid one (entry_mode="any"):  FVG midpoint > order-block midpoint > 70.5% retrace (OTE)
             or entry_mode in {fvg, ob, ote, mkt}; "mkt" = market buy at the OPEN of the bar after the MSS bar
  4. ENTRY   limit order at that level, STOP just below the sweep low (stop_mode="structure") or entry - atr_stop*ATR (stop_mode="atr"),
             TARGET at `rr` x risk

Everything used to ARM a setup is known at the close of the MSS bar (the 'signal bar'). The simulator then walks forward with
conservative rules: a bar that touches both stop and target counts as a stop; the target is not counted on a limit order's fill bar;
costs are charged in R (limit entry/target pay the maker fee; market entries, stops and timeouts pay taker fee + slippage).
AUDIT FIX: the original code cancelled a setup whenever the fill bar's high reached the target, which silently dropped some winners
while keeping fill-bar losers (legacy_cancel=True reproduces that). Now the order fills first and the trade continues, with the target
credited only from the NEXT bar. Optional adverse-execution switches (all off by default): fill_pen_bps = price must trade THROUGH a limit by
that many bps (touching is not enough, for entry and target), gap_stop = a stop that is gapped through exits at the open, and limit
orders fill at the open when it gaps through them. Unresolved trades at the end of the data are dropped.
"""
import numpy as np, pandas as pd
from smc import atr as _atr, confirmed_swings

DEFAULT = dict(sweep_len=4, mss_len=3, win=30, fill_win=24, rr=2.0, max_hold=48, min_risk=0.3, max_risk=5.0, buf=0.1,
               maker_bps=2.0, taker_bps=5.0, slip_bps=2.0, entry_mode="any", stop_mode="structure", atr_stop=1.5,
               fill_pen_bps=0.0, gap_stop=False, legacy_cancel=False)
MODES = {"fvg": 0, "ob": 1, "ote": 2, "mkt": 3}


def _mirror(df):
    m = pd.DataFrame(index=df.index)
    m["open"], m["high"], m["low"], m["close"] = -df.open, -df.low, -df.high, -df.close
    m["volume"] = df.volume
    return m


def _simulate(o, h, l, c, i, entry, stop, tp, rr, p, market=False):
    """walk forward from signal bar i. -> (fill_bar, exit_bar, gross_R, outcome, exit_px, fill_px) or None (never filled / missed / unresolved).
    R is measured against the planned risk (entry - stop), so a gap through the stop can lose MORE than 1R when gap_stop is on."""
    n, risk0, pen, gap = len(c), entry - stop, p["fill_pen_bps"] / 1e4, p["gap_stop"]
    if market:
        j = i + 1
        if j >= n:
            return None
        fill_px = entry                                   # market order at the open of bar j (entry == o[j])
    else:
        j = -1
        for t in range(i + 1, min(i + 1 + p["fill_win"], n)):
            filled = l[t] <= entry * (1 - pen)
            if p["legacy_cancel"] and h[t] >= tp:         # ORIGINAL behaviour: dropped any setup whose fill bar also reached the target
                return None
            if filled:
                j = t
                break
            if h[t] >= tp:                                # target traded without us ever getting in: setup is dead
                return None
        if j < 0:
            return None
        fill_px = min(entry, o[j]) if gap else entry      # a gap through a limit fills at the open
    end = min(j + p["max_hold"], n - 1)
    for t in range(j, end + 1):
        if l[t] <= stop or (gap and t == j and not market and fill_px <= stop):   # stop is checked before target (conservative)
            px = min(stop, o[t]) if gap else stop
            return j, t, (px - fill_px) / risk0, "sl", px, fill_px
        if (t > j or market) and h[t] >= tp * (1 + pen):  # the target does not count on a limit order's own fill bar
            return j, t, (tp - fill_px) / risk0, "tp", tp, fill_px
    if j + p["max_hold"] > n - 1:                         # data ended before the trade resolved
        return None
    return j, end, (c[end] - fill_px) / risk0, "timeout", c[end], fill_px


def _scan(df, p, fn):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    a = _atr(df).to_numpy()
    sh_s, sl_s = confirmed_swings(df, p["sweep_len"])[:2]
    sl_v, sh_v = sl_s.to_numpy(), sh_s.to_numpy()
    ish = confirmed_swings(df, p["mss_len"])[0].to_numpy()
    out, s, swing, ext, lvl = [], -1, np.nan, np.nan, np.nan
    for i in range(2, n):
        ai = a[i]
        if not ai > 0:
            continue
        if sl_v[i] == sl_v[i] and l[i] < sl_v[i] and c[i] > sl_v[i]:           # 1. sweep (re)starts the setup
            s, swing, ext = i, sl_v[i], l[i]
            fn["sweep"] += 1
            lvl = max(ish[i], h[i]) if ish[i] == ish[i] else h[i]
        elif s >= 0:
            ext = min(ext, l[i])
            if i - s > p["win"]:
                s = -1
                continue
            if c[i] > lvl:                                                       # 2. market-structure shift
                s0, s = s, -1
                fn["mss"] += 1
                zones = []
                if p["entry_mode"] == "mkt":
                    if i + 1 < n:
                        zones.append(("mkt", o[i + 1], 0.0))
                else:
                    for k in range(i, s0 + 1, -1):                               # FVG: latest bullish gap inside the move
                        if l[k] > h[k - 2]:
                            zones.append(("fvg", (h[k - 2] + l[k]) / 2, l[k] - h[k - 2]))
                            break
                    for k in range(i - 1, s0 - 1, -1):                           # OB: last down-candle before the move
                        if c[k] < o[k]:
                            zones.append(("ob", (h[k] + l[k]) / 2, h[k] - l[k]))
                            break
                    top = h[s0:i + 1].max()
                    zones.append(("ote", top - 0.705 * (top - ext), 0.0))
                    if p["entry_mode"] != "any":
                        zones = [z for z in zones if z[0] == p["entry_mode"]]
                pick = None
                for name, entry, width in zones:                                 # first valid zone in priority order
                    stop = entry - p["atr_stop"] * ai if p["stop_mode"] == "atr" else ext - p["buf"] * ai
                    risk = entry - stop
                    if (entry < c[i] or name == "mkt") and p["min_risk"] * ai <= risk <= p["max_risk"] * ai:
                        pick = (name, entry, width, stop, risk)
                        break
                if pick is None:
                    continue
                fn["zone"] += 1
                name, entry, width, stop, risk = pick
                tp = entry + p["rr"] * risk
                sim = _simulate(o, h, l, c, i, entry, stop, tp, p["rr"], p, market=(name == "mkt"))
                if sim is None:
                    continue
                fn["trade"] += 1
                fill, ex, rg, how, exit_px, fill_px = sim
                in_bps = p["taker_bps"] + p["slip_bps"] if name == "mkt" else p["maker_bps"]
                exit_bps = p["maker_bps"] if how == "tp" else p["taker_bps"] + p["slip_bps"]
                cost_r = (in_bps + exit_bps) / 1e4 * abs(entry) / risk
                tgt = (sh_v[i] - entry) / risk if sh_v[i] == sh_v[i] and sh_v[i] > entry else 10.0
                out.append(dict(
                    sig=i, entry=entry, stop=stop, tp=tp, risk=risk, fill=fill, exit=ex, outcome=how, exit_px=exit_px, fill_px=fill_px,
                    R_gross=rg, R=rg - cost_r, cost_R=cost_r,
                    s_mode=MODES[name], s_sweep_depth=(swing - ext) / ai, s_sweep_age=i - s0, s_mss_atr=(c[i] - lvl) / ai,
                    s_body_atr=(c[i] - o[i]) / ai, s_zone_atr=width / ai, s_risk_atr=risk / ai,
                    s_risk_pct=risk / abs(entry), s_retrace=(c[i] - entry) / risk, s_tgt_R=min(tgt, 10.0)))
    return out


def find_setups(df, funnel=None, **kw):
    """all simulated ICT setups (both directions), one row each, sorted by signal bar.
    funnel (optional dict) receives how many candidates survived each stage: sweep -> mss -> zone -> trade.
    Columns include signal/fill/exit timestamps, planned entry/stop/target, actual fill/exit prices, gross R, cost in R and net R."""
    p = {**DEFAULT, **kw}
    rows, fn = [], dict(sweep=0, mss=0, zone=0, trade=0)
    for d, src in ((1, df), (-1, _mirror(df))):
        for e in _scan(src, p, fn):
            e["dir"] = d
            if d < 0:
                for k in ("entry", "stop", "tp", "exit_px", "fill_px"):
                    e[k] = -e[k]
            rows.append(e)
    if funnel is not None:
        funnel.update(fn)
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev = ev.sort_values("sig", kind="stable").reset_index(drop=True)
    ev.insert(1, "ts", df.index[ev.sig.values])
    ev["fill_ts"], ev["exit_ts"] = df.index[ev.fill.values], df.index[ev.exit.values]
    return ev


# ---- context features taken from the main feature matrix at the signal bar, expressed relative to the trade direction ----
ALIGN = ["smc_trend", "smc_htf4h_trend", "smc_htf1d_trend", "smc_htf4h_pd", "smc_htf1d_pd", "smc_pd",
         "ret4", "ret16", "ret48", "rsi", "bb_z", "macd", "close_pos"]
SWAP = [("smc_d_asia_hi", "smc_d_asia_lo", "ctx_asia_ahead", "ctx_asia_behind"),
        ("smc_d_pdh", "smc_d_pdl", "ctx_pd_ahead", "ctx_pd_behind"),
        ("smc_sweep_pdh", "smc_sweep_pdl", "ctx_sweep_against", "ctx_sweep_with")]
NEUTRAL = ["atr", "vol16", "vol_ratio", "vol_z", "smc_kz_asia", "smc_kz_london", "smc_kz_ny_am", "smc_kz_ny_pm",
           "smc_bars_since_choch", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]


def attach_context(ev, F, index):
    """-> feature DataFrame X (one row per event). Drops nothing: rows without context get NaN (LightGBM handles it)."""
    Fa = F.reindex(index).iloc[ev.sig.values].reset_index(drop=True)
    d = ev["dir"].to_numpy(float)
    X = ev[[c for c in ev.columns if c.startswith("s_")]].reset_index(drop=True).copy()
    X["dir"] = d
    for c in ALIGN:
        if c in Fa:
            X[f"ctx_{c}"] = Fa[c].to_numpy() * d
    for hi, lo, n1, n2 in SWAP:
        if hi in Fa and lo in Fa:
            long = d > 0
            X[n1] = np.where(long, Fa[hi], Fa[lo])
            X[n2] = np.where(long, Fa[lo], Fa[hi])
    for c in NEUTRAL:
        if c in Fa:
            X[f"ctx_{c}"] = Fa[c].to_numpy()
    return X
