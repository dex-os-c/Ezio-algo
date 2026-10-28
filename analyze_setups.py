"""Which kinds of ICT setup (if any) actually pay? Slice analysis with multiple-testing protection.

  python analyze_setups.py --symbols BTC/USDT,ETH/USDT,... --exchange binance --timeframe 1h --years 20

Rules that keep it honest:
  * every slice is measured on the DEV period only (the last 15% of events by time is never touched while exploring)
  * each slice is traded as its own rule: one position at a time per symbol, costs included, results in R
  * p-values are Bonferroni-adjusted for the number of slices examined (looking at 40 slices WILL show a few lucky ones)
  * only the single best slice (highest 95% lower bound, enough trades) is then tested ONCE on the holdout
  * yearly breakdown shows whether a slice works across regimes or only in one lucky year
"""
import json, math, os
import numpy as np, pandas as pd
from config import parse
from train_setups import EXTRA, load_events, stats, seq_filter, breakeven
from setups import DEFAULT


def pval(R):
    """one-sided p that the true mean R is <= 0 (normal approximation)"""
    R = np.asarray(R, float)
    if len(R) < 30 or R.std(ddof=1) == 0:
        return 1.0
    t = R.mean() / (R.std(ddof=1) / math.sqrt(len(R)))
    return 0.5 * math.erfc(t / math.sqrt(2))


def slice_masks(ev, X, dev_idx):
    """name -> boolean mask over ALL events. Thresholds for terciles are learned on dev events only."""
    m, N = {}, len(ev)
    col = lambda c: X[c].to_numpy() if c in X else None
    m["long"], m["short"] = (ev.dir > 0).to_numpy(), (ev.dir < 0).to_numpy()
    for k, nme in enumerate(("entry=FVG", "entry=OB", "entry=OTE")):
        m[nme] = (ev.s_mode == k).to_numpy()
    for c, nme in (("ctx_smc_htf4h_trend", "4h trend"), ("ctx_smc_htf1d_trend", "1d trend"), ("ctx_smc_trend", "1h trend")):
        v = col(c)
        if v is not None:
            m[f"{nme} WITH trade"], m[f"{nme} AGAINST trade"] = v > 0, v < 0
    for c, nme in (("ctx_smc_pd", "1h range"), ("ctx_smc_htf4h_pd", "4h range"), ("ctx_smc_htf1d_pd", "1d range")):
        v = col(c)
        if v is not None:                                 # aligned value < 0 = buying discount / selling premium
            m[f"{nme}: right side (discount/premium)"], m[f"{nme}: wrong side"] = v < 0, v > 0
    v = col("ctx_sweep_with")
    if v is not None:
        m["same-day PDH/PDL sweep with trade"] = v > 0.5
    kz = {k: col(f"ctx_smc_kz_{k}") for k in ("asia", "london", "ny_am", "ny_pm")}
    if all(v is not None for v in kz.values()):
        for k, v in kz.items():
            m[f"killzone {k}"] = v > 0.5
        m["no killzone"] = ~np.any([v > 0.5 for v in kz.values()], axis=0)
    v = col("ctx_vol_ratio")
    if v is not None:
        m["vol expanding"], m["vol contracting"] = v > 1.0, v <= 1.0
    for c, nme in (("s_risk_atr", "stop size"), ("s_sweep_depth", "sweep depth"), ("s_zone_atr", "zone width")):
        v = ev[c].to_numpy() if c in ev else X[c].to_numpy()
        q1, q2 = np.quantile(v[dev_idx], [1 / 3, 2 / 3])
        m[f"{nme} small"], m[f"{nme} medium"], m[f"{nme} large"] = v <= q1, (v > q1) & (v <= q2), v > q2
    t = X["s_tgt_R"].to_numpy()
    m["liquidity target <2R away"], m["liquidity target 2-5R"], m["liquidity target >5R"] = t < 2, (t >= 2) & (t < 5), t >= 5
    a = X["s_sweep_age"].to_numpy()
    m["fast shift (<=3 bars)"], m["slow shift (>10 bars)"] = a <= 3, a > 10
    # ICT confluence score: how many of the textbook conditions line up
    parts = [m.get("4h trend WITH trade"), m.get("1d trend WITH trade"), m.get("1h trend WITH trade"),
             m.get("1h range: right side (discount/premium)"), m.get("same-day PDH/PDL sweep with trade")]
    parts = [p for p in parts if p is not None]
    if parts:
        score = np.sum(parts, axis=0)
        for s in range(0, len(parts) + 1):
            m[f"confluence score {s}/{len(parts)}"] = score == s
        m[f"confluence >= {max(len(parts) - 1, 1)}/{len(parts)}"] = score >= max(len(parts) - 1, 1)
    return m


def evaluate(ev, X, hold_frac=0.15, min_n=100, log=print):
    N = len(ev)
    R, sig, ex, sym = ev.R.to_numpy(), ev.sig_t.to_numpy(), ev.ex_t.to_numpy(), ev.sym.to_numpy()
    hold = int(N * hold_frac)
    dev_idx, ho_idx = np.arange(N - hold), np.arange(N - hold, N)
    masks = slice_masks(ev, X, dev_idx)
    base = stats(R[seq_filter(sig, ex, dev_idx, sym)])
    rows = []
    for name, mk in masks.items():
        idx = dev_idx[mk[dev_idx]]
        pick = seq_filter(sig, ex, idx, sym)
        if len(pick) >= min_n:
            s = stats(R[pick])
            rows.append(dict(name=name, p=pval(R[pick]), **s))
    m = len(rows)
    for r in rows:
        r["p_adj"] = min(1.0, r["p"] * m)
    rows.sort(key=lambda r: -r["exp_R"])
    f = lambda r: (f"{r['name'][:44]:44s} n={r['n']:4d} win {r['win']*100:2.0f}% exp {r['exp_R']:+.3f}R "
                   f"(low {r['lcb95']:+.3f}) adj-p {r['p_adj']:.3f}{'  <-- survives correction' if r['p_adj'] < 0.05 and r['lcb95'] > 0 else ''}")
    log(f"DEV baseline (all setups): n={base['n']} exp {base['exp_R']:+.3f}R (low {base['lcb95']:+.3f})")
    log(f"{m} slices with >= {min_n} trades; p-values multiplied by {m} (Bonferroni)\n")
    log("--- best 10 slices on DEV ---")
    for r in rows[:10]:
        log(f(r))
    log("--- worst 5 slices on DEV ---")
    for r in rows[-5:]:
        log(f(r))
    cands = [r for r in rows if r["n"] >= 150]
    out = dict(baseline=base, slices=rows, n_slices=m, best=None, holdout=None, survives=False, candidate=False)
    if not cands:
        log("\nno slice has enough trades to test")
        return out
    best = max(cands, key=lambda r: r["lcb95"])
    out["best"], out["survives"] = best, bool(best["p_adj"] < 0.05 and best["lcb95"] > 0)
    log(f"\nBEST by 95% lower bound: {best['name']}")
    mk = masks[best["name"]]
    yrs = ev.ts.dt.year.to_numpy()
    pick_b = seq_filter(sig, ex, dev_idx[mk[dev_idx]], sym)
    log("  expectancy by year (best slice): " + " | ".join(
        f"{y}: {R[pick_b][yrs[pick_b] == y].mean():+.2f}R (n={(yrs[pick_b] == y).sum()})" for y in sorted(set(yrs[pick_b])) if (yrs[pick_b] == y).sum() >= 10))
    pick_a = seq_filter(sig, ex, dev_idx, sym)
    log("  expectancy by year (all setups): " + " | ".join(
        f"{y}: {R[pick_a][yrs[pick_a] == y].mean():+.2f}R" for y in sorted(set(yrs[pick_a])) if (yrs[pick_a] == y).sum() >= 10))
    ho = seq_filter(sig, ex, ho_idx[mk[ho_idx]], sym)
    sh = stats(R[ho])
    out["holdout"] = sh
    log(f"\nHOLDOUT (looked at once): {best['name']}: n={sh['n']} win {sh['win']*100:.0f}% exp {sh['exp_R']:+.3f}R (low {sh['lcb95']:+.3f})")
    sb = stats(R[seq_filter(sig, ex, ho_idx, sym)])
    log(f"HOLDOUT all setups       : n={sb['n']} exp {sb['exp_R']:+.3f}R")
    out["candidate"] = bool(out["survives"] and sh["n"] >= 30 and sh["exp_R"] > 0)
    return out


def main():
    cfg = parse(__doc__, EXTRA)
    a = cfg.args
    sp = dict(rr=a.rr, win=a.win, fill_win=a.fill_win, max_hold=a.max_hold, sweep_len=a.sweep_len, max_risk=a.max_risk,
              entry_mode=a.entry_mode, maker_bps=a.maker_bps, taker_bps=cfg.fee_bps, slip_bps=cfg.slip_bps)
    ev, X, funnel = load_events(cfg, sp, print)
    print(f"\n{len(ev)} setups | break-even win rate {breakeven(ev, sp['rr'])*100:.0f}% at {sp['rr']:g}R\n")
    out = evaluate(ev, X)
    os.makedirs("runs", exist_ok=True)
    p = "runs/slice_looks.json"
    looks = json.load(open(p)) if os.path.exists(p) else {}
    key = f"{a.symbols or cfg.symbol}|{cfg.timeframe}"
    looks[key] = looks.get(key, 0) + 1
    json.dump(looks, open(p, "w"))
    print(f"\nholdout looks on this market set + timeframe so far: {looks[key]} (each extra look makes the holdout less trustworthy)")
    print("VERDICT:", "CANDIDATE: a slice survives correction AND is positive on the untouched holdout. Paper trade it before anything else."
          if out["candidate"] else "No slice passes both the multiple-testing correction and the holdout. No tradable edge found here.")
    os.makedirs(cfg.out, exist_ok=True)
    json.dump({k: v for k, v in out.items()}, open(f"{cfg.out}/slice_report.json", "w"), default=float)


if __name__ == "__main__":
    main()
