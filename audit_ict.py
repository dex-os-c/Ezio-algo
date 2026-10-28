"""One command: audit + reproduce + correct + validate the ICT research.

  python audit_ict.py --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,LINK/USDT \\
         --timeframes 1h,4h --start 2018-10-29 --end "2026-10-28 02:00" --out research_out

Steps: 1 unit tests -> 2 data integrity -> 3 reproduce the ORIGINAL report (original code path + original statistics) ->
4 corrected simulator vs original (before/after) -> 5 execution and cost sensitivity -> 6 breakdowns with dependence-aware CIs ->
7 pre-registered hypotheses with multiple-testing correction (exploration data only) -> 8 portfolio view (capital, concurrency, exposure) ->
9 baselines (buy&hold, MA trend, breakout, cash) and market-beta test -> 10 verdict. Outputs: CSV/JSON files + audit_report.md in --out.
Reproducibility: use a FIXED --start/--end (a rolling --years window moves every day), download with --years 8.1 or more."""
import copy, json, os, time
import numpy as np, pandas as pd
from config import parse
from data import load_history
from data_quality import validate_ohlcv, to_ns
from features import build_features, clean
from setups import find_setups, attach_context, DEFAULT
from train_setups import seq_filter
from research_stats import summarize, profit_factor
from portfolio import portfolio_sim
import baselines as bl
import hypotheses as hy
import ledger as lg
import manifest as mf

DEFAULT_SYMS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,LINK/USDT"
EXTRA = [(["--symbols"], dict(type=str, default=DEFAULT_SYMS)),
         (["--timeframes"], dict(type=str, default="1h,4h")),
         (["--pf_risk"], dict(type=float, default=0.005, help="equity risked per trade in the portfolio view")),
         (["--max_concurrent"], dict(type=int, default=4, help="max open/pending trades in the portfolio view")),
         (["--boot"], dict(type=int, default=2000, help="bootstrap resamples")),
         (["--ledger"], dict(type=str, default=lg.PATH)),
         (["--skip_tests"], dict(action="store_true"))]

# numbers printed by the ORIGINAL pooled 8-coin report (pasted during development) that step 3 tries to reproduce
REPORTED = {"1h": dict(setups=4945, dev_n=3406, dev_exp=-0.044, best="liquidity target >5R", best_n=387, best_exp=0.076, ho_n=69, ho_exp=0.097),
            "4h": dict(setups=1126, dev_n=791, dev_exp=-0.038, best="1d trend AGAINST trade", best_n=434, best_exp=0.146, ho_n=82, ho_exp=0.030)}
ORIGINAL_8 = set(DEFAULT_SYMS.split(","))
MIN_N = 100                      # no conclusions from fewer trades than this


def fnum(x):
    return "n/a" if x is None else f"{x:+.3f}R"


def banner(msg, log):
    log(f"\n{'=' * 78}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'=' * 78}")


# ------------------------------------------------------------------ data + events
def load_ctx(cfg, tf, symbols, log):
    """per symbol: cleaned candles, feature matrix and a data-integrity report"""
    ctx, quality, files = {}, {}, []
    for i, sym in enumerate(symbols):
        c = copy.copy(cfg)
        c.symbol, c.timeframe, c.seed = sym, tf, cfg.seed + i
        try:
            df = load_history(c, log)
        except Exception as e:
            log(f"  {sym} {tf}: SKIPPED ({type(e).__name__}: {str(e)[:80]})")
            continue
        quality[sym] = validate_ohlcv(df, tf)
        ctx[sym] = (df, clean(build_features(df, c)))
        files.append(f"data/{cfg.exchange}_{sym.replace('/', '-')}_{tf}.csv")
    return ctx, quality, files


def build_events(ctx, sp, with_X=False, cut_ns=None, side="explore"):
    """all simulated setups of all symbols, sorted by signal time. With a freeze cutoff: side='explore' keeps setups that signalled AND closed
    before it, side='confirm' keeps the setups that signalled at/after it."""
    evs, Xs, funnel = [], [], dict(sweep=0, mss=0, zone=0, trade=0)
    for sym, (df, F) in ctx.items():
        fn = {}
        ev = find_setups(df, funnel=fn, **sp)
        for k in funnel:
            funnel[k] += fn[k]
        if ev.empty:
            continue
        ev = ev[ev.sig >= df.index.get_loc(F.index[0])].reset_index(drop=True)
        if with_X:
            Xs.append(attach_context(ev, F, df.index))
        ev["sym"], ev["sig_t"], ev["ex_t"] = sym, to_ns(df.index[ev.sig.values]), to_ns(df.index[ev.exit.values])
        evs.append(ev)
    ev = pd.concat(evs, ignore_index=True)
    order = np.argsort(ev.sig_t.to_numpy(), kind="stable")
    ev = ev.iloc[order].reset_index(drop=True)
    X = pd.concat(Xs, ignore_index=True).iloc[order].reset_index(drop=True) if with_X else None
    if cut_ns is not None:
        keep = ((ev.sig_t < cut_ns) & (ev.ex_t < cut_ns)).to_numpy() if side == "explore" else (ev.sig_t >= cut_ns).to_numpy()
        ev = ev[keep].reset_index(drop=True)
        X = X[keep].reset_index(drop=True) if with_X else None
    return ev, X, funnel


def split(ev, hold=0.15):
    """dev = first 85% of events that had fully CLOSED before the holdout's first signal (purged); holdout = the rest"""
    N = len(ev)
    h = int(N * hold)
    first = ev.sig_t.iloc[N - h]
    idx = np.arange(N)
    dev = idx[(idx < N - h) & (ev.ex_t.to_numpy() < first)]
    return dev, idx[N - h:]


def traded(ev, idx):
    """one position per symbol at a time (the rule the original report used)"""
    return seq_filter(ev.sig_t.to_numpy(), ev.ex_t.to_numpy(), idx, ev.sym.to_numpy()) if len(idx) else idx   # returns EVENT indices


def r_stats(ev, idx, B):
    sub = ev.iloc[idx]
    n = len(sub)
    if n < 20:
        return dict(n=n, win=0.0, avg_win_R=0.0, avg_loss_R=0.0, exp_R=0.0, pf=0.0, total_R=0.0, maxdd_R=0.0, ci_lo=0.0, ci_hi=0.0, p=1.0,
                    p_iid=1.0, iid_lo=0.0, iid_hi=0.0, ess=float(n), design_effect=1.0)
    R = sub.R.to_numpy()
    s = summarize(R, pd.DatetimeIndex(sub.ts), B)
    cum = np.cumsum(R[np.argsort(sub.ex_t.to_numpy(), kind="stable")])      # drawdown in the order trades CLOSE
    return dict(n=n, win=float((R > 0).mean()), avg_win_R=float(R[R > 0].mean()) if (R > 0).any() else 0.0,
                avg_loss_R=float(R[R <= 0].mean()) if (R <= 0).any() else 0.0, exp_R=s["mean_R"], pf=profit_factor(R), total_R=float(R.sum()),
                maxdd_R=float((cum - np.maximum.accumulate(cum)).min()), ci_lo=s["lo"], ci_hi=s["hi"], p=s["p"], p_iid=s["p_iid"],
                iid_lo=s["iid_lo"], iid_hi=s["iid_hi"], ess=s["ess"], design_effect=s["design_effect"])


def line(name, s):
    return (f"{name:34s} n={s['n']:5d} win {s['win']*100:3.0f}% avgW {s['avg_win_R']:+.2f}R avgL {s['avg_loss_R']:+.2f}R exp {s['exp_R']:+.3f}R "
            f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] PF {s['pf']:.2f} p={s['p']:.3f} (iid p={s['p_iid']:.3f}) eff.n {s['ess']:.0f}")


def params(cfg, **over):
    return {**{k: v for k, v in DEFAULT.items()}, "taker_bps": cfg.fee_bps, "slip_bps": cfg.slip_bps, **over}


# ------------------------------------------------------------------ step 3: reproduce the original report
def reproduce(cfg, tf, ctx, symbols, log, be):
    """original code path: original simulator behaviour (legacy_cancel) + original statistics (iid, unpurged split, slice search)"""
    from analyze_setups import evaluate
    ev, X, _ = be(ctx, params(cfg, legacy_cancel=True), with_X=True)
    lines = []
    out = evaluate(ev, X, log=lines.append)
    row = dict(tf=tf, setups=len(ev), dev_n=out["baseline"]["n"], dev_exp=out["baseline"]["exp_R"],
               best=(out["best"] or {}).get("name"), best_n=(out["best"] or {}).get("n"), best_exp=(out["best"] or {}).get("exp_R"),
               ho_n=(out["holdout"] or {}).get("n"), ho_exp=(out["holdout"] or {}).get("exp_R"))
    exp = REPORTED.get(tf)
    comparable = cfg.source != "synthetic" and exp is not None and set(symbols) == ORIGINAL_8
    if not comparable:
        row["status"] = "NOT COMPARABLE (different data/symbols than the original report)"
    else:
        ok = (abs(row["setups"] / exp["setups"] - 1) <= 0.03 and abs(row["dev_exp"] - exp["dev_exp"]) <= 0.03 and
              abs(row["dev_n"] / exp["dev_n"] - 1) <= 0.05)
        row["status"] = "REPRODUCED" if ok else "NOT REPRODUCED (data window / exchange data differ?)"
        row.update({f"reported_{k}": v for k, v in exp.items()})
    log(f"  original code path on this data: {len(ev)} setups | dev baseline n={row['dev_n']} exp {row['dev_exp']:+.3f}R | best slice "
        f"{row['best']} n={row['best_n']} exp {fnum(row['best_exp'])} | holdout n={row['ho_n']} exp {fnum(row['ho_exp'])}")
    if comparable:
        log(f"  reported in the original run   : {exp['setups']} setups | dev n={exp['dev_n']} exp {exp['dev_exp']:+.3f}R | best {exp['best']} "
            f"n={exp['best_n']} exp {exp['best_exp']:+.3f} | holdout n={exp['ho_n']} exp {exp['ho_exp']:+.3f}")
    log(f"  -> {row['status']}")
    return row


# ------------------------------------------------------------------ steps 4-7
def slice_rule(rule, X):
    c = lambda k: X[k].to_numpy() if k in X else None
    if rule == "htf_aligned":
        parts = [c("ctx_smc_htf4h_trend"), c("ctx_smc_htf1d_trend")]
        parts = [p > 0 for p in parts if p is not None]
        return np.logical_and.reduce(parts) if parts else np.zeros(len(X), bool)
    if rule == "vol_expanding":
        return c("ctx_vol_ratio") > 1.0
    if rule == "killzone":
        z = [c(f"ctx_smc_kz_{k}") for k in ("london", "ny_am", "ny_pm")]
        return np.logical_or.reduce([p > 0.5 for p in z if p is not None]) if any(p is not None for p in z) else np.zeros(len(X), bool)
    raise ValueError(rule)


def breakdown(ev, X, idx, B, log):
    rows = []
    sub = ev.iloc[idx]
    groups = {"coin": sub.sym.to_numpy(), "year": sub.ts.dt.year.to_numpy(), "direction": np.where(sub.dir > 0, "long", "short"),
              "entry": sub.s_mode.map({0: "FVG", 1: "OB", 2: "OTE", 3: "market"}).to_numpy()}
    Xs = X.iloc[idx]
    if "ctx_smc_htf1d_trend" in Xs:
        groups["1d regime"] = np.where(Xs.ctx_smc_htf1d_trend.to_numpy() > 0, "with 1d trend", np.where(Xs.ctx_smc_htf1d_trend.to_numpy() < 0, "against 1d trend", "none"))
    if "ctx_vol_ratio" in Xs:
        groups["vol regime"] = np.where(Xs.ctx_vol_ratio.to_numpy() > 1, "expanding", "contracting")
    for g, labels in groups.items():
        for lab in sorted(set(labels), key=str):
            pos = idx[labels == lab]
            s = r_stats(ev, pos, B)
            rows.append(dict(group=g, level=str(lab), **s, enough=s["n"] >= MIN_N))
    return pd.DataFrame(rows)


def run_timeframe(cfg, a, tf, symbols, out, log, L):
    banner(f"TIMEFRAME {tf}", log)
    ctx, quality, files = load_ctx(cfg, tf, symbols, log)
    if not ctx:
        log("no data")
        return None
    res = dict(tf=tf, quality=quality, files=files)
    B = a.boot
    import freeze as fz
    cut = fz.cutoff() if cfg.confirm else None
    if cfg.confirm and cut is None:
        raise SystemExit("--confirm needs research/freeze.json (python freeze.py <date>)")
    cut_ns = None if cut is None else int(cut.value)
    from functools import partial
    be = partial(build_events, cut_ns=cut_ns)            # every exploratory computation is cut off before the frozen period
    # ---- step 2: data integrity
    banner(f"[{tf}] STEP 2: data integrity", log)
    for sym, q in quality.items():
        log(f"  {sym:10s} {q['rows']:6d} candles {q['start'][:10]} -> {q['end'][:10]} | gaps {q['gap_count']} ({q['missing_bars']} missing bars, largest {q['largest_gap_bars']}) | "
            f"dup {q['duplicates']} | malformed {q['bad_ohlc_rows']} | zero-vol {q['zero_volume_rows']} | forming-candle {q['incomplete_last_candle']} | coverage {q['coverage']*100:.2f}%")
    # ---- step 3: reproduce
    banner(f"[{tf}] STEP 3: reproduce the ORIGINAL report (original code path, original statistics)", log)
    res["reproduction"] = reproduce(cfg, tf, ctx, list(ctx), log, be)
    # ---- baseline events, corrected simulator
    ev0, X0, funnel = be(ctx, params(cfg), with_X=True)
    dev0, ho0 = split(ev0)
    log(f"\ncorrected simulator, H0: {funnel} -> {len(ev0)} setups ({int((ev0.dir > 0).sum())} long / {int((ev0.dir < 0).sum())} short)")
    # ---- step 4: before / after
    banner(f"[{tf}] STEP 4: before vs after the confirmed fixes (all events, one position per symbol, net of costs)", log)
    evl, _, _ = be(ctx, params(cfg, legacy_cancel=True))
    before, after = r_stats(evl, traded(evl, np.arange(len(evl))), B), r_stats(ev0, traded(ev0, np.arange(len(ev0))), B)
    log(line("BEFORE (fill-bar target cancel bug)", before))
    log(line("AFTER  (corrected simulator)", after))
    log(f"  iid p-value {after['p_iid']:.3f} vs cluster p-value {after['p']:.3f}; effective sample size {after['ess']:.0f} of {after['n']} trades "
        f"(design effect {after['design_effect']:.1f}): the original iid confidence intervals were {np.sqrt(after['design_effect']):.1f}x too narrow")
    res["before_after"] = dict(before=before, after=after)
    # ---- step 5: execution + cost sensitivity
    banner(f"[{tf}] STEP 5: execution and cost sensitivity (every scenario is a recorded comparison)", log)
    scen = [("S1 corrected default", {}, 1.0), ("S2 costs x2", {}, 2.0), ("S3 costs x3", {}, 3.0), ("S4 costs x0.5 (optimistic)", {}, 0.5),
            ("S5 limit must trade through by 2bps", dict(fill_pen_bps=2.0), 1.0), ("S6 limit must trade through by 5bps", dict(fill_pen_bps=5.0), 1.0),
            ("S7 gapped stops exit at the open", dict(gap_stop=True), 1.0), ("S8 adverse: through 2bps + gaps + costs x2", dict(fill_pen_bps=2.0, gap_stop=True), 2.0)]
    srows = []
    for name, over, cm in scen:
        e = ev0 if not over else be(ctx, params(cfg, **over))[0]
        e = e.copy()
        e["R"] = e.R_gross - e.cost_R * cm
        s = r_stats(e, traded(e, np.arange(len(e))), B)
        log(line(name, s))
        srows.append(dict(tf=tf, scenario=name, **s))
        lg.record("scenario", f"{tf}:{name}", path=a.ledger, n=s["n"], exp_R=s["exp_R"], p=s["p"], holdout_viewed=False)
    res["scenarios"] = srows
    # ---- step 6: breakdowns (all events; descriptive, not used for selection)
    banner(f"[{tf}] STEP 6: breakdowns of H0 (cluster-bootstrap CIs; groups with n < {MIN_N} are marked and not interpreted)", log)
    allidx = traded(ev0, np.arange(len(ev0)))
    bd = breakdown(ev0, X0, allidx, B, log)
    for r in bd.itertuples():
        flag = "" if r.enough else "   (n<100: ignore)"
        log(f"  {r.group:10s} {r.level:18s} n={r.n:5d} win {r.win*100:3.0f}% exp {r.exp_R:+.3f}R [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]{flag}")
    res["breakdown"] = bd.assign(tf=tf)
    # ---- step 7: pre-registered hypotheses, exploration (purged dev) only
    banner(f"[{tf}] STEP 7: {len(hy.HYPOTHESES)} pre-registered hypotheses on EXPLORATION data (registry {hy.registry_hash()}); Bonferroni x{len(hy.HYPOTHESES)}", log)
    m = len(hy.HYPOTHESES)
    hrows = []
    for h in hy.HYPOTHESES:
        if h["kind"] == "variant":
            e, _, _ = (ev0, X0, None) if not h["params"] else be(ctx, params(cfg, **h["params"]))
            d, _ = split(e)
            pos = traded(e, d)
        else:
            keep = slice_rule(h["rule"], X0)
            d = dev0[keep[dev0]]
            e, pos = ev0, traded(ev0, d)
        s = r_stats(e, pos, B)
        s["p_adj"] = min(1.0, s["p"] * m)
        s["passes"] = bool(s["n"] >= MIN_N and s["exp_R"] > 0 and s["ci_lo"] > 0 and s["p_adj"] < 0.05)
        hrows.append(dict(tf=tf, id=h["id"], text=h["text"], **s))
        log(line(f"{h['id']} {h['text'][:28]}", s) + f"  adj-p {s['p_adj']:.3f}" + ("  <-- PASSES exploration" if s["passes"] else ""))
        lg.record("hypothesis", f"{tf}:{h['id']}", path=a.ledger, n=s["n"], exp_R=s["exp_R"], ci=[s["ci_lo"], s["ci_hi"]], p=s["p"], p_adj=s["p_adj"],
                  passes=s["passes"], holdout_viewed=False, registry=hy.registry_hash())
    res["hypotheses"] = hrows
    res["confirmation"] = []
    if cfg.confirm:
        cand = [h for h in hrows if h["passes"]]
        banner(f"[{tf}] CONFIRMATORY TEST on the frozen period (from {cut.date()}): only hypotheses that passed exploration ({len(cand)})", log)
        if not cand:
            log("  no hypothesis passed exploration, so there is nothing to confirm")
        for h in cand:
            spec = next(x for x in hy.HYPOTHESES if x["id"] == h["id"])
            ec, Xc, _ = build_events(ctx, params(cfg, **spec.get("params", {})), with_X=True, cut_ns=cut_ns, side="confirm")
            pos = np.arange(len(ec)) if spec["kind"] == "variant" else np.flatnonzero(slice_rule(spec["rule"], Xc))
            sc = r_stats(ec, traded(ec, pos), B)
            sc["p_adj"] = min(1.0, sc["p"] * len(cand))
            sc["validated"] = bool(sc["n"] >= MIN_N and sc["exp_R"] > 0 and sc["ci_lo"] > 0 and sc["p_adj"] < 0.05)
            log(line(f"CONFIRM {h['id']}", sc) + f"  adj-p {sc['p_adj']:.3f}" + ("  <-- VALIDATED" if sc["validated"] else ""))
            lg.record("confirmation", f"{tf}:{h['id']}", path=a.ledger, n=sc["n"], exp_R=sc["exp_R"], p_adj=sc["p_adj"], validated=sc["validated"], holdout_viewed=True)
            res["confirmation"].append(dict(tf=tf, id=h["id"], **sc))
    # contaminated holdout: shown ONLY as a reference for the baseline, never used for selection
    sh = r_stats(ev0, traded(ev0, ho0), B)
    log(line("[contaminated holdout reference] H0", sh))
    lg.record("reference", f"{tf}:H0-contaminated-holdout", path=a.ledger, n=sh["n"], exp_R=sh["exp_R"], holdout_viewed=True)
    # ---- step 8: portfolio
    banner(f"[{tf}] STEP 8: portfolio view of H0 (risk {a.pf_risk:.2%} per trade, max {a.max_concurrent} concurrent, fixed fractional)", log)
    tr = ev0.iloc[allidx]
    T = pd.DataFrame(dict(sym=tr.sym.values, sig_ts=tr.ts.values, fill_ts=tr.fill_ts.values, exit_ts=tr.exit_ts.values, R=tr.R.values))
    for c in ("sig_ts", "fill_ts", "exit_ts"):
        T[c] = pd.to_datetime(T[c], utc=True)
    pf = portfolio_sim(T, a.pf_risk, a.max_concurrent)
    recon = pf["trades_taken"] and abs(pf["total_return"] - a.pf_risk * pf["taken"].R.sum())
    log(f"  trades taken {pf['trades_taken']} (skipped for capacity {pf['trades_skipped_capacity']}) | total return {pf['total_return']*100:+.1f}% | CAGR {pf['cagr']*100:+.1f}% | "
        f"maxDD {pf['max_dd']*100:.1f}% | Sharpe {pf['sharpe']:.2f} | exposure {pf['exposure']*100:.0f}% of time, avg {pf['avg_concurrent']:.2f} open")
    log(f"  reconciliation: portfolio return - risk_frac x sum(trade R) = {recon:.2e}  (must be ~0)")
    res["portfolio"] = {k: v for k, v in pf.items() if k in ("trades_taken", "trades_skipped_capacity", "total_return", "cagr", "max_dd", "sharpe", "exposure", "avg_concurrent")}
    res["portfolio"]["reconciliation_error"] = float(recon)
    res["trades"] = ev0.iloc[allidx][["sym", "dir", "ts", "fill_ts", "exit_ts", "entry", "stop", "tp", "fill_px", "exit_px", "outcome", "R_gross", "cost_R", "R", "s_mode"]]\
        .rename(columns={"ts": "signal_ts", "R": "net_R", "s_mode": "entry_mode"})
    # ---- step 9: baselines
    banner(f"[{tf}] STEP 9: baselines on daily closes, same costs (ICT = realised P&L of the portfolio above)", log)
    bpd = 1440 // (int(tf[:-1]) * (60 if tf.endswith("h") else 1))
    dc = {}
    for sym, (df, _) in ctx.items():
        cnt = df.close.resample("1D").count()
        full = cnt[cnt >= bpd].index
        dc[sym] = df.close.resample("1D").last().reindex(full)
    close_d = pd.DataFrame(dc).sort_index()
    rets = bl.all_baselines(close_d, cfg.fee_bps + cfg.slip_bps)
    if len(pf["daily"]):
        rets["ICT H0 (portfolio)"] = pf["daily"].tz_convert("UTC") if pf["daily"].index.tz is not None else pf["daily"].tz_localize("UTC")
    tbl = bl.compare(rets)
    cols = ["strategy", "days", "cagr", "vol", "sharpe", "max_dd", "cagr_at_20pct_vol", "max_dd_at_20pct_vol", "corr_to_market", "beta", "alpha_ann", "t_alpha"]
    log(tbl[[c for c in cols if c in tbl]].round(3).to_string(index=False))
    ict = tbl[tbl.strategy == "ICT H0 (portfolio)"]
    if len(ict) and "t_alpha" in ict:
        r0 = ict.iloc[0]
        log(f"  ICT vs market: beta {r0.beta:.2f}, alpha {r0.alpha_ann*100:+.1f}%/yr (Newey-West t = {r0.t_alpha:.2f}); "
            f"{'adds value beyond the market' if r0.t_alpha > 2 and r0.alpha_ann > 0 else 'no evidence of return beyond market exposure'}")
    res["baselines"] = tbl.assign(tf=tf)
    return res


def verdict(results, log):
    passing = [(r["tf"], h) for r in results for h in r["hypotheses"] if h["passes"]]
    conf = [c for r in results for c in r.get("confirmation", [])]
    if conf and any(c["validated"] for c in conf):
        v = "VALIDATED EDGE: " + ", ".join(f"{c['tf']}:{c['id']} ({c['exp_R']:+.3f}R, adj-p {c['p_adj']:.3f}, n={c['n']})" for c in conf if c["validated"]) + \
            " passed exploration AND the frozen confirmatory period. Next step: paper trade, then small size."
        log("\nFINAL VERDICT: " + v)
        return v
    if conf:
        v = "NO DEMONSTRATED EDGE: the exploratory candidate(s) " + ", ".join(f"{c['tf']}:{c['id']}" for c in conf) + " did NOT replicate on the frozen confirmatory period."
        log("\nFINAL VERDICT: " + v)
        return v
    h0 = [(r["tf"], next(h for h in r["hypotheses"] if h["id"] == "H0")) for r in results]
    if passing:
        v = ("INCONCLUSIVE: exploratory candidate(s) " + ", ".join(f"{t}:{h['id']} ({h['exp_R']:+.3f}R, adj-p {h['p_adj']:.3f})" for t, h in passing) +
             " passed the corrected exploration test. They are NOT validated: the old holdout is contaminated and no untouched period exists yet. "
             "Freeze new data (python freeze.py <date>), collect it, then run --confirm on exactly these hypotheses.")
    else:
        v = ("NO DEMONSTRATED EDGE: baseline H0 is " + ", ".join(f"{t} {h['exp_R']:+.3f}R [{h['ci_lo']:+.3f},{h['ci_hi']:+.3f}]" for t, h in h0) +
             " per trade net of costs, and none of the pre-registered variants has a positive expectancy that survives the multiple-testing correction.")
    log("\nFINAL VERDICT: " + v)
    return v


def write_report(out, results, repro, tests, v, a, man):
    L = ["# Ezio ICT/SMC audit report (generated by audit_ict.py, every number below was computed by this run)\n",
         f"- commit `{man['commit']}` | python {man['python']} | libraries {man['libraries']}",
         f"- symbols: {a.symbols}\n- timeframes: {a.timeframes}\n- fixed window: start `{man['args'].get('start')}` end `{man['args'].get('end')}`",
         f"- hypothesis registry hash `{hy.registry_hash()}` | holdout looks recorded in the ledger: **{lg.holdout_looks(a.ledger)}** | variants/scenarios recorded: {lg.n_tested(a.ledger)}\n",
         "## Tests\n" + "\n".join(f"- {r['status']}: {r['test']} - {r['what']}" for r in tests["tests"]) if tests else "## Tests\nskipped (--skip_tests)\n",
         "\n## Reproduction of the original report\n" + "\n".join(f"- {r['tf']}: **{r['status']}** (setups {r['setups']}, dev n={r['dev_n']} exp {r['dev_exp']:+.3f}R, best slice {r['best']} {fnum(r['best_exp'])})" for r in repro),
         "\n## Verdict\n**" + v + "**\n",
         "## Holdout contamination\nThe original 15% holdout was displayed repeatedly (ledger entries L1-L7 plus every run of this audit that prints the reference line). "
         "It is declared **contaminated**: it must not be used to select or confirm anything. A new untouched validation period must come from data that does not exist yet "
         "(`python freeze.py <date>`, then collect and use `--confirm`).\n"]
    for r in results:
        L.append(f"## {r['tf']}\n")
        ba = r["before_after"]
        L.append(f"- before fixes: n={ba['before']['n']} exp {ba['before']['exp_R']:+.3f}R | after: n={ba['after']['n']} exp {ba['after']['exp_R']:+.3f}R [{ba['after']['ci_lo']:+.3f},{ba['after']['ci_hi']:+.3f}]")
        L.append(f"- cluster vs iid: design effect {ba['after']['design_effect']:.1f}, effective n {ba['after']['ess']:.0f} of {ba['after']['n']}")
        L.append("\n| scenario | n | win | exp R | 95% CI | PF |\n|---|---|---|---|---|---|")
        L += [f"| {s['scenario']} | {s['n']} | {s['win']*100:.0f}% | {s['exp_R']:+.3f} | [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] | {s['pf']:.2f} |" for s in r["scenarios"]]
        L.append("\n| hypothesis | n | exp R | 95% CI | adj-p | passes |\n|---|---|---|---|---|---|")
        L += [f"| {h['id']} {h['text'][:60]} | {h['n']} | {h['exp_R']:+.3f} | [{h['ci_lo']:+.3f},{h['ci_hi']:+.3f}] | {h['p_adj']:.3f} | {'YES' if h['passes'] else 'no'} |" for h in r["hypotheses"]]
        p = r["portfolio"]
        L.append(f"\nPortfolio H0: return {p['total_return']*100:+.1f}%, maxDD {p['max_dd']*100:.1f}%, Sharpe {p['sharpe']:.2f}, exposure {p['exposure']*100:.0f}%, trades {p['trades_taken']} (skipped {p['trades_skipped_capacity']})\n")
        L.append(r["baselines"].round(3).to_markdown(index=False) if hasattr(r["baselines"], "to_markdown") else r["baselines"].round(3).to_string(index=False))
    if os.path.exists("research/AUDIT_FINDINGS.md"):
        L.append("\n---\n" + open("research/AUDIT_FINDINGS.md").read())
    open(f"{out}/audit_report.md", "w").write("\n".join(L))


def main():
    cfg = parse(__doc__, EXTRA)
    a = cfg.args
    out = cfg.out if cfg.out != "runs/default" else "research_out"
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/audit_log.txt", "w")

    def log(s=""):
        print(s, flush=True)
        logf.write(s + "\n")

    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in a.timeframes.split(",") if t.strip()]
    hy.write_registry(f"{out}/hypotheses.json")
    banner("STEP 1: unit tests", log)
    tests = None
    if not a.skip_tests:
        import run_tests
        tests = json.load(open(f"{out}/tests_report.json")) if run_tests.main(out) is not None else None
    results, files = [], []
    for tf in tfs:
        r = run_timeframe(cfg, a, tf, symbols, out, log, None)
        if r:
            results.append(r)
            files += r["files"]
    if not results:
        raise SystemExit("no data for any timeframe")
    v = verdict(results, log)
    pd.concat([r["trades"].assign(tf=r["tf"]) for r in results]).to_csv(f"{out}/trades_H0.csv", index=False)
    pd.DataFrame([h for r in results for h in r["hypotheses"]]).to_csv(f"{out}/hypotheses.csv", index=False)
    pd.DataFrame([s for r in results for s in r["scenarios"]]).to_csv(f"{out}/scenarios.csv", index=False)
    pd.concat([r["breakdown"] for r in results]).to_csv(f"{out}/breakdowns.csv", index=False)
    pd.concat([r["baselines"] for r in results]).to_csv(f"{out}/baselines.csv", index=False)
    pd.DataFrame([r["reproduction"] for r in results]).to_csv(f"{out}/reproduction.csv", index=False)
    summ = pd.DataFrame([dict(tf=r["tf"], strategy=k, **{kk: vv for kk, vv in s.items()}) for r in results
                         for k, s in (("H0_before_fixes", r["before_after"]["before"]), ("H0_after_fixes", r["before_after"]["after"]))] +
                        [dict(tf=h["tf"], strategy=h["id"], **{k: v_ for k, v_ in h.items() if k not in ("tf", "id", "text")}) for r in results for h in r["hypotheses"]])
    summ.to_csv(f"{out}/strategies_summary.csv", index=False)
    json.dump({r["tf"]: r["quality"] for r in results}, open(f"{out}/data_quality.json", "w"), indent=1, default=str)
    man = mf.write(f"{out}/manifest.json", args={**vars(a), **{k: v_ for k, v_ in vars(cfg).items() if k != "args"}},     # resolved config wins over raw CLI defaults data_files=sorted(set(files)),
                   extra=dict(hypothesis_registry=hy.registry_hash(), verdict=v))
    write_report(out, results, [r["reproduction"] for r in results], tests, v, a, man)
    json.dump(dict(verdict=v, timeframes=[r["tf"] for r in results]), open(f"{out}/summary.json", "w"), indent=1)
    log(f"\nwrote {out}/: audit_report.md, trades_H0.csv, strategies_summary.csv, hypotheses.csv, scenarios.csv, breakdowns.csv, baselines.csv, "
        f"reproduction.csv, data_quality.json, manifest.json, tests_report.json, audit_log.txt")


if __name__ == "__main__":
    main()
