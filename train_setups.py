"""Meta-labeling on ICT setups: the rules propose a trade (sweep -> MSS -> FVG retest), LightGBM only decides take / skip.

  python train_setups.py --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT --timeframe 1h --years 20 --out runs/pool_1h
  python train_setups.py --symbol BTC/USDT --timeframe 1h --years 20 --out runs/btc_setups
  python train_setups.py --source synthetic --bars 30000 --out runs/syn_setups          # plumbing test

Several symbols are POOLED into one model (all features are in ATR units, so markets are comparable); walk-forward folds and the
holdout are split by TIME across all symbols, and the one-trade-at-a-time rule is applied per symbol.

Honesty rules built in:
  * purged walk-forward (train events must have closed before the test block starts) + one untouched holdout
  * the take/skip threshold is picked on validation data and the model ABSTAINS unless validation expectancy is positive
  * results are in R (risk units, costs included) and are compared with the unfiltered setups and a random-selection test
  * a trial counter (runs/trials.json) deflates the p-value for how many times you have already tried this market
"""
import copy, json, os
import numpy as np, pandas as pd
from config import parse
from data import load_history
from features import build_features, clean
from setups import find_setups, attach_context, DEFAULT

try:
    import lightgbm as lgb
except Exception:                                     # pragma: no cover
    lgb = None

EXTRA = [(["--symbols"], dict(type=str, default="", help="comma list to pool, e.g. BTC/USDT,ETH/USDT,SOL/USDT (default: --symbol)")),
         (["--entry_mode"], dict(type=str, default=DEFAULT["entry_mode"], help="any | fvg | ob | ote")),
         (["--maker_bps"], dict(type=float, default=DEFAULT["maker_bps"], help="fee on limit entries / limit targets (bps)")),
         (["--max_risk"], dict(type=float, default=DEFAULT["max_risk"], help="widest allowed stop, in ATRs")),
         (["--rr"], dict(type=float, default=DEFAULT["rr"], help="target in R (take-profit = rr x risk)")),
         (["--win"], dict(type=int, default=DEFAULT["win"], help="bars allowed between sweep and structure shift")),
         (["--fill_win"], dict(type=int, default=DEFAULT["fill_win"], help="bars the limit order stays alive")),
         (["--max_hold"], dict(type=int, default=DEFAULT["max_hold"], help="max bars in a trade")),
         (["--sweep_len"], dict(type=int, default=DEFAULT["sweep_len"], help="swing length that defines a liquidity pool")),
         (["--min_trades"], dict(type=int, default=30, help="min selected validation trades to trust a threshold")),
         (["--perm"], dict(type=int, default=2000, help="random-selection permutations for the p-value"))]


def seq_filter(sig, ex, idx, sym=None):
    """one position at a time PER SYMBOL: skip any setup signalled while that symbol's previous accepted trade is still open"""
    keep, last = [], {}
    for k in idx:
        g = sym[k] if sym is not None else 0
        if sig[k] > last.get(g, -1):
            keep.append(k)
            last[g] = ex[k]
    return np.array(keep, int)


def stats(R):
    R = np.asarray(R, float)
    n = len(R)
    if n == 0:
        return dict(n=0, win=0.0, exp_R=0.0, pf=0.0, total_R=0.0, max_dd_R=0.0, se=0.0, lcb95=0.0)
    cum = np.cumsum(R)
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    se = R.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return dict(n=int(n), win=float((R > 0).mean()), exp_R=float(R.mean()), pf=float(gp / gl) if gl > 0 else float("inf"),
                total_R=float(cum[-1]), max_dd_R=float((cum - np.maximum.accumulate(cum)).min()), se=float(se),
                lcb95=float(R.mean() - 1.96 * se))


def fmt(s):
    return (f"trades {s['n']} | win {s['win']*100:.0f}% | expectancy {s['exp_R']:+.3f}R (95% low {s['lcb95']:+.3f}) | "
            f"profit factor {s['pf']:.2f} | total {s['total_R']:+.1f}R | maxDD {s['max_dd_R']:.1f}R")


def perm_p(R_pool, R_sel, B, seed=1):
    """P(random selection of the same number of setups does at least as well as the model's selection)"""
    k = len(R_sel)
    if k < 20 or len(R_pool) <= k:
        return 1.0
    rng, obs = np.random.default_rng(seed), np.mean(R_sel)
    hits = sum(np.mean(R_pool[rng.choice(len(R_pool), k, replace=False)]) >= obs for _ in range(B))
    return float((hits + 1) / (B + 1))


def fit_model(X, y, Xv, yv, seed):
    if len(set(y)) < 2 or len(set(yv)) < 2:
        return None
    params = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=seed, num_threads=0)
    return lgb.train(params, lgb.Dataset(X, y), 400, valid_sets=[lgb.Dataset(Xv, yv)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def pick_threshold(pv, Rv, min_n):
    """the cut-off with the best validation LOWER confidence bound of expectancy; inf (= abstain) if none is above zero"""
    best, thr = 0.0, np.inf
    for q in np.linspace(0.0, 0.9, 19):
        t = np.quantile(pv, q)
        r = Rv[pv >= t]
        if len(r) < min_n:
            continue
        lcb = r.mean() - r.std(ddof=1) / np.sqrt(len(r))
        if lcb > best:
            best, thr = lcb, t
    return float(thr)


def purge(pool, sig, ex, before):
    """events of `pool` that had fully closed before bar `before`"""
    return np.array([k for k in pool if ex[k] < before], int)


def train_block(X, R, y, sig, ex, tr_end, min_n, seed):
    """fit on events < tr_end (purged against bar sig[tr_end]); returns model, threshold, feature columns"""
    pool = purge(np.arange(tr_end), sig, ex, sig[tr_end] if tr_end < len(sig) else 10 ** 12)
    cut = int(len(pool) * 0.8)
    va = pool[cut:]
    fit = purge(pool[:cut], sig, ex, sig[va[0]]) if len(va) else pool[:cut]
    m = fit_model(X.iloc[fit], y[fit], X.iloc[va], y[va], seed)
    if m is None:
        return None, np.inf
    pv = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    return m, pick_threshold(pv, R[va], min_n)


def load_events(cfg, sp, log):
    """setups + context features for every symbol, pooled and sorted by signal TIME"""
    syms = [x.strip() for x in cfg.args.symbols.split(",") if x.strip()] or [cfg.symbol]
    evs, Xs, funnel = [], [], dict(sweep=0, mss=0, zone=0, trade=0)
    for i, sym in enumerate(syms):
        c = copy.copy(cfg)
        c.symbol = sym
        c.seed = cfg.seed + i                         # synthetic only: a different fake market per symbol
        try:
            df = load_history(c, log)
        except Exception as e:                        # one missing coin must not kill a pooled run
            log(f"  {sym}: skipped ({type(e).__name__}: {str(e)[:80]})")
            continue
        F = clean(build_features(df, c))
        fn = {}
        ev = find_setups(df, funnel=fn, **sp)
        for k in funnel:
            funnel[k] += fn[k]
        if ev.empty:
            continue
        ev = ev[ev.sig >= df.index.get_loc(F.index[0])].reset_index(drop=True)
        X = attach_context(ev, F, df.index)
        ev["sym"] = sym
        ev["sig_t"] = df.index[ev.sig.values].asi8
        ev["ex_t"] = df.index[ev.exit.values].asi8
        log(f"  {sym}: {len(df)} candles {df.index[0].date()} -> {df.index[-1].date()} | {len(ev)} setups")
        evs.append(ev)
        Xs.append(X)
    if not evs:
        raise SystemExit("no data loaded for any symbol")
    ev, X = pd.concat(evs, ignore_index=True), pd.concat(Xs, ignore_index=True)
    order = np.argsort(ev.sig_t.to_numpy(), kind="stable")
    return ev.iloc[order].reset_index(drop=True), X.iloc[order].reset_index(drop=True), funnel


def breakeven(ev, rr):
    """win rate needed to break even given the real maker/taker cost of wins and losses"""
    cw = float(ev.loc[ev.outcome == "tp", "cost_R"].median()) if (ev.outcome == "tp").any() else 0.0
    cl = float(ev.loc[ev.outcome == "sl", "cost_R"].median()) if (ev.outcome == "sl").any() else 0.0
    return (1 + cl) / (rr - cw + 1 + cl)


def run(cfg, log=print):
    a = cfg.args
    sp = dict(rr=a.rr, win=a.win, fill_win=a.fill_win, max_hold=a.max_hold, sweep_len=a.sweep_len, max_risk=a.max_risk,
              entry_mode=a.entry_mode, maker_bps=a.maker_bps, taker_bps=cfg.fee_bps, slip_bps=cfg.slip_bps)
    log(f"lightgbm={lgb is not None} | {cfg.source} {a.symbols or cfg.symbol} {cfg.timeframe} | setup params {sp}")
    ev, X, funnel = load_events(cfg, sp, log)
    log(f"funnel (all symbols): {funnel['sweep']} sweeps -> {funnel['mss']} structure shifts -> {funnel['zone']} valid entries -> "
        f"{funnel['trade']} filled & resolved trades")
    if len(ev) < 150:
        raise SystemExit(f"only {len(ev)} setups: add symbols (--symbols), more years, a lower timeframe, or loosen --win / --fill_win / --max_risk")
    R, sig, ex, sym = ev.R.to_numpy(), ev.sig_t.to_numpy(), ev.ex_t.to_numpy(), ev.sym.to_numpy()
    y = (R > 0).astype(int)
    N = len(ev)
    hold = int(N * cfg.holdout_frac)
    dev = N - hold
    be = breakeven(ev, sp["rr"])
    log(f"{N} setups ({int((ev.dir > 0).sum())} long / {int((ev.dir < 0).sum())} short) | {X.shape[1]} features | dev {dev} | holdout {hold}")
    log(f"break-even win rate at {sp['rr']:g}R after maker/taker costs: {be*100:.0f}%  (a win rate above that is what pays, not a round 90%)")
    log("ALL setups, no ML filter : " + fmt(stats(R[seq_filter(sig, ex, np.arange(N), sym)])))

    init = int(dev * 0.4)
    blk = (dev - init) // cfg.n_splits
    sel_R, sel_idx, pool_R, thr_log = [], [], [], []
    for k in range(cfg.n_splits):
        s, e = init + k * blk, min(init + (k + 1) * blk, dev)
        m, thr = train_block(X, R, y, sig, ex, s, a.min_trades, cfg.seed)
        test = np.arange(s, e)
        pool_R.append(R[test])
        if m is None or not np.isfinite(thr):
            log(f"fold {k+1}/{cfg.n_splits}: {len(test)} setups -> model abstains (no validation edge)")
            thr_log.append(None)
            continue
        p = m.predict(X.iloc[test], num_iteration=m.best_iteration)
        pick = seq_filter(sig, ex, test[p >= thr], sym)
        sel_R.append(R[pick])
        sel_idx.append(pick)
        thr_log.append(round(thr, 3))
        log(f"fold {k+1}/{cfg.n_splits}: {len(test)} setups, thr {thr:.3f} -> " + fmt(stats(R[pick])))
    wf = np.concatenate(sel_R) if sel_R else np.array([])
    wf_pool = np.concatenate(pool_R)
    swf = stats(wf)
    p_raw = perm_p(wf_pool, wf, a.perm)
    base = stats(R[seq_filter(sig, ex, np.arange(init, dev), sym)])
    log("WALK-FORWARD baseline (all setups): " + fmt(base))
    log("WALK-FORWARD filtered            : " + fmt(swf))
    if sel_idx and len(set(sym)) > 1:
        wi = np.concatenate(sel_idx)
        log("  per symbol (walk-forward, filtered): " + " | ".join(
            f"{g} n={stats(R[wi][sym[wi] == g])['n']} {stats(R[wi][sym[wi] == g])['exp_R']:+.2f}R" for g in sorted(set(sym))))

    os.makedirs("runs", exist_ok=True)
    tpath = "runs/trials.json"
    tr = json.load(open(tpath)) if os.path.exists(tpath) else {}
    key = f"{a.symbols or cfg.symbol}|{cfg.timeframe}|{cfg.exchange}"
    tr[key] = tr.get(key, 0) + 1
    json.dump(tr, open(tpath, "w"))
    p_adj = min(1.0, p_raw * tr[key])
    log(f"random-selection p-value {p_raw:.3f} | this is trial #{tr[key]} on this market set -> adjusted p {p_adj:.3f}")

    m, thr = train_block(X, R, y, sig, ex, dev, a.min_trades, cfg.seed)
    ho_idx = np.arange(dev, N)
    if m is None or not np.isfinite(thr):
        sho, ho_pick = stats(np.array([])), np.array([], int)
        log("HOLDOUT: model abstains (no validation edge), 0 trades")
    else:
        p = m.predict(X.iloc[ho_idx], num_iteration=m.best_iteration)
        ho_pick = seq_filter(sig, ex, ho_idx[p >= thr], sym)
        sho = stats(R[ho_pick])
        log("HOLDOUT filtered                : " + fmt(sho))
    log("HOLDOUT baseline (all setups)   : " + fmt(stats(R[seq_filter(sig, ex, ho_idx, sym)])))

    imp = {}
    if m is not None:
        g = m.feature_importance("gain")
        imp = {c: round(float(v / (g.sum() + 1e-12)), 4) for c, v in sorted(zip(X.columns, g), key=lambda t: -t[1])[:10]}
        log(f"top features: {imp}")
    edge = (swf["n"] >= 100 and swf["exp_R"] >= 0.1 and p_adj < 0.05 and sho["n"] >= 30 and sho["exp_R"] > 0)
    verdict = ("Candidate edge: paper trade it for weeks (setup signals only) before risking money." if edge else
               "No proven edge. Do not trade this yet: try more data, another market/timeframe, or new setup rules (each try counts as a trial).")
    log("VERDICT: " + verdict)

    os.makedirs(cfg.out, exist_ok=True)
    if m is not None:
        m.save_model(f"{cfg.out}/setup_model.txt", num_iteration=m.best_iteration)
    ev.iloc[ho_pick].to_csv(f"{cfg.out}/holdout_trades.csv", index=False)
    res = dict(symbol=a.symbols or cfg.symbol, funnel=funnel, timeframe=cfg.timeframe, setups=int(N), params=sp, break_even_win=be, thresholds=thr_log,
               holdout_threshold=None if not np.isfinite(thr) else thr, columns=list(X.columns),
               baseline_wf={k: v for k, v in base.items()}, walk_forward=swf, p_raw=p_raw, p_adjusted=p_adj, trial=tr[key],
               holdout=sho, importance=imp, edge=bool(edge), verdict=verdict)
    json.dump(res, open(f"{cfg.out}/setup_metrics.json", "w"), default=float)
    log(f"saved -> {cfg.out}/")
    return res


if __name__ == "__main__":
    run(parse(__doc__, EXTRA))
