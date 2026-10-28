"""Meta-labeling on ICT setups: the rules propose a trade (sweep -> MSS -> FVG retest), LightGBM only decides take / skip.

  python train_setups.py --symbol BTC/USDT --timeframe 1h --years 20 --out runs/btc_setups
  python train_setups.py --source synthetic --bars 30000 --out runs/syn_setups          # plumbing test

Honesty rules built in:
  * purged walk-forward (train events must have closed before the test block starts) + one untouched holdout
  * the take/skip threshold is picked on validation data and the model ABSTAINS unless validation expectancy is positive
  * results are in R (risk units, costs included) and are compared with the unfiltered setups and a random-selection test
  * a trial counter (runs/trials.json) deflates the p-value for how many times you have already tried this market
"""
import json, os
import numpy as np, pandas as pd
from config import parse
from data import load_history
from features import build_features, clean
from setups import find_setups, attach_context, DEFAULT

try:
    import lightgbm as lgb
except Exception:                                     # pragma: no cover
    lgb = None

EXTRA = [(["--rr"], dict(type=float, default=DEFAULT["rr"], help="target in R (take-profit = rr x risk)")),
         (["--win"], dict(type=int, default=DEFAULT["win"], help="bars allowed between sweep and structure shift")),
         (["--fill_win"], dict(type=int, default=DEFAULT["fill_win"], help="bars the limit order stays alive")),
         (["--max_hold"], dict(type=int, default=DEFAULT["max_hold"], help="max bars in a trade")),
         (["--sweep_len"], dict(type=int, default=DEFAULT["sweep_len"], help="swing length that defines a liquidity pool")),
         (["--min_trades"], dict(type=int, default=30, help="min selected validation trades to trust a threshold")),
         (["--perm"], dict(type=int, default=2000, help="random-selection permutations for the p-value"))]


def seq_filter(sig, ex, idx):
    """one position at a time: skip any setup signalled while the previous accepted trade is still open"""
    keep, last = [], -1
    for k in idx:
        if sig[k] > last:
            keep.append(k)
            last = ex[k]
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


def run(cfg, log=print):
    a = cfg.args
    sp = dict(rr=a.rr, win=a.win, fill_win=a.fill_win, max_hold=a.max_hold, sweep_len=a.sweep_len, cost_bps=cfg.fee_bps + cfg.slip_bps)
    log(f"lightgbm={lgb is not None} | {cfg.source} {cfg.symbol} {cfg.timeframe} | setup params {sp}")
    df = load_history(cfg, log)
    log(f"{len(df)} candles {df.index[0].date()} -> {df.index[-1].date()} | building features + finding ICT setups...")
    F = clean(build_features(df, cfg))
    ev = find_setups(df, **sp)
    ev = ev[ev.sig >= df.index.get_loc(F.index[0])].reset_index(drop=True)
    if len(ev) < 150:
        raise SystemExit(f"only {len(ev)} setups: need more data (--years), a lower timeframe, or a larger --win / --fill_win")
    X = attach_context(ev, F, df.index)
    R, sig, ex = ev.R.to_numpy(), ev.sig.to_numpy(), ev.exit.to_numpy()
    y = (R > 0).astype(int)
    N = len(ev)
    hold = int(N * cfg.holdout_frac)
    dev = N - hold
    be = (1 + float(ev.cost_R.median())) / (sp["rr"] + 1)
    log(f"{N} setups ({int((ev.dir > 0).sum())} long / {int((ev.dir < 0).sum())} short) | {X.shape[1]} features | dev {dev} | holdout {hold}")
    log(f"break-even win rate at {sp['rr']:g}R after costs: {be*100:.0f}%  (a win rate above that is what pays, not a round 90%)")
    log("ALL setups, no ML filter : " + fmt(stats(R[seq_filter(sig, ex, np.arange(N))])))

    init = int(dev * 0.4)
    blk = (dev - init) // cfg.n_splits
    sel_R, pool_R, thr_log = [], [], []
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
        pick = seq_filter(sig, ex, test[p >= thr])
        sel_R.append(R[pick])
        thr_log.append(round(thr, 3))
        log(f"fold {k+1}/{cfg.n_splits}: {len(test)} setups, thr {thr:.3f} -> " + fmt(stats(R[pick])))
    wf = np.concatenate(sel_R) if sel_R else np.array([])
    wf_pool = np.concatenate(pool_R)
    swf = stats(wf)
    p_raw = perm_p(wf_pool, wf, a.perm)
    base = stats(R[seq_filter(sig, ex, np.arange(init, dev))])
    log("WALK-FORWARD baseline (all setups): " + fmt(base))
    log("WALK-FORWARD filtered            : " + fmt(swf))

    os.makedirs("runs", exist_ok=True)
    tpath = "runs/trials.json"
    tr = json.load(open(tpath)) if os.path.exists(tpath) else {}
    key = f"{cfg.symbol}|{cfg.timeframe}|{cfg.exchange}"
    tr[key] = tr.get(key, 0) + 1
    json.dump(tr, open(tpath, "w"))
    p_adj = min(1.0, p_raw * tr[key])
    log(f"random-selection p-value {p_raw:.3f} | this is trial #{tr[key]} on {cfg.symbol} {cfg.timeframe} -> adjusted p {p_adj:.3f}")

    m, thr = train_block(X, R, y, sig, ex, dev, a.min_trades, cfg.seed)
    ho_idx = np.arange(dev, N)
    if m is None or not np.isfinite(thr):
        sho, ho_pick = stats(np.array([])), np.array([], int)
        log("HOLDOUT: model abstains (no validation edge), 0 trades")
    else:
        p = m.predict(X.iloc[ho_idx], num_iteration=m.best_iteration)
        ho_pick = seq_filter(sig, ex, ho_idx[p >= thr])
        sho = stats(R[ho_pick])
        log("HOLDOUT filtered                : " + fmt(sho))
    log("HOLDOUT baseline (all setups)   : " + fmt(stats(R[seq_filter(sig, ex, ho_idx)])))

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
    res = dict(symbol=cfg.symbol, timeframe=cfg.timeframe, setups=int(N), params=sp, break_even_win=be, thresholds=thr_log,
               holdout_threshold=None if not np.isfinite(thr) else thr, columns=list(X.columns),
               baseline_wf={k: v for k, v in base.items()}, walk_forward=swf, p_raw=p_raw, p_adjusted=p_adj, trial=tr[key],
               holdout=sho, importance=imp, edge=bool(edge), verdict=verdict)
    json.dump(res, open(f"{cfg.out}/setup_metrics.json", "w"), default=float)
    log(f"saved -> {cfg.out}/")
    return res


if __name__ == "__main__":
    run(parse(__doc__, EXTRA))
