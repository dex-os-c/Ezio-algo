"""Train with purged walk-forward validation, report honest out-of-sample results, save the model.

  python train.py --source synthetic --epochs 5                                  # plumbing test
  python train.py --symbol BTC/USDT --timeframe 1h --years 8 --out runs/btc      # real crypto data
  python train.py --symbol ETH/USDT --timeframe 15m --years 3 --features smc,amt --out runs/eth
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from config import parse
from data import load_history
from features import dataset
from models import Predictor, HAS_TORCH, HAS_LGB
from backtest import backtest, fmt


def score(p, y):
    m = ~np.isnan(p)
    if m.sum() < 50 or len(set(y[m])) < 2:
        return {}
    return {"auc": round(float(roc_auc_score(y[m], p[m])), 4), "acc": round(float(((p[m] > .5) == y[m]).mean()), 4),
            "logloss": round(float(log_loss(y[m], np.clip(p[m], 1e-4, 1 - 1e-4))), 4), "n": int(m.sum())}


def importance(P):
    """LightGBM gain by feature and by concept (base / smc / amt)"""
    if P.lgb is None:
        return {}
    g = P.lgb.feature_importance("gain")
    nf = len(P.cols)
    g = g.reshape(3, nf).sum(0)
    g = g / (g.sum() + 1e-12)
    groups = {"base": 0.0, "smc": 0.0, "amt": 0.0}
    for c, v in zip(P.cols, g):
        groups[c.split("_")[0] if c.split("_")[0] in ("smc", "amt") else "base"] += float(v)
    top = sorted(zip(P.cols, g), key=lambda x: -x[1])[:12]
    return {"by_concept": {k: round(v, 4) for k, v in groups.items()}, "top": [[c, round(float(v), 4)] for c, v in top]}


def run_training(cfg, log=print):
    S, H = cfg.seq_len, cfg.horizon
    log(f"torch={HAS_TORCH} lightgbm={HAS_LGB} | {cfg.source} {cfg.symbol} {cfg.timeframe} | features: {cfg.features}")
    df = load_history(cfg, log)
    log(f"{len(df)} candles {df.index[0].date()} -> {df.index[-1].date()} | building features (SMC / ICT / Auction)...")
    F, y = dataset(df, cfg)
    close, yv = df.close.reindex(F.index), y.values
    n = len(F)
    hold = int(n * cfg.holdout_frac)
    dev_n = n - hold
    log(f"{n} usable bars | {F.shape[1]} features | train/validate {dev_n} | untouched holdout {hold}")

    init = int(dev_n * 0.4)
    blk = (dev_n - init) // cfg.n_splits
    oos = np.full(n, np.nan)
    for k in range(cfg.n_splits):
        tr_end, te_e = init + k * blk, min(init + (k + 1) * blk, dev_n)
        te_s = tr_end + H                                   # embargo: labels look H bars ahead
        vs = int(tr_end * 0.85)
        P = Predictor(cfg)
        info = P.fit(F.iloc[:vs - H], yv[:vs - H], F.iloc[vs:tr_end], yv[vs:tr_end])
        oos[te_s:te_e] = P.predict(F.iloc[te_s - S + 1:te_e])[S - 1:]
        log(f"fold {k + 1}/{cfg.n_splits}: train {tr_end} -> test {te_e - te_s} bars | {info} | {score(oos[te_s:te_e], yv[te_s:te_e])}")
    m = ~np.isnan(oos)
    wf_score, bt = score(oos, yv), backtest(close.values[m], oos[m], cfg)
    log(f"WALK-FORWARD OOS  : {wf_score}")
    log(f"WALK-FORWARD trade: {fmt(bt)}")

    vs = int(dev_n * 0.85)
    P = Predictor(cfg)
    info = P.fit(F.iloc[:vs - H], yv[:vs - H], F.iloc[vs:dev_n], yv[vs:dev_n])
    ph = np.full(n, np.nan)
    ph[dev_n + H:] = P.predict(F.iloc[dev_n + H - S + 1:])[S - 1:]
    mh = ~np.isnan(ph)
    ho_score, bth = score(ph, yv), backtest(close.values[mh], ph[mh], cfg)
    log(f"HOLDOUT           : {ho_score}")
    log(f"HOLDOUT trade     : {fmt(bth)}")

    imp = importance(P)
    if imp:
        log(f"what the model uses (LightGBM gain): {imp['by_concept']}")
    P.version = 1
    os.makedirs(cfg.out, exist_ok=True)
    P.save(f"{cfg.out}/model")
    ts = close.index[mh]
    step = max(len(ts) // 600, 1)
    bh = (close.values[mh] / close.values[mh][0])
    curve = [[str(ts[i]), float(bth["equity"][i]), float(bh[i])] for i in range(0, len(ts), step)]
    edge = bt["sharpe"] > 0.5 and bth["net_return"] > 0 and bth["sharpe"] > 0.3
    verdict = ("Paper trade it for weeks across different market conditions before risking money." if edge else
               "Do not trade this. Change the features, look-ahead or timeframe, not the number of epochs.")
    res = {"symbol": cfg.symbol, "timeframe": cfg.timeframe, "exchange": cfg.exchange, "source": cfg.source,
           "features": cfg.features, "n_bars": n, "n_features": F.shape[1], "data_from": str(df.index[0]), "data_to": str(df.index[-1]),
           "holdout_start": str(F.index[dev_n + H]), "walk_forward": {**wf_score, **{k: v for k, v in bt.items() if k != "equity"}},
           "holdout": {**ho_score, **{k: v for k, v in bth.items() if k != "equity"}}, "importance": imp, "curve": curve,
           "edge": bool(edge), "verdict": verdict, "cfg": {k: v for k, v in vars(cfg).items() if k != "args"}}
    json.dump(res, open(f"{cfg.out}/metrics.json", "w"))
    log(f"saved -> {cfg.out}/model")
    log("VERDICT: " + verdict)
    return res


def main():
    cfg = parse(__doc__)
    run_training(cfg)


if __name__ == "__main__":
    main()
