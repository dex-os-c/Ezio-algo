"""Single config for the whole project. Override anything from the CLI: --horizon 12 --symbol ETH/USDT"""
import argparse, json
from dataclasses import dataclass, fields


@dataclass
class Cfg:
    # ---- data ----
    source: str = "ccxt"             # ccxt (crypto) | yfinance (stocks) | synthetic (plumbing test)
    exchange: str = "binance"        # any ccxt id: binance, bybit, okx, kraken, bitstamp (BTC/USD back to 2011, daily)
    symbol: str = "BTC/USDT"         # ETH/USDT, SOL/USDT ... ; yfinance example: RELIANCE.NS
    timeframe: str = "1h"            # 15m | 1h | 4h | 1d
    years: float = 8.0               # history to use. Binance starts 2017 (~8-9y); more is not available
    bars: int = 20000                # synthetic only
    features: str = "base,smc,amt"   # which feature sets (see features.py)
    # ---- target / model ----
    horizon: int = 8                 # predict direction of the next N bars
    label: str = "tb"                # tb = triple-barrier (ATR stop/target, next-open entry) | close = sign of return at horizon
    tb_mult: float = 1.0             # barrier distance in ATRs (symmetric, so p>0.5 still means long)
    seq_len: int = 48
    hidden: int = 64
    epochs: int = 12
    batch: int = 256
    lr: float = 1e-3
    use_gru: bool = True
    use_lgb: bool = True
    n_splits: int = 4                # walk-forward folds
    holdout_frac: float = 0.15       # untouched final test
    seed: int = 7
    # ---- trading rules ----
    fee_bps: float = 5.0             # per side
    slip_bps: float = 2.0
    allow_short: bool = True         # crypto perps: True. Spot / NSE cash equity: False
    long_th: float = 0.56
    short_th: float = 0.44
    exit_band: float = 0.50
    bars_per_year: int = 0           # 0 = auto
    # ---- paper trading ----
    start_cash: float = 10000.0
    risk_frac: float = 0.95
    stop_pct: float = 0.03
    max_dd_halt: float = 0.15
    daily_loss_halt: float = 0.04
    qty_step: float = 0.0            # 0 = fractional; 1 = whole shares
    poll_seconds: int = 20
    warm_bars: int = 2500            # history loaded at start (continual training needs ft_val+600 labelled bars)
    paper_bars: int = 3000           # synthetic replay length
    live_window: int = 6000          # candles kept in memory / re-featurised on each live bar
    max_bars: int = 0                # stop after N bars (0 = run forever)
    resume: bool = False
    # ---- continual training ----
    retrain_every: int = 96          # bars
    ft_window: int = 4000
    ft_val: int = 1000               # last half is the fresh comparison block: small blocks make promotion noisy
    ft_epochs: int = 3
    ft_lr: float = 2e-4
    out: str = "runs/default"


def tf_minutes(tf: str) -> int:
    n, u = int(tf[:-1]), tf[-1]
    return n * {"m": 1, "h": 60, "d": 1440, "w": 10080}[u]


def bars_per_year(cfg: Cfg) -> float:
    if cfg.bars_per_year:
        return cfg.bars_per_year
    m = tf_minutes(cfg.timeframe)
    if cfg.source == "yfinance":
        return 252 if m >= 1440 else (375 / m) * 248
    return 365 * 24 * 60 / m


def parse(desc="", extra=None) -> Cfg:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", help="optional JSON file with overrides")
    for f in fields(Cfg):
        if f.type is bool:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=None)
        else:
            ap.add_argument(f"--{f.name}", type=f.type, default=None)
    for e in extra or []:
        ap.add_argument(*e[0], **e[1])
    a = ap.parse_args()
    cfg = Cfg()
    if a.config:
        for k, v in json.load(open(a.config)).items():
            setattr(cfg, k, v)
    for f in fields(Cfg):
        v = getattr(a, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    cfg.args = a
    return cfg
