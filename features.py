"""Feature sets (all leak-safe: row t only uses data up to and including bar t) and labels.

  base : returns, volatility, RSI, ATR, Bollinger, MACD, volume z, time of day
  smc  : Smart Money Concepts / ICT  (smc.py)
  amt  : Auction Market Theory / volume profile  (amt.py)
Choose with --features base,smc,amt
"""
import numpy as np, pandas as pd
from config import tf_minutes


def base_features(df, tf_min):
    c, h, l, v = df.close, df.high, df.low, df.volume
    lc = np.log(c)
    r1 = lc.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 4, 16, 48):
        f[f"ret{k}"] = lc.diff(k)
    f["vol16"] = r1.rolling(16).std()
    f["vol64"] = r1.rolling(64).std()
    f["vol_ratio"] = f.vol16 / (f.vol64 + 1e-12)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = (100 - 100 / (1 + up / (dn + 1e-12))) / 100 - 0.5
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - ma20) / (sd20 + 1e-12)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["range"] = (h - l) / c
    f["close_pos"] = (c - l) / (h - l + 1e-12) - 0.5
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(96).mean()) / (lv.rolling(96).std() + 1e-12)
    if tf_min < 1440:
        m = df.index.hour * 60 + df.index.minute
        f["tod_sin"] = np.sin(2 * np.pi * m / 1440)
        f["tod_cos"] = np.cos(2 * np.pi * m / 1440)
        f["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        f["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return f.replace([np.inf, -np.inf], np.nan)


def build_features(df, cfg, overlays=False):
    tfm = tf_minutes(cfg.timeframe)
    sets = {s.strip() for s in cfg.features.split(",")}
    parts, ov = [], {}
    if "base" in sets:
        parts.append(base_features(df, tfm))
    if "smc" in sets:
        from smc import smc_features
        f, ov["smc"] = smc_features(df, tfm)
        parts.append(f)
    if "amt" in sets:
        from amt import amt_features
        f, ov["amt"] = amt_features(df, tfm)
        parts.append(f)
    F = pd.concat(parts, axis=1)
    return (F, ov) if overlays else F


def make_labels(df, horizon):
    fwd = df.close.shift(-horizon) / df.close - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y, fwd


def clean(F):
    """trim the indicator warm-up head; any rare later NaN (e.g. a day without an Asian session) is carried forward"""
    ok = F.notna().all(axis=1).values
    first = ok.argmax() if ok.any() else len(F)
    return F.iloc[first:].ffill().fillna(0.0)


def dataset(df, cfg):
    """features + labels aligned; head (warm-up) and tail (future unknown) removed"""
    F = clean(build_features(df, cfg))
    y, _ = make_labels(df, cfg.horizon)
    y = y.reindex(F.index)
    keep = y.notna().values
    last = len(keep) - keep[::-1].argmax() if keep.any() else 0
    return F.iloc[:last], y.iloc[:last]
