"""Sanity tests: python test_features.py
1. SMC building blocks behave on hand-made candles.
2. NO LOOKAHEAD: features computed on a prefix of the data must equal the same rows computed on the full data."""
import numpy as np, pandas as pd
from config import Cfg
from data import synthetic
from features import build_features
from smc import confirmed_swings, smc_features


def candles(rows, start="2024-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=100.0)


def test_swing_confirmation_lag():
    # a peak at bar 5; with L=3 it may only become known at bar 8
    hi = [1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1, 1]
    df = candles([(h - .5, h, h - 1, h - .5) for h in hi])
    sh, _, _, _ = confirmed_swings(df, 3)
    assert sh.iloc[:8].isna().all(), "swing visible before it is confirmed (lookahead!)"
    assert sh.iloc[8] == 9
    print("swing lag ok")


def test_fvg_and_structure():
    # bars 0..2 create a bullish FVG: low[2] > high[0]
    rows = [(10, 11, 9.5, 10.5), (10.5, 14, 10.4, 13.8), (13.8, 15, 12.5, 14.8)] + [(14.8, 15.2, 14.5, 15)] * 30
    df = candles(rows)
    df.loc[:, "volume"] = 100.0
    f, ov = smc_features(df, 60)
    assert any(z[1] == 1 and abs(z[2] - 11) < 1e-9 and abs(z[3] - 12.5) < 1e-9 for z in ov["fvg"]), ov["fvg"]
    print("fvg ok")


def test_no_lookahead(tf="1h", n=3000, features="base,smc,amt"):
    cfg = Cfg(timeframe=tf, features=features)
    df = synthetic(n, 3, tf)
    full = build_features(df, cfg)
    for cut in (1500, 2200, 2999):
        part = build_features(df.iloc[:cut], cfg)
        a, b = full.iloc[:cut], part
        bad = []
        for c in a.columns:
            x, y = a[c].values, b[c].values
            same = np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=True)
            if not same.all():
                bad.append((c, int((~same).sum()), int(np.where(~same)[0][0])))
        assert not bad, f"LOOKAHEAD in features at cut={cut}: {bad[:6]}"
    print(f"no-lookahead ok ({tf}, {full.shape[1]} features)")


if __name__ == "__main__":
    test_swing_confirmation_lag()
    test_fvg_and_structure()
    test_no_lookahead("1h")
    test_no_lookahead("15m", 2500)
    test_no_lookahead("1d", 1500)
    print("ALL OK")
