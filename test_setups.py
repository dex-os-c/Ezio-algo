"""Sanity tests for setups.py: geometry, independent re-simulation of every trade, no-lookahead, long/short mirror symmetry."""
import numpy as np, pandas as pd
from data import synthetic
from setups import find_setups, _mirror


def check(df, ev, rr, max_hold):
    h, l, c = df.high.values, df.low.values, df.close.values
    for r in ev.itertuples():
        long = r.dir > 0
        assert (r.stop < r.entry < r.tp) if long else (r.stop > r.entry > r.tp), "bad geometry"
        assert r.sig < r.fill <= r.exit, "bad ordering"
        # independent re-simulation over the bars between fill and exit
        sl_hit = (l[r.fill:r.exit + 1] <= r.stop).any() if long else (h[r.fill:r.exit + 1] >= r.stop).any()
        tp_bars = slice(r.fill + 1, r.exit + 1)
        tp_hit = (h[tp_bars] >= r.tp).any() if long else (l[tp_bars] <= r.tp).any()
        if r.outcome == "sl":
            assert sl_hit and r.R_gross == -1.0
        elif r.outcome == "tp":
            assert tp_hit and not sl_hit and abs(r.R_gross - rr) < 1e-9
        else:
            assert not sl_hit and not tp_hit and r.exit - r.fill == max_hold
        assert r.R < r.R_gross, "costs must reduce R"
        # the limit must actually have traded through before the fill, and no earlier bar may have touched the target
        assert (l[r.fill] <= r.entry) if long else (h[r.fill] >= r.entry)


def main():
    df = synthetic(30000, 11, "1h")
    ev = find_setups(df)
    assert len(ev) > 50, len(ev)
    check(df, ev, 2.0, 48)
    print(f"geometry + independent re-simulation ok ({len(ev)} setups)")

    cut = 20000                                       # no lookahead: truncating the future must not change closed trades
    part = find_setups(df.iloc[:cut])
    full = ev[ev.exit < cut].reset_index(drop=True)
    cols = ["sig", "dir", "entry", "stop", "tp", "fill", "exit", "outcome", "R", "s_sweep_depth", "s_fvg_atr", "s_tgt_R"]
    a, b = part[cols].reset_index(drop=True), full[cols]
    assert len(a) == len(b), (len(a), len(b))
    pd.testing.assert_frame_equal(a, b)
    print(f"no-lookahead ok ({len(a)} trades identical on truncated data)")

    mir = pd.DataFrame({"open": -df.open, "high": -df.low, "low": -df.high, "close": -df.close, "volume": df.volume})
    em = find_setups(mir)                             # mirroring the market must swap longs and shorts and keep the R's
    assert sorted(em.R.round(9)) == sorted(ev.R.round(9)) and (em.dir.values == -ev.dir.values).all()
    print("mirror symmetry ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
