"""Hand-built OHLC scenarios for the trade simulator: every execution rule is pinned down by an exact expected outcome.
Long trade used throughout: signal bar 0, limit entry 100, stop 98 (risk 2), target 104 (2R)."""
import numpy as np
from setups import _simulate, DEFAULT

P = dict(DEFAULT)
ENTRY, STOP, TP = 100.0, 98.0, 104.0


def run(bars, **kw):
    """bars: list of (open, high, low, close) for bar 0.. ; bar 0 is the signal bar"""
    a = np.array(bars, float)
    p = {**P, **kw}
    mkt = p.pop("market", False)
    return _simulate(a[:, 0], a[:, 1], a[:, 2], a[:, 3], 0, ENTRY, STOP, TP, 2.0, p, market=mkt)


SIG = (102, 103, 101, 102.5)           # signal bar: price above the limit
checks = 0


def ok(cond, msg):
    global checks
    assert cond, "FAILED: " + msg
    checks += 1


# 1. limit fills on bar 1, target hit on bar 2 -> +2R, exit at the target price
r = run([SIG, (101, 101.5, 99.9, 100.5), (100.5, 104.5, 100.2, 104)])
ok(r[:4] == (1, 2, 2.0, "tp") and r[4] == TP and r[5] == ENTRY, "limit fill then target")

# 2. the same candle touches both the stop and the target -> the stop wins (conservative)
r = run([SIG, (101, 101.5, 99.9, 100.5), (100.5, 105, 97.5, 101)])
ok(r[2] == -1.0 and r[3] == "sl" and r[1] == 2, "stop beats target inside one candle")

# 3. the fill candle itself pierces the stop -> filled and stopped in the same bar (-1R)
r = run([SIG, (101, 101.5, 97.0, 99)])
ok(r[:4] == (1, 1, -1.0, "sl"), "filled and stopped on the fill bar")

# 4. the target is NOT counted on a limit order's own fill bar, even if the high exceeds it
r = run([SIG, (101, 105, 99.9, 104.5), (103.5, 103.8, 103, 103.5)] + [(103, 103.5, 102.5, 103)] * 60)
ok(r[3] == "timeout" and r[0] == 1, "no target credit on the fill bar; the trade continues")
ok(run([SIG, (101, 105, 99.9, 104.5), (103.5, 103.8, 103, 103.5)] + [(103, 103.5, 102.5, 103)] * 60, legacy_cancel=True) is None,
   "legacy behaviour (dropped) is still reproducible")

# 5. the target trades before the order ever fills -> setup is dead (no trade)
ok(run([SIG, (101, 105, 100.5, 104.5)]) is None, "missed setup is cancelled")

# 6. never reaches the limit within fill_win -> no trade
ok(run([SIG] + [(102, 103, 101, 102)] * 30) is None, "unfilled order expires")

# 7. timeout exit at the close of the last allowed bar, R measured from the entry
bars = [SIG, (101, 101.5, 99.9, 100.5)] + [(100.5, 101, 99.5, 100.5)] * 47 + [(100.5, 101, 99.5, 101.0)]
r = run(bars)
ok(r[3] == "timeout" and r[1] == 49 and abs(r[2] - 0.5) < 1e-9 and r[4] == 101.0, "timeout exit")

# 8. gap through the stop: default caps the loss at -1R, gap_stop exits at the open (worse than -1R)
bars = [SIG, (101, 101.5, 99.9, 100.5), (97.0, 97.5, 96.0, 96.5)]
r0, r1 = run(bars), run(bars, gap_stop=True)
ok(r0[2] == -1.0 and r0[4] == STOP, "default stop fill is at the stop price")
ok(abs(r1[2] - (-1.5)) < 1e-9 and r1[4] == 97.0, "gap_stop exits at the gapped open (-1.5R)")

# 9. a limit order that gaps through the entry fills at the open when gap_stop is on
bars = [SIG, (99.0, 99.5, 98.5, 99.2), (99.2, 104.5, 99.0, 104)]
r0, r1 = run(bars), run(bars, gap_stop=True)
ok(r0[5] == ENTRY and r1[5] == 99.0, "gap-through limit fills at the open")
ok(r1[2] > r0[2], "better fill price -> more R")

# 10. touching the limit exactly fills by default but NOT when trade-through is required
bars = [SIG, (101, 101.5, 100.0, 100.5), (100.5, 104.5, 100.2, 104)]
ok(run(bars) is not None and run(bars, fill_pen_bps=2.0) is None, "trade-through requirement")

# 11. target touched exactly: counts by default, not when trade-through is required for the target
bars = [SIG, (101, 101.5, 99.9, 100.5), (100.5, 104.0, 100.2, 103), (103, 103.5, 100.5, 101)] + [(101, 101.5, 100.5, 101)] * 60
ok(run(bars)[3] == "tp" and run(bars, fill_pen_bps=2.0)[3] == "timeout", "target trade-through requirement")

# 12. market entry: filled at the next open, and the target CAN be hit on that same bar
r = run([SIG, (100.0, 104.5, 99.5, 104)], market=True)
ok(r[:4] == (1, 1, 2.0, "tp") and r[5] == 100.0, "market entry fills at the open; same-bar target allowed")
r = run([SIG, (100.0, 104.5, 97.5, 99)], market=True)
ok(r[3] == "sl", "market entry: same-bar stop beats target")

# 13. data ends before the trade resolves -> dropped, never counted
ok(run([SIG, (101, 101.5, 99.9, 100.5), (100.5, 101, 100, 100.5)]) is None, "unresolved trade at end of data is dropped")

print(f"{checks} execution checks passed")
print("ALL OK")
