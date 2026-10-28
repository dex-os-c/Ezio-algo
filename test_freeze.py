"""Freeze mechanism: exploratory loads never see the frozen period; --confirm does; the cutoff cannot be moved."""
import os, tempfile, json
import numpy as np, pandas as pd
import freeze
from config import Cfg
from data import window

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "freeze.json")
ok(freeze.cutoff(p) is None, "no file -> nothing frozen")
freeze.set_cutoff("2026-03-01", p)
ok(freeze.cutoff(p) == pd.Timestamp("2026-03-01", tz="UTC"), "cutoff stored")
try:
    freeze.set_cutoff("2026-06-01", p)
    ok(False, "moving the cutoff must be refused")
except SystemExit:
    ok(True, "moving the cutoff is refused")

idx = pd.date_range("2026-01-01", periods=24 * 90, freq="1h", tz="UTC")
df = pd.DataFrame({"open": 1., "high": 2., "low": .5, "close": 1.5, "volume": 1.}, index=idx)
freeze.PATH = p                                              # window() reads the cutoff through freeze.cutoff() -> default PATH
import freeze as fz
orig = fz.cutoff
fz.cutoff = lambda path=p: orig(path)
c = Cfg(); c.source = "ccxt"; c.start = "2026-01-01"
w = window(df, c, log=lambda *_: None)
ok(w.index.max() < pd.Timestamp("2026-03-01", tz="UTC"), "exploratory load stops before the cutoff")
c.confirm = True
w2 = window(df, c, log=lambda *_: None)
ok(w2.index.max() > pd.Timestamp("2026-03-15", tz="UTC"), "--confirm may read the frozen period")
print(f"{n_ok} freeze checks passed")
print("ALL OK")
