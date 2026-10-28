"""Regression tests for backtest.py: gaps between walk-forward folds, reversal costs, entry+exit cost attribution, trade-vs-portfolio P&L."""
import numpy as np
from config import Cfg
from backtest import backtest, backtest_legacy, segments

cfg = Cfg()
cost = (cfg.fee_bps + cfg.slip_bps) / 1e4
n_ok = 0


def ok(cond, msg):
    global n_ok
    assert cond, "FAILED: " + msg
    n_ok += 1


# --- 1. GAP BUG: two folds of 10 bars each, price jumps +50% in the unobserved gap between them; we are long throughout
c = np.r_[np.full(10, 100.0), np.full(10, 150.0)]
p = np.full(20, 0.9)
idx = np.r_[np.arange(10), np.arange(30, 40)]                  # the 20 missing bars sit between the folds
old = backtest_legacy(c, p, cfg)
new = backtest(c, p, cfg, idx=idx)
ok(old["net_return"] > 0.4, "legacy method books the +50% jump that happened inside the gap (the bug)")
ok(new["net_return"] < 0.0 and new["segments"] == 2, "fixed method earns nothing across the gap and only pays costs")
ok(abs(new["net_return"] - (-3 * cost)) < 5e-4, "3 cost legs: enter, flatten at the gap, re-enter (the open end is not liquidated)")
ok(segments(20, idx) == [(0, 10), (10, 20)] and segments(5) == [(0, 5)], "segment detection")

# --- 2. reversal pays BOTH legs: long -> short costs 2 units, split one to each trade
c = np.full(6, 100.0)
p = np.array([0.9, 0.9, 0.1, 0.1, 0.5, 0.5])                   # long, long, reverse to short, short, exit, flat
b = backtest(c, p, cfg, liquidate_end=True)
ok(b["trades"] == 2, "long trade + short trade")
ok(np.allclose(b["trade_pnl"], [-2 * cost, -2 * cost]), "flat prices: each trade pays its own entry AND exit cost")
ok(abs(b["net"].sum() - b["trade_pnl"].sum()) < 1e-12, "portfolio net == sum of trade P&L")

# --- 3. reconciliation on a random path with many trades (additive P&L must match exactly when the book ends flat)
rng = np.random.default_rng(3)
c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000)))
p = rng.random(4000)
idx = np.flatnonzero(np.r_[np.ones(1500, bool), np.zeros(200, bool), np.ones(2300, bool)])[:4000]   # gap of 200 rows
b = backtest(c[:len(idx)], p[:len(idx)], cfg, idx=idx, liquidate_end=True)
ok(b["segments"] == 2 and b["trades"] > 100, "many trades over two segments")
ok(abs(b["net"].sum() - b["trade_pnl"].sum()) < 1e-9, "random path: sum(net) == sum(trade P&L) incl. entry/exit costs")
ok(abs(b["open_pnl"]) < 1e-12, "nothing left open after liquidation")
ok(b["win_rate"] <= backtest_legacy(c[:len(idx)], p[:len(idx)], cfg)["win_rate"] + 0.02, "own exit cost can only lower the win rate")

# --- 4. without liquidation the last open trade is reported, not silently dropped
b2 = backtest(np.full(4, 100.0), np.array([0.9, 0.9, 0.9, 0.9]), cfg)
ok(b2["trades"] == 0 and abs(b2["open_pnl"] + cost) < 1e-12 and abs(b2["net"].sum() - b2["open_pnl"]) < 1e-12, "open position tracked as open P&L")

print(f"{n_ok} backtest checks passed")
print("ALL OK")
