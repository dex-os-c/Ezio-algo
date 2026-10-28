"""portfolio_sim: reconciliation with trade-level R, concurrency cap, compounding, exposure."""
import numpy as np, pandas as pd
from portfolio import portfolio_sim

n_ok = 0


def ok(c, m):
    global n_ok
    assert c, "FAILED: " + m
    n_ok += 1


def T(rows):
    d = pd.DataFrame(rows, columns=["sym", "sig", "fill", "exit", "R"])
    for c in ("sig", "fill", "exit"):
        d[c + "_ts"] = pd.to_datetime(d[c], utc=True)
    return d.drop(columns=["sig", "fill", "exit"])


# 1. reconciliation: no compounding -> final equity - 1 == risk_frac * sum(R)
rng = np.random.default_rng(5)
t0 = pd.Timestamp("2024-01-01", tz="UTC")
rows = []
for i in range(300):
    s = t0 + pd.Timedelta(hours=int(i * 7))
    rows.append(("A", s, s + pd.Timedelta(hours=1), s + pd.Timedelta(hours=int(rng.integers(2, 6))), float(rng.choice([-1.1, 2.0 - 0.1, 0.3]))))
df = T(rows)
r = portfolio_sim(df, risk_frac=0.01, max_concurrent=3)
ok(r["trades_taken"] == 300 and r["trades_skipped_capacity"] == 0, "no overlap -> all trades taken")
ok(abs((r["final"] - 1) - 0.01 * df.R.sum()) < 1e-12, "final equity reconciles with the sum of trade R")
ok(abs(r["total_return"] - 0.01 * df.R.sum()) < 1e-12, "total return == risk_frac * sum(R)")

# 2. concurrency cap: 10 simultaneous setups on different coins, cap 3 -> only the first 3 (by signal order) are taken
s = pd.Timestamp("2024-03-01", tz="UTC")
rows = [(f"C{i}", s + pd.Timedelta(minutes=i), s + pd.Timedelta(hours=1), s + pd.Timedelta(hours=10), 1.0) for i in range(10)]
r = portfolio_sim(T(rows), risk_frac=0.01, max_concurrent=3)
ok(r["trades_taken"] == 3 and r["trades_skipped_capacity"] == 7, "heat cap enforced")
ok(abs(r["total_return"] - 0.03) < 1e-12, "only taken trades count")
# after the exits the slots are free again
rows += [("Z", s + pd.Timedelta(hours=11), s + pd.Timedelta(hours=12), s + pd.Timedelta(hours=14), -1.0)]
r = portfolio_sim(T(rows), risk_frac=0.01, max_concurrent=3)
ok(r["trades_taken"] == 4, "slot is reusable after exits")

# 3. compounding: second trade is sized on equity realised before ITS signal
rows = [("A", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00", 2.0),
        ("A", "2024-01-02 00:00", "2024-01-02 01:00", "2024-01-02 05:00", 2.0)]
r = portfolio_sim(T(rows), risk_frac=0.10, max_concurrent=2, compound=True)
ok(abs(r["final"] - 1.44) < 1e-9, "1.0 -> 1.2 -> 1.44")
r = portfolio_sim(T(rows), risk_frac=0.10, max_concurrent=2, compound=False)
ok(abs(r["final"] - 1.40) < 1e-9, "fixed fractional: 1 + 0.2 + 0.2")

# 4. drawdown, exposure on a known path
rows = [("A", "2024-01-01 00:00", "2024-01-01 00:00", "2024-01-02 00:00", 1.0),
        ("A", "2024-01-03 00:00", "2024-01-03 00:00", "2024-01-04 00:00", -2.0)]
r = portfolio_sim(T(rows), risk_frac=0.01, max_concurrent=1)
ok(abs(r["max_dd"] - (-0.02 / 1.01)) < 1e-9, "max drawdown from a 1.01 peak down to 0.99")
ok(0.45 < r["exposure"] < 0.80, f"exposure measured over time ({r['exposure']:.2f})")
ok(r["trades_taken"] == 2, "both taken")

# 5. empty input
r = portfolio_sim(T([]), risk_frac=0.01)
ok(r["trades_taken"] == 0 and r["total_return"] == 0.0, "no trades -> flat")

print(f"{n_ok} portfolio checks passed")
print("ALL OK")
