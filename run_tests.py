"""Run every test script and report exactly which PASSED, FAILED or COULD NOT RUN (missing library / file). Nothing is skipped silently.
  python run_tests.py            -> prints a table, writes research_out/tests_report.json, exit code 1 if anything FAILED"""
import json, os, subprocess, sys, time

TESTS = [("test_features.py", "feature leak-safety (swings, FVG, HTF, killzones, volume profile)"),
         ("test_alignment.py", "leak-detector mutation test, HTF candle visibility, signal timing"),
         ("test_setups.py", "setup geometry, independent re-simulation, no-lookahead, mirror symmetry"),
         ("test_labels.py", "triple-barrier labels: next-open entry, stop wins ties, no leakage, NaN tail"),
         ("test_execution.py", "stop/target in one candle, fills, gaps, trade-through, timeout, market entry"),
         ("test_backtest.py", "gaps between folds, reversal costs, entry+exit cost attribution, trade vs portfolio P&L"),
         ("test_data_quality.py", "duplicates, gaps, forming candle, malformed rows, off-grid, naive timestamps"),
         ("test_portfolio.py", "reconciliation, concurrency cap, compounding, drawdown, exposure"),
         ("test_research_stats.py", "cluster bootstrap calibration vs iid under dependence"),
         ("test_baselines.py", "buy&hold / MA / breakout baselines, no look-ahead, alpha-beta"),
         ("test_train_setups.py", "take/skip gate finds planted edge, abstains on noise, per-symbol overlap rule"),
         ("test_audit_helpers.py", "purged dev/holdout split, per-subset one-position rule, small-sample guard"),
         ("test_freeze.py", "frozen confirmatory period is withheld from exploratory loads and cannot be moved"),
         ("test_analyze.py", "slice analysis finds planted slice, rejects noise")]


def main(out="research_out"):
    os.makedirs(out, exist_ok=True)
    rows = []
    for f, what in TESTS:
        if not os.path.exists(f):
            rows.append(dict(test=f, status="COULD_NOT_RUN", why="file missing", what=what, seconds=0))
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, f], capture_output=True, text=True)
        tail = (p.stdout + p.stderr).strip().splitlines()
        text = p.stdout + p.stderr
        if p.returncode == 0 and "ALL OK" in text:
            st, why = "PASS", next((l for l in reversed(tail) if "passed" in l or "ok" in l.lower()), "")
        elif "ModuleNotFoundError" in text or "ImportError" in text:
            st, why = "COULD_NOT_RUN", next((l for l in tail if "Error" in l), "import error")
        else:
            st, why = "FAIL", tail[-1] if tail else "no output"
        rows.append(dict(test=f, status=st, why=why[:160], what=what, seconds=round(time.time() - t0, 1)))
    for r in rows:
        print(f"{r['status']:14s} {r['test']:24s} {r['seconds']:6.1f}s  {r['what']}" + ("" if r["status"] == "PASS" else f"\n{'':15s}-> {r['why']}"))
    summary = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "COULD_NOT_RUN")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in summary.items()))
    json.dump(dict(summary=summary, tests=rows), open(f"{out}/tests_report.json", "w"), indent=1)
    return summary


if __name__ == "__main__":
    sys.exit(1 if main()["FAIL"] else 0)
