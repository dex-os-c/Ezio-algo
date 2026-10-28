# Reproducing the ICT audit

```bash
git clone -b ict-setups https://github.com/dex-os-c/Ezio-algo.git && cd Ezio-algo
pip install pandas numpy scikit-learn lightgbm ccxt

# 1. data (once; cached in data/). 8.1 years so a fixed --start of 2018-10-29 is covered
python download_data.py --timeframe 1h --years 8.1
python download_data.py --timeframe 4h --years 8.1

# 2. tests: prints PASS / FAIL / COULD_NOT_RUN for every test file, nothing is skipped silently
python run_tests.py

# 3. the whole audit (tests, data integrity, reproduction, before/after, sensitivity, breakdowns,
#    pre-registered hypotheses, portfolio, baselines, verdict). Fixed window = reproducible.
python audit_ict.py --timeframes 1h,4h --start 2018-10-29 --end "2026-10-28 02:00" --out research_out

# 4. much later, once data after research/freeze.json's cutoff exists: confirmatory test of the hypotheses that passed exploration
python download_data.py --timeframe 1h --years 8.1
python audit_ict.py --timeframes 1h --start 2018-10-29 --confirm true --out research_confirm
```

Outputs in `research_out/`: `audit_report.md`, `trades_H0.csv` (every trade: signal/fill/exit timestamps, entry, stop, target, fill and exit prices, outcome, gross R, cost in R, net R),
`strategies_summary.csv`, `hypotheses.csv`, `scenarios.csv`, `breakdowns.csv`, `baselines.csv`, `reproduction.csv`, `data_quality.json`, `manifest.json` (commit, library versions, data hashes, arguments), `tests_report.json`, `audit_log.txt`.
Commit `research/ledger.jsonl` after every run: it is the append-only record of everything tried and every holdout look.

Rules enforced by code: data at/after `research/freeze.json` is withheld unless `--confirm true`; the cutoff cannot be moved; hypotheses are fixed in `hypotheses.py` (hash in the ledger); every p-value is corrected for all registered hypotheses; the verdict can only be VALIDATED after a passing confirmatory test.
