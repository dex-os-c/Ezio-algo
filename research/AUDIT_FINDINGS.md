# Audit findings

Legend: **CONFIRMED** = reproduced by a test that fails on the old code, then fixed. **RISK** = an assumption that could flatter or hurt results; now measured as an explicit scenario instead of being silently baked in. **VERIFIED** = inspected and covered by a passing test.
Effect sizes quoted below come from *synthetic* data (the sandbox that wrote this has no exchange access). The real-data effect of each fix is computed by `audit_ict.py` step 4 on your candles.

## Confirmed bugs (all fixed, all have a regression test)

| # | Where | What was wrong | Test | Direction / size |
|---|---|---|---|---|
| B1 | `setups.py::_simulate` | A setup was cancelled whenever the *fill bar's high* reached the target, before checking whether the limit had filled on that bar. Fill-bar winners were dropped while fill-bar stop-outs were kept. Contradicted the documented rule ("target not credited on the fill bar"). | `test_execution.py` #4 | Synthetic: 2.3% of trades dropped, their mean was -0.36R, so the old numbers were slightly *flattered*. `legacy_cancel=True` reproduces the old behaviour. |
| B2 | `train.py` + `backtest.py` | Walk-forward folds are separated by H-bar embargo gaps but were concatenated and back-tested as consecutive bars: returns were earned across unobserved intervals and positions were carried across them. Affects the probability-model pipeline (`train.py`), not the setup/slice reports. | `test_backtest.py` #1 (a +50% jump inside a gap is booked by the old code, not by the new) | Unknown sign; large when a position is held across a gap. `train.py` now prints both methods. |
| B3 | `backtest.py` | A trade's own exit cost was charged one bar late, so trade-level P&L excluded it and did not reconcile with portfolio P&L; win rate was slightly optimistic. | `test_backtest.py` #2-#3 (sum of trade P&L == sum of bar P&L exactly) | Win rate can only fall. |
| B4 | `analyze_setups.py`, `train_setups.py` | Standard errors, CIs and p-values treated trades as independent, but setups on 8 correlated coins arrive in bursts. | `test_research_stats.py`: with weekly clustering the iid test rejects a true zero 25% of the time at a nominal 5%; the cluster bootstrap 6.7%. Design effect 6.0. | Overconfident in both directions. The negative conclusions stand; any positive-looking slice deserved more doubt. |
| B5 | `analyze_setups.py` | The dev set was not purged: dev trades could still be open when the holdout began, so holdout prices influenced dev statistics. | `test_audit_helpers.py` (split) | Small leakage; fixed by `audit_ict.split`. |
| B6 | `train_setups.stats`, `analyze_setups` | Max drawdown in R was computed in signal order, not in the order trades close. | `r_stats` orders by exit time | Drawdown slightly misstated. |
| B7 | `train_setups.py` | `Index.asi8` returns the index's own time unit (ns or us depending on pandas), used to order events across symbols. | `test_data_quality.py` (found by it) | Latent; replaced by `data_quality.to_ns`. |
| B8 | `data.py` | Rolling `--years` window moved every day (runs not reproducible), dedupe kept the oldest copy of a duplicate and used an unstable sort, no integrity checks. | `test_data_quality.py` | Added `--start/--end`, newest-copy-wins stable dedupe, `sanitize`. |
| B9 | `audit_ict.py` (development) | My own `traded()` helper indexed the subset twice; only correct when the subset was everything. Caught by a crash before any result was reported. | `test_audit_helpers.py` | n/a |
| B10 | `setups.py::_simulate` | `fill_pen_bps` (trade-through requirement) used `entry*(1-pen)` and `tp*(1+pen)`. Shorts are simulated on NEGATED prices, so for shorts the penalty made fills and targets EASIER instead of harder. Found in review of the real-data run; S5/S6/S8 were too optimistic on the short side. | `test_execution.py` #14 (short mirror of the long scenarios; fails on the old code) | Real data (run of 2026-10-28): S5/S6 moved expectancy by only 0.002-0.006R, verdict unchanged. Now `entry - pen*abs(entry)`, `tp + pen*abs(tp)`. |
| B11 | `audit_ict.py::main` | `data_files=` was swallowed by a trailing `#` comment, so `manifest.json` recorded no data hashes (0 entries in the real-data run). The run could not prove which candles produced it. | `test_audit_helpers.py` (manifest hashes + AST check that main passes `data_files`) | Reproducibility only. |
| B12 | `audit_ict.py` step 7 | (a) Each variant (H1-H3, H7) got its own 85/15 split by event COUNT, so variants were judged on different periods than H0 and the slices. (b) Bonferroni used x8 although every hypothesis is tested on each timeframe: x16 for 1h+4h. | `test_audit_helpers.py` (`split_at`) | The 4h killzone candidate (H6, raw p=0.019) had adj-p 0.156 under x8; it is weaker still under x16. No hypothesis changes status. |

**Do these explain the negative ICT result?** No. The two bugs that touch the ICT reports (B1, B4) either flattered the numbers slightly or made them look *more* certain; fixing them cannot turn about -0.04R per trade into a robust positive. The measured size on real data is in `reproduction.csv` / `scenarios.csv` after you run the audit.

Real-data result (Binance, 8 coins, 2018-10-29 to 2026-10-28): verdict NO DEMONSTRATED EDGE. 1h H0 -0.046R, 4h H0 -0.025R per trade net of costs; both timeframes reproduced the original report exactly. B10-B12 were found after that run and do not change it.

## Risks (assumptions now measured, not hidden)

| Risk | How it is handled |
|---|---|
| Limit orders fill when price merely touches them (optimistic: no queue position) | Scenario S5/S6 require trade-through by 2/5 bps for entry and target |
| A stop that is gapped through is capped at -1R | Scenario S7 exits at the open |
| Maker fee assumed on limit entries and targets | Costs x2/x3 and x0.5 scenarios; adverse combined scenario S8 |
| Perpetual funding not charged in the ICT simulator (setups last at most ~3 days) | Not modelled: listed as a limitation |
| Parameters (`sweep_len=4, win=30, fill_win=24, max_risk=5`) were loosened by me after the first BTC-only run produced too few setups | Researcher degree of freedom, disclosed; the 8 hypotheses are registered with a hash before results are viewed |
| Coin list chosen today (survivors) | Survivorship bias, listed as a limitation |
| Eight crypto coins are highly correlated | Week-clustered bootstrap; effective sample size reported next to n |

## Verified correct (inspected + tested)

- Swing confirmation delay: a swing is invisible for L bars after it forms (`test_features.py`).
- No-lookahead of every feature set, and **the detector itself is proven capable of failing**: it flags three deliberately leaky features (`test_alignment.py`).
- Higher-timeframe 4h/1d structure is invisible until the candle's last base bar has closed; a later candle cannot leak backwards (`test_alignment.py`).
- Prior-day high/low and Asian-range levels use only completed sessions.
- Signal timing: setups arm on closed candles only; entry is strictly after the signal bar; arming features are identical whether or not the future exists (`test_alignment.py`, `test_setups.py`).
- Stop/target in one candle resolves to the stop; target not credited on a limit's fill bar; timeouts; market entry (`test_execution.py`, 17 checks).
- Entry and exit costs on every trade, exact formula and price reconciliation of gross R (`test_setups.py`).
- Triple-barrier labels: next-open entry, stop wins ties, no leakage, NaN tail (`test_labels.py`).
- Candle completion at download (`fetch_ccxt` drops the forming candle), now also enforced on cached data (`sanitize`).
- Trade-level P&L reconciles with portfolio P&L to 1e-15, concurrency cap and compounding (`test_portfolio.py`, audit step 8).

## Remaining limitations

1. **The real-data reproduction was not run by the author of this audit** (no exchange access); `audit_ict.py` step 3 does it on your data and prints REPRODUCED / NOT REPRODUCED with tolerances.
2. The old 15% holdout is **contaminated** (ledger entries L1-L7, plus every later look). No clean validation period exists until after the freeze cutoff.
3. Only 8 pre-registered hypotheses were tested. Other ICT concepts (breaker blocks, SMT divergence, session opens, discretionary context) are untested; a negative result for these mechanical definitions does not prove discretionary ICT cannot work.
4. Intrabar order of highs and lows is unknowable at candle resolution; all ambiguity is resolved conservatively and bounded by scenarios.
5. Tests prove the logic on synthetic data, not that any market behaves in a particular way.
6. No order-book, queue-position or latency model; no funding for perps; survivorship bias in the coin list.
