"""PRE-REGISTERED hypotheses. This list is fixed BEFORE any corrected results are looked at; its hash is written to the ledger so a later
edit is visible. One hypothesis = one change relative to the baseline H0 (sweep -> confirmed market-structure shift -> best available zone,
structure stop, 2R target). Each is tested on exploration data only and corrected for len(HYPOTHESES) comparisons (Bonferroni).
kind='variant' re-simulates the strategy with different parameters; kind='slice' keeps the baseline trades that satisfy a rule."""
import hashlib, json

HYPOTHESES = [
    dict(id="H0", kind="variant", params={}, text="Baseline: liquidity sweep + confirmed structure shift, entry at FVG>OB>OTE, stop beyond the sweep, 2R target"),
    dict(id="H1", kind="variant", params=dict(entry_mode="fvg"), text="Enter only at fair-value-gap midpoints (vs best available zone)"),
    dict(id="H2", kind="variant", params=dict(entry_mode="ob"), text="Enter only at order-block midpoints (vs best available zone)"),
    dict(id="H3", kind="variant", params=dict(entry_mode="mkt"), text="Market entry at the next open after the structure shift (vs limit at a zone)"),
    dict(id="H4", kind="slice", rule="htf_aligned", text="Only trades aligned with BOTH higher-timeframe structure trends (4h and 1d)"),
    dict(id="H5", kind="slice", rule="vol_expanding", text="Only when short-term volatility is expanding (vol16 > vol64)"),
    dict(id="H6", kind="slice", rule="killzone", text="Only inside the London / New York killzones"),
    dict(id="H7", kind="variant", params=dict(stop_mode="atr", atr_stop=1.5), text="Volatility-adjusted exits: stop 1.5 ATR / target 3 ATR (vs structure stop, 2R)"),
]


def registry_hash():
    return hashlib.sha256(json.dumps(HYPOTHESES, sort_keys=True).encode()).hexdigest()[:16]


def write_registry(path="research/hypotheses.json"):
    json.dump(dict(hash=registry_hash(), hypotheses=HYPOTHESES, correction="Bonferroni over all %d" % len(HYPOTHESES)), open(path, "w"), indent=1)
