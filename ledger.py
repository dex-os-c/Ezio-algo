"""Append-only research ledger (research/ledger.jsonl). Every strategy variant, scenario and holdout look is recorded, so the number of
things tried and the number of times each holdout was viewed can never be quietly forgotten. Commit this file to git."""
import json, os, subprocess, time

PATH = "research/ledger.jsonl"

# Looks at the old holdout that are visible in the project history (reconstructed from the run logs pasted during development; the
# commit hashes of the very first runs are unknown). They are why the existing holdout is declared CONTAMINATED.
LEGACY = [
    dict(kind="legacy", id="L1", what="train.py probability model BTC/USDT 1h: walk-forward + holdout printed", holdout_viewed=True, markets="BTC/USDT 1h"),
    dict(kind="legacy", id="L2", what="train_setups.py BTC/USDT 1h, 293 setups (first version): holdout printed", holdout_viewed=True, markets="BTC/USDT 1h"),
    dict(kind="legacy", id="L3", what="train_setups.py BTC/USDT 1h, 676 setups: holdout printed", holdout_viewed=True, markets="BTC/USDT 1h"),
    dict(kind="legacy", id="L4", what="same run repeated (identical output pasted twice)", holdout_viewed=True, markets="BTC/USDT 1h"),
    dict(kind="legacy", id="L5", what="train_setups.py pooled 8 coins 1h: holdout printed", holdout_viewed=True, markets="8 coins 1h"),
    dict(kind="legacy", id="L6", what="analyze_setups.py pooled 8 coins 1h: best-slice holdout printed", holdout_viewed=True, markets="8 coins 1h"),
    dict(kind="legacy", id="L7", what="analyze_setups.py pooled 8 coins 4h: best-slice holdout printed", holdout_viewed=True, markets="8 coins 4h"),
]


def _git():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip())
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def entries(path=PATH):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def record(kind, id, path=PATH, **fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        for e in LEGACY:
            open(path, "a").write(json.dumps({**e, "ts": "reconstructed", "commit": "unknown"}) + "\n")
    e = dict(kind=kind, id=id, ts=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), commit=_git(), **fields)
    open(path, "a").write(json.dumps(e, default=float) + "\n")
    return e


def holdout_looks(path=PATH):
    return sum(1 for e in entries(path) if e.get("holdout_viewed"))


def n_tested(path=PATH):
    return sum(1 for e in entries(path) if e["kind"] in ("hypothesis", "scenario"))
