"""Frozen confirmatory period. `research/freeze.json` holds {"cutoff": "<UTC timestamp>"}. Every exploratory run (the default) is
denied data at/after the cutoff; only a run with --confirm may read it, and only strategies that were registered BEFORE the cutoff
was reached (research/hypotheses.json) may be judged on it. Create it once with:  python freeze.py 2026-11-01"""
import json, os, sys
import pandas as pd

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "freeze.json")


def cutoff(path=PATH):
    if not os.path.exists(path):
        return None
    return pd.Timestamp(json.load(open(path))["cutoff"], tz="UTC")


def set_cutoff(date, path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists ({cutoff(path)}). The cutoff may not be moved: that would un-freeze the confirmatory data.")
    json.dump({"cutoff": str(pd.Timestamp(date, tz="UTC")), "note": "confirmatory period starts here; never use it for exploration"}, open(path, "w"))
    print("frozen from", pd.Timestamp(date, tz="UTC"))


if __name__ == "__main__":
    set_cutoff(sys.argv[1]) if len(sys.argv) > 1 else print("cutoff:", cutoff())
