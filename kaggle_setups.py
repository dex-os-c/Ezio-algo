"""Kaggle runner for the pooled ICT-setup trainer. Notebook: Accelerator any, Internet On (for pip), your candle CSVs attached as a dataset.

  !python kaggle_setups.py                    # 1h
  !TF=15m python kaggle_setups.py             # other timeframe
It finds every  <exchange>_<BASE>-<QUOTE>_<tf>.csv  under /kaggle/input (the names download_data.py produces), pools them, trains, and zips runs/.
"""
import glob, os, re, shutil, subprocess, sys

TF = os.environ.get("TF", "1h")
ROOT = os.environ.get("INPUT", "/kaggle/input")
OUT = os.environ.get("OUT", f"runs/pool_{TF}")
PAT = re.compile(r"^([a-z0-9]+)_([A-Z0-9]+)-([A-Z0-9]+)_" + re.escape(TF) + r"\.csv$")


def discover(root=ROOT):
    found = {}
    for f in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
        m = PAT.match(os.path.basename(f))
        if m:
            found[f"{m.group(2)}/{m.group(3)}"] = (m.group(1), f)
    return found


if __name__ == "__main__":
    found = discover()
    if not found:
        raise SystemExit(f"no files named exchange_BASE-QUOTE_{TF}.csv under {ROOT}. Attach your dataset (Add Input) or set TF=...")
    os.makedirs("data", exist_ok=True)
    exchanges = {ex for ex, _ in found.values()}
    for sym, (ex, f) in found.items():
        shutil.copy(f, f"data/{os.path.basename(f)}")
    print(f"pooling {len(found)} symbols on {TF}: {', '.join(found)}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "ccxt"], check=False)
    cmd = [sys.executable, "-u", "train_setups.py", "--symbols", ",".join(found), "--exchange", sorted(exchanges)[0],
           "--timeframe", TF, "--years", "20", "--out", OUT] + sys.argv[1:]
    subprocess.run(cmd)
    shutil.make_archive("/kaggle/working/runs" if os.path.isdir("/kaggle/working") else "runs_out", "zip", ".", "runs")
    print("\nDone. Download runs.zip from the Output tab.")
