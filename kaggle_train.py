# ONE-CELL KAGGLE TRAINER for algo_ai.   Notebook settings: Accelerator = GPU, Internet = On.
# Add the algo_ai.zip you downloaded as a Kaggle Dataset (Add Input -> Upload), then paste this into a cell and run.
import os, sys, glob, shutil, subprocess

SYMBOLS   = ["BTC", "ETH"]
TIMEFRAME = "1h"          # 15m | 1h | 4h | 1d
YEARS     = 8
EPOCHS    = 20
FEATURES  = "base,smc,amt"
# Kaggle runs on US servers: Binance/Bybit/OKX often refuse US IPs (HTTP 451), so fall back to Bitstamp (USD pairs).
EXCHANGES = [("binance", "USDT"), ("bybit", "USDT"), ("okx", "USDT"), ("bitstamp", "USD")]

work = "/kaggle/working"
proj = f"{work}/algo_ai"
if not os.path.exists(f"{proj}/train.py"):
    hits = glob.glob("/kaggle/input/**/train.py", recursive=True)
    assert hits, "algo_ai not found: add the zip as a dataset (Add Input -> Upload)"
    shutil.copytree(os.path.dirname(hits[0]), proj, dirs_exist_ok=True)
os.chdir(proj)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ccxt", "lightgbm"], check=False)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (turn the accelerator on)")

for sym in SYMBOLS:
    for ex, quote in EXCHANGES:
        cmd = [sys.executable, "-u", "train.py", "--symbol", f"{sym}/{quote}", "--exchange", ex, "--timeframe", TIMEFRAME,
               "--years", str(YEARS), "--epochs", str(EPOCHS), "--features", FEATURES,
               "--out", f"runs/{sym}-{quote}_{TIMEFRAME}"]
        print(f"\n===== {sym}/{quote} on {ex} =====", flush=True)
        if subprocess.run(cmd).returncode == 0:
            break
        print(f"{ex} failed, trying the next exchange...")

shutil.make_archive(f"{work}/runs", "zip", proj, "runs")
print("\nDone. Download /kaggle/working/runs.zip from the Output tab (or Save Version -> Save & Run All).")
