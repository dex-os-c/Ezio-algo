# ONE-CELL KAGGLE TRAINER for algo_ai: pulls the code from GitHub, trains, packs the results.
# Notebook settings: Accelerator = GPU, Internet = On.
# Private repo? Add a read-only token as a Kaggle Secret named GITHUB_TOKEN (Add-ons -> Secrets) and attach it to the notebook.
import os, sys, glob, shutil, subprocess, base64

REPO      = os.environ.get("ALGO_REPO_URL", "https://github.com/dex-os-c/ezio-algo.git")
BRANCH    = "main"
SYMBOLS   = ["BTC", "ETH"]
TIMEFRAME = "1h"          # 15m | 1h | 4h | 1d
YEARS     = 8
EPOCHS    = 20
FEATURES  = "base,smc,amt"
# Kaggle runs on US servers: Binance/Bybit/OKX often refuse US IPs (HTTP 451), so fall back to Bitstamp (USD pairs).
EXCHANGES = [("binance", "USDT"), ("bybit", "USDT"), ("okx", "USDT"), ("bitstamp", "USD")]

work = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
proj = f"{work}/algo_ai"


def git(*args):
    """run git; never print the command (it may carry the token), only git's own error text"""
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass
    pre = []
    if token and REPO.startswith("https://github.com/"):
        pre = ["-c", "http.extraheader=AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()]
    r = subprocess.run(["git", *pre, *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()[-400:]


def get_code():
    if os.path.isdir(f"{proj}/.git"):
        ok, msg = git("-C", proj, "pull", "--ff-only")
        print("updated code" if ok else f"pull failed, using the existing copy: {msg}")
        return True
    ok, msg = git("clone", "--depth", "1", "-b", BRANCH, REPO, proj)
    if ok:
        print("cloned", REPO)
        return True
    print("git clone failed:", msg)
    print("If the repo is private, add a GITHUB_TOKEN secret. Falling back to an uploaded dataset in /kaggle/input...")
    hits = glob.glob("/kaggle/input/**/train.py", recursive=True)
    if hits:
        shutil.copytree(os.path.dirname(hits[0]), proj, dirs_exist_ok=True)
        return True
    return False


if not get_code():
    raise SystemExit("Could not get the code: fix the repo URL / token, or upload the zip as a dataset.")
os.chdir(proj)
if os.environ.get("ALGO_DRY"):
    print("dry run: code fetched at", proj, "->", sorted(os.listdir(proj))[:6])
    raise SystemExit

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
print(f"\nDone. Download {work}/runs.zip from the Output tab (or Save Version -> Save & Run All).")
