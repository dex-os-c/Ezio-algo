"""Download candles for several symbols into data/ (same cache format the trainers read).

  python download_data.py --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT --timeframe 1h --years 8

Run it where Binance is reachable (your PC; Kaggle/Colab US servers are often blocked), then upload the data/ folder as a Kaggle Dataset.
"""
import copy
from config import parse
from data import load_history

EXTRA = [(["--symbols"], dict(type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,LINK/USDT"))]

if __name__ == "__main__":
    cfg = parse(__doc__, EXTRA)
    for sym in [s.strip() for s in cfg.args.symbols.split(",") if s.strip()]:
        c = copy.copy(cfg)
        c.symbol = sym
        try:
            df = load_history(c)
            print(f"{sym}: {len(df)} candles {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            print(f"{sym}: FAILED ({type(e).__name__}: {e})")
