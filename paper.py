"""Paper trading from the command line (the web UI uses the same engine).

  python paper.py --out runs/BTC-USDT_1h                       # live candles from the exchange
  python paper.py --out runs/BTC-USDT_1h --mode replay --speed 0   # replay the untouched holdout as fast as possible
  python paper.py --out runs/BTC-USDT_1h --continual true      # keep fine-tuning from the same point
  touch runs/BTC-USDT_1h/STOP                                  # kill switch: flatten and exit

NO real orders are ever sent. Fills are simulated at the last close +/- slippage, with fees.
"""
import time
from config import parse
from engine import RunData, Session


def main():
    cfg = parse(__doc__, extra=[
        (["--mode"], dict(default="live", choices=["live", "replay"])),
        (["--speed"], dict(type=float, default=0.0, help="replay bars/second (0 = max)")),
        (["--continual"], dict(type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=False)),
        (["--manual_auto"], dict(type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True, help="let the model place trades")),
    ])
    a = cfg.args
    rd = RunData(cfg.out, from_latest=(a.mode == "live")).load(print)
    over = {k: getattr(cfg, k) for k in ("start_cash", "risk_frac", "stop_pct", "max_dd_halt", "daily_loss_halt", "retrain_every",
                                         "ft_window", "ft_val", "ft_epochs", "ft_lr", "max_bars", "poll_seconds", "fee_bps", "slip_bps")}
    s = Session(rd, mode=a.mode, speed=a.speed, auto=a.manual_auto, continual=a.continual, **over)
    try:
        s.run()
    except KeyboardInterrupt:
        s.stop_ev.set()
    snap = s.snapshot()
    print(f"\n{snap['status']}: {snap['bars']} bars | equity {snap['equity']:.2f} ({snap['ret_pct']:+.2f}%) | buy&hold {snap['buy_hold_pct']:+.2f}% | "
          f"sharpe {snap['stats']['sharpe']:.2f} | maxDD {snap['stats']['max_dd'] * 100:.1f}% | trades {snap['n_trades']} | model v{snap['model_version']}")


if __name__ == "__main__":
    main()
