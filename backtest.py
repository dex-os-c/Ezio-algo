"""Cost-aware backtest. Signal at the close of bar t -> position held through bar t+1. Fees + slippage charged on every position change."""
import numpy as np
from config import bars_per_year


def next_pos(prev, p, cfg):
    """position rule with hysteresis (enter on strong signal, exit when it fades) to cut churn. Shared with the paper trader."""
    if p != p:
        return 0
    if p >= cfg.long_th:
        return 1
    if cfg.allow_short and p <= cfg.short_th:
        return -1
    if prev == 1 and p > cfg.exit_band:
        return 1
    if prev == -1 and cfg.allow_short and p < cfg.exit_band:
        return -1
    return 0


def backtest(close, prob, cfg):
    c, p, n = np.asarray(close, float), np.asarray(prob, float), len(close)
    pos, prev = np.zeros(n), 0
    for t in range(n):
        prev = next_pos(prev, p[t], cfg)
        pos[t] = prev
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1
    turn = np.abs(np.diff(pos, prepend=0.0))
    cost = (cfg.fee_bps + cfg.slip_bps) / 1e4
    net = np.zeros(n)
    net[1:] = pos[:-1] * ret[1:] - cost * turn[:-1]
    eq = np.cumprod(1 + net)
    # per-trade results
    acc, trades = 0.0, []
    for t in range(1, n):
        if pos[t - 1] != 0:
            acc += net[t]
        if pos[t] != pos[t - 1] and pos[t - 1] != 0:
            trades.append(acc)
            acc = 0.0
    bpy = bars_per_year(cfg)
    sd = net.std()
    return {
        "net_return": float(eq[-1] - 1),
        "buy_hold": float(c[-1] / c[0] - 1),
        "sharpe": float(net.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "trades": len(trades),
        "win_rate": float(np.mean(np.array(trades) > 0)) if trades else 0.0,
        "exposure": float(np.abs(pos).mean()),
        "equity": eq,
    }


def fmt(m):
    return (f"net {m['net_return']*100:+.1f}% | buy&hold {m['buy_hold']*100:+.1f}% | sharpe {m['sharpe']:.2f} | "
            f"maxDD {m['max_drawdown']*100:.1f}% | trades {m['trades']} | win {m['win_rate']*100:.0f}% | exposure {m['exposure']*100:.0f}%")
