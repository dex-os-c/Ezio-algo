"""Cost-aware backtest. Signal at the close of bar t -> position held through bar t+1. Fees + slippage are charged on the bar where the
position changes (so entry AND exit costs belong to their own trade). Rows that are not consecutive bars (idx jumps, e.g. between
walk-forward folds) are separate SEGMENTS: no return is ever earned across a gap and each segment starts and ends flat."""
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


def segments(n, idx=None):
    """[(start, end), ...] runs of consecutive bars; idx = original bar positions of the rows (None = all consecutive)"""
    if idx is None:
        return [(0, n)]
    cuts = np.flatnonzero(np.diff(np.asarray(idx)) != 1) + 1
    b = np.r_[0, cuts, n]
    return list(zip(b[:-1], b[1:]))


def backtest(close, prob, cfg, idx=None, liquidate_end=False):
    c, p, n = np.asarray(close, float), np.asarray(prob, float), len(close)
    cost = (cfg.fee_bps + cfg.slip_bps) / 1e4
    segs = segments(n, idx)
    nets, trades, open_pnl = [], [], 0.0
    expo = 0.0
    for k, (a, b) in enumerate(segs):
        cc, pp, m = c[a:b], p[a:b], b - a
        pos, prev = np.zeros(m), 0
        for t in range(m):
            prev = next_pos(prev, pp[t], cfg)
            pos[t] = prev
        ret = np.zeros(m)
        ret[1:] = cc[1:] / cc[:-1] - 1
        flatten = k < len(segs) - 1 or liquidate_end     # a gap forces a flat book; the very end is left as is unless asked
        net = np.zeros(m)
        net[1:] = pos[:-1] * ret[1:]
        net -= cost * np.abs(np.diff(pos, prepend=0.0))   # cost on the bar where the position changes
        if flatten and pos[-1] != 0:
            net[-1] -= cost * abs(pos[-1])
        nets.append(net)
        expo += np.abs(pos).sum()
        # trade-level accounting: each trade carries its own entry and exit cost, so sum(trades) + open == sum(net) exactly
        acc, last = 0.0, 0.0
        for t in range(m):
            if t >= 1 and pos[t - 1] != 0:
                acc += pos[t - 1] * ret[t]
            if pos[t] != last:
                if last != 0:
                    acc -= cost * abs(last)
                    trades.append(acc)
                    acc = 0.0
                if pos[t] != 0:
                    acc -= cost * abs(pos[t])
            last = pos[t]
        if flatten and last != 0:
            acc -= cost * abs(last)
            trades.append(acc)
            acc = 0.0
        open_pnl += acc
    net = np.concatenate(nets)
    eq = np.cumprod(1 + net)
    bpy = bars_per_year(cfg)
    sd = net.std()
    return {
        "net_return": float(eq[-1] - 1),
        "buy_hold": float(c[-1] / c[0] - 1),
        "sharpe": float(net.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "trades": len(trades),
        "win_rate": float(np.mean(np.array(trades) > 0)) if trades else 0.0,
        "exposure": float(expo / n),
        "equity": eq,
        "net": net,
        "trade_pnl": np.array(trades),
        "open_pnl": float(open_pnl),
        "segments": len(segs),
    }


def backtest_legacy(close, prob, cfg):
    """ORIGINAL implementation, kept only to measure the effect of the audit fixes. Known issues: (1) it treats rows separated by gaps as
    consecutive bars (walk-forward folds are glued across their embargo gaps), (2) a trade's own exit cost is charged one bar late, so
    trade-level P&L omits it and no longer reconciles with portfolio P&L."""
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
