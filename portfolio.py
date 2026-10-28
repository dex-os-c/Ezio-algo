"""Turn trade-level results (net R per trade) into a time-based portfolio.

Every trade risks `risk_frac` of the equity known when its signal fired (equity only changes when trades CLOSE). A setup is skipped when
`max_concurrent` trades are already open or pending (a pending limit order reserves its slot), so portfolio heat is capped at
max_concurrent * risk_frac. One position per symbol at a time is enforced by train_setups.seq_filter before trades reach this function.
Trades need: sig_ts (signal time), exit_ts, R (net of costs), sym.  With compound=False the identity
    final_equity - 1 == risk_frac * sum(R of taken trades)
holds exactly and is tested."""
import numpy as np, pandas as pd

BPY = 365


def portfolio_sim(trades, risk_frac=0.005, max_concurrent=4, compound=False, start=1.0):
    tr = trades.sort_values(["sig_ts", "exit_ts"], kind="stable").reset_index(drop=True)
    taken, open_exits = [], []                         # open_exits: exit times of trades occupying a slot
    for k, r in tr.iterrows():
        open_exits = [e for e in open_exits if e > r.sig_ts]
        if len(open_exits) < max_concurrent:
            taken.append(k)
            open_exits.append(r.exit_ts)
    T = tr.loc[taken].copy()
    skipped = len(tr) - len(T)
    # sizing: equity realised before the signal (compound) or the starting equity (fixed fractional on the initial capital)
    T = T.sort_values("exit_ts", kind="stable")
    eq_at_exit = []
    equity = start
    realised = []                                       # (exit_ts, pnl)
    for _, r in T.sort_values("sig_ts", kind="stable").iterrows():
        base = start
        if compound:
            base = start + sum(p for t, p in realised if t <= r.sig_ts)
            base = max(base, 0.0)
        realised.append((r.exit_ts, risk_frac * base * r.R))
    pnl = pd.Series({t: 0.0 for t, _ in realised})
    pnl = pd.Series([p for _, p in realised], index=pd.DatetimeIndex([t for t, _ in realised])).groupby(level=0).sum().sort_index()
    eq = start + pnl.cumsum()
    out = dict(trades_taken=int(len(T)), trades_skipped_capacity=int(skipped), equity=eq, pnl=pnl, taken=T)
    if len(T) == 0:
        out.update(final=start, total_return=0.0, max_dd=0.0, sharpe=0.0, exposure=0.0, avg_concurrent=0.0, cagr=0.0, daily=pd.Series(dtype=float))
        return out
    # daily equity (realised P&L booked on the day a trade closes) -> daily returns, drawdown, Sharpe
    days = pd.date_range(T.sig_ts.min().normalize(), T.exit_ts.max().normalize(), freq="D", tz="UTC")
    de = eq.groupby(eq.index.normalize()).last().reindex(days).ffill().fillna(start)
    daily = de.pct_change().fillna(de.iloc[0] / start - 1)
    dd = float((de / de.cummax() - 1).min())
    yrs = max(len(days) / BPY, 1e-9)
    # exposure: share of time with >= 1 open position; average number of open positions
    ev = pd.concat([pd.Series(1, index=T.fill_ts if "fill_ts" in T else T.sig_ts), pd.Series(-1, index=T.exit_ts)]).sort_index()
    openn = ev.groupby(level=0).sum().cumsum()
    dur = openn.index.to_series().shift(-1) - openn.index.to_series()
    span = (openn.index[-1] - openn.index[0]).total_seconds() or 1.0
    exposure = float(dur[openn > 0].dt.total_seconds().sum() / span)
    avgc = float((openn * dur.dt.total_seconds().fillna(0)).sum() / span)
    out.update(final=float(eq.iloc[-1]), total_return=float(eq.iloc[-1] / start - 1), max_dd=dd,
               cagr=float((eq.iloc[-1] / start) ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
               sharpe=float(daily.mean() / daily.std() * np.sqrt(BPY)) if daily.std() > 0 else 0.0,
               exposure=exposure, avg_concurrent=avgc, daily=daily)
    return out
