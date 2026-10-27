"""Paper-trading engine shared by the CLI (paper.py) and the web server (server.py).

RunData : a trained model + candles + features/overlays + model probabilities (cached)
Session : simulated broker + risk layer + replay/live loop + continual training
No real orders are ever sent.
"""
import copy, json, math, os, threading, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from config import Cfg, bars_per_year, tf_minutes
from data import load_history, LiveFeed
from features import build_features, clean, make_labels
from models import Predictor
from backtest import next_pos
from smc import LEN_EXT


def cfg_from_dict(d):
    c = Cfg()
    for k, v in d.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c


def jclean(x):
    """make numpy / NaN values JSON safe"""
    if isinstance(x, dict):
        return {k: jclean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jclean(v) for v in x]
    if isinstance(x, np.ndarray):
        return jclean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ------------------------------------------------------------------ data + model
class RunData:
    def __init__(self, run_dir, from_latest=False):
        self.dir = run_dir
        self.metrics = json.load(open(f"{run_dir}/metrics.json"))
        self.cfg = cfg_from_dict(self.metrics["cfg"])
        self.cfg.out = run_dir
        mdir = f"{run_dir}/model_latest" if from_latest and os.path.exists(f"{run_dir}/model_latest/meta.json") else f"{run_dir}/model"
        self.P = Predictor.load(mdir, self.cfg)
        self.df = self.F = self.prob = None
        self.ov = {}
        self.lock = threading.RLock()

    def load(self, log=print):
        self.set_df(load_history(self.cfg, log))
        return self

    def set_df(self, df):
        with self.lock:
            F, ov = build_features(df, self.cfg, overlays=True)
            self.df, self.F, self.ov = df, clean(F), ov
            self.prob = np.full(len(df), np.nan)
            self.repredict(0)

    def repredict(self, from_pos=0):
        """recompute model probabilities for bars >= from_pos (after a model promotion only the future changes)"""
        S = self.cfg.seq_len
        a = max(self.F.index.searchsorted(self.df.index[from_pos]) - (S - 1), 0)
        p = self.P.predict(self.F.iloc[a:])
        pos = self.df.index.get_indexer(self.F.index[a:])
        ok = ~np.isnan(p)
        self.prob[pos[ok]] = p[ok]

    def append(self, bars):
        df = pd.concat([self.df, bars])
        df = df[~df.index.duplicated()].tail(self.cfg.live_window)
        self.set_df(df)

    @staticmethod
    def _zones(zs, vis, X, end, keep_active=10, keep_done=6):
        """declutter: newest still-open zones plus a few recently mitigated ones"""
        zs = [z for z in zs if z[5] <= end and vis(z[0], z[4] if z[4] is not None and z[4] <= end else None)]
        act = [z for z in zs if z[4] is None or z[4] > end][-keep_active:]
        done = [z for z in zs if z[4] is not None and z[4] <= end][-keep_done:]
        return [[X(z[0]), z[1], z[2], z[3], None if z[4] is None or z[4] > end else X(z[4])] for z in done + act]

    def chart(self, end, n=300, trades=None):
        """candles + SMC / auction overlays + model probability for bars (end-n, end]"""
        with self.lock:
            df, end = self.df, min(end, len(self.df) - 1)
            start = max(0, end - n + 1)
            w = df.iloc[start:end + 1]
            X = lambda i: int(i) - start                       # absolute bar -> x in the window
            smc, amt = self.ov.get("smc", {}), self.ov.get("amt", {})
            vis = lambda t0, t1: t0 <= end and (t1 is None or t1 >= start)
            out = {
                "start_ts": str(w.index[0]), "end_ts": str(w.index[-1]), "n": len(w),
                "candles": [[int(t.timestamp() * 1000), o, h, l, c, v] for t, o, h, l, c, v in
                            zip(w.index, w.open, w.high, w.low, w.close, w.volume)],
                "prob": self.prob[start:end + 1],
                "swings": [[X(i), k, p] for i, k, p in smc.get("swings", []) if start <= i <= end and i + LEN_EXT <= end],
                "bos": [[X(i), k, d, p] for i, k, d, p in smc.get("bos", []) if start <= i <= end],
                "sweeps": [[X(i), k, p] for i, k, p in smc.get("sweeps", []) if start <= i <= end],
                "fvg": self._zones(smc.get("fvg", []), vis, X, end),
                "ob": self._zones(smc.get("ob", []), vis, X, end),
                "amt": {k: {m: v[start:end + 1] for m, v in d.items()} for k, d in amt.get("series", {}).items()},
            }
            if trades:
                idx = w.index
                out["trades"] = [[int(idx.searchsorted(pd.Timestamp(t["ts"]))), t["qty"] > 0, t["px"]] for t in trades
                                 if idx[0] <= pd.Timestamp(t["ts"]) <= idx[-1]]
            return jclean(out)


# ------------------------------------------------------------------ simulated broker
class Broker:
    def __init__(self, cfg):
        self.cfg, self.cash, self.qty, self.entry = cfg, cfg.start_cash, 0.0, 0.0
        self.fee, self.slip = cfg.fee_bps / 1e4, cfg.slip_bps / 1e4

    def equity(self, px):
        return self.cash + self.qty * px

    def sign(self):
        return int(np.sign(self.qty))

    def target(self, frac, px):
        if frac != 0 and self.qty != 0 and np.sign(frac) == np.sign(self.qty):
            return None                                   # already positioned that way: hold, don't burn fees rebalancing
        tq = frac * self.equity(px) / px
        if self.cfg.qty_step:
            tq = math.floor(abs(tq) / self.cfg.qty_step) * self.cfg.qty_step * np.sign(tq)
        dq = tq - self.qty
        if abs(dq) * px < 1e-4 * self.equity(px):
            return None
        fill = px * (1 + np.sign(dq) * self.slip)
        fee = abs(dq) * fill * self.fee
        self.cash -= dq * fill + fee
        if np.sign(tq) != np.sign(self.qty):
            self.entry = fill
        self.qty = tq
        return dq, fee, fill

    def state(self):
        return {"cash": self.cash, "qty": self.qty, "entry": self.entry}

    def load(self, s):
        self.cash, self.qty, self.entry = s["cash"], s["qty"], s["entry"]


def _eval(P, F, y, S):
    p = P.predict(F)[S - 1:]
    yy = y[S - 1:]
    return {"logloss": float(log_loss(yy, np.clip(p, 1e-4, 1 - 1e-4))), "auc": float(roc_auc_score(yy, p)) if len(set(yy)) > 1 else 0.5}


# ------------------------------------------------------------------ session
class Session:
    def __init__(self, rd, mode="replay", speed=5.0, auto=True, continual=False, **over):
        self.rd, self.mode, self.speed, self.auto, self.continual = rd, mode, float(speed), auto, continual
        self.cfg = copy.copy(rd.cfg)
        for k, v in over.items():
            if v is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
        c = self.cfg
        self.br = Broker(c)
        self.lock, self.stop_ev, self.thread = threading.RLock(), threading.Event(), None
        self.status, self.halted, self.halted_day, self.cooldown = "idle", False, None, 0
        self.trades, self.eq, self.retrains, self.logs = [], [], [], []
        self.bars, self.peak, self.day, self.day_eq = 0, c.start_cash, None, c.start_cash
        self.last_ts, self.last_px, self.last_p, self.last_tick = None, None, None, None
        if mode == "replay":
            self.pos = int(rd.df.index.searchsorted(pd.Timestamp(rd.metrics["holdout_start"])))
            self.first_px = float(rd.df.close.iloc[self.pos])
        else:
            self.feed = LiveFeed(c)
            self.feed.hist = rd.df.tail(c.warm_bars)
            self.feed.last_ts = rd.df.index[-1]
            self.pos = len(rd.df) - 1
            self.first_px = float(rd.df.close.iloc[-1])
        self.last_px, self.last_ts = self.first_px, rd.df.index[self.pos]
        self.last_p = None if np.isnan(rd.prob[self.pos]) else float(rd.prob[self.pos])
        self.log(f"session {mode} | {c.symbol} {c.timeframe} | model v{rd.P.version} | cash {c.start_cash:.0f} | auto={auto} continual={continual}")

    # ---------- helpers
    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.logs.append(line)
        del self.logs[:-200]
        with open(f"{self.rd.dir}/paper.log", "a") as f:
            f.write(line + "\n")

    def _trade(self, ts, frac, px, reason, p):
        r = self.br.target(frac, px)
        if not r:
            return
        dq, fee, fill = r
        t = {"ts": str(ts), "qty": float(dq), "px": float(fill), "fee": float(fee), "reason": reason,
             "p": None if p is None or p != p else float(p), "equity": float(self.br.equity(px))}
        self.trades.append(t)
        self.log(f"{'BUY ' if dq > 0 else 'SELL'} {abs(dq):.5f} @ {fill:.2f} ({reason})")

    # ---------- one closed bar
    def _process(self, ts, px, p):
        c, br = self.cfg, self.br
        eq = br.equity(px)
        self.peak = max(self.peak, eq)
        if self.day != ts.date():
            self.day, self.day_eq = ts.date(), eq
        if eq / self.peak - 1 < -c.max_dd_halt:
            self._trade(ts, 0, px, "max drawdown halt", p)
            self.halted, self.status = True, "halted: max drawdown"
            self.log("max drawdown reached: flattened, trading halted")
            return
        if eq / self.day_eq - 1 < -c.daily_loss_halt:
            self.halted_day = self.day
        want, reason = None, "model"
        if self.halted_day == self.day:
            want, reason = 0, "daily loss halt"
        elif br.qty and (px / br.entry - 1) * br.sign() < -c.stop_pct:
            want, reason, self.cooldown = 0, "stop loss", c.horizon
        elif self.cooldown > 0:
            want, reason, self.cooldown = 0, "cooldown", self.cooldown - 1
        elif self.auto:
            want = next_pos(br.sign(), p if p is not None else float("nan"), c)
        if want is not None:
            self._trade(ts, want * c.risk_frac, px, reason, p)

    def _step_bar(self, i):
        df = self.rd.df
        ts, px = df.index[i], float(df.close.iloc[i])
        p = None if np.isnan(self.rd.prob[i]) else float(self.rd.prob[i])
        self.last_ts, self.last_px, self.last_p, self.pos = ts, px, p, i
        self._process(ts, px, p)
        self.bars += 1
        eq = self.br.equity(px)
        self.eq.append((str(ts), px, p, self.br.sign(), float(eq)))
        if self.continual and self.bars % self.cfg.retrain_every == 0 and not self.halted:
            try:
                self._continual(i)
            except Exception as e:
                self.log(f"retrain failed: {type(e).__name__}: {e}")
        if self.mode == "live":
            json.dump({"broker": self.br.state(), "peak": self.peak, "bars": self.bars}, open(f"{self.rd.dir}/paper_state.json", "w"))

    # ---------- continual training: fine-tune a copy from the current weights, promote only if it wins on fresh data
    def _continual(self, i):
        c, rd = self.cfg, self.rd
        S, H = c.seq_len, c.horizon
        d = rd.df.iloc[max(0, i + 1 - (c.ft_window + 600)):i + 1]
        F = clean(build_features(d, c))
        y, _ = make_labels(d, H)
        y = y.reindex(F.index)
        ok = y.notna().values
        F, y = F[ok], y[ok].values
        if len(F) < c.ft_val + 600:
            self.log(f"retrain skipped: only {len(F)} labelled bars")
            return
        es_n = c.ft_val // 2
        tr_end = len(F) - c.ft_val - H
        es = slice(len(F) - c.ft_val, len(F) - es_n)
        cmp_ = slice(len(F) - es_n - S + 1, len(F))
        ch = copy.deepcopy(rd.P)
        ch.fit(F.iloc[:tr_end], y[:tr_end], F.iloc[es], y[es], epochs=c.ft_epochs, lr=c.ft_lr, warm=True)
        a, b = _eval(rd.P, F.iloc[cmp_], y[cmp_], S), _eval(ch, F.iloc[cmp_], y[cmp_], S)
        win = b["logloss"] < a["logloss"] - 5e-4 and b["auc"] >= a["auc"] - 0.005
        ev = {"ts": str(self.last_ts), "from": rd.P.version, "champ_ll": a["logloss"], "chall_ll": b["logloss"],
              "champ_auc": a["auc"], "chall_auc": b["auc"], "promoted": bool(win)}
        if win:
            ch.version = rd.P.version + 1
            ch.save(f"{rd.dir}/model_v{ch.version}")
            ch.save(f"{rd.dir}/model_latest")
            rd.P = ch
            rd.repredict(i)
        self.retrains.append(ev)
        self.log(f"retrain: champion ll={a['logloss']:.4f} auc={a['auc']:.3f} | challenger ll={b['logloss']:.4f} auc={b['auc']:.3f} -> "
                 + (f"PROMOTED v{rd.P.version}" if win else "rejected"))

    # ---------- manual orders
    def manual(self, side, frac=None):
        with self.lock:
            c, px = self.cfg, self.last_tick or self.last_px
            frac = c.risk_frac if frac is None else float(frac)
            target = {"buy": frac, "sell": -frac if c.allow_short else 0.0, "flat": 0.0}[side]
            self._trade(self.last_ts, target, px, "manual", self.last_p)

    # ---------- loops
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.status = "stopped" if self.status == "running" else self.status

    def run(self):
        self.status = "running"
        try:
            while not self.stop_ev.is_set() and not self.halted:
                if os.path.exists(f"{self.rd.dir}/STOP"):
                    self._trade(self.last_ts, 0, self.last_px, "STOP file", self.last_p)
                    self.status = "stopped (STOP file)"
                    return
                if self.cfg.max_bars and self.bars >= self.cfg.max_bars:
                    self.status = "finished"
                    return
                if self.mode == "replay":
                    if self.pos + 1 >= len(self.rd.df):
                        self.status = "finished"
                        return
                    with self.lock:
                        self._step_bar(self.pos + 1)
                    if self.speed > 0:
                        self.stop_ev.wait(1.0 / self.speed)
                else:
                    new = self.feed.poll()
                    tick = self.feed.last_price()
                    if tick:
                        self.last_tick = tick
                    if new.empty:
                        self.stop_ev.wait(self.cfg.poll_seconds)
                        continue
                    with self.lock:
                        for k in range(len(new)):
                            self.rd.append(new.iloc[k:k + 1])
                            self._step_bar(len(self.rd.df) - 1)
        except Exception as e:
            self.status = f"error: {type(e).__name__}: {e}"
            self.log(self.status)
        else:
            if self.halted:
                pass
        finally:
            if self.status == "running":
                self.status = "stopped"

    # ---------- reporting
    def stats(self):
        e = np.array([x[4] for x in self.eq], float)
        out = {"sharpe": 0.0, "max_dd": 0.0}
        if len(e) > 3:
            r = np.diff(e) / e[:-1]
            if r.std() > 0:
                out["sharpe"] = float(r.mean() / r.std() * np.sqrt(bars_per_year(self.cfg)))
            out["max_dd"] = float((e / np.maximum.accumulate(e) - 1).min())
        wins = []
        return out

    def snapshot(self, curve_points=800):
        with self.lock:
            c, br = self.cfg, self.br
            px = self.last_tick or self.last_px
            eq = br.equity(px)
            side = br.sign()
            step = max(len(self.eq) // curve_points, 1)
            return jclean({
                "status": self.status, "mode": self.mode, "speed": self.speed, "auto": self.auto, "continual": self.continual,
                "bars": self.bars, "pos": self.pos, "ts": str(self.last_ts), "price": px, "prob": self.last_p,
                "equity": eq, "cash": br.cash, "start_cash": c.start_cash, "ret_pct": (eq / c.start_cash - 1) * 100,
                "buy_hold_pct": (px / self.first_px - 1) * 100, "drawdown_pct": (eq / self.peak - 1) * 100,
                "position": {"side": "long" if side > 0 else "short" if side < 0 else "flat", "qty": br.qty, "entry": br.entry,
                             "upnl": br.qty * (px - br.entry) if br.qty else 0.0,
                             "upnl_pct": (px / br.entry - 1) * side * 100 if br.qty else 0.0},
                "model_version": self.rd.P.version, "halted": self.halted, "daily_halt": self.halted_day is not None and self.halted_day == self.day,
                "cooldown": self.cooldown, "stats": self.stats(), "n_trades": len(self.trades),
                "trades": self.trades[-150:], "retrains": self.retrains[-15:], "log": self.logs[-40:],
                "curve": [list(x) for x in self.eq[::step]] + ([list(self.eq[-1])] if self.eq and (len(self.eq) - 1) % step else []),
                "cfg": {k: getattr(c, k) for k in ("symbol", "timeframe", "exchange", "risk_frac", "stop_pct", "max_dd_halt", "daily_loss_halt",
                                                    "long_th", "short_th", "allow_short", "fee_bps", "slip_bps", "retrain_every", "horizon")},
            })
