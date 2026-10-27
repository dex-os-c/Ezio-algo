"""Predictor = small GRU (uses the GPU when present) + LightGBM, averaged. Supports warm-start fine-tuning from the current weights."""
import copy, json, os
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def windows(X, seq):
    """(n,f) -> (n-seq+1, seq, f); window k ends at row k+seq-1"""
    return np.transpose(sliding_window_view(X, seq, axis=0), (0, 2, 1))


if HAS_TORCH:
    class GRUNet(nn.Module):
        def __init__(self, nf, hid, layers=2, drop=0.2):
            super().__init__()
            self.gru = nn.GRU(nf, hid, layers, batch_first=True, dropout=drop)
            self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(drop), nn.Linear(hid, 1))

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1]).squeeze(-1)


def _flat(X):
    lag = lambda k: np.vstack([np.full((k, X.shape[1]), np.nan, np.float32), X[:-k]])
    return np.hstack([X, lag(1), lag(4)])


class Predictor:
    def __init__(self, cfg):
        self.cfg, self.mu, self.sd, self.cols = cfg, None, None, None
        self.gru, self.lgb, self.version = None, None, 0

    # ---------- helpers ----------
    def _prep(self, F):
        return ((F[self.cols].values.astype(np.float32) - self.mu) / self.sd).clip(-5, 5)

    @staticmethod
    def _dev(n=10 ** 9):
        return "cuda" if HAS_TORCH and torch.cuda.is_available() and n > 1000 else "cpu"

    # ---------- training ----------
    def fit(self, F, y, Fv=None, yv=None, epochs=None, lr=None, warm=False):
        cfg = self.cfg
        if not warm or self.mu is None:
            warm, self.cols = False, list(F.columns)
            X = F.values.astype(np.float32)
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xt, yt = self._prep(F), np.asarray(y, np.float32)
        Xv = self._prep(Fv) if Fv is not None else None
        yv = np.asarray(yv, np.float32) if yv is not None else None
        info = {}
        if cfg.use_gru and HAS_TORCH:
            info["gru"] = self._fit_gru(Xt, yt, Xv, yv, epochs or cfg.epochs, lr or cfg.lr, warm)
        if cfg.use_lgb and HAS_LGB:
            info["lgb"] = self._fit_lgb(Xt, yt, Xv, yv, warm)
        return info

    def _fit_gru(self, Xt, yt, Xv, yv, epochs, lr, warm):
        cfg, S = self.cfg, self.cfg.seq_len
        torch.manual_seed(cfg.seed)
        if self.gru is None or not warm:
            self.gru = GRUNet(Xt.shape[1], cfg.hidden)
        dev = self._dev()
        m = self.gru.to(dev)
        Wt, yw = windows(Xt, S), yt[S - 1:]
        Wv = windows(Xv, S) if Xv is not None else None
        yvw = yv[S - 1:] if yv is not None else None
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad, ep = 1e9, None, 0, 0
        for ep in range(epochs):
            m.train()
            perm = np.random.default_rng(cfg.seed + ep).permutation(len(Wt))
            for i in range(0, len(Wt), cfg.batch):
                idx = perm[i:i + cfg.batch]
                xb = torch.from_numpy(np.ascontiguousarray(Wt[idx])).to(dev)
                yb = torch.from_numpy(yw[idx]).to(dev)
                opt.zero_grad()
                lossf(m(xb), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            if Wv is not None:
                m.eval()
                with torch.no_grad():
                    lo = [m(torch.from_numpy(np.ascontiguousarray(Wv[j:j + 2048])).to(dev)).cpu().numpy()
                          for j in range(0, len(Wv), 2048)]
                lo = np.concatenate(lo)
                vl = float(np.mean(np.logaddexp(0, lo) - yvw * lo))
                if vl < best - 1e-4:
                    best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
                else:
                    bad += 1
                    if bad >= 3:
                        break
        if best_state is not None:
            m.load_state_dict(best_state)
        self.gru = m.cpu()
        return {"val_loss": None if best_state is None else round(best, 5), "epochs": ep + 1, "device": dev}

    def _fit_lgb(self, Xt, yt, Xv, yv, warm):
        cfg = self.cfg
        params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, seed=cfg.seed)
        dt = lgb.Dataset(_flat(Xt), yt)
        if warm and self.lgb is not None:
            self.lgb = lgb.train(params, dt, num_boost_round=40, init_model=self.lgb)
        else:
            vs, cb = None, []
            if Xv is not None:
                vs, cb = [lgb.Dataset(_flat(Xv), yv)], [lgb.early_stopping(50, verbose=False)]
            self.lgb = lgb.train(params, dt, num_boost_round=600, valid_sets=vs, callbacks=cb)
        return {"trees": self.lgb.current_iteration()}

    # ---------- inference ----------
    def predict(self, F):
        """P(up over next `horizon` bars) per row; first seq_len-1 rows are NaN (not enough history)."""
        S, X = self.cfg.seq_len, self._prep(F)
        ps = []
        if self.gru is not None:
            W = windows(X, S)
            dev = self._dev(len(W))
            m = self.gru.to(dev).eval()
            with torch.no_grad():
                lo = np.concatenate([m(torch.from_numpy(np.ascontiguousarray(W[j:j + 2048])).to(dev)).cpu().numpy()
                                     for j in range(0, len(W), 2048)])
            self.gru = m.cpu()
            ps.append(np.r_[np.full(S - 1, np.nan), 1 / (1 + np.exp(-lo))])
        if self.lgb is not None:
            ps.append(self.lgb.predict(_flat(X)))
        out = np.mean(ps, axis=0)
        out[:S - 1] = np.nan
        return out

    # ---------- persistence ----------
    def save(self, d):
        os.makedirs(d, exist_ok=True)
        if self.gru is not None:
            torch.save({"state": self.gru.state_dict(), "nf": len(self.cols)}, f"{d}/gru.pt")
        if self.lgb is not None:
            self.lgb.save_model(f"{d}/lgb.txt")
        json.dump({"mu": self.mu.tolist(), "sd": self.sd.tolist(), "cols": self.cols, "version": self.version},
                  open(f"{d}/meta.json", "w"))

    @classmethod
    def load(cls, d, cfg):
        p = cls(cfg)
        meta = json.load(open(f"{d}/meta.json"))
        p.mu, p.sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
        p.cols, p.version = meta["cols"], meta.get("version", 0)
        if os.path.exists(f"{d}/gru.pt") and HAS_TORCH:
            ck = torch.load(f"{d}/gru.pt", map_location="cpu")
            p.gru = GRUNet(ck["nf"], cfg.hidden)
            p.gru.load_state_dict(ck["state"])
        if os.path.exists(f"{d}/lgb.txt") and HAS_LGB:
            p.lgb = lgb.Booster(model_file=f"{d}/lgb.txt")
        return p
