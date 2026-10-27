"""Web backend + UI.   python server.py   ->  open http://localhost:8000

Everything is paper trading: no exchange keys are used and no orders are sent anywhere.
"""
import json, os, re, threading, time, traceback, uuid
from dataclasses import fields
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from config import Cfg
from engine import RunData, Session, jclean
from train import run_training

app = FastAPI(title="algo_ai")
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)
JOBS, DATA, STATE = {}, {}, {"session": None}
LOCK = threading.RLock()


def run_dir(symbol, tf, source="ccxt"):
    tag = re.sub(r"[^A-Za-z0-9]+", "-", symbol).strip("-")
    return os.path.join(RUNS, f"{tag}_{tf}" + ("_synthetic" if source == "synthetic" else ""))


def get_rd(name, from_latest=False, refresh=False):
    key = (name, from_latest)
    with LOCK:
        if refresh or key not in DATA:
            d = os.path.join(RUNS, name)
            if not os.path.exists(f"{d}/metrics.json"):
                raise HTTPException(404, f"no trained model '{name}'")
            DATA[key] = RunData(d, from_latest).load(lambda *_: None)
        return DATA[key]


# ------------------------------------------------------------------ models / training
class TrainReq(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    exchange: str = "binance"
    source: str = "ccxt"
    years: float = 8.0
    features: str = "base,smc,amt"
    horizon: int = 8
    epochs: int = 12
    allow_short: bool = True
    fee_bps: float = 5.0
    slip_bps: float = 2.0
    use_gru: bool = True
    bars: int = 15000


@app.post("/api/train")
def train(req: TrainReq):
    cfg = Cfg()
    for k, v in req.model_dump().items():
        setattr(cfg, k, v)
    cfg.out = run_dir(cfg.symbol, cfg.timeframe, cfg.source)
    jid = uuid.uuid4().hex[:8]
    job = JOBS[jid] = {"id": jid, "status": "running", "log": [], "run": os.path.basename(cfg.out), "result": None, "t0": time.time()}

    def work():
        try:
            job["result"] = run_training(cfg, lambda m: job["log"].append(str(m)))
            job["status"] = "done"
            with LOCK:
                for k in [k for k in DATA if k[0] == job["run"]]:
                    DATA.pop(k)
        except Exception as e:
            job["status"] = "error"
            job["log"].append(f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()

    threading.Thread(target=work, daemon=True).start()
    return {"job": jid, "run": job["run"]}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return jclean({**j, "log": j["log"][-80:], "elapsed": time.time() - j["t0"]})


@app.get("/api/models")
def models():
    out = []
    for n in sorted(os.listdir(RUNS)):
        p = f"{RUNS}/{n}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            m.pop("curve", None)
            out.append({"run": n, **{k: m.get(k) for k in ("symbol", "timeframe", "exchange", "source", "features", "n_bars", "data_from", "data_to",
                                                         "walk_forward", "holdout", "importance", "edge", "verdict")},
                        "has_latest": os.path.exists(f"{RUNS}/{n}/model_latest/meta.json")})
    return jclean(out)


@app.get("/api/models/{name}")
def model_detail(name: str):
    p = f"{RUNS}/{name}/metrics.json"
    if not os.path.exists(p):
        raise HTTPException(404)
    return jclean(json.load(open(p)))


# ------------------------------------------------------------------ chart
@app.get("/api/chart")
def chart(run: str, n: int = 300, end: int = -1, latest: bool = False):
    s = STATE["session"]
    rd = s.rd if s and s.rd.dir.endswith(run) else get_rd(run, latest)
    pos = end if end >= 0 else (s.pos if s and s.rd is rd else len(rd.df) - 1)
    return rd.chart(pos, max(50, min(n, 1200)), s.trades if s and s.rd is rd else None)


# ------------------------------------------------------------------ paper trading session
class SessionReq(BaseModel):
    run: str
    mode: str = "replay"
    speed: float = 8.0
    auto: bool = True
    continual: bool = False
    start_cash: float = 10000.0
    risk_frac: float = 0.95
    stop_pct: float = 0.03
    max_dd_halt: float = 0.15
    daily_loss_halt: float = 0.04
    allow_short: bool = True
    long_th: float = 0.56
    short_th: float = 0.44
    fee_bps: float = 5.0
    slip_bps: float = 2.0
    retrain_every: int = 96
    poll_seconds: int = 20


@app.post("/api/session/start")
def session_start(req: SessionReq):
    with LOCK:
        if STATE["session"]:
            STATE["session"].stop()
        rd = RunData(os.path.join(RUNS, req.run), from_latest=(req.mode == "live")).load(lambda *_: None)
        over = {k: v for k, v in req.model_dump().items() if k not in ("run", "mode", "speed", "auto", "continual")}
        s = Session(rd, req.mode, req.speed, req.auto, req.continual, **over)
        STATE["session"] = s
        for f in ("equity.csv", "trades.csv"):
            pass
        s.start()
    return s.snapshot()


def cur():
    s = STATE["session"]
    if not s:
        raise HTTPException(404, "no session")
    return s


@app.get("/api/session")
def session_get():
    s = STATE["session"]
    return s.snapshot() if s else JSONResponse({"status": "none"})


class Patch(BaseModel):
    speed: float | None = None
    auto: bool | None = None
    continual: bool | None = None


@app.post("/api/session/patch")
def session_patch(p: Patch):
    s = cur()
    for k, v in p.model_dump().items():
        if v is not None:
            setattr(s, k, v)
    return s.snapshot()


@app.post("/api/session/pause")
def session_pause():
    s = cur()
    s.stop()
    s.status = "paused"
    return s.snapshot()


@app.post("/api/session/resume")
def session_resume():
    s = cur()
    if s.halted:
        raise HTTPException(400, "halted by the risk layer; start a new session")
    s.start()
    return s.snapshot()


@app.post("/api/session/stop")
def session_stop():
    s = cur()
    s.stop()
    s._trade(s.last_ts, 0, s.last_tick or s.last_px, "session stopped", s.last_p)
    s.status = "stopped"
    return s.snapshot()


class Order(BaseModel):
    side: str
    frac: float | None = None


@app.post("/api/session/order")
def session_order(o: Order):
    if o.side not in ("buy", "sell", "flat"):
        raise HTTPException(400, "side must be buy / sell / flat")
    s = cur()
    s.manual(o.side, o.frac)
    return s.snapshot()


@app.get("/api/symbols")
def symbols():
    return {"crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"],
            "timeframes": ["15m", "1h", "4h", "1d"], "exchanges": ["binance", "bybit", "okx", "kraken", "bitstamp", "coinbase"]}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  algo_ai  ->  http://localhost:{port}\n")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")
