"""Everything needed to reproduce a research run: code version, library versions, arguments, seeds and a fingerprint of every data file."""
import hashlib, json, os, platform, sys, time
from ledger import _git


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def build(args=None, data_files=(), extra=None):
    libs = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "ccxt", "torch"):
        try:
            libs[m] = __import__(m).__version__
        except Exception:
            libs[m] = None
    files = {p: dict(sha256_16=sha(p), bytes=os.path.getsize(p)) for p in data_files if os.path.exists(p)}
    return dict(created=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), commit=_git(), python=sys.version.split()[0],
                platform=platform.platform(), libraries=libs, args=args or {}, data=files, **(extra or {}))


def write(path, **kw):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    m = build(**kw)
    json.dump(m, open(path, "w"), indent=1, default=str)
    return m
