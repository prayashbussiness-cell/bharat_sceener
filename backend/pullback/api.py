"""FastAPI router for the Pull Back screener - mounted at /api/pullback/*.
Mirrors the host app's async job/poll pattern (POST /run starts a background thread,
GET /status polls it, GET /results reads the cache) so it behaves like the other
screeners on the page, but never shares state or scoring code with them.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from . import backtest as B
from . import data_source as DS
from . import engine as E

log = logging.getLogger("pullback.api")
router = APIRouter(prefix="/api/pullback", tags=["pullback"])

_DATA_DIR = os.environ.get("PULLBACK_DATA_DIR", "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "last_result_pullback.json")
_UNIVERSE = os.environ.get("SCREEN_UNIVERSE_PULLBACK", "quicklist")
_TOP_N = int(os.environ.get("SCREEN_TOP_N_PULLBACK", "40"))

_lock = threading.Lock()
_job = {"status": "idle", "started": None, "finished": None, "error": None, "progress": {"done": 0, "total": 0}}


def _load_cache():
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(payload):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(payload, f)


def _run_scan(universe, top_n, quick):
    symbols = DS.universe_symbols("quicklist" if quick else universe)
    results, errors = [], []
    _job["progress"] = {"done": 0, "total": len(symbols)}
    DS.bulk_prefetch(symbols)  # one batched request for the whole universe, see data_source.py
    for sym in symbols:
        try:
            df = DS.get_ohlcv(sym)
            if df is None or len(df) < 60:
                errors.append({"symbol": sym, "reason": "no_data"})
            else:
                r = E.evaluate(df)
                r["symbol"] = sym
                results.append(r)
        except Exception as e:
            log.warning("scan failed for %s: %s", sym, e)
            errors.append({"symbol": sym, "reason": str(e)})
        _job["progress"]["done"] += 1
    order = {s: i for i, s in enumerate(E.rules_config()["states"])}
    results.sort(key=lambda r: (-order.get(r["state"], 0), -r["score"]))
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe": universe, "quick": quick,
               "count": len(results), "results": results[:top_n] if top_n else results,
               "all_symbol_states": {r["symbol"]: r["state"] for r in results}, "errors": errors}
    _save_cache(payload)
    return payload


def _job_worker(universe, top_n, quick):
    try:
        _run_scan(universe, top_n, quick)
        _job["status"] = "done"
    except Exception as e:
        log.exception("pullback scan crashed")
        _job["status"] = "error"
        _job["error"] = str(e)
    finally:
        _job["finished"] = datetime.now(timezone.utc).isoformat()


@router.post("/run")
async def run(universe: str = _UNIVERSE, top_n: int = _TOP_N, quick: bool = False):
    with _lock:
        if _job["status"] == "running":
            return {"status": "already_running", "progress": _job["progress"]}
        _job.update(status="running", started=datetime.now(timezone.utc).isoformat(), finished=None, error=None,
                    progress={"done": 0, "total": 0})
    threading.Thread(target=_job_worker, args=(universe, top_n, quick), daemon=True).start()
    return {"status": "started"}


@router.get("/status")
async def status():
    return _job


@router.get("/results")
async def results():
    data = _load_cache()
    if data is None:
        raise HTTPException(404, "No completed scan yet - POST /api/pullback/run first.")
    return data


@router.get("/lookup/{symbol}")
async def lookup(symbol: str):
    df = await run_in_threadpool(DS.get_ohlcv, symbol)
    if df is None or len(df) < 60:
        raise HTTPException(404, f"No usable price data for {symbol}")
    return await run_in_threadpool(E.evaluate, df)


@router.get("/config")
async def config():
    return E.rules_config()


@router.post("/backtest")
async def backtest(symbols: list[str] | None = None, hold_days: int = 20):
    syms = symbols or DS.universe_symbols("quicklist")
    price_data = {}
    for s in syms:
        df = await run_in_threadpool(DS.get_ohlcv, s)
        if df is not None and len(df) > 250:
            price_data[s] = df
    if not price_data:
        raise HTTPException(400, "No usable price history for the requested symbols.")
    return await run_in_threadpool(B.run_backtest, price_data, None, hold_days)
