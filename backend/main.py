"""
Bharat Top Performing Stocks - live screener web app.

Design note on the button:
A full fundamentals scan takes several minutes, which is too long for a
single HTTP request (Render, browsers, and load balancers all time out long
before that). So each "run" button starts a BACKGROUND JOB and the page
polls /api/status until it's done, then fetches /api/results for that mode.

Two independent screeners are supported (see MODES below): "primary"
(Nifty 100, large-cap - closest to the real strategy's disclosed config)
and "broad" (Nifty 250, more names/more risk). "quick" is a small utility
mode for a fast sub-minute sanity check. Only one scan runs at a time
system-wide (Render free tier has limited CPU), but each mode's last
COMPLETED result is cached separately on disk, so running one never
overwrites the other's results - you can view both independently.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.engine_core import run_screen, BUCKET_WEIGHTS
from backend.newflow_engine import run_newflow_screen

from backend.gemini_summary import generate_commentary

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Config via env vars, so Render deploy settings can tune this without a code change.
UNIVERSE_PRIMARY = os.environ.get("SCREEN_UNIVERSE_PRIMARY", "nifty100")
UNIVERSE_BROAD = os.environ.get("SCREEN_UNIVERSE_BROAD", "nifty250")
UNIVERSE_NEWFLOW = os.environ.get("SCREEN_UNIVERSE_NEWFLOW", "nifty250")
TOP_N = int(os.environ.get("SCREEN_TOP_N", "20"))
MAX_PER_SECTOR = int(os.environ.get("SCREEN_MAX_PER_SECTOR", "4"))
FAST_MODE = os.environ.get("SCREEN_FAST_MODE", "false").lower() == "true"
ENABLE_GEMINI = os.environ.get("ENABLE_GEMINI", "true").lower() == "true"

# Two Bharat Screener universes + one fast test mode, all using the original
# scoring engine, plus "newflow" - a completely separate, deeper scoring
# engine (its own factor set, hard filters, red-flag penalties - see
# newflow_engine.py). "engine" tags which scoring code a mode uses; the
# original three never touch newflow_engine and vice versa. Each mode gets
# its own result cache file.
MODES = {
    "primary": {"universe": UNIVERSE_PRIMARY, "fast": FAST_MODE, "label": "Nifty 100 (primary)", "engine": "bharat"},
    "broad":   {"universe": UNIVERSE_BROAD,   "fast": FAST_MODE, "label": "Nifty 250 (broad)", "engine": "bharat"},
    "quick":   {"universe": "quicklist",      "fast": False,     "label": "Quick test (30 stocks)", "engine": "bharat"},
    "newflow": {"universe": UNIVERSE_NEWFLOW, "fast": False,     "label": "New Flow 0.1", "engine": "newflow"},
}


def _result_file(mode: str) -> str:
    return os.path.join(DATA_DIR, f"last_result_{mode}.json")


app = FastAPI(title="Bharat Top Performing Stocks")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# In-memory job state (single worker process assumption - see README for
# what changes if you scale Render to multiple instances/workers). Only one
# scan runs at a time system-wide, but the "mode" field says which one, and
# results are written to that mode's own file so the other mode's last
# result is never touched.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_job = {
    "status": "idle",      # idle | running | done | error
    "mode": None,
    "progress": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _log(msg: str) -> None:
    with _lock:
        _job["progress"] = msg
    print(f"[screen] {msg}", flush=True)


def _run_job(mode: str) -> None:
    cfg = MODES[mode]
    with _lock:
        if _job["status"] == "running":
            return
        _job.update(status="running", mode=mode, progress="starting",
                    started_at=time.time(), finished_at=None, error=None)
    try:
        if cfg["engine"] == "newflow":
            result = run_newflow_screen(
                universe=cfg["universe"], top=TOP_N,
                max_per_sector=MAX_PER_SECTOR, log=_log,
            )
            # New Flow already produces deterministic, auditable Key
            # Strengths/Risks per stock (see newflow_engine.py) - it doesn't
            # need or use Gemini commentary on top of that.
            result["commentary"] = {"per_stock": {}, "overall": ""}
        else:
            result = run_screen(
                universe=cfg["universe"], top=TOP_N, fast=cfg["fast"],
                max_per_sector=MAX_PER_SECTOR, weights=BUCKET_WEIGHTS, log=_log,
            )
            if ENABLE_GEMINI and result["picks"]:
                _log("Asking Gemini for commentary")
                commentary = generate_commentary(result["picks"])
                result["commentary"] = commentary
            else:
                result["commentary"] = {"per_stock": {}, "overall": ""}

        result["mode"] = mode
        result["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(_result_file(mode), "w") as f:
            json.dump(result, f, indent=2, default=str)

        with _lock:
            _job.update(status="done", progress="complete", finished_at=time.time())
    except Exception as e:
        traceback.print_exc()
        with _lock:
            _job.update(status="error", progress="failed",
                        error=f"{type(e).__name__}: {e}", finished_at=time.time())


@app.get("/api/status")
def api_status():
    with _lock:
        return dict(_job)


@app.post("/api/run")
def api_run(mode: str = Query("broad", enum=list(MODES))):
    """
    mode=primary -> Nifty 100, large-cap - closest to the real disclosed
                    Bharat Market Outperformers config
    mode=broad   -> Nifty 250, wider sweep, more names, more risk
    mode=quick   -> ~30-stock curated watchlist, no NSE download needed,
                    sub-minute - good for testing the button/Gemini wiring
    """
    with _lock:
        if _job["status"] == "running":
            return {"started": False, "reason": "already running",
                    "running_mode": _job["mode"], **_job}
    t = threading.Thread(target=_run_job, args=(mode,), daemon=True)
    t.start()
    return {"started": True, "mode": mode, **MODES[mode]}


@app.get("/api/results")
def api_results(mode: str = Query("broad", enum=list(MODES))):
    path = _result_file(mode)
    if not os.path.exists(path):
        raise HTTPException(404, f"No completed '{mode}' scan yet - POST /api/run?mode={mode} first")
    with open(path) as f:
        return json.load(f)


@app.get("/api/lookup/{symbol}")
def api_lookup(symbol: str, mode: str = Query("broad", enum=list(MODES))):
    """Answers 'why isn't <symbol> in the list' directly: eligible or not,
    which rule it failed if not, and its full score breakdown if it was
    scored - regardless of whether it made the final top N. Checked against
    the given mode's last completed scan (default: the Nifty 250 'broad' run).
    """
    path = _result_file(mode)
    if not os.path.exists(path):
        raise HTTPException(404, f"No completed '{mode}' scan yet - POST /api/run?mode={mode} first")
    with open(path) as f:
        data = json.load(f)
    lookup = data.get("lookup", {})
    sym = symbol.strip().upper()
    if sym not in lookup:
        raise HTTPException(404, f"'{sym}' was not in the scanned universe "
                            f"({data.get('universe')}, {data.get('universe_size')} names) "
                            f"for the last '{mode}' run.")
    return {"symbol": sym, "mode": mode, "scan_generated_at": data.get("generated_at"),
            **lookup[sym]}


@app.get("/api/config")
def api_config():
    from dataclasses import asdict
    from backend.engine_core import Eligibility, FACTORS
    from backend.newflow_engine import HardFilters as NewFlowHardFilters, \
        BUCKET_WEIGHTS_V2, FACTORS_V2, RED_FLAG_PENALTIES

    rules = asdict(Eligibility())
    factors_by_bucket: dict[str, list[str]] = {}
    for col, higher, bucket in FACTORS:
        factors_by_bucket.setdefault(bucket, []).append(
            f"{col} ({'higher is better' if higher else 'lower is better'})")

    newflow_rules = asdict(NewFlowHardFilters())
    newflow_factors_by_bucket: dict[str, list[str]] = {}
    for col, higher, bucket in FACTORS_V2:
        newflow_factors_by_bucket.setdefault(bucket, []).append(
            f"{col} ({'higher is better' if higher else 'lower is better'})")

    return {
        "modes": MODES,
        "top_n": TOP_N,
        "max_per_sector": MAX_PER_SECTOR,
        "fast_mode": FAST_MODE,
        "gemini_enabled": ENABLE_GEMINI,
        "weights": BUCKET_WEIGHTS,
        "eligibility_rules": rules,
        "factors_by_bucket": factors_by_bucket,
        "newflow": {
            "hard_filters": newflow_rules,
            "factors_by_bucket": newflow_factors_by_bucket,
            "weights": BUCKET_WEIGHTS_V2,
            "red_flag_penalties": RED_FLAG_PENALTIES,
        },
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
