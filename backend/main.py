"""
Bharat Top Performing Stocks - live screener web app.

Design note on the button:
A full Nifty-250 fundamentals scan takes several minutes, which is too long
for a single HTTP request (Render, browsers, and load balancers all time out
long before that). So the button starts a BACKGROUND JOB and the page polls
/api/status until it's done, then fetches /api/results. This also means only
one scan runs at a time - a second click while a scan is running attaches to
the same job instead of starting a duplicate.

Last completed result is cached to disk (data/last_result.json) so the page
has something to show immediately on load, before anyone clicks the button.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.engine_core import run_screen, BUCKET_WEIGHTS
from backend.gemini_summary import generate_commentary

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULT_FILE = os.path.join(DATA_DIR, "last_result.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Config via env vars, so Render deploy settings can tune this without a code change.
UNIVERSE = os.environ.get("SCREEN_UNIVERSE", "nifty250")
TOP_N = int(os.environ.get("SCREEN_TOP_N", "15"))
MAX_PER_SECTOR = int(os.environ.get("SCREEN_MAX_PER_SECTOR", "4"))
FAST_MODE = os.environ.get("SCREEN_FAST_MODE", "false").lower() == "true"
ENABLE_GEMINI = os.environ.get("ENABLE_GEMINI", "true").lower() == "true"

app = FastAPI(title="Bharat Top Performing Stocks")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# In-memory job state (single worker process assumption - see README for
# what changes if you scale Render to multiple instances/workers)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_job = {
    "status": "idle",      # idle | running | done | error
    "progress": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _log(msg: str) -> None:
    with _lock:
        _job["progress"] = msg
    print(f"[screen] {msg}", flush=True)


def _run_job() -> None:
    with _lock:
        if _job["status"] == "running":
            return
        _job.update(status="running", progress="starting", started_at=time.time(),
                    finished_at=None, error=None)
    try:
        result = run_screen(
            universe=UNIVERSE, top=TOP_N, fast=FAST_MODE,
            max_per_sector=MAX_PER_SECTOR, weights=BUCKET_WEIGHTS, log=_log,
        )
        if ENABLE_GEMINI and result["picks"]:
            _log("Asking Gemini for commentary")
            commentary = generate_commentary(result["picks"])
            result["commentary"] = commentary
        else:
            result["commentary"] = {"per_stock": {}, "overall": ""}

        result["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(RESULT_FILE, "w") as f:
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
def api_run():
    with _lock:
        if _job["status"] == "running":
            return {"started": False, "reason": "already running", **_job}
    t = threading.Thread(target=_run_job, daemon=True)
    t.start()
    return {"started": True}


@app.get("/api/results")
def api_results():
    if not os.path.exists(RESULT_FILE):
        raise HTTPException(404, "No completed scan yet - POST /api/run first")
    with open(RESULT_FILE) as f:
        return json.load(f)


@app.get("/api/config")
def api_config():
    return {
        "universe": UNIVERSE,
        "top_n": TOP_N,
        "max_per_sector": MAX_PER_SECTOR,
        "fast_mode": FAST_MODE,
        "gemini_enabled": ENABLE_GEMINI,
        "weights": BUCKET_WEIGHTS,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
