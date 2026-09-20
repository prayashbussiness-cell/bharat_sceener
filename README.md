# Bharat Top Performing Stocks — live screener web app

A FastAPI service with one button. Click it, and it fetches live prices and
fundamentals from Yahoo Finance for the Nifty LargeMidcap 250, runs them
through the same two-stage eligibility + sector-neutral scoring engine from
the earlier CLI version, and has Gemini write a one-line rationale for each
pick — grounded strictly in the numbers the engine computed, nothing else.

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo. `render.yaml` sets
   everything up automatically (build command, start command, health check).
3. Render will ask you to fill in `GEMINI_API_KEY` (marked `sync: false` in
   the blueprint so it's never committed to git). Get a free key at
   https://aistudio.google.com/apikey.
4. Deploy. First load will be empty — click **Run today's scan**.

No blueprint? Manually create a Web Service with:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env var: `GEMINI_API_KEY`

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GEMINI_API_KEY
export $(cat .env | grep -v '^#' | xargs)   # or use python-dotenv / direnv
uvicorn app.main:app --reload
```

Open http://localhost:8000 and click the button. Or use the original CLI:

```bash
python run_cli.py --universe nifty250 --top 20
```

## How the button actually works

A full fundamentals scan of 250 stocks takes several minutes — too long for
one HTTP request (browsers, Render, and any load balancer in between will
time out first). So:

1. Click → `POST /api/run` starts a background thread, returns immediately.
2. The page polls `GET /api/status` every ~2s (`idle` → `running` → `done`).
3. On `done`, it fetches `GET /api/results` and renders the table.
4. The result is also cached to `data/last_result.json`, so reloading the
   page shows the last completed scan instantly without re-running anything.

A second click while a scan is already running attaches to the same job
instead of starting a duplicate — there's a lock around the single job slot.

**This job model assumes one server process.** If you ever scale the Render
service to more than one instance, or more than one uvicorn worker, each
process gets its own job state and its own cache file, and the button will
behave inconsistently (poll one instance, get run on another). Keep this at
one instance/one worker unless you move job state into something shared
(Redis, a database row) — flagged here so it doesn't surprise you later.

## Configuration (env vars, all optional beyond `GEMINI_API_KEY`)

| Var | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | — | required for AI commentary; app still works without it, just skips the rationale text |
| `GEMINI_MODEL` | `gemini-2.5-flash` | swap for `gemini-2.5-pro` for slightly better prose, slower/costlier |
| `ENABLE_GEMINI` | `true` | set `false` to skip the Gemini call entirely |
| `SCREEN_UNIVERSE` | `nifty250` | or `nifty200`, `nifty500`, `midsmall400` |
| `SCREEN_TOP_N` | `15` | how many picks to return |
| `SCREEN_MAX_PER_SECTOR` | `4` | sector concentration cap |
| `SCREEN_FAST_MODE` | `false` | `true` = momentum-only, skips fundamentals entirely (~1 min instead of ~5–10 min) |

## Two real risks, already handled but worth knowing about

**1. NSE blocks cloud IPs.** I tested the live NSE constituent-list URL from
this build environment and it returned `403 Forbidden` — NSE rate-limits and
blocks traffic from data-center IP ranges, which is exactly what Render runs
on. `app/nifty250_fallback.csv` is a bundled, manually curated snapshot of
~225 Nifty LargeMidcap names that the app automatically falls back to when
the live download fails, so the button still works — it just won't reflect
the *exact* current index membership. Refresh that CSV occasionally by
downloading the real list from nseindia.com on a residential connection and
replacing the file. The app logs a warning (visible in Render's log tab)
whenever it's using the fallback, so you'll know when this is happening.

**2. Yahoo Finance rate limits.** Fetching fundamentals for 250 tickers
one-by-one is the slow step and can occasionally get throttled. If a run
errors out, click the button again — fundamentals are cached for 12 hours
(`cache_hours` in `engine_core.run_screen`), so a retry after a partial
failure is much faster than the first attempt.

## What Gemini is and isn't doing here

The prompt in `app/gemini_summary.py` hands Gemini only the already-computed
factor scores (growth/quality/value/momentum/risk sub-scores, ROCE, P/E,
etc.) and instructs it to explain the ranking in plain English — not to
introduce outside facts, not to predict returns, not to say buy/sell/hold.
If the API key is missing or the call fails for any reason, the app falls
back to showing raw scores with no commentary rather than breaking the page.

## Still on the "add later" list

- Separate scoring path for banks/NBFCs (current model uses EV/EBITDA and
  ROCE, which don't mean much for lenders)
- A scheduled nightly refresh (Render Cron Job hitting `POST /api/run`) so
  the cache is always warm before anyone visits
- Point-in-time fundamentals and a real backtest harness — see the caveats
  in the original screener README, they still apply here
- Persisting scan history (e.g. to Supabase, if you want to reuse the
  project pattern from PalmAI) instead of only keeping the latest run
