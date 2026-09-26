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
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Env var: `GEMINI_API_KEY`

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GEMINI_API_KEY
export $(cat .env | grep -v '^#' | xargs)   # or use python-dotenv / direnv
uvicorn backend.main:app --reload
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
| `GEMINI_API_KEY` | — | required for AI commentary on the Bharat screeners; app still works without it, just skips the rationale text. Not used by New Flow 0.1 (it never calls Gemini). |
| `GEMINI_MODEL` | `gemini-2.5-flash` | swap for `gemini-2.5-pro` for slightly better prose, slower/costlier |
| `ENABLE_GEMINI` | `true` | set `false` to skip the Gemini call entirely |
| `SCREEN_UNIVERSE_PRIMARY` | `nifty100` | universe for the "Primary" button |
| `SCREEN_UNIVERSE_BROAD` | `nifty250` | universe for the "Broad" button; or `nifty200`, `nifty500`, `midsmall400` |
| `SCREEN_UNIVERSE_NEWFLOW` | `nifty250` | universe for the New Flow 0.1 button |
| `SCREEN_TOP_N` | `20` | how many picks to return, per screen |
| `SCREEN_MAX_PER_SECTOR` | `4` | sector concentration cap |
| `SCREEN_FAST_MODE` | `false` | `true` = momentum-only for the Bharat screeners, skips fundamentals entirely (~1 min instead of ~5–10 min). Doesn't apply to New Flow 0.1, which always needs fundamentals for its hard filters. |

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

## Fixes and additions since the first version

**Bug fix: banks were being wrongly excluded.** Yahoo reports a bank's
customer deposits as balance-sheet "debt," so `debtToEquity` for a healthy
bank routinely reads 400-900%+ - nothing to do with financial distress, just
how banking works. The debt/equity and interest-coverage eligibility checks
now skip `Financial Services` entirely (`Eligibility.skip_leverage_checks_for_sectors`).
Verified with a test that gives a bank and a non-financial company an
identical D/E and confirms only the non-financial one gets excluded.

**Bug fix: Yahoo fundamentals timing out.** Yahoo's crumb-gated fundamentals
endpoints get blocked from cloud IPs the same way NSE blocks the constituent
list. Previously the app would grind through all 250 names discovering this
one-by-one, taking many minutes and producing nothing visible. It now runs
one preflight check first - if fundamentals are blocked, it falls back to
momentum-only scoring immediately instead of wasting the time, and the UI
shows a banner explaining why. Fundamentals fetching is also parallelized
(`ThreadPoolExecutor`) for the runs where it does work.

**New: "why isn't X in the list" lookup.** `GET /api/lookup/{symbol}` (and a
search box on the page) tells you directly whether a stock was eligible, why
not if it wasn't, and its full score breakdown if it was - whether or not it
made the final top N.

**New: the actual screening thresholds are on the page**, pulled live from
`/api/config` (which introspects the `Eligibility` dataclass directly), so
the displayed rules can never drift out of sync with what the code enforces.

**New: `quicklist` universe.** A ~30-stock curated watchlist that skips the
NSE download entirely. Use the "Quick scan" button, or `mode=quick` on
`POST /api/run`, for a sub-minute sanity check instead of waiting on a full
250-stock scan.

**New factors, from a supplied factor-importance sheet:**
- `golden_cross_num` (Momentum) - the "50 DMA > 200 DMA" trend signal was
  already computed but wasn't feeding the score; now it does.
- `eps_acceleration` (Growth) - an annual proxy for "is EPS growth speeding
  up or fading," computed as this year's growth minus the 3-year CAGR. A
  cleaner quarter-over-quarter version would need quarterly statements,
  which are exposed to the same Yahoo-blocking risk as everything else here.

**Factors considered and deliberately left out:** ROIC (needs an invested-
capital breakdown Yahoo doesn't cleanly expose for Indian filings), earnings
estimate revisions (needs analyst consensus history, not reliably available
free), and promoter pledge (still requires the manual `pledge_overrides.csv`
- no free reliable source found).

**On using Gemini to do the filtering itself:** deliberately not done. The
screen's value depends on being reproducible and auditable - same inputs,
same output, and you can trace exactly why a stock ranked where it did. An
LLM making filtering decisions adds variance and occasional hallucination to
that, which is the wrong trade-off for something informing real money
decisions. Gemini's role stays explanatory: it now also flags in its
commentary when a pick's `data_coverage` is low or a sub-score is missing,
so a thin-data pick doesn't read as a fully-informed one.

## Strategy revision: two screeners, reweighted toward momentum

After comparing our output against publicly disclosed Bharat Market
Outperformers holdings (PSU banks, NBFCs, defense - large-cap, momentum-
heavy names), two things changed:

**Two independent screeners instead of one**, each with its own button,
own cached result, own lookup scope:
- **Primary (Nifty 100)** - large-cap only. Matches a leaked config summary
  describing the real strategy as "Large-Cap, Low risk, Max 20 holdings."
  This is the closer match to what the real product actually holds.
- **Broad (Nifty 250)** - the original wider sweep, kept as a second,
  higher-risk/higher-reach option. Running one never overwrites the other's
  results - they're cached separately (`data/last_result_primary.json` /
  `data/last_result_broad.json`) and both stay viewable on the page at once.

**Bucket weights rebalanced** from 20/25/20/25/10 to
**Growth 20 / Quality 20 / Value 15 / Momentum 35 / Risk 10**. The real
disclosed portfolio (Union Bank of India, Bank of Maharashtra, L&T Finance,
HAL, PNB) reads as momentum/re-rating-driven more than classic cheap-quality
compounding - a screener.in screen sorted primarily by 1-year return and
built on a large-cap universe empirically overlapped with real disclosed
holdings (Union Bank of India and Bank of Maharashtra both appeared in it),
while a separate small-cap "quality+growth" screen sharing the "bharat" name
shared no holdings with the real portfolio at all. That's the evidence behind
the reweight - not a hunch.

**Top N raised from 15 to 20**, matching the disclosed max-holdings figure.

**Caveat, stated plainly:** this reweighting is still evidence-based
inference from public marketing pages and a handful of disclosed names, not
the actual InvestingPro model weights, which remain undisclosed. Treat the
new weights as a better-informed starting point, not a confirmed replica -
the backtesting caveats in the original README still apply in full.

### New API shape (mode-based)

- `POST /api/run?mode=primary|broad|quick`
- `GET /api/results?mode=primary|broad|quick` (default `broad`)
- `GET /api/lookup/{symbol}?mode=primary|broad|quick` (default `broad`)
- `GET /api/status` now also returns `mode`, showing which screener (if any)
  is currently running - only one scan runs system-wide at a time regardless
  of mode, to stay within Render free-tier CPU limits.

New env vars: `SCREEN_UNIVERSE_PRIMARY` (default `nifty100`) and
`SCREEN_UNIVERSE_BROAD` (default `nifty250`), replacing the old single
`SCREEN_UNIVERSE`.

## New Flow 0.1 - a separate, deeper screener

A completely independent third screener, built from a supplied spec (100-
factor engine, hard filters, red-flag penalties, Key Strengths/Risks
output). It has its own button, its own factor set, its own hard filters,
and its own cache file - it never shares scoring code with the Bharat
Screener (primary/broad/quick), so changes to one can never silently affect
the other. It reuses only the market-data plumbing (NSE download + fallback,
Yahoo price/fundamentals fetch) from `engine_core.py`.

### What's real vs. what's a documented no-op

The supplied spec assumes some data this free pipeline doesn't have access
to. Rather than fake those with a shaky proxy, they're implemented as
explicit no-ops - visible in the API response (`hard_filters_not_enforced`)
and in the UI's "Hard filters & red-flag penalties" panel:

- **Hard filters (8 of 10 enforced):** market cap, daily turnover, listing
  history, promoter pledge (only when you supply `pledge_overrides.csv` -
  otherwise inert), extreme debt/equity (500% cutoff, with the same
  Financial-Services exemption as the original screener), extreme 1-year
  dilution, and negative equity are all real, computed checks.
  `accounting_red_flag` and `extreme_circuit_frequency` are **not
  enforced** - they need auditor-opinion data and NSE surveillance/circuit
  history that no free source reliably provides.
- **Red flags (6 of 7 fire for real):** promoter pledge, major dilution,
  multi-year negative FCF, debt explosion (year-over-year, exempting
  banks), earnings/cashflow divergence (via the accrual ratio), and extreme
  valuation all compute from real fetched data. `major_auditor_issue`
  (-15 pts) **never fires** - same reason as above, and it's tested to
  confirm it stays inert rather than silently triggering on bad data.
- **Approximated, not exact:** ROIC uses a flat 25% assumed Indian
  corporate tax rate (no reliable effective-tax-rate field from Yahoo's
  free statements), and `profit_cagr_5y` often computes over 3-4 years in
  practice since Yahoo's free annual statements rarely go back a full 5.
  `cash_conversion_cycle` is left `None` always - receivables/inventory/
  payables aren't reliably available, and a rough approximation there would
  be more misleading than an honest gap.

### Scoring

7 buckets: **Growth 20% / Quality 20% / Earnings Quality 15% / Momentum
20% / Value 10% / Risk 10% / Ownership 5%**. Each factor is the same
sector-neutral winsorized z-score as the original screener (shared code,
`engine_core.score()`, now parametrized by factor list). The 0-100 SCORE is
a percentile rank of the composite z, then red-flag penalty points are
subtracted and the result is clipped back to [0, 100].

**Status label**, deterministic from the score and quality/momentum
sub-scores: `HIGH-QUALITY MOMENTUM` (score ≥80, strong quality AND
momentum) → `OUTPERFORMER` (≥75) → `WATCHLIST` (≥55) → `BELOW THRESHOLD`.

**Key Strengths / Key Risks** are generated deterministically from the
actual per-factor z-scores and triggered red flags - not by an LLM. This
was a deliberate choice, consistent with the rest of this project: the
screening logic stays auditable and reproducible end-to-end, so New Flow
doesn't call Gemini at all.

### API

`POST /api/run?mode=newflow`, `GET /api/results?mode=newflow`,
`GET /api/lookup/{symbol}?mode=newflow` - same shape as the other modes.
`GET /api/config` includes a `newflow` block with its hard filters,
weights, and red-flag penalty table.

New env var: `SCREEN_UNIVERSE_NEWFLOW` (default `nifty250`).

## Progress bar

Every panel now shows an actual progress bar during a run, not just a
status dot. The backend doesn't have each engine report a percentage
directly - instead, `main.py` maps known log-message patterns from any
engine (universe loading, price download, fundamentals fetch progress like
"fundamentals 75/250", scoring, Gemini commentary) to a rough 0-100 stage
estimate in one place (`_progress_from_message`), so it works for all three
engines without each one needing to agree on a shared scale. The percentage
never moves backwards even if a later message maps to an earlier stage.

## Technical Analysis - a third, fully independent engine

A fourth button, built from a supplied ~30-indicator spec. This one is
meaningfully more robust to host than the other two: it needs only daily
OHLCV price history, which comes from a different, unauthenticated Yahoo
endpoint that has stayed reachable even during the fundamentals-blocking
issues documented earlier in this README. No crumb dependency, no NSE
constituent-list dependency beyond the shared universe loader (same
fallback mechanism as the others).

**6 buckets, weights exactly as specified:** Trend 35% / Momentum 20% /
Volume 15% / Volatility 10% / Breakout 10% / Correction 10%.

**Design choice worth knowing:** unlike Bharat Screener and New Flow, this
engine's z-scores are **not sector-neutral** - they're computed across the
whole scanned universe directly. Sector-relative scoring exists to stop a
screen rewarding whichever sector has structurally different valuation or
margin norms; that reasoning doesn't apply to RSI, MACD, or distance from a
moving average, which mean the same thing regardless of what business a
company is in.

**What's computed for real** (all ~30 factors from the spec, organised into
the 6 buckets): price vs 20 EMA/50/100/200 DMA, 20 EMA vs 50 EMA, golden
cross, 50/200 DMA slope, Supertrend (daily *and* weekly, resampled), ADX
and +DI/-DI, RSI(14) and its slope, MACD line/signal/histogram and
histogram trend, Stochastic %K-%D, 20D/60D rate of change, volume vs
20-day average, breakout volume ratio, up/down volume ratio, OBV trend and
its alignment with price (bearish-divergence check), Accumulation/
Distribution line trend, ATR% and 12-month realised volatility, distance
from 52-week high, and ATR-scaled distance from the 20 EMA (used once as a
breakout-strength signal, once as an overextension-risk signal - same
underlying number, opposite framing, kept under separate column names so
the two buckets can't collide on a shared z-score).

**Correction risk labels are relative, not fixed thresholds.** The spec's
example table (+3% healthy, +8% strong, +15% stretched, +25% very
stretched) also said "the exact thresholds should be backtested rather
than assumed" - so rather than hard-code guessed cutoffs, the LOW/MODERATE/
HIGH label is relative to the current scan's own distribution of
overextension, consistent with how every other bucket in this project
scores. Worth knowing when comparing labels across two different runs.

**Per-stock output** is a deterministic summary block in the requested
style (Supertrend/RSI/MACD/50DMA-200DMA/Volume/price-vs-EMA/Correction
risk) - not LLM-generated, same design philosophy as New Flow. No Gemini
call for this engine either.

### API

`POST /api/run?mode=technical`, `GET /api/results?mode=technical`,
`GET /api/lookup/{symbol}?mode=technical`. `GET /api/config` includes a
`technical` block with its filters and bucket weights.

New env var: `SCREEN_UNIVERSE_TECHNICAL` (default `nifty250`).
