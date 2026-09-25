# Pull Back screener - technical only

A new, fully independent screener implementing the "Bharat Market — Pullback Setup
Screener | Technical Only" spec (Trend → Impulse → Controlled Pullback → Seller
Exhaustion → Trigger), built to drop into `prayashbussiness-cell/bharat_sceener`
alongside the existing Primary / Broad / Quick / New Flow screeners.

**It shares no scoring code, no job state, and no cache file with the other
screeners** — same design principle the repo's own README uses for New Flow ("it
never shares scoring code ... so changes to one can never silently affect the
other"). It reuses only market-data plumbing (`engine_core`) when that module is
importable, and falls back to its own `yfinance` fetcher otherwise, so it also works
as a standalone package.

## What's in this zip

```
backend/pullback/
  __init__.py
  indicators.py     # pure pandas/numpy TA library (EMA/SMA/RSI/MACD/ADX/ATR/
                     # Supertrend/OBV/AD-line/Stochastic/pivots) - no TA-Lib dependency
  engine.py          # the spec itself: hard filters, pullback/support/momentum/
                     # volume/ATR engines, reversal triggers, 100-pt score,
                     # anti-chasing filter, invalidation, 0-6 state machine
  data_source.py     # fetch + 12h cache; reuses engine_core if present
  backtest.py        # section 20: walk-forward, regime split, parameter
                     # sensitivity, ablation - all replaying the same evaluate_at()
  api.py             # FastAPI router: /api/pullback/run|status|results|lookup|
                     # config|backtest  (same async job/poll pattern as the host app)
  mount_snippet.py    # 2-line instructions to mount the router into backend/main.py
  ui/
    pullback_tab.html      # the "Pull Back" tab: table + symbol lookup + detail panel
    nav_button_snippet.html # button + tab-list wiring for the existing nav
tests/
  test_pullback_engine.py  # 11 tests: causality/no-lookahead, hard filters, scoring
                            # bounds, invalidation, output shape, Supertrend sanity
requirements-pullback.txt
README_PULLBACK.md (this file)
```

## Every condition from the PDF, and where it lives

| Spec section | Implemented as |
|---|---|
| §1 Five-stage sequence + decision hierarchy | `engine.evaluate_at` builds each stage's boolean, feeding the §15 state machine in order |
| §2 Timeframe architecture (Weekly regime / Daily main / 4H-1H optional / 15m-5m excluded) | Daily is the core engine; weekly EMA20/EMA50/SMA200/RSI/Supertrend computed in `prepare()` and mapped onto each daily bar without look-ahead (only weeks fully complete by that date). 4H/1H/intraday intentionally out of scope, per spec |
| §3 Hard filters HF01–HF10 | `engine.evaluate_at` → `hf` dict, all ten, each individually reported in the API output. All thresholds are `PullbackParams` fields (parameterized, not hard-coded, as the spec requires for HF06–HF09) |
| §4 Trend qualification indicators (EMA20/SMA50/SMA200, slopes, golden-cross, Supertrend, ADX, +DI/-DI) | `indicators.py` + `trend` block of the output. Golden cross is treated as background regime, not an entry signal — matches the spec's explicit warning |
| §5 Pullback measurement engine (all 6 formulas) | `pullback` block: `pullback_pct`, `bounce_from_low_pct`, `distance_to_20/50/200_pct`, `atr_pct`, `pullback_depth_atr`, plus the 6 zone bands (0-3/3-8/8-12/12-15/15-20/>20%) |
| §6 Support-zone engine (EMA20, SMA50, prior breakout, prior swing high, horizontal, volume-node, Fib 38.2/50/61.8, VWAP) | `_levels()` computes all of these; confluence is scored by counting **independent groups** (moving_average / structure / volume_profile / vwap / fibonacci), matching the spec's "don't just vote every MA separately" principle |
| §7 Momentum stabilization (RSI, RSI slope, RSI>50, MACD line/hist/cross, ROC20, Stochastic, ADX) | `momentum` block + `rsi_stab`/`rsi_regime`/`macd_stab`/`macd_cross` score factors. RSI>70/<30 explicitly NOT auto-bearish/bullish (spec's "important RSI rule") — only slope/stabilization drive the score |
| §8 Volume & selling-pressure engine | `volume` block: pullback volume contraction, later-vs-earlier down-volume, reversal volume ratio, OBV/AD state, bearish OBV divergence, breakdown-volume and distribution-at-support detectors |
| §9 ATR & volatility engine | `atr` block: ATR14, ATR%, pullback-depth-in-ATR, distance-to-EMA20/SMA50-in-ATR, ATR14/ATR50 expansion, range compression |
| §10 Price-action reversal triggers | `_patterns()` (bullish engulfing, hammer, pin bar, inside-bar breakout, high-volume bullish close) + higher-low, lower-high break, EMA20/SMA50 reclaim, trendline break — all in `price_action` |
| §11 Fibonacci & swing structure | Fib 38.2/50/61.8 in `_levels()`, 78.6% flagged as "usually too deep" per spec; higher-highs/higher-lows structure feeds the higher-low/lower-high detectors |
| §12 100-point score + classification bands | `finalize()`: 7 weighted buckets (20/10/20/15/15/10/10 = 100), bands 80-100/70-79/60-69/50-59/<50 → ENTRY_TRIGGER/PULLBACK_SETUP/PULLBACK_DEVELOPING/WATCHLIST/NO_SETUP. Invalidation always overrides the score, exactly as the spec insists |
| §13 Detailed point allocation | `MAX_PTS` — all 20 factors at the spec's exact point values, normalized 0-1 internally as the spec recommends ("normalize ... rather than a large collection of binary if/else rules") |
| §14 Invalidation & false-pullback protection | All 10 invalidation signals checked in `evaluate_at`; weekly-bearish downgrades (doesn't use daily signal alone) rather than a hard stop-loss %, per spec |
| §15 State machine (0-6 + INVALIDATED from any state) | `machine_state` field walks states 0→6 sequentially and can be overridden to INVALIDATED from any point |
| §16 Claude-ready pseudocode | `engine.py` is a direct, faithful expansion of the pseudocode's structure (indicators → trend → swing/impulse → support → seller exhaustion → price action → invalidation → score → classify) |
| §17 Entry trigger tiers (Early/Standard/Confirmed/Strong) | `entry.trigger_tier`; default minimum is Standard, configurable via `PullbackParams.min_trigger_tier` |
| §18 Anti-chasing / overextension filter | `overextension` block: EMA20/SMA50 distance, RSI, ATR-normalized extension, %-from-52W-high, range expansion, exhaustion-volume spike → HEALTHY/CAUTION/AVOID_CHASING |
| §19 Screener output shape | API `/lookup/{symbol}` and each row of `/results` match the spec's JSON shape field-for-field (symbol/state/score/trend/pullback/support/momentum/volume/price_action/entry/invalidation) |
| §20 Backtesting requirements | `backtest.py`: walk-forward (time-ordered train/test split per symbol), regime split, look-ahead control (enter next bar's open only), signal-overlap control (no new trade until prior exits), fees+slippage, parameter sensitivity, ablation test |
| §21 P0/P1/P2 checklist | All P0 + P1 items implemented; P2 items (Anchored VWAP, volume-profile/HVN) are also implemented (bonus, not just stubbed) — only true intraday (4H/1H) execution refinement is left as future work, as the spec allows |

Every numeric threshold from the PDF is a `PullbackParams` field (not hard-coded) —
per the spec's own instruction not to freeze HF06-HF09 and the score bands
permanently. Defaults match the PDF's suggested values.

## Wiring into the existing UI

1. Copy `backend/pullback/` into the repo's `backend/` folder.
2. In `backend/main.py`, add:
   ```python
   from backend.pullback.api import router as pullback_router
   app.include_router(pullback_router)
   ```
3. In `static/index.html` (or wherever the Primary/Broad/New Flow tab buttons and
   `<div>` panels live), paste `backend/pullback/ui/pullback_tab.html`'s `<section>`
   into the tab-panel area, and add the button from
   `backend/pullback/ui/nav_button_snippet.html` next to the existing tab buttons.
4. Add `"pullback-tab"` to whatever array/list your existing tab-switch JS uses to
   show/hide panels (the fragment's own JS only handles fetching + rendering, not
   which panel is visible).
5. `pip install -r requirements-pullback.txt` (only adds anything beyond the host
   repo's existing `requirements.txt` if `yfinance` isn't already there).

New env vars (all optional): `SCREEN_UNIVERSE_PULLBACK` (default `quicklist`),
`SCREEN_TOP_N_PULLBACK` (default `40`), `PULLBACK_DATA_DIR` (default `data`).

## API

- `POST /api/pullback/run?universe=...&top_n=...&quick=false` — starts a background scan
- `GET /api/pullback/status` — `{status, progress:{done,total}}`
- `GET /api/pullback/results` — cached last-run results (§19 shape, sorted state→score)
- `GET /api/pullback/lookup/{symbol}` — single-stock diagnostic, whether or not it made the top N
- `GET /api/pullback/config` — live thresholds, bucket weights, point model (introspected from `PullbackParams`, so the UI can never drift out of sync with the code)
- `POST /api/pullback/backtest?hold_days=20` — runs `backtest.run_backtest` over a symbol list

## Tests

```
pip install pytest   # optional - the file also runs standalone
python tests/test_pullback_engine.py
# or
pytest tests/test_pullback_engine.py -v
```

11 tests: no-look-ahead causality, HF10 on short history, trend hard filters on a
genuine uptrend, seller-exhaustion pullbacks scoring above panic-selling pullbacks,
crashes never being labelled a normal pullback, a close below the 200DMA forcing
INVALIDATED/NO_SETUP, RSI>70 not being auto-penalized, score always in [0,100],
§19 output shape, and a Supertrend sanity check against a hand-built monotonic series.

## Known simplifications (stated plainly, as the spec asks)

- Intraday (4H/1H/15m/5m) refinement is out of scope, matching the spec's own
  "avoid for core screener" / "optional" guidance — the engine is daily-only with a
  weekly regime filter, as recommended.
- `data_source.py`'s standalone fallback uses `yfinance`; if the host repo's
  `engine_core` already has a working NSE-blocked-IP fallback (per the main
  README's "NSE blocks cloud IPs" section), that's reused automatically and this
  fallback never triggers.
- Backtest regime labels (bull/sideways/correction/bear) are supplied by the
  caller (`regime_split(price_data, regime_labels, ...)`) — the engine doesn't
  invent its own regime classifier, since that's a separate research question
  from the pullback logic itself.

This is a technical research/development tool, not investment advice. Validate
thresholds on your target universe with out-of-sample testing before relying on it,
exactly as the source spec says.
