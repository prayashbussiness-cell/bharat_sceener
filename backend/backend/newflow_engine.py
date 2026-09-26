#!/usr/bin/env python3
"""
New Flow 0.1 - a separate, deeper Indian-equity scoring engine.

This is intentionally NOT a variant of the original Bharat Screener - it has
its own factor set, its own hard filters, and a red-flag penalty system on
top of the score. It reuses only the market-data PLUMBING from engine_core
(NSE universe download + fallback, Yahoo price/fundamentals fetch, the
sector-neutral z-scoring math) - never its scoring config, so changes here
can never affect the original screener's output.

Honesty note on data coverage: the spec this was built from names 10 hard
filters and several red flags that assume data this free pipeline doesn't
have (auditor opinions, NSE circuit-filter history). Those are implemented
as documented no-ops below rather than faked with a shaky proxy - see
HardFilters and RED_FLAG_PENALTIES for exactly which ones are real.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from backend.engine_core import (
    load_universe, fetch_prices, price_factors, fetch_fundamentals,
    fundamentals_available, sector_neutral_z, score as _score_generic,
    CACHE_DIR,
)

# ----------------------------------------------------------------------------
# Factor set - organised into the 7 buckets from the spec. Each tuple is
# (column, higher_is_better, bucket). Columns are computed in
# engine_core._fetch_one_fundamental() (New Flow section) or reused directly
# from the original price/fundamentals fetch (momentum, ROCE/ROE, etc.) -
# nothing here triggers a second network fetch.
# ----------------------------------------------------------------------------
FACTORS_V2 = [
    # --- Growth (20%) --------------------------------------------------
    ("revenue_growth_consistency", True,  "growth"),
    ("eps_growth_consistency",     True,  "growth"),
    ("profit_cagr_5y",             True,  "growth"),
    ("rev_growth_yoy",             True,  "growth"),
    ("eps_growth_yoy",             True,  "growth"),
    ("eps_acceleration",           True,  "growth"),

    # --- Quality (20%) --------------------------------------------------
    ("roic",                       True,  "quality"),
    ("incremental_roce",           True,  "quality"),
    ("asset_turnover",             True,  "quality"),
    ("roce",                       True,  "quality"),
    ("roe",                        True,  "quality"),
    # cash_conversion_cycle deliberately excluded - always None, see
    # engine_core._fetch_one_fundamental for why (no reliable free source)

    # --- Earnings quality / cash flow (15%) ------------------------------
    ("cfo_to_pat",                 True,  "earnings_quality"),
    ("fcf_growth_3y",              True,  "earnings_quality"),
    ("accrual_ratio",              False, "earnings_quality"),

    # --- Momentum (20%) - reusing the same price-trend factors as the
    # original screener; this bucket's inputs are identical, only its
    # weight within the overall composite differs.
    ("rs_6m",                      True,  "momentum"),
    ("rs_12m",                     True,  "momentum"),
    ("ret_3m",                     True,  "momentum"),
    ("px_vs_200dma",               True,  "momentum"),
    ("slope_200dma",               True,  "momentum"),
    ("dist_52w_high",              True,  "momentum"),
    ("vol_12m",                    False, "momentum"),
    ("golden_cross_num",           True,  "momentum"),

    # --- Value (10%) ------------------------------------------------------
    ("price_to_book",              False, "value"),
    ("price_to_sales",             False, "value"),
    ("dividend_yield",             True,  "value"),
    ("peg",                        False, "value"),
    ("pe_vs_own_5y",               False, "value"),

    # --- Risk (10%) ---------------------------------------------------------
    ("net_debt_to_fcf",            False, "risk"),
    ("fcf_to_debt",                True,  "risk"),
    ("debt_to_equity",             False, "risk"),
    ("max_drawdown_1y",            True,  "risk"),

    # --- Ownership (5%) - limited by data availability: promoter pledge
    # needs the manual pledge_overrides.csv (see original README) or it's
    # inert (all-NaN, silently skipped by the scorer, never faked). Share
    # dilution is the one ownership signal available from Yahoo directly.
    ("promoter_pledge_pct",        False, "ownership"),
    ("share_dilution_1y",          False, "ownership"),
]

BUCKET_WEIGHTS_V2 = {
    "growth": 0.20,
    "quality": 0.20,
    "earnings_quality": 0.15,
    "momentum": 0.20,
    "value": 0.10,
    "risk": 0.10,
    "ownership": 0.05,
}


# ----------------------------------------------------------------------------
# Stage 1 - hard filters. 10 named, matching the spec's HARD_FILTERS list.
# Each is either genuinely enforced from this data pipeline, or documented
# as a no-op where no reliable free data source exists - never faked.
# ----------------------------------------------------------------------------
@dataclass
class HardFilters:
    minimum_market_cap_cr: float = 1500.0
    minimum_daily_turnover_cr: float = 2.0
    minimum_listing_history_days: int = 300
    minimum_trading_days: int = 300          # same underlying signal (n_days) as listing history in this data source
    max_promoter_pledge_pct: float = 50.0    # only enforced when pledge_overrides.csv is supplied - see note below
    extreme_debt_to_equity: float = 500.0    # stricter absolute cutoff than the original screener's 300% - meant to catch only the tail
    extreme_dilution_1y: float = 0.20        # >20% share count growth in a year
    require_positive_equity: bool = True     # stands in for the spec's "negative_equity" hard cutoff
    skip_leverage_checks_for_sectors: list = field(default_factory=lambda: ["Financial Services"])
    # NOT enforced - no reliable free data source available to this pipeline,
    # left as documented no-ops rather than a fake heuristic:
    #   accounting_red_flag        (needs auditor opinion / forensic-audit data)
    #   extreme_circuit_frequency  (needs NSE circuit-filter/surveillance history)


def apply_hard_filters(df: pd.DataFrame, rules: HardFilters) -> pd.DataFrame:
    def ok(col, test, label):
        if col not in df.columns:
            df[f"fail_{label}"] = False
            return
        v = pd.to_numeric(df[col], errors="coerce")
        df[f"fail_{label}"] = v.notna() & ~test(v)

    ok("market_cap_cr", lambda v: v >= rules.minimum_market_cap_cr, "market_cap")
    ok("adtv_cr", lambda v: v >= rules.minimum_daily_turnover_cr, "daily_turnover")
    ok("n_days", lambda v: v >= rules.minimum_listing_history_days, "listing_history")
    ok("promoter_pledge_pct", lambda v: v <= rules.max_promoter_pledge_pct, "promoter_pledge")
    ok("debt_to_equity", lambda v: v <= rules.extreme_debt_to_equity, "extreme_debt")
    ok("share_dilution_1y", lambda v: v <= rules.extreme_dilution_1y, "extreme_dilution")
    if rules.require_positive_equity:
        ok("total_equity", lambda v: v > 0, "negative_equity")

    if rules.skip_leverage_checks_for_sectors and "sector" in df.columns:
        exempt = df["sector"].isin(rules.skip_leverage_checks_for_sectors)
        if "fail_extreme_debt" in df.columns:
            df.loc[exempt, "fail_extreme_debt"] = False

    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    df["n_fails"] = df[fail_cols].sum(axis=1)
    df["eligible"] = df["n_fails"] == 0
    df["fail_reasons"] = df[fail_cols].apply(
        lambda r: ",".join(c[5:] for c in fail_cols if r[c]), axis=1)
    return df


# ----------------------------------------------------------------------------
# Red-flag penalties - subtracted from the final 0-100 score after ranking.
# Point values exactly as specified. Triggers only fire on data this
# pipeline actually has; the two that need data we don't have never fire
# (documented, not faked).
# ----------------------------------------------------------------------------
RED_FLAG_PENALTIES = {
    "high_promoter_pledge":         -10,
    "major_dilution":               -8,
    "fcf_negative_multiple_years":  -8,
    "debt_explosion":               -8,
    "earnings_cashflow_divergence": -7,
    "major_auditor_issue":          -15,   # never fires - no auditor-opinion data source
    "extreme_valuation":            -5,
}


def compute_red_flags(row: pd.Series) -> dict:
    flags = {}

    pledge = row.get("promoter_pledge_pct")
    flags["high_promoter_pledge"] = bool(pd.notna(pledge) and pledge > 50)

    dilution = row.get("share_dilution_1y")
    flags["major_dilution"] = bool(pd.notna(dilution) and dilution > 0.15)

    neg_years = row.get("fcf_negative_years_recent")
    flags["fcf_negative_multiple_years"] = bool(pd.notna(neg_years) and neg_years >= 2)

    de_now = row.get("debt_to_equity")
    de_prior = row.get("debt_to_equity_prior")
    flags["debt_explosion"] = bool(
        pd.notna(de_now) and pd.notna(de_prior) and de_prior > 0
        and de_now > de_prior * 1.5 and de_now > 150
        and row.get("sector") != "Financial Services"  # structurally high/volatile D/E for lenders
    )

    accrual = row.get("accrual_ratio")
    cfo_pat = row.get("cfo_to_pat")
    flags["earnings_cashflow_divergence"] = bool(
        (pd.notna(accrual) and accrual > 0.10)
        or (pd.notna(cfo_pat) and cfo_pat < 0.5 and pd.notna(row.get("net_income")) and row.get("net_income", 0) > 0)
    )

    flags["major_auditor_issue"] = False  # no data source - always inert, see module docstring

    peg = row.get("peg")
    pe_hist = row.get("pe_vs_own_5y")
    flags["extreme_valuation"] = bool(
        (pd.notna(peg) and peg > 3) or (pd.notna(pe_hist) and pe_hist > 2.0)
    )

    return flags


def _status_label(score: float, quality_score, momentum_score) -> str:
    q = quality_score if quality_score is not None and not pd.isna(quality_score) else 0
    m = momentum_score if momentum_score is not None and not pd.isna(momentum_score) else 0
    if score >= 80 and q > 0.5 and m > 0.5:
        return "HIGH-QUALITY MOMENTUM"
    if score >= 75:
        return "OUTPERFORMER"
    if score >= 55:
        return "WATCHLIST"
    return "BELOW THRESHOLD"


def _build_strengths_risks(row: pd.Series, flags: dict) -> tuple[list, list]:
    """Deterministic, auditable - not LLM-generated. Picks the strongest and
    weakest available z-scored factors plus any triggered red flags, so the
    output can always be traced back to a specific number.
    """
    z_items = []
    for col, higher, bucket in FACTORS_V2:
        zcol = f"z_{col}"
        if zcol in row.index and pd.notna(row[zcol]):
            z_items.append((col, bucket, float(row[zcol])))
    z_items.sort(key=lambda x: x[2], reverse=True)

    strengths = [f"{col.replace('_', ' ')} strong for its sector ({bucket})"
                for col, bucket, z in z_items[:3] if z > 0.5]
    risks = [f"{col.replace('_', ' ')} weak for its sector ({bucket})"
            for col, bucket, z in z_items[-2:] if z < -0.5]

    for name, triggered in flags.items():
        if triggered and name != "major_auditor_issue":
            risks.append(name.replace("_", " "))

    return strengths, risks


def score_newflow(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    df = _score_generic(df, sector_col=sector_col, weights=BUCKET_WEIGHTS_V2, factors=FACTORS_V2)

    penalties = []
    all_flags = []
    for sym, row in df.iterrows():
        flags = compute_red_flags(row)
        pts = sum(RED_FLAG_PENALTIES[name] for name, hit in flags.items() if hit)
        penalties.append(pts)
        all_flags.append(flags)
    df["red_flag_penalty"] = penalties
    df["red_flags"] = all_flags

    # SCORE is a percentile of the composite z (0-100), then red-flag points
    # are subtracted directly and clipped back into range.
    df["SCORE_pre_penalty"] = (df["composite_z"].rank(pct=True) * 100).round(1)
    df["SCORE"] = (df["SCORE_pre_penalty"] + df["red_flag_penalty"]).clip(0, 100).round(1)
    df["RANK"] = df["SCORE"].rank(ascending=False, method="min")
    df["data_coverage"] = df["composite_z"].notna().astype(float)  # coarse; per-bucket cov also available
    return df


def run_newflow_screen(
    universe: str = "nifty250",
    top: int = 20,
    max_per_sector: int = 4,
    pledge_csv: str | None = None,
    cache_hours: float = 12.0,
    log=print,
) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    log(f"[New Flow 0.1] Loading universe: {universe}")
    uni = load_universe(universe)
    symbols = sorted(uni["symbol"].unique().tolist())
    log(f"{len(symbols)} symbols in universe")

    log("[New Flow 0.1] Downloading prices ...")
    px = fetch_prices(symbols)
    pf = price_factors(px, symbols)

    data_mode = "full"
    cache = os.path.join(CACHE_DIR, f"fund_{universe}.parquet")
    fresh = (os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < cache_hours * 3600)
    if fresh:
        log("[New Flow 0.1] Loading cached fundamentals (shared cache with Bharat Screener)")
        fund = pd.read_parquet(cache)
    else:
        log("[New Flow 0.1] Checking whether Yahoo fundamentals are reachable ...")
        if not fundamentals_available():
            log("[New Flow 0.1] Fundamentals blocked from this server - falling back to "
                "momentum-only scoring for this run.")
            fund = pd.DataFrame(index=pf.index)
            data_mode = "momentum_only_fallback"
        else:
            log(f"[New Flow 0.1] Downloading fundamentals for {len(symbols)} names ...")
            fund = fetch_fundamentals(symbols, log=log)
            try:
                fund.to_parquet(cache)
            except Exception:
                fund.to_csv(cache.replace(".parquet", ".csv"))

    df = pf.join(fund, how="left")
    df = df.join(uni.set_index("symbol")[["name", "nse_sector"]], how="left")
    df["sector"] = df.get("yf_sector", pd.Series(index=df.index, dtype=object))
    df["sector"] = df["sector"].fillna(df["nse_sector"]).fillna("Unknown")

    if pledge_csv and os.path.exists(pledge_csv):
        pl = pd.read_csv(pledge_csv).set_index("symbol")
        df["promoter_pledge_pct"] = pl["promoter_pledge_pct"].reindex(df.index)

    log("[New Flow 0.1] Applying hard filters")
    df = apply_hard_filters(df, HardFilters())
    elig = df[df["eligible"]].copy()
    log(f"{len(elig)}/{len(df)} names passed hard filters")

    log("[New Flow 0.1] Scoring + red-flag penalties")
    elig = score_newflow(elig, "sector")
    elig = elig.sort_values("SCORE", ascending=False)

    keep, counts = [], {}
    for sym, row in elig.iterrows():
        sec = row["sector"]
        if counts.get(sec, 0) >= max_per_sector:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep.append(sym)
    picks_df = elig.loc[keep].head(top)

    records = []
    for sym, row in picks_df.iterrows():
        flags = row.get("red_flags", {}) or {}
        strengths, risks = _build_strengths_risks(row, flags)
        rec = {
            "symbol": sym,
            "name": None if pd.isna(row.get("name")) else row.get("name"),
            "sector": None if pd.isna(row.get("sector")) else row.get("sector"),
            "SCORE": float(row["SCORE"]) if pd.notna(row["SCORE"]) else None,
            "score_growth": _f(row.get("score_growth")),
            "score_quality": _f(row.get("score_quality")),
            "score_earnings_quality": _f(row.get("score_earnings_quality")),
            "score_momentum": _f(row.get("score_momentum")),
            "score_value": _f(row.get("score_value")),
            "score_risk": _f(row.get("score_risk")),
            "score_ownership": _f(row.get("score_ownership")),
            "red_flag_penalty": _f(row.get("red_flag_penalty")),
            "triggered_flags": [k for k, v in flags.items() if v],
            "key_strengths": strengths,
            "key_risks": risks,
            "status": _status_label(row["SCORE"], row.get("score_quality"), row.get("score_momentum")),
            "roce": _f(row.get("roce")),
            "roic": _f(row.get("roic")),
            "fcf_to_pat": _f(row.get("fcf_to_pat")),
            "pe": _f(row.get("pe")),
            "pe_vs_own_5y": _f(row.get("pe_vs_own_5y")),
            "rs_6m": _f(row.get("rs_6m")),
            "rs_12m": _f(row.get("rs_12m")),
            "price": _f(row.get("price")),
            "market_cap_cr": _f(row.get("market_cap_cr")),
        }
        records.append(rec)

    score_cols = [c for c in elig.columns if c.startswith("score_") or c == "SCORE"]
    df = df.join(elig[score_cols], how="left")
    lookup = {}
    for sym, row in df.iterrows():
        entry = {
            "name": None if pd.isna(row.get("name")) else row.get("name"),
            "sector": None if pd.isna(row.get("sector")) else row.get("sector"),
            "eligible": bool(row.get("eligible", False)),
            "fail_reasons": row.get("fail_reasons", "") or None,
            "made_top_n": sym in set(picks_df.index),
        }
        for c in score_cols:
            entry[c] = _f(row.get(c))
        lookup[sym] = entry

    return {
        "engine": "newflow-0.1",
        "generated_at": stamp,
        "universe": universe,
        "universe_size": len(symbols),
        "data_mode": data_mode,
        "eligible_count": int(len(elig)),
        "weights": BUCKET_WEIGHTS_V2,
        "hard_filters_enforced": [
            "minimum_market_cap", "minimum_daily_turnover", "minimum_listing_history",
            "minimum_trading_days", "promoter_pledge_limit (only when pledge data supplied)",
            "extreme_debt", "extreme_dilution", "negative_equity",
        ],
        "hard_filters_not_enforced": ["accounting_red_flag", "extreme_circuit_frequency"],
        "red_flag_penalties": RED_FLAG_PENALTIES,
        "picks": records,
        "lookup": lookup,
    }


def _f(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return v
