#!/usr/bin/env python3
"""
Technical Analysis screener - a third, fully independent engine.

Unlike Bharat Screener and New Flow 0.1, this one never touches Yahoo's
crumb-gated fundamentals endpoints at all - it only needs daily OHLCV price
history, which comes from a different, unauthenticated Yahoo endpoint that
has stayed reliable even when fundamentals were blocked. That makes this
screener meaningfully more robust to run on a cloud host than the other two.

Design choice worth stating plainly: factors here are NOT sector-neutral
z-scores like the other two engines. Sector-relative scoring exists to stop
a screen from just rewarding whichever sector has structurally different
valuation or margin norms - that reasoning doesn't apply to RSI, MACD, or
distance from a moving average, which mean the same thing regardless of
what business a company is in. So this engine uses a plain universe-wide
winsorized z-score instead (see score_technical()).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from backend.engine_core import load_universe, fetch_prices, _series, winsorized_z, CACHE_DIR
from backend.technical_indicators import (
    ema, sma, rsi, macd, atr, adx, supertrend, stochastic, obv, ad_line, roc,
    linreg_slope_pct,
)

MIN_BARS = 210  # need ~200 trading days for a 200 DMA plus warm-up room


# ----------------------------------------------------------------------------
# Factor set - 6 buckets. Weights sum to 100, matching the supplied spec.
# ----------------------------------------------------------------------------
BUCKET_WEIGHTS_TECH = {
    "trend": 0.35,
    "momentum": 0.20,
    "volume": 0.15,
    "volatility": 0.10,
    "breakout": 0.10,
    "correction": 0.10,
}

FACTORS_TECH = [
    # --- Trend (35%) - 12 factors from the spec ---------------------------
    ("price_vs_20ema",        True, "trend"),
    ("price_vs_50dma",        True, "trend"),
    ("price_vs_100dma",       True, "trend"),
    ("price_vs_200dma",       True, "trend"),
    ("ema20_vs_ema50",        True, "trend"),
    ("golden_cross_num",      True, "trend"),   # 50 DMA > 200 DMA
    ("slope_200dma_tech",     True, "trend"),
    ("slope_50dma",           True, "trend"),
    ("supertrend_daily_num",  True, "trend"),
    ("supertrend_weekly_num", True, "trend"),
    ("adx14",                 True, "trend"),
    ("plus_minus_di_num",     True, "trend"),

    # --- Momentum (20%) - 9 factors -----------------------------------------
    ("rsi14",                 True, "momentum"),
    ("rsi_above_50_num",      True, "momentum"),
    ("rsi_slope",             True, "momentum"),
    ("macd_gt_signal_num",    True, "momentum"),
    ("macd_hist",             True, "momentum"),
    ("macd_hist_rising_num",  True, "momentum"),
    ("stoch_signal",          True, "momentum"),
    ("roc_20d",                True, "momentum"),
    ("roc_60d",                True, "momentum"),

    # --- Volume (15%) - 6 factors --------------------------------------------
    ("vol_vs_20d_avg",        True, "volume"),
    ("breakout_volume_ratio", True, "volume"),
    ("up_down_vol_ratio",     True, "volume"),
    ("obv_trend",             True, "volume"),
    ("obv_price_alignment",   True, "volume"),
    ("ad_line_trend",         True, "volume"),

    # --- Volatility (10%) ----------------------------------------------------
    ("atr_pct",                False, "volatility"),  # lower realised vol = calmer, more tradeable trend
    ("vol_12m_tech",           False, "volatility"),

    # --- Breakout (10%) - stretch above 20 EMA, scaled by ATR, is a good
    # thing here: a strong, volatility-adjusted breakout -----------------
    ("dist_52w_high_tech",     True, "breakout"),
    ("stretch_20ema_atr",      True, "breakout"),

    # --- Correction / overextension risk (10%) - the SAME underlying
    # numbers as breakout_stretch, but interpreted as risk here (lower is
    # safer) - kept under distinct column names so the two buckets can't
    # collide on a shared z-score cache (see score_technical()). -----------
    ("overext_20ema_atr",      False, "correction"),
    ("overext_50dma_pct",      False, "correction"),
    ("overext_200dma_pct",     False, "correction"),
]


@dataclass
class TechnicalFilters:
    min_price: float = 20.0
    min_adtv_cr: float = 2.0
    min_history_days: int = MIN_BARS


def apply_technical_filters(df: pd.DataFrame, rules: TechnicalFilters) -> pd.DataFrame:
    def ok(col, test, label):
        if col not in df.columns:
            df[f"fail_{label}"] = False
            return
        v = pd.to_numeric(df[col], errors="coerce")
        df[f"fail_{label}"] = v.notna() & ~test(v)

    ok("price", lambda v: v >= rules.min_price, "price")
    ok("adtv_cr", lambda v: v >= rules.min_adtv_cr, "liquidity")
    ok("n_days", lambda v: v >= rules.min_history_days, "history")

    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    df["n_fails"] = df[fail_cols].sum(axis=1)
    df["eligible"] = df["n_fails"] == 0
    df["fail_reasons"] = df[fail_cols].apply(
        lambda r: ",".join(c[5:] for c in fail_cols if r[c]), axis=1)
    return df


def _one_stock_technical(sym: str, data: pd.DataFrame) -> dict:
    rec: dict = {"symbol": sym}
    close = _series(data, sym + ".NS", "Close")
    high = _series(data, sym + ".NS", "High")
    low = _series(data, sym + ".NS", "Low")
    vol = _series(data, sym + ".NS", "Volume")

    rec["n_days"] = 0 if close is None else len(close)
    if close is None or len(close) < 60 or high is None or low is None or vol is None:
        return rec  # too little history - eligibility will filter it out

    last = float(close.iloc[-1])
    rec["price"] = last
    tv = (close * vol).dropna().tail(60)
    rec["adtv_cr"] = float(tv.mean()) / 1e7 if len(tv) else None

    # --- Trend ---------------------------------------------------------
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    dma50 = sma(close, 50)
    dma100 = sma(close, 100)
    dma200 = sma(close, 200)

    def pct_vs(ma: pd.Series):
        v = ma.iloc[-1]
        return (last / float(v) - 1) if pd.notna(v) and v else None

    rec["price_vs_20ema"] = pct_vs(ema20)
    rec["price_vs_50dma"] = pct_vs(dma50)
    rec["price_vs_100dma"] = pct_vs(dma100)
    rec["price_vs_200dma"] = pct_vs(dma200)
    rec["ema20_vs_ema50"] = (
        float(ema20.iloc[-1]) / float(ema50.iloc[-1]) - 1
        if pd.notna(ema20.iloc[-1]) and pd.notna(ema50.iloc[-1]) and ema50.iloc[-1] else None
    )
    rec["golden_cross_num"] = (
        1.0 if pd.notna(dma50.iloc[-1]) and pd.notna(dma200.iloc[-1]) and dma50.iloc[-1] > dma200.iloc[-1]
        else (0.0 if pd.notna(dma50.iloc[-1]) and pd.notna(dma200.iloc[-1]) else None)
    )
    rec["slope_200dma_tech"] = linreg_slope_pct(dma200, 40)
    rec["slope_50dma"] = linreg_slope_pct(dma50, 20)

    st_daily = supertrend(high, low, close)
    rec["supertrend_daily_num"] = float(st_daily.iloc[-1])
    rec["supertrend_daily_label"] = "GREEN" if st_daily.iloc[-1] > 0 else "RED"

    # weekly Supertrend from resampled OHLC - needs enough weekly bars
    weekly = pd.DataFrame({
        "High": high.resample("W").max(), "Low": low.resample("W").min(),
        "Close": close.resample("W").last(),
    }).dropna()
    if len(weekly) >= 15:
        st_weekly = supertrend(weekly["High"], weekly["Low"], weekly["Close"], period=10, multiplier=3.0)
        rec["supertrend_weekly_num"] = float(st_weekly.iloc[-1])
        rec["supertrend_weekly_label"] = "GREEN" if st_weekly.iloc[-1] > 0 else "RED"
    else:
        rec["supertrend_weekly_num"] = None
        rec["supertrend_weekly_label"] = None

    adx_, pdi, mdi = adx(high, low, close, 14)
    rec["adx14"] = float(adx_.iloc[-1]) if pd.notna(adx_.iloc[-1]) else None
    rec["plus_minus_di_num"] = (
        1.0 if pd.notna(pdi.iloc[-1]) and pd.notna(mdi.iloc[-1]) and pdi.iloc[-1] > mdi.iloc[-1]
        else (0.0 if pd.notna(pdi.iloc[-1]) and pd.notna(mdi.iloc[-1]) else None)
    )

    # --- Momentum ---------------------------------------------------------
    rsi14 = rsi(close, 14)
    rec["rsi14"] = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else None
    rec["rsi_above_50_num"] = None if rec["rsi14"] is None else (1.0 if rec["rsi14"] > 50 else 0.0)
    if len(rsi14.dropna()) > 5:
        rec["rsi_slope"] = float(rsi14.iloc[-1] - rsi14.iloc[-6])
    else:
        rec["rsi_slope"] = None

    macd_line, macd_signal, macd_hist = macd(close)
    ml, msig, mh = macd_line.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1]
    rec["macd_gt_signal_num"] = None if pd.isna(ml) or pd.isna(msig) else (1.0 if ml > msig else 0.0)
    rec["macd_hist"] = float(mh) if pd.notna(mh) else None
    if len(macd_hist.dropna()) > 3:
        rec["macd_hist_rising_num"] = 1.0 if macd_hist.iloc[-1] > macd_hist.iloc[-4] else 0.0
    else:
        rec["macd_hist_rising_num"] = None
    rec["macd_label"] = None if rec["macd_gt_signal_num"] is None else (
        "Bullish" if rec["macd_gt_signal_num"] else "Bearish")

    k, d = stochastic(high, low, close)
    rec["stoch_signal"] = (
        float(k.iloc[-1] - d.iloc[-1]) if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]) else None
    )

    roc20 = roc(close, 20)
    roc60 = roc(close, 60)
    rec["roc_20d"] = float(roc20.iloc[-1]) if pd.notna(roc20.iloc[-1]) else None
    rec["roc_60d"] = float(roc60.iloc[-1]) if pd.notna(roc60.iloc[-1]) else None

    # --- Volume ---------------------------------------------------------
    vol20 = sma(vol, 20)
    v20 = vol20.iloc[-1]
    rec["vol_vs_20d_avg"] = float(vol.iloc[-1] / v20) if pd.notna(v20) and v20 else None
    recent_ratio = (vol.tail(3) / vol20.tail(3)).dropna()
    rec["breakout_volume_ratio"] = float(recent_ratio.max()) if len(recent_ratio) else None

    tail20 = pd.DataFrame({"c": close.tail(21).diff().dropna(), "v": vol.tail(20)})
    up_vol = tail20.loc[tail20["c"] > 0, "v"].sum()
    down_vol = tail20.loc[tail20["c"] < 0, "v"].sum()
    rec["up_down_vol_ratio"] = float(up_vol / down_vol) if down_vol > 0 else (
        None if up_vol == 0 else 5.0)  # cap: no down-volume days is an extreme, not infinite, signal

    obv_ = obv(close, vol)
    rec["obv_trend"] = linreg_slope_pct(obv_, 20)
    price_roc20 = rec["roc_20d"]
    if rec["obv_trend"] is not None and price_roc20 is not None:
        aligned = np.sign(price_roc20) == np.sign(rec["obv_trend"])
        rec["obv_price_alignment"] = 1.0 if aligned else (0.0 if price_roc20 != 0 else 0.5)
    else:
        rec["obv_price_alignment"] = None

    ad_ = ad_line(high, low, close, vol)
    rec["ad_line_trend"] = linreg_slope_pct(ad_, 20)

    # --- Volatility ---------------------------------------------------------
    atr14 = atr(high, low, close, 14)
    a14 = atr14.iloc[-1]
    rec["atr_pct"] = float(a14 / last * 100) if pd.notna(a14) and last else None
    daily_ret = close.pct_change().dropna().tail(252)
    rec["vol_12m_tech"] = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 60 else None

    # --- Breakout / correction (overextension) ---------------------------
    hi52 = float(close.tail(252).max()) if len(close) >= 30 else None
    rec["dist_52w_high_tech"] = (last / hi52 - 1) if hi52 else None

    stretch = None
    if rec["price_vs_20ema"] is not None and pd.notna(a14) and a14 and last:
        stretch = ((last - float(ema20.iloc[-1])) / float(a14))
    rec["stretch_20ema_atr"] = stretch          # breakout framing: higher = stronger breakout
    rec["overext_20ema_atr"] = stretch          # same number, correction framing: higher = more risk
    rec["overext_50dma_pct"] = rec["price_vs_50dma"]
    rec["overext_200dma_pct"] = rec["price_vs_200dma"]

    return rec


def technical_factors(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    rows = [_one_stock_technical(sym, data) for sym in symbols]
    return pd.DataFrame(rows).set_index("symbol")


def score_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Plain universe-wide winsorized z-score per factor (not sector-neutral
    - see module docstring for why), averaged per bucket, then a weighted
    composite converted to a 0-100 percentile.
    """
    zcols: dict[str, list[str]] = {b: [] for b in BUCKET_WEIGHTS_TECH}
    for col, higher, bucket in FACTORS_TECH:
        if col not in df.columns or df[col].notna().sum() < 5:
            continue
        z = winsorized_z(df[col]).clip(-3, 3)
        zname = f"z_{col}"
        df[zname] = z if higher else -z
        zcols[bucket].append(zname)

    for bucket, cols in zcols.items():
        df[f"score_{bucket}"] = df[cols].mean(axis=1, skipna=True) if cols else np.nan
        df[f"cov_{bucket}"] = df[cols].notna().sum(axis=1) / max(len(cols), 1) if cols else 0.0

    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for bucket, w in BUCKET_WEIGHTS_TECH.items():
        b = df[f"score_{bucket}"]
        num = num.add((b * w).fillna(0.0))
        den = den.add(pd.Series(np.where(b.notna(), w, 0.0), index=df.index))
    df["composite_z"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    df["data_coverage"] = den / sum(BUCKET_WEIGHTS_TECH.values())
    df["SCORE"] = (pd.Series(df["composite_z"], index=df.index).rank(pct=True) * 100).round(1)
    df["RANK"] = df["SCORE"].rank(ascending=False, method="min")
    return df


def _status(score: float) -> str:
    if score >= 80:
        return "STRONG + HEALTHY"
    if score >= 65:
        return "STRONG"
    if score >= 45:
        return "NEUTRAL"
    return "WEAK"


def _correction_risk_label(overext_score) -> str:
    if overext_score is None or pd.isna(overext_score):
        return "UNKNOWN"
    if overext_score > 0.5:
        return "LOW"
    if overext_score > -0.5:
        return "MODERATE"
    return "HIGH"


def _summary_lines(row: pd.Series) -> list[str]:
    """Deterministic text block in the exact style requested - not
    LLM-generated, every line traces to a specific computed number."""
    lines = []
    if row.get("supertrend_daily_label"):
        lines.append(f"Supertrend: {row['supertrend_daily_label']}")
    if row.get("rsi14") is not None:
        lines.append(f"RSI: {row['rsi14']:.0f}")
    if row.get("macd_label"):
        lines.append(f"MACD: {row['macd_label']}")
    if row.get("golden_cross_num") is not None:
        lines.append("50DMA > 200DMA" if row["golden_cross_num"] else "50DMA < 200DMA")
    if row.get("slope_200dma_tech") is not None:
        lines.append(f"200DMA: {'Rising' if row['slope_200dma_tech'] > 0 else 'Falling'}")
    if row.get("vol_vs_20d_avg") is not None:
        lines.append("Volume: Confirmed" if row["vol_vs_20d_avg"] > 1.0 else "Volume: Below average")
    if row.get("price_vs_20ema") is not None:
        lines.append(f"Price vs 20EMA: {row['price_vs_20ema']*100:+.1f}%")
    if row.get("price_vs_50dma") is not None:
        lines.append(f"Price vs 50DMA: {row['price_vs_50dma']*100:+.1f}%")
    lines.append(f"Correction risk: {_correction_risk_label(row.get('score_correction'))}")
    return lines


def run_technical_screen(
    universe: str = "nifty250",
    top: int = 20,
    max_per_sector: int = 4,
    cache_hours: float = 6.0,
    log=print,
) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    log(f"[Technical Analysis] Loading universe: {universe}")
    uni = load_universe(universe)
    symbols = sorted(uni["symbol"].unique().tolist())
    log(f"{len(symbols)} symbols in universe")

    log("[Technical Analysis] Downloading prices (OHLCV) ...")
    data = fetch_prices(symbols, period="2y")

    log("[Technical Analysis] Computing ~30 indicators per stock ...")
    tf = technical_factors(data, symbols)
    tf = tf.join(uni.set_index("symbol")[["name", "nse_sector"]], how="left")
    tf["sector"] = tf["nse_sector"].fillna("Unknown")

    log("[Technical Analysis] Applying filters")
    tf = apply_technical_filters(tf, TechnicalFilters())
    elig = tf[tf["eligible"]].copy()
    log(f"{len(elig)}/{len(tf)} names passed filters")

    log("[Technical Analysis] Scoring")
    elig = score_technical(elig)
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
        rec = {
            "symbol": sym,
            "name": None if pd.isna(row.get("name")) else row.get("name"),
            "sector": None if pd.isna(row.get("sector")) else row.get("sector"),
            "SCORE": float(row["SCORE"]) if pd.notna(row["SCORE"]) else None,
            "status": _status(row["SCORE"]) if pd.notna(row["SCORE"]) else None,
            "score_trend": _f(row.get("score_trend")),
            "score_momentum": _f(row.get("score_momentum")),
            "score_volume": _f(row.get("score_volume")),
            "score_volatility": _f(row.get("score_volatility")),
            "score_breakout": _f(row.get("score_breakout")),
            "score_correction": _f(row.get("score_correction")),
            "data_coverage": _f(row.get("data_coverage")),
            "summary_lines": _summary_lines(row),
            "price": _f(row.get("price")),
            "rsi14": _f(row.get("rsi14")),
            "adx14": _f(row.get("adx14")),
        }
        records.append(rec)

    score_cols = [c for c in elig.columns if c.startswith("score_") or c == "SCORE"]
    tf = tf.join(elig[score_cols], how="left")
    lookup = {}
    for sym, row in tf.iterrows():
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
        "engine": "technical",
        "generated_at": stamp,
        "universe": universe,
        "universe_size": len(symbols),
        "data_mode": "full",  # no fundamentals dependency, so no fallback mode to report
        "eligible_count": int(len(elig)),
        "weights": BUCKET_WEIGHTS_TECH,
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
