#!/usr/bin/env python3
"""
Bharat Outperformer-style multi-factor screener for Indian equities.

Two-stage design:
  Stage 1  Eligibility  -> hard rules remove illiquid / distressed / too-new names
  Stage 2  Scoring      -> sector-neutral z-scores across 5 factor buckets

Run:
    python bmo_screener.py                 # full run, Nifty LargeMidcap 250
    python bmo_screener.py --fast          # price/momentum only (~1 min)
    python bmo_screener.py --universe nifty500
    python bmo_screener.py --top 25 --out picks.csv

Data source: Yahoo Finance via yfinance (free, patchy on Indian fundamentals).
Swap in a paid feed by replacing fetch_fundamentals() only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
BENCHMARK = "^CRSLDX"           # Nifty 500 TRI proxy; falls back to ^NSEI
BENCHMARK_FALLBACK = "^NSEI"    # Nifty 50

# NSE publishes constituent CSVs at these stable URLs.
UNIVERSE_URLS = {
    "nifty250": "https://nsearchives.nseindia.com/content/indices/ind_niftylargemidcap250list.csv",
    "nifty500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "nifty200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "midsmall400": "https://nsearchives.nseindia.com/content/indices/ind_niftymidsmallcap400list.csv",
}

# "quicklist" needs no NSE download at all - useful when NSE is blocking the
# server's IP (see load_universe), or just for a fast sub-minute test/demo run.
# It's the same ~30 large, liquid names across sectors used as a sanity-check
# watchlist, not a substitute for scanning the full index.
QUICK_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN",
    "ITC", "HINDUNILVR", "LT", "KOTAKBANK", "BAJFINANCE", "AXISBANK", "MARUTI",
    "SUNPHARMA", "HCLTECH", "TITAN", "ULTRACEMCO", "ASIANPAINT", "TATAMOTORS",
    "JSWSTEEL", "NTPC", "POWERGRID", "M&M", "WIPRO", "NESTLEIND", "TATASTEEL",
    "ADANIENT", "DIVISLAB", "CIPLA",
]


# ----------------------------------------------------------------------------
# Stage 1 - eligibility rules (hard filters, applied before any scoring)
# ----------------------------------------------------------------------------
@dataclass
class Eligibility:
    min_market_cap_cr: float = 1500.0      # INR crore
    min_adtv_cr: float = 2.0               # 60-day avg daily traded value, INR crore
    min_price: float = 20.0                # avoid sub-20 penny names
    min_history_days: int = 300            # ~15 months listed
    max_debt_to_equity: float = 300.0      # % ; excludes balance-sheet blowups
    min_interest_coverage: float = 1.5     # EBIT / interest expense
    require_positive_equity: bool = True
    require_positive_ebitda: bool = True
    exclude_sectors: list[str] = field(default_factory=list)
    # Banks/NBFCs carry customer deposits as balance-sheet liabilities, so
    # Yahoo's debt/equity for them routinely reads 400-900%+ - that's how
    # banking works, not distress. Applying a manufacturing-style leverage
    # cap to them was silently disqualifying every healthy bank before
    # scoring even started. Same logic for interest coverage: a bank's
    # "interest expense" is its cost of deposits, not debt service risk.
    skip_leverage_checks_for_sectors: list[str] = field(
        default_factory=lambda: ["Financial Services"])


# ----------------------------------------------------------------------------
# Stage 2 - factor definitions
#   key     : column name produced by build_factor_table()
#   higher  : True if a larger raw value is better
#   bucket  : which score bucket it feeds
# ----------------------------------------------------------------------------
FACTORS = [
    # --- Growth (20%) -------------------------------------------------------
    ("rev_growth_yoy",      True,  "growth"),
    ("rev_cagr_3y",         True,  "growth"),
    ("eps_growth_yoy",      True,  "growth"),
    ("eps_cagr_3y",         True,  "growth"),
    ("ebitda_growth_yoy",   True,  "growth"),
    ("eps_acceleration",    True,  "growth"),     # is growth speeding up or fading

    # --- Quality (25%) ------------------------------------------------------
    ("roce",                True,  "quality"),
    ("roe",                 True,  "quality"),
    ("ebitda_margin",       True,  "quality"),
    ("margin_delta_3y",     True,  "quality"),     # margin expansion
    ("net_debt_to_ebitda",  False, "quality"),
    ("interest_coverage",   True,  "quality"),
    ("fcf_to_pat",          True,  "quality"),     # cash conversion
    ("fcf_margin",          True,  "quality"),

    # --- Value (20%) --------------------------------------------------------
    ("peg",                 False, "value"),
    ("ev_to_ebitda",        False, "value"),
    ("pe_vs_own_5y",        False, "value"),       # current P/E / 5Y median P/E
    ("fcf_yield",           True,  "value"),
    ("earnings_yield",      True,  "value"),

    # --- Momentum (25%) -----------------------------------------------------
    ("rs_6m",               True,  "momentum"),    # 6m return minus benchmark
    ("rs_12m",               True, "momentum"),
    ("ret_3m",               True, "momentum"),
    ("px_vs_200dma",        True,  "momentum"),
    ("slope_200dma",        True,  "momentum"),
    ("dist_52w_high",       True,  "momentum"),    # negative number, closer to 0 = better
    ("vol_12m",             False, "momentum"),    # lower realised vol preferred
    ("golden_cross_num",    True,  "momentum"),    # "50 DMA > 200 DMA" trend signal

    # --- Risk / ownership (10%) --------------------------------------------
    ("promoter_pledge_pct", False, "risk"),        # needs pledge_overrides.csv
    ("share_dilution_1y",   False, "risk"),
    ("debt_to_equity",      False, "risk"),
    ("max_drawdown_1y",     True,  "risk"),        # less negative = better
]

BUCKET_WEIGHTS = {
    "growth": 0.20,
    "quality": 0.25,
    "value": 0.20,
    "momentum": 0.25,
    "risk": 0.10,
}


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
def load_universe(name: str, custom_csv: str | None = None) -> pd.DataFrame:
    """Return DataFrame with columns: symbol, name, nse_sector."""
    if custom_csv:
        df = pd.read_csv(custom_csv)
        col = next(c for c in df.columns if c.strip().lower() in ("symbol", "ticker"))
        out = pd.DataFrame({"symbol": df[col].astype(str).str.strip()})
        out["name"] = df.get("Company Name", out["symbol"])
        out["nse_sector"] = df.get("Industry", "Unknown")
        return out

    if name == "quicklist":
        return pd.DataFrame({
            "symbol": QUICK_WATCHLIST,
            "name": QUICK_WATCHLIST,
            "nse_sector": "Unknown",
        })

    url = UNIVERSE_URLS[name]
    import requests
    hdrs = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=hdrs, timeout=20)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            return pd.DataFrame({
                "symbol": df["Symbol"].astype(str).str.strip(),
                "name": df["Company Name"].astype(str).str.strip(),
                "nse_sector": df.get("Industry", pd.Series(["Unknown"] * len(df))),
            })
        except Exception as e:
            last_err = e
            time.sleep(2)

    # Live NSE fetch failed on all attempts (common from cloud-hosted IPs, which
    # NSE frequently blocks). Fall back to the bundled snapshot rather than
    # failing the whole scan - it's a smaller, manually-curated list and will
    # drift out of date, but keeps the app usable. Update
    # app/nifty250_fallback.csv periodically to refresh it.
    fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "nifty250_fallback.csv")
    if os.path.exists(fallback_path):
        print(f"WARNING: live NSE fetch failed ({last_err}); using bundled fallback list "
              f"({fallback_path}). This list is a manually curated snapshot, not the live "
              f"index membership - update it periodically.", file=sys.stderr)
        df = pd.read_csv(fallback_path)
        return pd.DataFrame({
            "symbol": df["Symbol"].astype(str).str.strip(),
            "name": df["Company Name"].astype(str).str.strip(),
            "nse_sector": df.get("Industry", pd.Series(["Unknown"] * len(df))),
        })

    raise SystemExit(
        f"Could not download {name} constituents ({last_err}), and no fallback "
        f"list was found at {fallback_path}."
    )


# ----------------------------------------------------------------------------
# Price data + price-derived factors
# ----------------------------------------------------------------------------
def fetch_prices(symbols: list[str], period: str = "3y") -> pd.DataFrame:
    import yfinance as yf
    tickers = [s + ".NS" for s in symbols] + [BENCHMARK, BENCHMARK_FALLBACK]
    print(f"  downloading {len(tickers)} price series ...")
    data = yf.download(
        tickers, period=period, interval="1d",
        auto_adjust=True, group_by="ticker", threads=True, progress=False,
    )
    return data


def _series(data: pd.DataFrame, ticker: str, field: str) -> pd.Series | None:
    try:
        s = data[ticker][field].dropna()
        return s if len(s) else None
    except Exception:
        return None


def price_factors(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    bench = _series(data, BENCHMARK, "Close")
    if bench is None or len(bench) < 250:
        bench = _series(data, BENCHMARK_FALLBACK, "Close")
    if bench is None:
        raise SystemExit("No benchmark price data - check your internet connection.")

    def bret(days: int) -> float:
        if len(bench) <= days:
            return 0.0
        return float(bench.iloc[-1] / bench.iloc[-days - 1] - 1)

    b6, b12 = bret(126), bret(252)
    rows = []

    for sym in symbols:
        px = _series(data, sym + ".NS", "Close")
        vol = _series(data, sym + ".NS", "Volume")
        if px is None or len(px) < 60:
            rows.append({"symbol": sym, "n_days": 0 if px is None else len(px)})
            continue

        def ret(days: int) -> float | None:
            if len(px) <= days:
                return None
            return float(px.iloc[-1] / px.iloc[-days - 1] - 1)

        r3, r6, r12 = ret(63), ret(126), ret(252)
        dma200 = px.rolling(200).mean()
        dma50 = px.rolling(50).mean()
        last = float(px.iloc[-1])

        slope200 = None
        if dma200.notna().sum() > 45:
            d_now, d_prev = dma200.iloc[-1], dma200.iloc[-45]
            if pd.notna(d_now) and pd.notna(d_prev) and d_prev > 0:
                slope200 = float(d_now / d_prev - 1)

        hi52 = float(px.tail(252).max())
        daily = px.pct_change().dropna().tail(252)
        adtv = None
        if vol is not None:
            tv = (px * vol).dropna().tail(60)
            if len(tv):
                adtv = float(tv.mean()) / 1e7   # INR crore

        roll_max = px.tail(252).cummax()
        mdd = float((px.tail(252) / roll_max - 1).min()) if len(px) >= 30 else None

        golden_cross = (
            bool(dma50.iloc[-1] > dma200.iloc[-1])
            if pd.notna(dma50.iloc[-1]) and pd.notna(dma200.iloc[-1]) else None
        )

        rows.append({
            "symbol": sym,
            "n_days": len(px),
            "price": last,
            "adtv_cr": adtv,
            "ret_3m": r3,
            "rs_6m": None if r6 is None else r6 - b6,
            "rs_12m": None if r12 is None else r12 - b12,
            "px_vs_200dma": (last / float(dma200.iloc[-1]) - 1) if pd.notna(dma200.iloc[-1]) else None,
            "px_vs_50dma": (last / float(dma50.iloc[-1]) - 1) if pd.notna(dma50.iloc[-1]) else None,
            "golden_cross": golden_cross,
            # numeric form (1.0/0.0) so the scorer can use it as a factor;
            # "50 DMA > 200 DMA" from the factor sheet - a trend-following signal
            "golden_cross_num": None if golden_cross is None else (1.0 if golden_cross else 0.0),
            "slope_200dma": slope200,
            "dist_52w_high": last / hi52 - 1 if hi52 else None,
            "vol_12m": float(daily.std() * np.sqrt(252)) if len(daily) > 60 else None,
            "max_drawdown_1y": mdd,
        })
    return pd.DataFrame(rows).set_index("symbol")


# ----------------------------------------------------------------------------
# Fundamentals
# ----------------------------------------------------------------------------
BS_KEYS = {
    "total_assets": ["Total Assets"],
    "current_liab": ["Current Liabilities", "Total Current Liabilities"],
    "total_equity": ["Stockholders Equity", "Total Stockholder Equity",
                     "Common Stock Equity"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
}
IS_KEYS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest": ["Interest Expense", "Interest Expense Non Operating"],
    "diluted_eps": ["Diluted EPS", "Basic EPS"],
}
CF_KEYS = {
    "cfo": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure"],
}


def _pick(df: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s.dropna()
    return None


def _at(s: pd.Series | None, i: int = 0) -> float | None:
    if s is None or len(s) <= i:
        return None
    v = s.iloc[i]
    return None if pd.isna(v) else float(v)


def _cagr(s: pd.Series | None, years: int) -> float | None:
    """s is newest-first (yfinance convention)."""
    if s is None or len(s) <= years:
        return None
    new, old = _at(s, 0), _at(s, years)
    if new is None or old is None or old <= 0 or new <= 0:
        return None
    return (new / old) ** (1 / years) - 1


def _growth(s: pd.Series | None) -> float | None:
    new, old = _at(s, 0), _at(s, 1)
    if new is None or old is None or old == 0:
        return None
    return new / abs(old) - 1


def _fetch_one_fundamental(sym: str) -> dict:
    import yfinance as yf
    rec: dict = {"symbol": sym}
    try:
        t = yf.Ticker(sym + ".NS")
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            pass

        bs = getattr(t, "balance_sheet", None)
        isx = getattr(t, "income_stmt", None)
        cf = getattr(t, "cashflow", None)

        rec["yf_sector"] = info.get("sector") or "Unknown"
        rec["yf_industry"] = info.get("industry") or "Unknown"
        mcap = info.get("marketCap")
        rec["market_cap_cr"] = mcap / 1e7 if mcap else None

        rev = _pick(isx, IS_KEYS["revenue"])
        ebit = _pick(isx, IS_KEYS["ebit"])
        ebitda = _pick(isx, IS_KEYS["ebitda"])
        ni = _pick(isx, IS_KEYS["net_income"])
        eps = _pick(isx, IS_KEYS["diluted_eps"])
        interest = _pick(isx, IS_KEYS["interest"])
        assets = _pick(bs, BS_KEYS["total_assets"])
        cl = _pick(bs, BS_KEYS["current_liab"])
        eq = _pick(bs, BS_KEYS["total_equity"])
        debt = _pick(bs, BS_KEYS["total_debt"])
        cash = _pick(bs, BS_KEYS["cash"])
        shares = _pick(bs, BS_KEYS["shares"])
        cfo = _pick(cf, CF_KEYS["cfo"])
        capex = _pick(cf, CF_KEYS["capex"])

        rev0, ebit0, ebitda0, ni0 = _at(rev), _at(ebit), _at(ebitda), _at(ni)
        eq0, debt0, cash0 = _at(eq), _at(debt) or 0.0, _at(cash) or 0.0
        cfo0, capex0 = _at(cfo), _at(capex) or 0.0

        if ebitda0 is None and ebit0 is not None:
            ebitda0 = ebit0  # crude fallback; understates margin

        rec["revenue"] = rev0
        rec["ebitda"] = ebitda0
        rec["net_income"] = ni0
        rec["total_equity"] = eq0

        # growth
        rec["rev_growth_yoy"] = _growth(rev) or info.get("revenueGrowth")
        rec["rev_cagr_3y"] = _cagr(rev, 3)
        rec["eps_growth_yoy"] = _growth(eps) or info.get("earningsGrowth")
        rec["eps_cagr_3y"] = _cagr(eps, 3)
        # "EPS acceleration" from the factor sheet: is the latest year's growth
        # outrunning the stock's own 3-year trend, or decelerating? A cleaner
        # QoQ version would need quarterly statements, which are even more
        # exposed to the same Yahoo crumb-blocking issue - this annual proxy
        # is a reasonable middle ground.
        rec["eps_acceleration"] = (
            rec["eps_growth_yoy"] - rec["eps_cagr_3y"]
            if rec["eps_growth_yoy"] is not None and rec["eps_cagr_3y"] is not None
            else None
        )
        rec["ebitda_growth_yoy"] = _growth(ebitda)

        # quality
        capital_employed = None
        if _at(assets) is not None and _at(cl) is not None:
            capital_employed = _at(assets) - _at(cl)
        rec["roce"] = (ebit0 / capital_employed) if (ebit0 and capital_employed and capital_employed > 0) else None
        rec["roe"] = info.get("returnOnEquity") or ((ni0 / eq0) if (ni0 and eq0 and eq0 > 0) else None)
        rec["ebitda_margin"] = (ebitda0 / rev0) if (ebitda0 is not None and rev0) else None
        m3 = None
        if ebitda is not None and rev is not None and len(ebitda) > 3 and len(rev) > 3:
            old_m = _at(ebitda, 3) / _at(rev, 3) if _at(rev, 3) else None
            if old_m is not None and rec["ebitda_margin"] is not None:
                m3 = rec["ebitda_margin"] - old_m
        rec["margin_delta_3y"] = m3

        net_debt = debt0 - cash0
        rec["net_debt_to_ebitda"] = (net_debt / ebitda0) if (ebitda0 and ebitda0 > 0) else None
        rec["debt_to_equity"] = info.get("debtToEquity") or (
            (debt0 / eq0 * 100) if (eq0 and eq0 > 0) else None)
        int0 = abs(_at(interest) or 0.0)
        rec["interest_coverage"] = (ebit0 / int0) if (ebit0 is not None and int0 > 0) else (
            99.0 if ebit0 and ebit0 > 0 else None)

        fcf = (cfo0 + capex0) if cfo0 is not None else info.get("freeCashflow")
        rec["fcf"] = fcf
        rec["fcf_to_pat"] = (fcf / ni0) if (fcf is not None and ni0 and ni0 > 0) else None
        rec["fcf_margin"] = (fcf / rev0) if (fcf is not None and rev0) else None

        # value
        pe = info.get("trailingPE")
        rec["pe"] = pe
        rec["peg"] = info.get("pegRatio") or info.get("trailingPegRatio")
        if rec["peg"] is None and pe and rec.get("eps_cagr_3y"):
            g = rec["eps_cagr_3y"] * 100
            rec["peg"] = pe / g if g > 0 else None
        ev = info.get("enterpriseValue")
        rec["ev_to_ebitda"] = (ev / ebitda0) if (ev and ebitda0 and ebitda0 > 0) else None
        rec["earnings_yield"] = (1 / pe) if (pe and pe > 0) else None
        rec["fcf_yield"] = (fcf / mcap) if (fcf is not None and mcap) else None
        # own-history valuation: current P/E vs 5y median P/E from annual EPS
        rec["pe_vs_own_5y"] = None
        if pe and eps is not None and len(eps) >= 3:
            hist_pe = []
            price_now = info.get("currentPrice") or info.get("regularMarketPrice")
            if price_now:
                for j in range(min(5, len(eps))):
                    e = _at(eps, j)
                    if e and e > 0:
                        hist_pe.append(price_now / e)
            if len(hist_pe) >= 3:
                med = float(np.median(hist_pe))
                rec["pe_vs_own_5y"] = pe / med if med > 0 else None

        # risk
        rec["share_dilution_1y"] = _growth(shares)
        rec["promoter_pledge_pct"] = None  # populated from overrides file

    except Exception as e:
        rec["error"] = str(e)[:120]
    return rec


def fundamentals_available(probe_symbol: str = "RELIANCE") -> bool:
    """Quick check for whether Yahoo's fundamentals (quoteSummary/crumb-gated)
    endpoints are reachable from this server, as opposed to the plain price
    history endpoint which is unauthenticated and usually still works.

    Yahoo has increasingly blocked the crumb-gated endpoints from datacenter
    IPs (Render, AWS, etc.) even when price downloads succeed. Fetching 250
    stocks sequentially only to discover this 250 times over wastes many
    minutes for nothing - so we check once, up front, with a single symbol.
    """
    rec = _fetch_one_fundamental(probe_symbol)
    # A working call returns a real market cap; a blocked one returns almost
    # nothing but "symbol" and possibly "error".
    return rec.get("market_cap_cr") is not None or rec.get("revenue") is not None


def fetch_fundamentals(symbols: list[str], sleep: float = 0.0,
                       max_workers: int = 10, log=print) -> pd.DataFrame:
    """Fetch fundamentals in parallel (network-bound, so threads help a lot),
    with per-symbol failures isolated - one blocked/slow ticker never blocks
    the rest.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one_fundamental, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 25 == 0 or done == total:
                log(f"fundamentals {done}/{total}")
            if sleep:
                time.sleep(sleep)
    return pd.DataFrame(rows).set_index("symbol")


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def winsorized_z(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)
    a, b = s.quantile(lo), s.quantile(hi)
    c = s.clip(a, b)
    sd = c.std(ddof=0)
    if not sd or np.isnan(sd) or sd == 0:
        return pd.Series(0.0, index=s.index).where(s.notna())
    return (c - c.mean()) / sd


def sector_neutral_z(df: pd.DataFrame, col: str, sector_col: str,
                     higher_is_better: bool, min_n: int = 6) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for sec, grp in df.groupby(sector_col):
        z = winsorized_z(grp[col]) if grp[col].notna().sum() >= min_n else pd.Series(np.nan, index=grp.index)
        out.loc[grp.index] = z
    # fall back to whole-universe z where the sector was too small / sparse
    uni = winsorized_z(df[col])
    out = out.fillna(uni)
    return out if higher_is_better else -out


def score(df: pd.DataFrame, sector_col: str = "sector",
          weights: dict | None = None) -> pd.DataFrame:
    weights = weights or BUCKET_WEIGHTS
    zcols: dict[str, list[str]] = {b: [] for b in weights}

    for col, higher, bucket in FACTORS:
        if col not in df.columns:
            continue
        if df[col].notna().sum() < 5:
            continue
        zname = f"z_{col}"
        df[zname] = sector_neutral_z(df, col, sector_col, higher).clip(-3, 3)
        zcols[bucket].append(zname)

    for bucket, cols in zcols.items():
        df[f"score_{bucket}"] = df[cols].mean(axis=1, skipna=True) if cols else np.nan
        df[f"cov_{bucket}"] = df[cols].notna().sum(axis=1) / max(len(cols), 1) if cols else 0.0

    # weighted composite, renormalised over the buckets a stock actually has data for
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for bucket, w in weights.items():
        b = df[f"score_{bucket}"]
        num = num.add((b * w).fillna(0.0))
        den = den.add(pd.Series(np.where(b.notna(), w, 0.0), index=df.index))
    df["composite_z"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    df["data_coverage"] = den / sum(weights.values())

    # 0-100 percentile score, easier to read than a z
    df["SCORE"] = (df["composite_z"].rank(pct=True) * 100).round(1)
    df["RANK"] = df["SCORE"].rank(ascending=False, method="min")
    return df


# ----------------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------------
def apply_eligibility(df: pd.DataFrame, rules: Eligibility) -> pd.DataFrame:
    def ok(col, test, label):
        if col not in df.columns:
            df[f"fail_{label}"] = False
            return
        v = pd.to_numeric(df[col], errors="coerce")
        df[f"fail_{label}"] = v.notna() & ~test(v)

    ok("market_cap_cr", lambda v: v >= rules.min_market_cap_cr, "mcap")
    ok("adtv_cr", lambda v: v >= rules.min_adtv_cr, "liquidity")
    ok("price", lambda v: v >= rules.min_price, "price")
    ok("n_days", lambda v: v >= rules.min_history_days, "history")
    ok("debt_to_equity", lambda v: v <= rules.max_debt_to_equity, "leverage")
    ok("interest_coverage", lambda v: v >= rules.min_interest_coverage, "intcov")
    if rules.require_positive_equity:
        ok("total_equity", lambda v: v > 0, "equity")
    if rules.require_positive_ebitda:
        ok("ebitda", lambda v: v > 0, "ebitda")

    if rules.skip_leverage_checks_for_sectors and "sector" in df.columns:
        exempt = df["sector"].isin(rules.skip_leverage_checks_for_sectors)
        for col in ("fail_leverage", "fail_intcov"):
            if col in df.columns:
                df.loc[exempt, col] = False

    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    df["n_fails"] = df[fail_cols].sum(axis=1)
    df["eligible"] = df["n_fails"] == 0
    df["fail_reasons"] = df[fail_cols].apply(
        lambda r: ",".join(c[5:] for c in fail_cols if r[c]), axis=1)
    if rules.exclude_sectors:
        bad = df["sector"].isin(rules.exclude_sectors)
        df.loc[bad, "eligible"] = False
        df.loc[bad, "fail_reasons"] = df.loc[bad, "fail_reasons"] + ",sector"
    return df


# ----------------------------------------------------------------------------
# Reusable entry point (used by both the CLI and the web app)
# ----------------------------------------------------------------------------
def run_screen(
    universe: str = "nifty250",
    universe_csv: str | None = None,
    top: int = 20,
    fast: bool = False,
    max_per_sector: int = 4,
    pledge_csv: str | None = None,
    cache_hours: float = 12.0,
    weights: dict | None = None,
    log=print,
) -> dict:
    """Run the full two-stage screen and return a JSON-able dict.

    log: callable(str) -> None, used to report progress (e.g. into a job-status
    object from a web server instead of stdout).
    """
    weights = weights or BUCKET_WEIGHTS
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    log(f"Loading universe: {universe}")
    uni = load_universe(universe, universe_csv)
    symbols = sorted(uni["symbol"].unique().tolist())
    log(f"{len(symbols)} symbols in universe")

    log("Downloading prices from Yahoo Finance ...")
    px = fetch_prices(symbols)
    pf = price_factors(px, symbols)

    data_mode = "fast" if fast else "full"
    if fast:
        fund = pd.DataFrame(index=pf.index)
    else:
        cache = os.path.join(CACHE_DIR, f"fund_{universe}.parquet")
        fresh = (os.path.exists(cache) and
                 (time.time() - os.path.getmtime(cache)) < cache_hours * 3600)
        if fresh:
            log("Loading cached fundamentals")
            fund = pd.read_parquet(cache)
        else:
            log("Checking whether Yahoo fundamentals are reachable from this server ...")
            if not fundamentals_available():
                log("Fundamentals endpoint is blocked from this server (common on cloud "
                    "hosts) - falling back to momentum-only scoring instead of spending "
                    "several minutes failing on every one of "
                    f"{len(symbols)} names.")
                fund = pd.DataFrame(index=pf.index)
                data_mode = "momentum_only_fallback"
            else:
                log(f"Downloading fundamentals for {len(symbols)} names (parallel) ...")
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

    log("Applying eligibility rules")
    df = apply_eligibility(df, Eligibility())
    elig = df[df["eligible"]].copy()
    log(f"{len(elig)}/{len(df)} names passed eligibility")

    log("Scoring")
    elig = score(elig, "sector", weights)
    elig = elig.sort_values("SCORE", ascending=False)

    keep, counts = [], {}
    for sym, row in elig.iterrows():
        sec = row["sector"]
        if counts.get(sec, 0) >= max_per_sector:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep.append(sym)
    picks = elig.loc[keep].head(top)

    cols = ["name", "sector", "SCORE", "score_growth", "score_quality", "score_value",
            "score_momentum", "score_risk", "data_coverage", "price", "market_cap_cr",
            "adtv_cr", "roce", "roe", "rev_growth_yoy", "eps_growth_yoy",
            "rs_6m", "rs_12m", "pe", "peg", "fcf_yield", "net_debt_to_ebitda",
            "debt_to_equity", "dist_52w_high"]
    cols = [c for c in cols if c in picks.columns]

    records = []
    for sym, row in picks[cols].iterrows():
        rec = {"symbol": sym}
        for c in cols:
            v = row[c]
            rec[c] = None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else v)
        records.append(rec)

    # Full-universe lookup for debugging/transparency: every symbol scanned,
    # whether it made eligibility, why not if it didn't, and its full score
    # breakdown if it did - regardless of whether it made the final top N.
    # This is what answers "why isn't <stock> in the list" without guessing.
    score_cols = [c for c in elig.columns if c.startswith("score_") or c == "SCORE"
                 or c == "data_coverage"]
    df = df.join(elig[score_cols], how="left", rsuffix="_elig")
    lookup = {}
    for sym, row in df.iterrows():
        entry = {
            "name": None if pd.isna(row.get("name")) else row.get("name"),
            "sector": None if pd.isna(row.get("sector")) else row.get("sector"),
            "eligible": bool(row.get("eligible", False)),
            "fail_reasons": row.get("fail_reasons", "") or None,
        }
        for c in score_cols:
            v = row.get(c)
            entry[c] = None if pd.isna(v) else float(v)
        entry["made_top_n"] = sym in set(picks.index)
        lookup[sym] = entry

    return {
        "generated_at": stamp,
        "universe": universe,
        "universe_size": len(symbols),
        "data_mode": data_mode,
        "eligible_count": int(len(elig)),
        "weights": weights,
        "picks": records,
        "lookup": lookup,
    }

