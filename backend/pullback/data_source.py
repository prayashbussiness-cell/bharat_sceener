"""Market data plumbing for the Pull Back screener.

Tries to reuse the host repo's engine_core (NSE list + Yahoo fetch + fallback CSV +
12h cache) if it's importable; otherwise falls back to a small self-contained
yfinance-based fetcher so this module works standalone.
"""
import logging
import time
import pandas as pd

log = logging.getLogger("pullback.data_source")
_CACHE = {}
_CACHE_TTL = 12 * 3600


def _host_fetch(symbol):
    try:
        from backend import engine_core as core  # host repo module, if present
    except Exception:
        return None
    for fn in ("fetch_price_history", "get_price_history", "fetch_ohlcv"):
        f = getattr(core, fn, None)
        if callable(f):
            try:
                return f(symbol)
            except Exception as e:
                log.warning("host fetch %s failed for %s: %s", fn, symbol, e)
    return None


def _yf_fetch(symbol):
    import yfinance as yf
    tk = symbol if symbol.endswith((".NS", ".BO")) else symbol + ".NS"
    df = yf.Ticker(tk).history(period="3y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df


def get_ohlcv(symbol, cache_hours=12):
    key = symbol.upper()
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < cache_hours * 3600:
        return hit[1]
    df = _host_fetch(symbol)
    if df is None or getattr(df, "empty", True):
        df = _yf_fetch(symbol)
    if df is None or df.empty:
        return None
    _CACHE[key] = (now, df)
    return df


def universe_symbols(name="quicklist"):
    try:
        from backend import engine_core as core
        for fn in ("get_universe", "load_universe", "universe_symbols"):
            f = getattr(core, fn, None)
            if callable(f):
                return list(f(name))
    except Exception as e:
        log.warning("host universe lookup failed: %s", e)
    quicklist = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "LT", "SBIN",
                 "BHARTIARTL", "ITC", "AXISBANK", "KOTAKBANK", "HINDUNILVR",
                 "MARUTI", "BAJFINANCE", "SUNPHARMA", "TITAN", "ULTRACEMCO",
                 "NTPC", "POWERGRID", "TATAMOTORS", "M&M", "HAL", "PNB",
                 "BANKBARODA", "UNIONBANK", "TATASTEEL", "ONGC", "COALINDIA",
                 "ADANIENT", "WIPRO"]
    return quicklist
