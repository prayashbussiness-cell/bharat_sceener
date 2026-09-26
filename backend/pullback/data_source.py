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


def bulk_prefetch(symbols, period="3y"):
    """Warm the cache for many symbols with ONE batched yfinance call instead
    of one yf.Ticker(...).history() call per symbol. Scanning a couple hundred
    symbols one at a time is exactly the pattern that trips Yahoo's crumb/
    rate-limiting (HTTP 429/401) - batching, like the host app's own
    fetch_prices() already does for the primary/broad screeners, is far less
    likely to get rate-limited. Any symbol the batch call misses just falls
    through to the normal per-symbol fetch in get_ohlcv() as before, so this
    is purely an optimization: it never removes a symbol from the scan.
    """
    if not symbols:
        return
    try:
        import yfinance as yf
        tickers = [s if s.endswith((".NS", ".BO")) else s + ".NS" for s in symbols]
        data = yf.download(tickers, period=period, interval="1d",
                            auto_adjust=False, group_by="ticker",
                            threads=True, progress=False)
        if data is None or data.empty:
            return
        now = time.time()
        multi = len(tickers) > 1
        for sym, tk in zip(symbols, tickers):
            try:
                sub = data[tk] if multi else data
                sub = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(how="all")
                if not sub.empty:
                    _CACHE[sym.upper()] = (now, sub)
            except Exception:
                continue  # this symbol wasn't in the batch result - fine, per-symbol fetch will catch it
    except Exception as e:
        log.warning("bulk prefetch failed, falling back to per-symbol fetch: %s", e)


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
                result = f(name)
                # load_universe() (the function that actually exists on the
                # host's engine_core) returns a DataFrame with columns
                # symbol/name/nse_sector - list(df) on a DataFrame yields its
                # COLUMN NAMES ('symbol','name','nse_sector'), not the rows,
                # so pull the symbol column explicitly. Any other callable
                # that already returns a plain list/iterable of tickers is
                # passed through unchanged.
                if hasattr(result, "columns"):
                    col = "symbol" if "symbol" in result.columns else result.columns[0]
                    return [str(s).strip().upper() for s in result[col].tolist() if str(s).strip()]
                return [str(s).strip().upper() for s in result]
    except Exception as e:
        log.warning("host universe lookup failed: %s", e)
    quicklist = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "LT", "SBIN",
                 "BHARTIARTL", "ITC", "AXISBANK", "KOTAKBANK", "HINDUNILVR",
                 "MARUTI", "BAJFINANCE", "SUNPHARMA", "TITAN", "ULTRACEMCO",
                 "NTPC", "POWERGRID", "TATAMOTORS", "M&M", "HAL", "PNB",
                 "BANKBARODA", "UNIONBANK", "TATASTEEL", "ONGC", "COALINDIA",
                 "ADANIENT", "WIPRO"]
    return quicklist
