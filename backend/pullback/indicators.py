"""Indicator library (pure pandas/numpy - no TA-Lib). Every indicator is causal:
value at bar t only uses bars <= t."""
import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    au, ad = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return 100.0 * au / (au + ad).replace(0, np.nan)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return wilder(true_range(df), n)


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = wilder(true_range(df), n)
    pdi, mdi = 100 * wilder(pdm, n) / a, 100 * wilder(mdm, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi


def supertrend(df, period=10, mult=3.0):
    """Returns (line, direction) direction: 1 bullish/green, -1 bearish/red, 0 unavailable."""
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    a = atr(df, period).to_numpy(float)
    n = len(c)
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = np.full(n, np.nan), np.full(n, np.nan)
    d, st = np.zeros(n, dtype=int), np.full(n, np.nan)
    started = False
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if not started:
            fub[i], flb[i] = ub[i], lb[i]
            d[i] = 1 if c[i] > hl2[i] else -1
            started = True
        else:
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
            if d[i - 1] == -1 and c[i] > fub[i - 1]:
                d[i] = 1
            elif d[i - 1] == 1 and c[i] < flb[i - 1]:
                d[i] = -1
            else:
                d[i] = d[i - 1]
        st[i] = flb[i] if d[i] == 1 else fub[i]
    return pd.Series(st, index=df.index), pd.Series(d, index=df.index)


def obv(df):
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def ad_line(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng).fillna(0)
    return (clv * df["volume"]).cumsum()


def stochastic(df, k=14, d=3):
    ll, hh = df["low"].rolling(k, min_periods=k).min(), df["high"].rolling(k, min_periods=k).max()
    pk = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return pk, pk.rolling(d, min_periods=d).mean()


def pivots(high, low, k=2):
    """Boolean pivot arrays. Pivot at i is only *confirmed* at bar i+k - callers must
    ignore pivots with i > t-k (look-ahead control)."""
    n = len(high)
    ph, pl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(k, n - k):
        w_h, w_l = high[i - k:i + k + 1], low[i - k:i + k + 1]
        if high[i] >= w_h.max() and high[i] > high[i - 1]:
            ph[i] = True
        if low[i] <= w_l.min() and low[i] < low[i - 1]:
            pl[i] = True
    return ph, pl


def normalize_ohlcv(df):
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def weekly_frame(df):
    """Weekly OHLCV whose index is the *last actual trading date* of each week."""
    w = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    last = df.index.to_series().resample("W-FRI").max()
    w = w.dropna(subset=["close"])
    w.index = last.loc[w.index].values
    return w
