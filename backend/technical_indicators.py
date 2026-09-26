"""
Technical indicator math, kept separate from the engine that uses it so the
formulas can be checked/tested in isolation.

All functions take pandas Series in ASCENDING date order (oldest first) -
the standard order yfinance's daily price history comes back in. This is
the opposite convention from the annual-statement Series used elsewhere in
this project (which are newest-first, per yfinance's statement convention) -
worth remembering if you're moving values between the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing = an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0, 100.0)  # no losses in the window -> RSI 100


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Returns (adx, plus_di, minus_di), standard Wilder's method."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_, plus_di, minus_di


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Returns a Series of +1 (uptrend/green) / -1 (downtrend/red), same
    index as input. Standard Supertrend algorithm.
    """
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_
    lower_band = hl2 - multiplier * atr_

    direction = pd.Series(1, index=close.index)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(close)):
        if close.iloc[i - 1] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i - 1] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

    return direction


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
              k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Chaikin Accumulation/Distribution line."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume
    return mfv.cumsum()


def roc(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def linreg_slope_pct(s: pd.Series, window: int) -> float | None:
    """Slope of a linear fit over the last `window` points, as a % of the
    series' mean level over that window - makes the slope comparable across
    stocks at very different price levels."""
    tail = s.tail(window).dropna()
    if len(tail) < max(5, window // 2):
        return None
    x = np.arange(len(tail))
    y = tail.values
    mean_y = float(np.mean(y))
    if mean_y == 0:
        return None
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / abs(mean_y) * 100
