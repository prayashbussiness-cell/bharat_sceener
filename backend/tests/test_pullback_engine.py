"""Tests for the Pull Back screener engine.
Run with: pytest tests/test_pullback_engine.py -v
(or, without pytest installed: python tests/test_pullback_engine.py)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.pullback import engine as E
from backend.pullback import indicators as I


def _flat_uptrend_df(n=400, seed=1, dip_start=None, dip_len=25, dip_pct=6.0, seller_exhaustion=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.empty(n)
    close[0] = 100.0
    for i in range(1, n):
        drift = 0.12
        if dip_start and dip_start <= i < dip_start + dip_len:
            frac = (i - dip_start) / dip_len
            drift = -dip_pct / dip_len * (1.4 - frac) if seller_exhaustion else -dip_pct / dip_len
        close[i] = close[i - 1] * (1 + drift / 100 + rng.normal(0, 0.35) / 100)
    noise = np.abs(rng.normal(0, 0.3, n)) / 100
    high = close * (1 + noise)
    low = close * (1 - noise)
    openp = close * (1 + rng.normal(0, 0.15, n) / 100)
    high = np.maximum.reduce([high, close, openp])
    low = np.minimum.reduce([low, close, openp])
    vol = rng.integers(100000, 400000, n).astype(float)
    if dip_start:
        vol[dip_start:dip_start + dip_len] *= 0.7 if seller_exhaustion else 1.6
        vol[dip_start + dip_len:dip_start + dip_len + 5] *= 1.6
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=dates)


def test_indicators_are_causal_no_lookahead():
    df = _flat_uptrend_df(300)
    A = E.prepare(df)
    # truncating history after t must not change any indicator value AT t
    t = 250
    A_full = A
    A_trunc = E.prepare(df.iloc[: t + 1])
    for k in ("ema20", "sma50", "sma200", "rsi", "macd", "atr14", "st_dir"):
        a, b = A_full[k][t], A_trunc[k][-1]
        if np.isfinite(a) and np.isfinite(b):
            assert abs(a - b) < 1e-6, f"{k} leaks future information"


def test_insufficient_history_flags_hf10():
    df = _flat_uptrend_df(80)
    r = E.evaluate(df)
    assert r["state"] == "NO_SETUP"
    assert r["hard_filters"].get("HF10") is False


def test_established_uptrend_qualifies_trend_hard_filters():
    df = _flat_uptrend_df(400)
    r = E.evaluate(df)
    assert r["hard_filters"]["HF01"] is True     # close > 200DMA
    assert r["hard_filters"]["HF03"] is True     # 50DMA > 200DMA
    assert r["trend"]["sma200_slope"] == "positive"


def test_pullback_with_seller_exhaustion_scores_higher_than_without():
    df_good = _flat_uptrend_df(400, seed=5, dip_start=350, dip_len=25, dip_pct=7, seller_exhaustion=True)
    df_bad = _flat_uptrend_df(400, seed=5, dip_start=350, dip_len=25, dip_pct=7, seller_exhaustion=False)
    r_good, r_bad = E.evaluate(df_good), E.evaluate(df_bad)
    assert r_good["score"] >= r_bad["score"]


def test_deep_crash_is_never_labelled_a_normal_pullback():
    df = _flat_uptrend_df(400, seed=9, dip_start=370, dip_len=8, dip_pct=28, seller_exhaustion=False)
    r = E.evaluate(df)
    assert r["state"] in ("NO_SETUP", "WATCHLIST", "INVALIDATED")
    assert r["pullback"]["pullback_pct"] > 15


def test_close_below_200dma_forces_invalidated_or_no_setup():
    df = _flat_uptrend_df(400, seed=2)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 0.7  # crash the last close hard
    df2.iloc[-1, df2.columns.get_loc("low")] = df2.iloc[-1]["close"] * 0.98
    r = E.evaluate(df2)
    assert r["invalidation"]["trend_broken"] is True
    assert r["state"] in ("INVALIDATED", "NO_SETUP")


def test_rsi_above_70_is_not_automatically_penalized_as_bearish():
    # Spec: "RSI >70 is not automatically bearish" - construct a strong steady uptrend
    # (no pullback) and confirm the engine doesn't invalidate purely on elevated RSI.
    df = _flat_uptrend_df(400, seed=11)
    r = E.evaluate(df)
    assert "RSI" not in " ".join(r["warnings"]) or r["state"] != "INVALIDATED"


def test_score_never_exceeds_100_or_goes_negative():
    for seed in range(5):
        df = _flat_uptrend_df(400, seed=seed, dip_start=360, dip_len=20, dip_pct=6)
        r = E.evaluate(df)
        assert 0 <= r["score"] <= 100


def test_output_shape_matches_spec_section_19():
    df = _flat_uptrend_df(400, seed=3, dip_start=350, dip_len=20, dip_pct=6)
    r = E.evaluate(df)
    for k in ("state", "score", "trend", "pullback", "support", "momentum", "volume", "price_action",
              "entry", "invalidation"):
        assert k in r, f"missing top-level key {k}"
    assert set(r["support"].keys()) >= {"near_ema20", "near_sma50", "prior_breakout_support",
                                         "fib_confluence", "support_confluence_count"}


def test_supertrend_matches_reference_small_case():
    # Sanity check against a hand-computed small monotonic uptrend: Supertrend should be green.
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.arange(n) * 0.8
    df = pd.DataFrame({"open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close,
                        "volume": np.full(n, 1e5)}, index=dates)
    _, d = I.supertrend(df)
    assert d.iloc[-1] == 1


def test_rules_config_exposes_thresholds():
    cfg = E.rules_config()
    assert "params" in cfg and "buckets" in cfg and "max_points" in cfg
    total_weight = sum(v["weight"] for v in cfg["buckets"].values())
    assert total_weight == 100


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, "-", e)
            except Exception as e:
                fails += 1
                print("ERROR", name, "-", repr(e))
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
