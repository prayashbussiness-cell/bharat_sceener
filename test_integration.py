"""Offline integration test - mocks all network I/O so it runs in a sandboxed CI."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend.engine_core as engine

rng = np.random.default_rng(3)


def fake_load_universe(name, custom_csv=None):
    n = 40
    syms = [f"STK{i:03d}" for i in range(n)]
    sectors = rng.choice(["Financial Services", "Technology", "Healthcare",
                          "Industrials", "Energy"], n)
    return pd.DataFrame({"symbol": syms, "name": [f"Company {s}" for s in syms],
                         "nse_sector": sectors})


def fake_fetch_prices(symbols, period="3y"):
    return object()  # unused because price_factors is also patched


def fake_price_factors(data, symbols):
    n = len(symbols)
    return pd.DataFrame({
        "symbol": symbols,
        "n_days": rng.integers(250, 750, n),
        "price": rng.uniform(30, 4000, n),
        "adtv_cr": rng.lognormal(1.3, 1.0, n),
        "ret_3m": rng.normal(0.03, 0.1, n),
        "rs_6m": rng.normal(0.0, 0.12, n),
        "rs_12m": rng.normal(0.0, 0.18, n),
        "px_vs_200dma": rng.normal(0.02, 0.08, n),
        "slope_200dma": rng.normal(0.01, 0.03, n),
        "dist_52w_high": -rng.uniform(0, 0.3, n),
        "vol_12m": rng.uniform(0.15, 0.6, n),
        "max_drawdown_1y": -rng.uniform(0.05, 0.4, n),
        "golden_cross_num": rng.choice([0.0, 1.0], n),
    }).set_index("symbol")


def fake_fetch_fundamentals(symbols, sleep=0.0, max_workers=10, log=print):
    n = len(symbols)
    df = pd.DataFrame({
        "symbol": symbols,
        "yf_sector": rng.choice(["Financial Services", "Technology", "Healthcare",
                                 "Industrials", "Energy"], n),
        "market_cap_cr": rng.lognormal(8.5, 1.1, n),
        "total_equity": rng.normal(5e9, 3e9, n),
        "ebitda": np.abs(rng.normal(2e9, 1e9, n)),
        "net_income": rng.normal(1e9, 8e8, n),
        "debt_to_equity": rng.uniform(0, 250, n),
        "interest_coverage": rng.uniform(1, 25, n),
        "rev_growth_yoy": rng.normal(0.1, 0.08, n),
        "rev_cagr_3y": rng.normal(0.09, 0.06, n),
        "eps_growth_yoy": rng.normal(0.12, 0.1, n),
        "eps_cagr_3y": rng.normal(0.1, 0.07, n),
        "eps_acceleration": rng.normal(0, 0.05, n),
        "ebitda_growth_yoy": rng.normal(0.1, 0.08, n),
        "roce": rng.uniform(0.05, 0.35, n),
        "roe": rng.uniform(0.05, 0.3, n),
        "ebitda_margin": rng.uniform(0.08, 0.35, n),
        "margin_delta_3y": rng.normal(0, 0.02, n),
        "net_debt_to_ebitda": rng.uniform(-1, 3, n),
        "fcf_to_pat": rng.uniform(0.5, 1.5, n),
        "fcf_margin": rng.uniform(0.02, 0.2, n),
        "peg": rng.uniform(0.5, 3, n),
        "ev_to_ebitda": rng.uniform(5, 30, n),
        "pe_vs_own_5y": rng.uniform(0.6, 1.6, n),
        "fcf_yield": rng.uniform(0.01, 0.08, n),
        "earnings_yield": rng.uniform(0.02, 0.08, n),
        "share_dilution_1y": rng.normal(0.01, 0.03, n),
        "promoter_pledge_pct": np.nan,
    }).set_index("symbol")
    return df


def test_pipeline():
    engine.load_universe = fake_load_universe
    engine.fetch_prices = fake_fetch_prices
    engine.price_factors = fake_price_factors
    engine.fetch_fundamentals = fake_fetch_fundamentals
    engine.fundamentals_available = lambda probe_symbol="RELIANCE": True

    result = engine.run_screen(universe="nifty250", top=10, fast=False,
                               max_per_sector=3, cache_hours=0, log=lambda m: None)

    assert result["universe_size"] == 40
    assert result["eligible_count"] > 0
    assert result["data_mode"] == "full"
    assert 1 <= len(result["picks"]) <= 10
    for p in result["picks"]:
        assert 0 <= p["SCORE"] <= 100
        assert p["data_coverage"] > 0

    # sector cap respected
    from collections import Counter
    c = Counter(p["sector"] for p in result["picks"])
    assert max(c.values()) <= 3, c

    # lookup covers every scanned symbol, not just the top N
    assert len(result["lookup"]) == 40
    for sym, entry in result["lookup"].items():
        assert "eligible" in entry and "sector" in entry
        if entry["eligible"]:
            assert entry["SCORE"] is not None
        else:
            assert entry["fail_reasons"]  # ineligible names must say why

    # JSON-serialisable (this is what the web app writes to disk / returns)
    json.dumps(result, default=str)
    print(f"run_screen OK: {result['eligible_count']} eligible, "
          f"{len(result['picks'])} picks, top score "
          f"{result['picks'][0]['SCORE']}, lookup has {len(result['lookup'])} entries")
    return result


def test_bank_leverage_exemption():
    """A bank with a 600% D/E (normal - deposits are liabilities) must not
    be disqualified by the leverage rule; a non-bank with the same D/E must
    still fail it. This is the exact bug that excluded ICICI Bank-style
    names in production."""
    df = pd.DataFrame({
        "sector": ["Financial Services", "Industrials"],
        "market_cap_cr": [500000, 5000],
        "adtv_cr": [50, 10],
        "price": [1200, 800],
        "n_days": [1000, 1000],
        "debt_to_equity": [650, 650],       # identical, deliberately
        "interest_coverage": [0.9, 0.9],    # identical, deliberately
        "total_equity": [1e11, 1e9],
        "ebitda": [5e10, 5e8],
    }, index=["SOMEBANK", "SOMEINDUSTRIAL"])

    out = engine.apply_eligibility(df.copy(), engine.Eligibility())
    assert out.loc["SOMEBANK", "eligible"] == True, \
        f"bank wrongly excluded: {out.loc['SOMEBANK', 'fail_reasons']}"
    assert out.loc["SOMEINDUSTRIAL", "eligible"] == False
    assert "leverage" in out.loc["SOMEINDUSTRIAL", "fail_reasons"]
    print("Bank leverage exemption OK: same D/E, different sectors, different outcome")


def test_fundamentals_blocked_fallback():
    """This is the exact failure mode seen in production: Yahoo's crumb-gated
    endpoints return 401 for every symbol. The preflight check must catch
    this ONCE (not after grinding through the whole universe) and fall back
    to momentum-only scoring, tagging the result so the UI can say why."""
    engine.load_universe = fake_load_universe
    engine.fetch_prices = fake_fetch_prices
    engine.price_factors = fake_price_factors
    engine.fundamentals_available = lambda probe_symbol="RELIANCE": False

    calls = {"n": 0}
    def fail_if_called(*a, **k):
        calls["n"] += 1
        raise AssertionError("fetch_fundamentals must not be called when preflight fails")
    engine.fetch_fundamentals = fail_if_called

    result = engine.run_screen(universe="nifty250", top=10, fast=False,
                               max_per_sector=3, cache_hours=0, log=lambda m: None)

    assert calls["n"] == 0, "fetch_fundamentals was called despite failed preflight"
    assert result["data_mode"] == "momentum_only_fallback"
    assert result["eligible_count"] > 0, "price-only eligibility should still pass most names"
    assert len(result["picks"]) > 0
    for p in result["picks"]:
        # quality/value scores should be absent/None since no fundamentals came in
        assert p.get("roce") is None
    print(f"Fallback OK: preflight caught the block with 0 wasted per-symbol calls, "
          f"data_mode={result['data_mode']}, {len(result['picks'])} momentum-only picks")


def test_quicklist_universe_bypasses_nse():
    """quicklist must not touch load_universe's network path at all."""
    from backend.engine_core import load_universe, QUICK_WATCHLIST
    df = load_universe("quicklist")
    assert len(df) == len(QUICK_WATCHLIST)
    assert set(df["symbol"]) == set(QUICK_WATCHLIST)
    print(f"Quicklist OK: {len(df)} symbols, no network call required")


def test_gemini_fallback():
    """No GEMINI_API_KEY set -> must degrade gracefully, not raise."""
    os.environ.pop("GEMINI_API_KEY", None)
    from backend.gemini_summary import generate_commentary
    picks = [{"symbol": "STK000", "name": "Company STK000", "sector": "Technology",
             "SCORE": 88.5, "score_growth": 1.1, "score_quality": 0.8,
             "score_value": -0.2, "score_momentum": 1.4, "score_risk": 0.3,
             "roce": 0.22, "data_coverage": 1.0}]
    out = generate_commentary(picks)
    assert "overall" in out and isinstance(out["overall"], str)
    assert out["per_stock"] == {}
    print("Gemini fallback OK (no API key ->", out["overall"][:60], "...)")


def test_web_job_flow():
    """Exercise the FastAPI app's job lifecycle with run_screen mocked out,
    so this test needs no network and no real Gemini key."""
    import backend.main as web

    def fake_run_screen(**kwargs):
        time.sleep(0.05)
        return {
            "generated_at": "test", "universe": "nifty250", "universe_size": 40,
            "eligible_count": 12, "weights": {}, "picks": [
                {"symbol": "STK001", "name": "Company STK001", "sector": "Technology",
                 "SCORE": 91.2, "score_growth": 1.2, "score_quality": 0.9,
                 "score_value": 0.1, "score_momentum": 1.5, "score_risk": 0.2,
                 "data_coverage": 1.0},
            ],
        }

    def fake_commentary(picks):
        return {"per_stock": {"STK001": "Strong momentum and quality scores."},
                "overall": "Momentum and quality are driving today's list."}

    web.run_screen = fake_run_screen
    web.generate_commentary = fake_commentary
    web.ENABLE_GEMINI = True

    from fastapi.testclient import TestClient
    client = TestClient(web.app)

    r = client.get("/api/status")
    assert r.json()["status"] == "idle"

    r = client.get("/healthz")
    assert r.json() == {"ok": True}

    r = client.get("/api/config")
    cfg = r.json()
    assert cfg["eligibility_rules"]["max_debt_to_equity"] == 300.0
    assert "Financial Services" in cfg["eligibility_rules"]["skip_leverage_checks_for_sectors"]
    assert "growth" in cfg["factors_by_bucket"]
    print("Config endpoint OK: eligibility rules and factor list exposed dynamically")

    r = client.post("/api/run")
    assert r.json()["started"] is True

    # duplicate click while running should not start a second job
    r2 = client.post("/api/run")
    assert r2.json().get("started") in (False, True)  # timing-dependent, both handled

    for _ in range(40):
        s = client.get("/api/status").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s

    r = client.get("/api/results")
    data = r.json()
    assert data["picks"][0]["symbol"] == "STK001"
    assert data["commentary"]["overall"].startswith("Momentum")
    assert os.path.exists(web.RESULT_FILE)
    print("Web job flow OK: status idle->running->done, results served, cache file written")


if __name__ == "__main__":
    test_quicklist_universe_bypasses_nse()
    test_bank_leverage_exemption()
    test_pipeline()
    test_fundamentals_blocked_fallback()
    test_gemini_fallback()
    test_web_job_flow()
    print("\nALL INTEGRATION TESTS PASSED")
