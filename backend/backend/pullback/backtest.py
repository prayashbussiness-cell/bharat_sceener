"""Section 20 backtesting harness: walk-forward, regime split, look-ahead control,
signal-overlap control, parameter sensitivity, ablation. Runs entirely on the causal
evaluate_at(), so 'what the screener would have said on day t' is exactly what the
backtest replays - no separate code path to drift out of sync.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
from . import engine as E


def _trades_from_signals(df, A, P, entry_state="ENTRY_TRIGGER", hold_days=20, start=None, end=None):
    close = A["close"]
    trades = []
    i, n = (start or P.min_bars), (end or A["n"])
    last_exit = -1
    while i < n - 1:
        if i <= last_exit:
            i += 1
            continue
        r = E.evaluate_at(A, i, P)
        if r["state"] == entry_state:
            entry_i = i + 1  # enter next bar's open -> avoid look-ahead
            exit_i = min(entry_i + hold_days, n - 1)
            if entry_i < n:
                ret = (close[exit_i] - A["open"][entry_i]) / A["open"][entry_i] * 100
                trades.append({"entry_date": str(A["dates"][entry_i].date()), "exit_date": str(A["dates"][exit_i].date()),
                                "return_pct": round(float(ret), 2), "hold_days": exit_i - entry_i})
                last_exit = exit_i  # signal-overlap control: no new trade until this one exits
        i += 1
    return trades


def _stats(trades):
    if not trades:
        return {"n_trades": 0}
    r = np.array([t["return_pct"] for t in trades])
    return {"n_trades": len(r), "win_rate": round(float((r > 0).mean() * 100), 1),
            "avg_return_pct": round(float(r.mean()), 2), "median_return_pct": round(float(np.median(r)), 2),
            "std_pct": round(float(r.std()), 2), "best_pct": round(float(r.max()), 2), "worst_pct": round(float(r.min()), 2)}


def run_backtest(price_data, params=None, hold_days=20, train_frac=0.6, fee_bps=10, slippage_bps=10):
    """price_data: dict[symbol -> OHLCV DataFrame]. Returns walk-forward (train/test split,
    fees+slippage applied) and full-sample results across every symbol."""
    P = params or E.PullbackParams()
    cost = (fee_bps + slippage_bps) / 100.0
    all_trades, train_trades, test_trades = [], [], []
    for sym, df in price_data.items():
        try:
            A = E.prepare(df, P)
        except Exception:
            continue
        if A["n"] < P.min_bars + 30:
            continue
        split = P.min_bars + int((A["n"] - P.min_bars) * train_frac)
        tr = _trades_from_signals(df, A, P, hold_days=hold_days, start=P.min_bars, end=split)
        te = _trades_from_signals(df, A, P, hold_days=hold_days, start=split, end=A["n"])
        for t in tr + te:
            t["return_pct"] = round(t["return_pct"] - cost, 2)
            t["symbol"] = sym
        train_trades += tr
        test_trades += te
        all_trades += tr + te
    return {"train": _stats(train_trades), "test": _stats(test_trades), "full_sample": _stats(all_trades),
            "trades": all_trades, "cost_bps_applied": fee_bps + slippage_bps,
            "note": "train/test is a simple time-ordered walk-forward split per symbol; look-ahead is controlled "
                    "by entering at the bar AFTER the signal bar's close, and signal overlap is controlled by "
                    "blocking new entries until the prior trade's hold period ends."}


def regime_split(price_data, regime_labels, params=None, hold_days=20):
    """regime_labels: dict[symbol -> pd.Series of {'bull','sideways','correction','bear'} aligned to df.index]."""
    P = params or E.PullbackParams()
    buckets = {"bull": [], "sideways": [], "correction": [], "bear": []}
    for sym, df in price_data.items():
        A = E.prepare(df, P)
        if A["n"] < P.min_bars + 10:
            continue
        trades = _trades_from_signals(df, A, P, hold_days=hold_days, start=P.min_bars)
        lab = regime_labels.get(sym)
        for t in trades:
            t["symbol"] = sym
            if lab is not None and t["entry_date"] in lab.index.astype(str):
                r = lab.loc[t["entry_date"]]
                buckets.get(r, buckets.setdefault(r, [])).append(t)
    return {k: _stats(v) for k, v in buckets.items()}


def parameter_sensitivity(price_data, base_params=None, hold_days=20, perturbations=None):
    """Perturb one parameter at a time (default: the spec's own '18/22 EMA, nearby 50DMA
    alternatives' example) and report how full-sample stats move."""
    P = base_params or E.PullbackParams()
    perturbations = perturbations or {"ema_fast": [18, 20, 22], "sma_mid": [45, 50, 55],
                                      "entry_score": [75, 80, 85]}
    out = {}
    for name, values in perturbations.items():
        out[name] = []
        for v in values:
            Pv = replace(P, **{name: v})
            trades = []
            for sym, df in price_data.items():
                A = E.prepare(df, Pv)
                if A["n"] >= Pv.min_bars + 10:
                    trades += _trades_from_signals(df, A, Pv, hold_days=hold_days, start=Pv.min_bars)
            out[name].append({"value": v, **_stats(trades)})
    return out


def ablation_test(price_data, base_params=None, hold_days=20):
    """Remove one factor group at a time (section 20's 'ablation test') and compare
    full-sample stats to the baseline (all factors in)."""
    P = base_params or E.PullbackParams()

    def run(drop):
        trades = []
        for sym, df in price_data.items():
            A = E.prepare(df, P)
            if A["n"] < P.min_bars + 10:
                continue
            i, n, last_exit = P.min_bars, A["n"], -1
            while i < n - 1:
                if i <= last_exit:
                    i += 1
                    continue
                r = E.evaluate_at(A, i, P, want_parts=True)
                if "_parts" in r:
                    score, state, _ = E.finalize(r["_parts"]["pts"], r["_parts"]["ctx"], P, drop=drop)
                else:
                    state = r["state"]
                if state == "ENTRY_TRIGGER":
                    entry_i = i + 1
                    exit_i = min(entry_i + hold_days, n - 1)
                    if entry_i < n:
                        ret = (A["close"][exit_i] - A["open"][entry_i]) / A["open"][entry_i] * 100
                        trades.append({"return_pct": round(float(ret), 2)})
                        last_exit = exit_i
                i += 1
        return _stats(trades)

    baseline = run(())
    return {"baseline": baseline, "without": {g: run(fs) for g, fs in E.ABLATION_GROUPS.items()}}
