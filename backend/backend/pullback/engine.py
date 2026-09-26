"""Pull Back setup engine - technical only.

Implements the 5-stage sequence from the spec: Trend -> Impulse -> Controlled pullback
-> Seller exhaustion -> Trigger, with hard filters HF01-HF10, support-zone engine,
momentum / volume / ATR engines, reversal triggers, 100-point score, anti-chasing
filter, invalidation rules and the 0-6 state machine.

All logic is causal: evaluate_at(A, t) only reads bars <= t (pivots need k bars of
confirmation and are masked accordingly), so the same code powers the live screener
and the walk-forward backtest.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from . import indicators as I

TIER = {"none": 0, "early": 1, "standard": 2, "confirmed": 3, "strong": 4}
MACHINE = ["NO_SETUP", "TREND_QUALIFIED", "IMPULSE_CONFIRMED", "PULLBACK_DEVELOPING",
           "SUPPORT_TEST", "REVERSAL_FORMING", "ENTRY_TRIGGER"]


@dataclass
class PullbackParams:
    # indicators
    ema_fast: int = 20
    sma_mid: int = 50
    sma_long: int = 200
    st_period: int = 10
    st_mult: float = 3.0
    min_bars: int = 230                      # HF10
    slope_lookback: int = 20                 # HF02
    sma50_slope_lookback: int = 10
    # swing / impulse (HF06, parameterised)
    swing_lookback: int = 60
    impulse_lookback: int = 60
    impulse_min_pct: float = 12.0
    impulse_min_atr: float = 4.0
    impulse_min_bars: int = 5
    impulse_max_day_share: float = 0.5       # avoid one-day spikes
    impulse_full_pct: float = 25.0
    # pullback zones (HF07)
    min_pullback_pct: float = 0.5
    shallow_pct: float = 3.0
    healthy_pct: float = 8.0
    moderate_pct: float = 12.0
    deep_pct: float = 15.0
    max_pullback_pct: float = 20.0
    max_pullback_atr: float = 10.0
    deep_atr: float = 6.0
    # support (HF09)
    support_atr_band: float = 1.0
    support_tol_pct: float = 3.0
    breakout_lookback: int = 80
    struct_lookback: int = 150
    hvn_lookback: int = 120
    # invalidation
    death_cross_lookback: int = 10
    decisive_below_pct: float = 1.0
    breakdown_vol_ratio: float = 1.5
    atr_expansion: float = 1.5
    weekly_bear_mult: float = 0.6
    distribution_penalty: float = 10.0
    macd_bear_penalty: float = 8.0
    # triggers
    trigger_fresh_bars: int = 3
    reversal_vol_ratio: float = 1.2
    strong_vol_ratio: float = 1.5
    min_trigger_tier: str = "standard"       # early | standard | confirmed | strong
    adx_min: float = 20.0
    # anti-chasing (section 18)
    ext_ema20_caution: float = 5.0
    ext_ema20_avoid: float = 12.0
    ext_sma50_caution: float = 8.0
    ext_sma50_avoid: float = 15.0
    ext_rsi_caution: float = 65.0
    ext_rsi_avoid: float = 70.0
    ext_atr_caution: float = 2.0
    ext_atr_avoid: float = 3.5
    near_52w_pct: float = 2.0
    # classification
    entry_score: float = 80.0
    setup_score: float = 70.0
    developing_score: float = 60.0
    watch_score: float = 50.0


# ---- section 13 point model (raw points) and section 12 bucket weights ----------
MAX_PTS = dict(close_200dma=4, slope_200=4, ma_50_200=4, supertrend=4, weekly=4,
               impulse=10,
               pb_depth=6, pb_duration=4, support_prox=6,
               support_confluence=5, breakout_retest=4,
               rsi_stab=5, rsi_regime=3, macd_stab=4, macd_cross=3,
               vol_contract=4, rev_volume=3, obv_ad=3,
               higher_low=4, lh_break=4, rev_candle=2)
BUCKETS = {  # bucket -> (weight, [factors])
    "trend_regime": (20, ["close_200dma", "slope_200", "ma_50_200", "supertrend", "weekly"]),
    "prior_impulse": (10, ["impulse"]),
    "pullback_quality": (20, ["pb_depth", "pb_duration", "support_prox"]),
    "support_confluence": (15, ["support_confluence", "breakout_retest"]),
    "momentum": (15, ["rsi_stab", "rsi_regime", "macd_stab", "macd_cross"]),
    "volume_accumulation": (10, ["vol_contract", "rev_volume", "obv_ad"]),
    "price_action_trigger": (10, ["higher_low", "lh_break", "rev_candle"]),
}
ABLATION_GROUPS = {"rsi": ["rsi_stab", "rsi_regime"], "macd": ["macd_stab", "macd_cross"],
                   "volume": ["vol_contract", "rev_volume", "obv_ad"], "support_confluence": ["support_confluence", "breakout_retest"],
                   "price_action": ["higher_low", "lh_break", "rev_candle"], "weekly": ["weekly"], "supertrend": ["supertrend"]}


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _ok(x):
    return x is not None and np.isfinite(x)


# ------------------------------------------------------------------ preparation
def prepare(df, P=None):
    P = P or PullbackParams()
    df = I.normalize_ohlcv(df)
    d = pd.DataFrame(index=df.index)
    for k in ("open", "high", "low", "close", "volume"):
        d[k] = df[k]
    c = df["close"]
    d["ema20"], d["sma50"], d["sma200"] = I.ema(c, P.ema_fast), I.sma(c, P.sma_mid), I.sma(c, P.sma_long)
    d["rsi"] = I.rsi(c, 14)
    d["macd"], d["macd_sig"], d["macd_hist"] = I.macd(c)
    d["adx"], d["pdi"], d["mdi"] = I.adx(df, 14)
    d["tr"] = I.true_range(df)
    d["atr14"], d["atr50"] = I.wilder(d["tr"], 14), I.wilder(d["tr"], 50)
    d["st"], d["st_dir"] = I.supertrend(df, P.st_period, P.st_mult)
    d["vol20"] = I.sma(df["volume"], 20)
    d["vr"] = df["volume"] / d["vol20"].replace(0, np.nan)
    d["obv"], d["ad"] = I.obv(df), I.ad_line(df)
    d["roc20"] = c.pct_change(20) * 100
    d["stk"], d["std"] = I.stochastic(df)
    d["hi52"] = df["high"].rolling(252, min_periods=60).max()
    # weekly regime, mapped to daily using only weeks completed on/before each date
    wk = I.weekly_frame(df)
    if len(wk) >= 10:
        w = pd.DataFrame(index=wk.index)
        w["close"], w["ema20"], w["ema50"] = wk["close"], I.ema(wk["close"], 20), I.ema(wk["close"], 50)
        w["sma200"], w["rsi"] = I.sma(wk["close"], 200), I.rsi(wk["close"], 14)
        w["st_dir"] = I.supertrend(wk, 10, 3.0)[1].astype(float)
        ix = np.searchsorted(wk.index.values, df.index.values, side="right") - 1
        for k in w.columns:
            col = w[k].to_numpy(float)
            d["w_" + k] = np.where(ix >= 0, col[np.clip(ix, 0, None)], np.nan)
    else:
        for k in ("close", "ema20", "ema50", "sma200", "rsi", "st_dir"):
            d["w_" + k] = np.nan
    A = {k: d[k].to_numpy(float) for k in d.columns}
    A["ph"], A["pl"] = I.pivots(A["high"], A["low"], 2)
    A["dates"], A["n"] = d.index, len(d)
    return A


# --------------------------------------------------------------- helper detectors
def _patterns(A, i, P):
    O, H, L, C = A["open"], A["high"], A["low"], A["close"]
    o, h, l, c = O[i], H[i], L[i], C[i]
    rng, body = h - l, abs(c - o)
    if rng <= 0:
        return []
    lw = min(o, c) - l
    p = []
    if C[i - 1] < O[i - 1] and c > o and o <= C[i - 1] and c >= O[i - 1]:
        p.append("bullish_engulfing")
    if lw >= 2 * body and lw >= 0.5 * rng and c >= l + 0.6 * rng:
        p.append("hammer")
    elif lw >= 0.6 * rng and c >= l + 0.5 * rng:
        p.append("bullish_pin_bar")
    if H[i - 1] < H[i - 2] and L[i - 1] > L[i - 2] and c > H[i - 1]:
        p.append("inside_bar_breakout")
    a = A["atr14"][i - 1]
    if c > o and c > C[i - 1] and _ok(a) and (c - o) >= 0.8 * a and _ok(A["vr"][i]) and A["vr"][i] >= P.reversal_vol_ratio:
        p.append("high_volume_bullish_close")
    return p


def _crossed_up(a, b, t, n):
    return any(a[i] > b[i] and a[i - 1] <= b[i - 1] for i in range(t - n + 1, t + 1))


def _levels(A, t, P, sh_idx, sh, il_idx, il, atr):
    """Support reference levels (section 6 / 11)."""
    H, L, C, V = A["high"], A["low"], A["close"], A["volume"]
    c, band = C[t], P.support_atr_band * atr
    lv = {"ema20": A["ema20"][t], "sma50": A["sma50"][t]}
    pre_lo = max(0, il_idx - P.breakout_lookback)
    if il_idx - pre_lo >= 10:
        b = H[pre_lo:il_idx + 1].max()
        if sh > b * 1.01:
            lv["breakout"] = b
    a0 = max(0, t - P.struct_lookback)
    if sh_idx - 3 > a0:
        idx = [i for i in np.flatnonzero(A["ph"][a0:sh_idx - 3]) + a0 if H[i] < sh * 0.999]
        if idx:
            j = min(idx, key=lambda i: abs(H[i] - c))
            lv["prior_swing_high"] = H[j]
    idx = np.flatnonzero(A["pl"][a0:max(a0, t - 2)]) + a0
    if len(idx) >= 2:
        pls = L[idx]
        best = None
        for p in pls:
            m = pls[np.abs(pls - p) <= band]
            if len(m) >= 2 and (best is None or abs(m.mean() - c) < abs(best - c)):
                best = m.mean()
        if best is not None:
            lv["horizontal"] = best
    w0 = max(0, t - P.hvn_lookback + 1)
    lo_p, hi_p = L[w0:t + 1].min(), H[w0:t + 1].max()
    if hi_p > lo_p:
        edges = np.linspace(lo_p, hi_p, 25)
        b = np.clip(np.digitize(C[w0:t + 1], edges) - 1, 0, 23)
        vb = np.bincount(b, weights=V[w0:t + 1], minlength=24)
        if (vb > 0).any():
            cent = (edges[:-1] + edges[1:]) / 2
            hv = cent[vb >= 1.5 * vb[vb > 0].mean()]
            if len(hv):
                lv["hvn"] = hv[np.argmin(np.abs(hv - c))]
    tp = (H[il_idx:t + 1] + L[il_idx:t + 1] + C[il_idx:t + 1]) / 3
    vv = V[il_idx:t + 1]
    if vv.sum() > 0:
        lv["avwap"] = float((tp * vv).sum() / vv.sum())
    for r in (0.382, 0.5, 0.618):
        lv[f"fib_{r}"] = sh - r * (sh - il)
    return lv


GROUP = {"ema20": "moving_average", "sma50": "moving_average", "breakout": "structure", "prior_swing_high": "structure",
         "horizontal": "structure", "avwap": "vwap", "hvn": "volume_profile"}


def _group(name):
    return "fibonacci" if name.startswith("fib_") else GROUP[name]


# ------------------------------------------------------------------ scoring/state
def finalize(pts, ctx, P, drop=()):
    """pts: raw factor points; ctx: flags. Returns (score, state, bucket_scores)."""
    drop = set(drop)
    buckets, total = {}, 0.0
    for name, (w, facs) in BUCKETS.items():
        use = [f for f in facs if f not in drop]
        mx = sum(MAX_PTS[f] for f in use)
        s = (sum(pts.get(f, 0.0) for f in use) / mx * w) if mx else 0.0
        buckets[name] = round(s, 2)
        total += s
    total -= ctx["penalties"]
    if ctx["weekly"] == "BEARISH":
        total *= P.weekly_bear_mult
    total = _clip(total, 0, 100)
    hf = ctx["hf"]
    if any(hf.get(k) is False for k in ("HF01", "HF03", "HF08", "HF10")):
        total = min(total, P.watch_score - 1)
    elif any(v is False for v in hf.values()):
        total = min(total, P.developing_score - 1)
    if not ctx["pullback_active"]:
        total = min(total, P.watch_score - 1)
    if ctx["invalid"]:
        state = "INVALIDATED" if ctx["was_trending"] else "NO_SETUP"
    elif total >= P.entry_score and ctx["machine_idx"] >= 6 and ctx["ext"] != "AVOID_CHASING" \
            and ctx["weekly"] != "BEARISH":
        state = "ENTRY_TRIGGER"
    elif total >= P.setup_score:
        state = "PULLBACK_SETUP"
    elif total >= P.developing_score:
        state = "PULLBACK_DEVELOPING"
    elif total >= P.watch_score:
        state = "WATCHLIST"
    else:
        state = "NO_SETUP"
    if ctx["weekly"] == "BEARISH" and state in ("PULLBACK_SETUP", "PULLBACK_DEVELOPING"):
        state = "WATCHLIST"                    # weekly bearish: downgrade heavily
    if ctx["ext"] == "AVOID_CHASING" and state in ("PULLBACK_SETUP", "PULLBACK_DEVELOPING"):
        state = "WATCHLIST"
    return round(total, 1), state, buckets


# ------------------------------------------------------------------- main evaluator
def evaluate_at(A, t, P=None, want_parts=False):
    P = P or PullbackParams()
    O, H, L, C, V = A["open"], A["high"], A["low"], A["close"], A["volume"]
    ema20, sma50, sma200 = A["ema20"], A["sma50"], A["sma200"]
    rsi, hist, macd, msig = A["rsi"], A["macd_hist"], A["macd"], A["macd_sig"]
    vr, atrs = A["vr"], A["atr14"]
    c, atr = C[t], atrs[t]
    base = {"date": str(A["dates"][t].date()), "close": _f(c)}
    need = (ema20[t], sma50[t], sma200[t], atr, rsi[t], hist[t], A["atr50"][t])
    if t + 1 < P.min_bars or not all(np.isfinite(need)) or not np.isfinite(sma200[t - P.slope_lookback]):
        return {**base, "state": "NO_SETUP", "score": 0, "machine_state": "NO_SETUP",
                "hard_filters": {"HF10": False}, "warnings": ["Insufficient history for all indicators (HF10)"]}
    band = P.support_atr_band * atr
    warnings = []

    # ---- swing high, impulse, pullback low (section 5)
    lo = max(0, t - P.swing_lookback + 1)
    seg = H[lo:t + 1]
    sh_idx = lo + len(seg) - 1 - int(np.argmax(seg[::-1]))
    sh = H[sh_idx]
    pb_bars = t - sh_idx
    pb_pct, pb_atr = (sh - c) / sh * 100, (sh - c) / atr
    il_lo = max(0, sh_idx - P.impulse_lookback)
    sg = L[il_lo:sh_idx + 1]
    il_idx = il_lo + len(sg) - 1 - int(np.argmin(sg[::-1]))
    il = L[il_idx]
    imp_pct, imp_bars = (sh - il) / il * 100, sh_idx - il_idx
    a_sh = atrs[sh_idx]
    imp_atr = (sh - il) / a_sh if np.isfinite(a_sh) and a_sh > 0 else np.nan
    leg = C[il_idx:sh_idx + 1]
    total_move = leg[-1] - leg[0]
    share = float(np.diff(leg).max() / total_move) if len(leg) > 1 and total_move > 0 else 1.0
    impulse_ok = (imp_pct >= P.impulse_min_pct and imp_bars >= P.impulse_min_bars and share <= P.impulse_max_day_share
                  and _ok(imp_atr) and imp_atr >= P.impulse_min_atr)
    s2 = L[sh_idx:t + 1]
    pl_idx = sh_idx + len(s2) - 1 - int(np.argmin(s2[::-1]))
    pl = L[pl_idx]
    bounce_pct = (c - pl) / pl * 100
    pullback_active = pb_bars >= 1 and pb_pct >= P.min_pullback_pct

    # ---- support zones (section 6/11)
    lv = _levels(A, t, P, sh_idx, sh, il_idx, il, atr)
    near = {k: (abs(c - x) <= band or abs(L[t] - x) <= band) for k, x in lv.items()}
    refs = [k for k, v in near.items() if v]
    groups = sorted({_group(k) for k in refs})
    dist_atr = {k: abs(c - x) / atr for k, x in lv.items()}
    nearest = min(dist_atr.values())
    fib786 = sh - 0.786 * (sh - il)
    if abs(c - fib786) <= band:
        warnings.append("Price at 78.6% Fibonacci retracement: usually too deep for a normal pullback")
    brk = lv.get("breakout")
    retest_hold = brk is not None and near["breakout"] and not (C[sh_idx:t + 1] < brk - band).any()

    # ---- trend hard filters (section 3)
    dcross = any(sma50[i - 1] >= A["sma200"][i - 1] and sma50[i] < sma200[i]
                 for i in range(t - P.death_cross_lookback + 1, t + 1) if np.isfinite(sma200[i - 1]))
    slope200 = sma200[t] - sma200[t - P.slope_lookback]
    w_close, w_e50, w_e20 = A["w_close"][t], A["w_ema50"][t], A["w_ema20"][t]
    w_st, w_s200 = A["w_st_dir"][t], A["w_sma200"][t]
    if not np.isfinite(w_close) or not np.isfinite(w_e50):
        weekly, hf05 = "UNKNOWN", None
    else:
        hf05 = bool(w_close > w_e50)
        if not hf05 and (w_st == -1 or w_e20 < w_e50):
            weekly = "BEARISH"
        elif hf05 and w_e20 >= w_e50 and w_st == 1 and not (np.isfinite(w_s200) and w_close < w_s200):
            weekly = "BULLISH"
        else:
            weekly = "NEUTRAL"
    hf = {"HF01": bool(c > sma200[t]), "HF02": bool(slope200 > 0), "HF03": bool(sma50[t] > sma200[t]),
          "HF04": bool(A["st_dir"][t] == 1), "HF05": hf05, "HF06": bool(impulse_ok),
          "HF07": bool(pb_pct < P.max_pullback_pct), "HF08": not dcross,
          "HF09": bool(c >= sma50[t] * (1 - P.support_tol_pct / 100) or (len(refs) > 0 and c >= sma200[t])),
          "HF10": True}
    was_trending = bool(np.any((C[t - 39:t + 1] > sma200[t - 39:t + 1]) & (sma50[t - 39:t + 1] > sma200[t - 39:t + 1])))

    # ---- momentum (section 7)
    rsi_imp, rsi_rise = rsi[t] > rsi[t - 1], rsi[t] > rsi[t - 3]
    macd_imp = hist[t] > hist[t - 1]
    macd_expand_bear = hist[t] < hist[t - 1] < hist[t - 2] < 0
    persist_weak = bool((rsi[t - 4:t + 1] < 40).all() and rsi[t] <= rsi[t - 4])
    bull_cross = _crossed_up(macd, msig, t, 5)
    roc = A["roc20"]
    roc_imp = bool(np.isfinite(roc[t]) and np.isfinite(roc[t - 3]) and (roc[t] > roc[t - 3] or roc[t] > 0))
    di_imp = bool(A["pdi"][t] > A["pdi"][t - 3])
    stoch_turn = bool(np.isfinite(A["stk"][t]) and A["stk"][t] > A["std"][t] and np.nanmin(A["stk"][t - 5:t + 1]) < 30)

    # ---- volume (section 8)
    pbv = vr[sh_idx + 1:t + 1]
    pbv = pbv[np.isfinite(pbv)]
    mean_pbv = float(pbv.mean()) if len(pbv) else np.nan
    dn = [i for i in range(sh_idx + 1, t + 1) if C[i] < C[i - 1]]
    later_lower = None
    if len(dn) >= 4:
        h_ = len(dn) // 2
        later_lower = bool(V[dn[h_:]].mean() < V[dn[:h_]].mean())
    span = max(1, pb_bars)
    v20 = A["vol20"][t]
    obv_z = (A["obv"][t] - A["obv"][sh_idx]) / (v20 * span) if v20 > 0 else 0.0
    ad_z = (A["ad"][t] - A["ad"][sh_idx]) / (v20 * span) if v20 > 0 else 0.0
    obv_state = "RISING" if obv_z > 0.05 else ("STABLE" if obv_z > -0.3 else "FALLING")
    w1 = max(0, sh_idx - 60)
    diverg = False
    if sh_idx - 10 > w1 and H[sh_idx] > H[w1:sh_idx - 10].max() and A["obv"][sh_idx] < A["obv"][w1:sh_idx - 10].max():
        diverg = True
        warnings.append("Bearish divergence: price made a higher high while OBV did not")
    breakdown_vol = any(C[i] < sma50[i] - band and vr[i] >= P.breakdown_vol_ratio and C[i] < C[i - 1]
                        for i in range(t - 2, t + 1) if np.isfinite(vr[i]))
    distribution = any(C[i] < C[i - 1] and C[i] < O[i] and vr[i] >= P.breakdown_vol_ratio and
                       min(abs(C[i] - x) for x in (sma50[i], ema20[i])) <= band
                       for i in range(t - 2, t + 1) if np.isfinite(vr[i])) and bool(refs)
    vol_confirm = bool(np.nanmax(vr[t - 2:t + 1]) >= P.reversal_vol_ratio)

    # ---- ATR / volatility (section 9)
    atr_ratio = atr / A["atr50"][t]
    tr = A["tr"]
    compress = bool(np.nanmean(tr[t - 4:t + 1]) < 0.85 * np.nanmean(tr[t - 14:t - 4]))
    if atr_ratio >= P.atr_expansion:
        warnings.append("ATR expansion: volatility shock (ATR14/ATR50 = %.2f)" % atr_ratio)

    # ---- price action / triggers (section 10, 17)
    ph_after = [i for i in np.flatnonzero(A["ph"][sh_idx + 1:max(sh_idx + 1, t - 1)]) + sh_idx + 1 if i <= t - 2]
    lh_level = np.nan
    if ph_after:
        lh_level = H[ph_after[-1]]
    elif pb_bars >= 2:
        lh_level = H[max(sh_idx + 1, t - 5):t].max()
    fr = P.trigger_fresh_bars
    lh_break = bool(np.isfinite(lh_level) and c > lh_level and
                    any(C[i] > lh_level and C[i - 1] <= lh_level for i in range(t - fr + 1, t + 1)))
    tl_break, tl_val = False, np.nan
    pk = [sh_idx] + ph_after
    if len(pk) >= 2 and H[pk[-1]] < H[pk[-2]]:
        x1, x2 = pk[-2], pk[-1]
        sl_ = (H[x2] - H[x1]) / (x2 - x1)
        line = lambda x: H[x2] + sl_ * (x - x2)
        tl_val = line(t)
        tl_break = bool(c > line(t) and any(C[t - k] <= line(t - k) for k in range(1, fr + 1)))
    pls_after = [i for i in np.flatnonzero(A["pl"][pl_idx + 1:max(pl_idx + 1, t - 1)]) + pl_idx + 1 if i <= t - 2]
    higher_low = bool(pls_after and any(L[i] > pl for i in pls_after) and L[pl_idx + 1:t + 1].min() > pl)
    hl_dev = bool(not higher_low and t - pl_idx >= 2 and L[pl_idx + 1:t + 1].min() > pl and c > C[t - 1])
    pats = []
    for off in (0, 1):
        pats += [p for p in _patterns(A, t - off, P) if p not in pats]
    ema_reclaim = bool(c > ema20[t] and any(C[t - k] <= ema20[t - k] for k in range(1, fr + 1)) and pullback_active)
    sma_reclaim = bool(c > sma50[t] and any(C[t - k] < sma50[t - k] for k in range(1, 6)) and pullback_active)
    at_support = bool(refs)
    support_hold = bool(at_support and c >= pl and c > sma200[t] * (1 - P.decisive_below_pct / 100))
    improving = bool(rsi_imp or macd_imp)
    early = support_hold and (bool(pats) or ema_reclaim or sma_reclaim) and improving
    standard = (lh_break or tl_break) and improving
    confirmed = standard and c > ema20[t] and (not np.isfinite(lh_level) or c > lh_level) and vol_confirm
    strong = confirmed and higher_low and len(groups) >= 2 and (hist[t] > 0 or bull_cross or (rsi[t] > 50 and rsi_imp))
    tier = "strong" if strong else "confirmed" if confirmed else "standard" if standard else "early" if early else "none"
    trigger_ok = TIER[tier] >= TIER[P.min_trigger_tier] and tier != "none"

    # ---- anti-chasing / overextension (section 18)
    d20, d50 = (c - ema20[t]) / ema20[t] * 100, (c - sma50[t]) / sma50[t] * 100
    d20_atr = (c - ema20[t]) / atr
    off52 = (A["hi52"][t] - c) / A["hi52"][t] * 100 if np.isfinite(A["hi52"][t]) else np.nan
    exp_rng = int((tr[t - 4:t + 1] > 1.5 * atr).sum())
    spike = bool(vr[t] >= 2.5 and rsi[t] > P.ext_rsi_avoid)
    avoid = (d20 > P.ext_ema20_avoid or d50 > P.ext_sma50_avoid or d20_atr > P.ext_atr_avoid or
             (rsi[t] > P.ext_rsi_avoid and d20 > P.ext_ema20_caution) or exp_rng >= 3 or spike or
             (np.isfinite(off52) and off52 < 1.0 and d20 > 8))
    caution = (d20 > P.ext_ema20_caution or d50 > P.ext_sma50_caution or rsi[t] > P.ext_rsi_caution or
               d20_atr > P.ext_atr_caution or (np.isfinite(off52) and off52 < P.near_52w_pct) or
               vr[t] >= 2.0 or exp_rng >= 2)
    ext = "AVOID_CHASING" if avoid else "CAUTION" if caution else "HEALTHY"

    # ---- invalidation (section 14)
    inv = []
    below200 = C[t - 1:t + 1] < sma200[t - 1:t + 1]
    if c < sma200[t] * (1 - P.decisive_below_pct / 100) or below200.all():
        inv.append("Daily close decisively below 200 DMA")
    if dcross:
        inv.append("50 DMA crossed below 200 DMA")
    st = A["st_dir"]
    if st[t] == -1 and 1 in st[t - 5:t]:
        flip = max(i for i in range(t - 5, t) if st[i] == 1) + 1
        if c < C[flip]:
            inv.append("Supertrend flipped red and price continues lower")
    prior_pl = [i for i in np.flatnonzero(A["pl"][il_idx:max(il_idx, sh_idx - 2)]) + il_idx]
    major_low = L[prior_pl[-1]] if prior_pl else il
    if any(C[i] < major_low and vr[i] >= P.breakdown_vol_ratio for i in range(t - 1, t + 1) if np.isfinite(vr[i])):
        inv.append("Break of major swing low on strong volume")
    elif (C[t - 2:t + 1] < major_low).all():
        inv.append("Lower low below major swing with no quick reclaim")
    if pb_pct >= P.max_pullback_pct or pb_atr >= P.max_pullback_atr:
        inv.append("Pullback exceeds configured maximum depth")
    support_broken = bool(c < sma50[t] - band or c < major_low)
    if atr_ratio >= P.atr_expansion and support_broken:
        inv.append("ATR expanded sharply while support broke")
    if macd_expand_bear and support_broken:
        inv.append("Bearish MACD expansion with price breakdown")
    pls_seq = [i for i in np.flatnonzero(A["pl"][sh_idx:max(sh_idx, t - 1)]) + sh_idx if i <= t - 2]
    if len(pls_seq) >= 2 and len(pk) >= 2 and L[pls_seq[-1]] < L[pls_seq[-2]] and H[pk[-1]] < H[pk[-2]] \
            and c < L[pls_seq[-1]] and c < sma50[t]:
        inv.append("Lower low, lower high, then another breakdown")
    if weekly == "BEARISH":
        warnings.append("Weekly regime bearish: daily signal must not be used alone (score heavily downgraded)")
    penalties = (P.distribution_penalty if distribution else 0) + (P.macd_bear_penalty if macd_expand_bear else 0)
    if distribution:
        warnings.append("Heavy-volume distribution near support")
    if breakdown_vol:
        warnings.append("Abnormal-volume close below support")
    if macd_expand_bear:
        warnings.append("MACD histogram expanding negative")
    if persist_weak:
        warnings.append("RSI persistently below 40 with no recovery")
    if A["adx"][t] < P.adx_min:
        warnings.append("ADX below %.0f: weak trend strength" % P.adx_min)
    if A["pdi"][t] < A["mdi"][t]:
        warnings.append("-DI above +DI: supply dominating")
    if not np.isfinite(A["w_close"][t]):
        warnings.append("Weekly history too short for weekly regime check")

    # ---- component scores (section 13), each normalised 0-1 x points
    p = {}
    p["close_200dma"] = 4.0 * hf["HF01"]
    ds = slope200 / sma200[t]
    p["slope_200"] = 4.0 if ds > 0.002 else 2.0 if abs(ds) <= 0.002 else 0.0
    p["ma_50_200"] = 4.0 * hf["HF03"]
    p["supertrend"] = 4.0 * hf["HF04"]
    p["weekly"] = {"BULLISH": 4.0, "NEUTRAL": 2.0, "BEARISH": 0.0, "UNKNOWN": 2.0}[weekly]
    p["impulse"] = 10.0 * (0.0 if imp_bars < 3 else (0.4 * _clip(imp_pct / P.impulse_full_pct) + 0.3 * _clip(imp_bars / 20) +
                                                    0.3 * (1 - _clip((share - 0.15) / 0.35)))) if impulse_ok else \
        10.0 * 0.25 * _clip(imp_pct / P.impulse_min_pct) * (imp_bars >= 3)
    strong_support_reversal = bool(len(groups) >= 2 and (pats or higher_low or lh_break))
    if pb_pct < P.shallow_pct:
        pz = 0.8 if (len(groups) >= 1 and TIER[tier] >= 2) else 0.5
    elif pb_pct < P.healthy_pct:
        pz = 1.0
    elif pb_pct < P.moderate_pct:
        pz = 0.85 if c >= sma50[t] * 0.99 or len(groups) else 0.5
    elif pb_pct < P.deep_pct:
        pz = 0.6 if (pats or higher_low or lh_break) else 0.3
    elif pb_pct < P.max_pullback_pct:
        pz = 0.5 if strong_support_reversal else 0.25
    else:
        pz = 0.0
    az = 1.0 if 1.0 <= pb_atr <= 5.0 else 0.6 if pb_atr < 1.0 else 0.7 if pb_atr <= P.deep_atr else 0.3 if pb_atr < P.max_pullback_atr else 0.0
    p["pb_depth"] = 6.0 * (0.5 * pz + 0.5 * az) * pullback_active
    crash = pb_pct >= 10 and pb_bars <= 3
    dz = 0.0 if crash else 1.0 if 3 <= pb_bars <= 15 else 0.4 if pb_bars < 3 else 0.7 if pb_bars <= 30 else 0.4
    p["pb_duration"] = 4.0 * dz * pullback_active
    p["support_prox"] = 6.0 * (1.0 if nearest <= 1.0 else 0.7 if nearest <= 1.5 else 0.3 if nearest <= 2.5 else 0.0) * pullback_active
    p["support_confluence"] = 5.0 * {0: 0.0, 1: 0.4, 2: 0.8}.get(len(groups), 1.0) * pullback_active
    p["breakout_retest"] = 4.0 * (1.0 if retest_hold else 0.4 if (brk is not None and near["breakout"]) else 0.0) * pullback_active
    stab = (0.4 * rsi_imp + 0.2 * rsi_rise) + (0.4 if 45 <= rsi[t] <= 60 else 0.25 if 40 <= rsi[t] < 45 or 60 < rsi[t] <= 65 else 0.0)
    p["rsi_stab"] = 0.0 if persist_weak else 5.0 * _clip(stab)
    p["rsi_regime"] = 3.0 * (1.0 if rsi[t] >= 50 else 0.6 if rsi[t] >= 45 else 0.3 if rsi[t] >= 40 else 0.0)
    if macd_expand_bear:
        ms = 0.0
    else:
        ms = 0.35 * macd_imp + 0.25 * ((hist[t] < 0 and hist[t] > hist[t - 1] > hist[t - 2]) or (hist[t] >= 0 and macd_imp)) + \
            0.25 * (macd[t] > macd[t - 1])
        ms += 0.15 * (roc_imp + di_imp + stoch_turn) / 3
    p["macd_stab"] = 4.0 * _clip(ms)
    p["macd_cross"] = 3.0 * ((1.0 if at_support else 0.5) if bull_cross else 0.0)
    if np.isfinite(mean_pbv):
        vc = _clip((1.1 - mean_pbv) / 0.5)
        vcs = 0.6 * vc + 0.4 * (1.0 if later_lower else 0.0 if later_lower is False else 0.5)
    else:
        vcs = 0.3
    p["vol_contract"] = 4.0 * vcs * pullback_active
    rv = 0.0
    for i in range(t - 2, t + 1):
        if C[i] > O[i] and C[i] > C[i - 1] and np.isfinite(vr[i]):
            rv = max(rv, 1.0 if vr[i] >= P.strong_vol_ratio else 0.6 if vr[i] >= P.reversal_vol_ratio else 0.0)
    p["rev_volume"] = 3.0 * rv * pullback_active
    zs = lambda z: 1.0 if z >= 0 else _clip(1 + z / 0.5)
    p["obv_ad"] = 3.0 * (0.5 * zs(obv_z) + 0.5 * zs(ad_z)) * (0.5 if diverg else 1.0) * pullback_active
    p["higher_low"] = 4.0 * (1.0 if higher_low else 0.3 if hl_dev else 0.0) * pullback_active
    p["lh_break"] = 4.0 * (1.0 if lh_break else 0.75 if tl_break else 0.5 if (ema_reclaim or sma_reclaim) else 0.0) * pullback_active
    p["rev_candle"] = 2.0 * ((1.0 if at_support else 0.5) if pats else 0.0) * pullback_active
    # ADX/DI/ROC/stochastic (spec section 4/7) act as a small momentum-bucket modifier
    mod = 0.25 * ((A["adx"][t] >= P.adx_min and A["pdi"][t] > A["mdi"][t]) + roc_imp + di_imp + stoch_turn)
    mod -= 0.5 * (A["adx"][t] < 15 and support_broken) + 0.5 * (np.isfinite(roc[t]) and roc[t] < roc[t - 3] < 0)
    p["macd_stab"] = _clip(p["macd_stab"] + mod * 0.5, 0, 4.0)
    p = {k: float(v) for k, v in p.items()}

    # ---- state machine (section 15)
    seller_ex = (not breakdown_vol) and ((np.isfinite(mean_pbv) and mean_pbv < 1.0) or bool(later_lower) or obv_state != "FALLING"
                                          or compress or improving)
    m = 0
    trend_q = hf["HF01"] and hf["HF02"] and hf["HF03"] and hf["HF04"] and hf05 is not False
    conds = [trend_q, impulse_ok, pullback_active and hf["HF07"], bool(refs) and seller_ex,
             improving and (higher_low or bool(pats) or ema_reclaim or lh_break or tl_break or sma_reclaim), trigger_ok]
    for ok_ in conds:
        if not ok_:
            break
        m += 1
    ctx = {"penalties": penalties, "weekly": weekly, "hf": hf, "pullback_active": bool(pullback_active), "invalid": bool(inv),
           "was_trending": was_trending, "machine_idx": m, "ext": ext}
    score, state, buckets = finalize(p, ctx, P)

    if state == "ENTRY_TRIGGER":
        status = "TRIGGERED_" + tier.upper()
    elif state == "INVALIDATED":
        status = "INVALIDATED"
    elif ext == "AVOID_CHASING":
        status = "AVOID_CHASING_WAIT_FOR_BASE"
    elif not pullback_active:
        status = "NO_PULLBACK_YET"
    elif not refs:
        status = "WAIT_FOR_SUPPORT"
    elif not (higher_low or pats):
        status = "WAIT_FOR_STABILIZATION"
    else:
        status = "WAIT_FOR_BREAKOUT"

    out = {
        **base, "state": state, "score": score, "machine_state": MACHINE[m], "score_buckets": buckets,
        "trend": {"close_above_200dma": hf["HF01"], "sma50_above_sma200": hf["HF03"],
                  "sma200_slope": "positive" if ds > 0.002 else "flat" if abs(ds) <= 0.002 else "negative",
                  "supertrend_daily": "GREEN" if hf["HF04"] else "RED", "weekly_regime": weekly,
                  "ema20_above_sma50": bool(ema20[t] > sma50[t]), "sma50_slope_positive": bool(sma50[t] > sma50[t - P.sma50_slope_lookback]),
                  "adx14": _f(A["adx"][t], 1), "plus_di": _f(A["pdi"][t], 1), "minus_di": _f(A["mdi"][t], 1),
                  "death_cross_recent": dcross},
        "hard_filters": hf,
        "impulse": {"low": _f(il), "high": _f(sh), "pct": _f(imp_pct, 1), "bars": int(imp_bars), "atr": _f(imp_atr, 1),
                    "max_single_day_share": _f(share), "valid": bool(impulse_ok)},
        "pullback": {"swing_high": _f(sh), "swing_high_date": str(A["dates"][sh_idx].date()), "current_close": _f(c),
                     "pullback_low": _f(pl), "pullback_pct": _f(pb_pct, 1), "pullback_atr": _f(pb_atr, 1),
                     "bounce_from_low_pct": _f(bounce_pct, 1), "duration_days": int(pb_bars),
                     "zone": ("shallow" if pb_pct < P.shallow_pct else "healthy" if pb_pct < P.healthy_pct else "moderate" if pb_pct < P.moderate_pct
                              else "deep" if pb_pct < P.deep_pct else "very_deep" if pb_pct < P.max_pullback_pct else "structural_risk"),
                     "distance_to_20ema_pct": _f(d20, 1), "distance_to_50dma_pct": _f(d50, 1),
                     "distance_to_200dma_pct": _f((c - sma200[t]) / sma200[t] * 100, 1), "atr_pct": _f(atr / c * 100, 2)},
        "support": {"near_ema20": bool(near["ema20"]), "near_sma50": bool(near["sma50"]),
                    "prior_breakout_support": bool(near.get("breakout", False)), "breakout_retest_holding": bool(retest_hold),
                    "near_prior_swing_high": bool(near.get("prior_swing_high", False)), "near_horizontal": bool(near.get("horizontal", False)),
                    "near_volume_node": bool(near.get("hvn", False)), "near_anchored_vwap": bool(near.get("avwap", False)),
                    "fib_confluence": [k for k in refs if k.startswith("fib_")], "support_confluence_count": len(refs),
                    "independent_groups": groups, "nearest_support_atr": _f(nearest, 2),
                    "levels": {k: _f(v) for k, v in lv.items()}},
        "momentum": {"rsi14": _f(rsi[t], 1), "rsi_slope": "RISING" if rsi_rise else "FALLING" if rsi[t] < rsi[t - 3] else "FLAT",
                     "macd_histogram": "IMPROVING" if macd_imp else "WEAKENING", "macd_bull_cross_5d": bool(bull_cross),
                     "roc20": _f(roc[t], 1), "stochastic_k": _f(A["stk"][t], 1), "stochastic_bull_turn": stoch_turn,
                     "adx14": _f(A["adx"][t], 1)},
        "volume": {"pullback_volume": "CONTRACTING" if (np.isfinite(mean_pbv) and mean_pbv < 1.0) else "NOT_CONTRACTING",
                   "avg_pullback_volume_ratio": _f(mean_pbv), "down_volume_later_lower": later_lower,
                   "volume_ratio_today": _f(vr[t]), "volume_ratio_trigger": _f(np.nanmax(vr[t - 2:t + 1])) if vol_confirm else None,
                   "obv": obv_state, "ad_line": "RISING" if ad_z > 0.05 else "STABLE" if ad_z > -0.3 else "FALLING",
                   "bearish_divergence": diverg, "breakdown_volume": breakdown_vol, "distribution_at_support": distribution},
        "atr": {"atr14": _f(atr), "atr_pct": _f(atr / c * 100), "pullback_depth_atr": _f(pb_atr, 1), "dist_ema20_atr": _f(abs(c - ema20[t]) / atr, 2),
                "dist_sma50_atr": _f(abs(c - sma50[t]) / atr, 2), "atr14_vs_atr50": _f(atr_ratio), "expansion": bool(atr_ratio >= P.atr_expansion),
                "range_compression": compress},
        "price_action": {"higher_low": higher_low, "bullish_reversal": bool(pats), "patterns": pats, "lower_high_level": _f(lh_level),
                         "lower_high_break": lh_break, "trendline_break": tl_break, "close_above_ema20_after_test": ema_reclaim,
                         "reclaim_50dma": sma_reclaim},
        "entry": {"trigger_tier": tier, "trigger_confirmed": bool(trigger_ok), "status": status},
        "overextension": {"status": ext, "close_vs_ema20_pct": _f(d20, 1), "close_vs_sma50_pct": _f(d50, 1), "ema20_extension_atr": _f(d20_atr, 2),
                          "pct_below_52w_high": _f(off52, 1), "recent_range_expansions_5d": exp_rng, "exhaustion_spike": spike},
        "invalidation": {"major_support_broken": support_broken, "trend_broken": bool(not hf["HF01"] or dcross), "reasons": inv},
        "warnings": warnings,
    }
    if want_parts:
        out["_parts"] = {"pts": p, "ctx": ctx, "sh_idx": int(sh_idx), "pl_price": float(pl)}
    return out


def evaluate(df, P=None):
    P = P or PullbackParams()
    A = prepare(df, P)
    if A["n"] < 5:
        return {"state": "NO_SETUP", "score": 0, "warnings": ["No data"]}
    return evaluate_at(A, A["n"] - 1, P)


def rules_config(P=None):
    P = P or PullbackParams()
    return {"params": asdict(P), "max_points": MAX_PTS,
            "buckets": {k: {"weight": w, "factors": f} for k, (w, f) in BUCKETS.items()},
            "states": ["NO_SETUP", "WATCHLIST", "PULLBACK_DEVELOPING", "PULLBACK_SETUP", "ENTRY_TRIGGER", "INVALIDATED"],
            "machine": MACHINE,
            "note": "Detailed factor points in the spec sum to 90 while bucket weights sum to 100; factor points are normalised "
                    "inside each bucket and multiplied by the bucket weight."}
