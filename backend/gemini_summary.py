"""
Gemini commentary layer.

IMPORTANT: Gemini is only ever shown the numbers the screener already computed.
It is explicitly instructed not to introduce any fact, price, or claim not in
that data - its job is to explain the ranking in plain English, not to pick
stocks or add outside "knowledge" that could be wrong or stale.

Uses the current `google-genai` SDK (`pip install google-genai`), not the
older/deprecated `google-generativeai` package.
"""
from __future__ import annotations

import json
import os

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _compact(picks: list[dict], fields: list[str]) -> list[dict]:
    out = []
    for p in picks:
        row = {"symbol": p.get("symbol"), "name": p.get("name")}
        for f in fields:
            if f in p and p[f] is not None:
                row[f] = round(p[f], 3) if isinstance(p[f], float) else p[f]
        out.append(row)
    return out


PROMPT_TEMPLATE = """You are annotating the output of a quantitative Indian-equity \
screener for someone reviewing the list, not a financial advisor giving \
recommendations.

Below is JSON for the top-ranked stocks from a rules-based, sector-neutral \
multi-factor score (0-100). Each stock includes only the fields the model \
actually used: growth, quality, value, momentum and risk sub-scores plus a \
handful of raw metrics.

Rules you must follow:
- Base every sentence ONLY on the numbers given below. Do not add facts, \
news, price targets, or opinions about the company that are not derivable \
from this data.
- For each stock, write exactly one sentence explaining WHY it scored where \
it did, referencing its strongest 1-2 sub-scores or metrics by name.
- Then write one short overall paragraph (3-4 sentences) describing the \
shape of today's list as a whole (e.g. which buckets are driving the top \
names, any sector concentration visible in the data, anything that stands \
out about coverage or risk scores).
- Do not tell the reader to buy, sell, or hold anything. Do not predict \
future returns.
- Output strict JSON only, matching this schema, no markdown fences:
{{"per_stock": {{"<symbol>": "<one sentence>", ...}}, "overall": "<paragraph>"}}

DATA:
{data}
"""


def generate_commentary(picks: list[dict]) -> dict:
    """Returns {"per_stock": {symbol: sentence}, "overall": paragraph}.

    On any failure (no API key, network issue, bad response) this returns a
    safe fallback instead of raising, so the screener page still works
    without Gemini configured.
    """
    fields = ["sector", "SCORE", "score_growth", "score_quality", "score_value",
              "score_momentum", "score_risk", "roce", "roe", "rev_growth_yoy",
              "eps_growth_yoy", "rs_6m", "rs_12m", "pe", "peg",
              "net_debt_to_ebitda", "data_coverage"]
    compact = _compact(picks, fields)

    try:
        client = _client()
        prompt = PROMPT_TEMPLATE.format(data=json.dumps(compact, indent=2))
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        text = (resp.text or "").strip()
        parsed = json.loads(text)
        if "per_stock" not in parsed or "overall" not in parsed:
            raise ValueError("unexpected Gemini response shape")
        return parsed
    except Exception as e:
        return {
            "per_stock": {},
            "overall": (
                "Commentary unavailable right now "
                f"({type(e).__name__}). Showing raw factor scores only."
            ),
            "error": str(e)[:200],
        }
