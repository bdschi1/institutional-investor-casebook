"""Rule-based scoring of model outputs against golden answers.

Compares LLM-generated PM analyses to expert golden answers using
numerical extraction, keyword overlap, and structural matching.
No GPU or ML model required — runs anywhere.
"""

from __future__ import annotations

import re
import statistics
from typing import Sequence


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

# Matches: $21M, -$10.5M, $500M, $400M, +$10.4M, -$21.1M
_DOLLAR_RE = re.compile(
    r"[+-]?\$[\d,]+(?:\.\d+)?\s*(?:[BMKbmk](?:illion|illion)?)?",
)

# Matches: 1.4, -3%, 50bps, 0.50%, 2.6%, 7 (standalone numbers in context)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])"          # not preceded by a letter
    r"[+-]?[\d,]+(?:\.\d+)?"  # number with optional decimal
    r"\s*(?:%|bps|bp)?"       # optional unit
    r"(?![A-Za-z])",          # not followed by a letter
)


def _extract_numbers(text: str) -> list[float]:
    """Extract numerical values from financial text.

    Handles dollar amounts ($21M → 21.0), percentages (3% → 3.0),
    basis points (50bps → 50.0), and plain numbers.

    Returns a deduplicated list of floats in order of appearance.
    """
    raw_matches: list[str] = []
    raw_matches.extend(_DOLLAR_RE.findall(text))
    raw_matches.extend(_NUMBER_RE.findall(text))

    numbers: list[float] = []
    seen: set[float] = set()

    for raw in raw_matches:
        # Strip dollar signs, commas, whitespace, units
        cleaned = raw.replace("$", "").replace(",", "").strip()
        cleaned = re.sub(r"[BMKbmk](?:illion|illion)?$", "", cleaned).strip()
        cleaned = cleaned.replace("%", "").replace("bps", "").replace("bp", "").strip()

        if not cleaned or cleaned in (".", "+", "-"):
            continue

        try:
            val = float(cleaned)
        except ValueError:
            continue

        if val not in seen:
            seen.add(val)
            numbers.append(val)

    return numbers


# ---------------------------------------------------------------------------
# Key-term extraction
# ---------------------------------------------------------------------------

# Financial terms that indicate analytical structure
_STRUCTURE_TERMS = [
    "beta", "duration", "hedge", "p&l", "pnl", "sleeve",
    "correlation", "drawdown", "factor", "alpha", "exposure",
    "volatility", "sharpe", "risk", "return", "portfolio",
    "position", "sizing", "notional", "delta", "gamma",
    "var", "cvar", "stress", "scenario", "decompos",
    "attribution", "kelly", "information ratio",
    "mnpi", "material non-public", "information barrier",
    "compliance", "regulatory",
]


def _extract_key_terms(text: str) -> set[str]:
    """Extract financial key terms found in text (case-insensitive)."""
    text_lower = text.lower()
    return {term for term in _STRUCTURE_TERMS if term in text_lower}


# ---------------------------------------------------------------------------
# Likert scale mapping
# ---------------------------------------------------------------------------

_LIKERT_LABELS = {
    1: "Fail",
    2: "Below Expectations",
    3: "Meets Expectations",
    4: "Above Expectations",
    5: "Excellent",
}


def _to_likert(overall: float) -> int:
    """Map a 0-1 overall score to a 1-5 Likert rating.

    Thresholds calibrated for financial analysis quality:
      0.80+ = Excellent (5) — nearly all key numbers and structure present
      0.60+ = Above Expectations (4) — most content captured
      0.40+ = Meets Expectations (3) — partial but reasonable
      0.20+ = Below Expectations (2) — significant gaps
      <0.20 = Fail (1) — output missed the point
    """
    if overall >= 0.80:
        return 5
    if overall >= 0.60:
        return 4
    if overall >= 0.40:
        return 3
    if overall >= 0.20:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class CaseScorer:
    """Score model outputs against golden answers."""

    def score_case(self, model_output: str, golden_answer: str) -> dict:
        """Score a single model output against its golden answer.

        Returns
        -------
        dict with keys: completeness, numerical_accuracy, structure, overall
        Each is a float in [0, 1].
        """
        golden_nums = _extract_numbers(golden_answer)
        model_nums = _extract_numbers(model_output)

        golden_terms = _extract_key_terms(golden_answer)
        model_terms = _extract_key_terms(model_output)

        # --- Completeness: fraction of golden numbers found in output ---
        if golden_nums:
            found = 0
            for gn in golden_nums:
                for mn in model_nums:
                    if gn == 0 and mn == 0:
                        found += 1
                        break
                    if gn != 0 and abs(mn - gn) / abs(gn) < 0.30:
                        found += 1
                        break
            completeness = found / len(golden_nums)
        else:
            completeness = 1.0 if not model_nums else 0.5

        # --- Numerical accuracy: for matched numbers, how close? ---
        if golden_nums and model_nums:
            accuracies = []
            for gn in golden_nums:
                best_score = 0.0
                for mn in model_nums:
                    if gn == 0:
                        acc = 1.0 if mn == 0 else 0.0
                    else:
                        pct_diff = abs(mn - gn) / abs(gn)
                        if pct_diff < 0.10:
                            acc = 1.0
                        elif pct_diff < 0.25:
                            acc = 0.5
                        else:
                            acc = 0.0
                    best_score = max(best_score, acc)
                accuracies.append(best_score)
            numerical_accuracy = sum(accuracies) / len(accuracies)
        else:
            numerical_accuracy = 0.0

        # --- Structure: fraction of golden key terms in output ---
        if golden_terms:
            structure = len(golden_terms & model_terms) / len(golden_terms)
        else:
            structure = 1.0

        # --- Overall: weighted average ---
        overall = (
            0.40 * completeness
            + 0.40 * numerical_accuracy
            + 0.20 * structure
        )

        # --- Likert rating (1-5) mapped from overall score ---
        likert = _to_likert(overall)

        return {
            "completeness": round(completeness, 3),
            "numerical_accuracy": round(numerical_accuracy, 3),
            "structure": round(structure, 3),
            "overall": round(overall, 3),
            "likert": likert,
            "likert_label": _LIKERT_LABELS[likert],
        }

    def score_all(self, results: Sequence[dict]) -> dict:
        """Score all benchmark results and return per-case + aggregate.

        Parameters
        ----------
        results : list[dict]
            Each must have ``model_output`` and ``golden_answer`` keys.

        Returns
        -------
        dict with ``per_case`` (list) and ``aggregate`` (dict) keys.
        """
        per_case = []
        overalls = []
        likerts = []

        for r in results:
            scores = self.score_case(
                r.get("model_output", ""),
                r.get("golden_answer", ""),
            )
            per_case.append(
                {
                    "id": r.get("id", "Unknown"),
                    "category": r.get("category", ""),
                    **scores,
                }
            )
            overalls.append(scores["overall"])
            likerts.append(scores["likert"])

        aggregate = {}
        if overalls:
            aggregate = {
                "mean": round(statistics.mean(overalls), 3),
                "median": round(statistics.median(overalls), 3),
                "min": round(min(overalls), 3),
                "max": round(max(overalls), 3),
                "count": len(overalls),
                "likert_mean": round(statistics.mean(likerts), 1),
                "likert_distribution": {
                    label: likerts.count(rating)
                    for rating, label in _LIKERT_LABELS.items()
                },
            }

        return {"per_case": per_case, "aggregate": aggregate}
