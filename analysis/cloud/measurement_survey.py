"""Pure (Spark-free) helpers for the OMOP measurement-domain survey.

Split out from ``measurement_survey_cloud.py`` (the Spark driver) exactly as
``anchor_selection.py`` is split from ``anchor_selection_cloud.py``: everything
here is unit-testable without a BigQuery read or a SparkSession.

Two jobs:

- **Disclosure floor** (``apply_floor``): drop any grouped row whose count is
  below the small-cell threshold BEFORE it is written or printed, per the AoU
  small-cell rule. Callers pass the driver-collected agg rows through this so
  nothing sub-floor reaches disk or the paste-back digest.
- **Interpretation** (``derive_concept_summary``, ``classify_representation``):
  turn the raw per-concept aggregate counts into the fractions/shares the
  representation decision turns on, and tag each lab with the value-aware
  representation its coverage supports. The tag is a heuristic reading aid, not a
  commitment — the survey reports the raw fractions alongside it.
"""
from __future__ import annotations

from typing import Any


def safe_div(num: float, den: float) -> float:
    """num/den, or 0.0 when den is 0/None. Keeps every derived fraction finite
    even for a concept whose relevant denominator is empty."""
    if not den:
        return 0.0
    return num / den


def apply_floor(
    rows: list[dict[str, Any]], count_key: str, floor: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Drop rows whose ``row[count_key]`` is below ``floor``.

    Returns ``(kept, n_suppressed, suppressed_count_total)``. The two counters
    let the driver report *that* small cells were dropped (and roughly how much
    volume) without disclosing any individual sub-floor count — the totals are
    themselves only reported when they clear the floor (the driver's call).

    A row missing ``count_key`` or carrying a null count is treated as 0 and
    suppressed: absence of a count is never evidence of a safe-to-show cell.
    """
    kept: list[dict[str, Any]] = []
    n_suppressed = 0
    suppressed_total = 0
    for r in rows:
        c = r.get(count_key)
        c = 0 if c is None else int(c)
        if c < floor:
            n_suppressed += 1
            suppressed_total += c
        else:
            kept.append(r)
    return kept, n_suppressed, suppressed_total


def derive_concept_summary(agg: dict[str, Any]) -> dict[str, Any]:
    """Turn one concept's raw aggregate counts into the decision fractions.

    ``agg`` carries the driver's group-by outputs for a single
    ``measurement_concept_id`` (all integer counts):
      n_rows, n_persons, n_val_number, n_val_concept, n_range, n_unit,
      n_operator, n_feasible (numeric AND range present), n_low, n_high,
      n_distinct_units, top_unit_n (rows in the single most common unit).

    Returns ``agg`` shallow-copied with derived fields added:
      pct_val_number, pct_val_concept, pct_range, pct_unit, pct_operator
        — fractions of the concept's rows (denominator n_rows).
      pct_feasible — fraction of rows codable as low/normal/high from a range.
      top_unit_share — fraction of rows in the single most common unit
        (1.0 = one clean unit; low = unit chaos).
      frac_low / frac_normal / frac_high — split of the *feasible* rows
        (denominator n_feasible; the abnormality base rate).
    Divide-by-zero is impossible (safe_div guards every ratio).
    """
    out = dict(agg)
    n_rows = agg.get("n_rows") or 0
    n_feasible = agg.get("n_feasible") or 0
    n_low = agg.get("n_low") or 0
    n_high = agg.get("n_high") or 0
    n_normal = max(n_feasible - n_low - n_high, 0)

    out["pct_val_number"] = safe_div(agg.get("n_val_number") or 0, n_rows)
    out["pct_val_concept"] = safe_div(agg.get("n_val_concept") or 0, n_rows)
    out["pct_range"] = safe_div(agg.get("n_range") or 0, n_rows)
    out["pct_unit"] = safe_div(agg.get("n_unit") or 0, n_rows)
    out["pct_operator"] = safe_div(agg.get("n_operator") or 0, n_rows)
    out["pct_feasible"] = safe_div(n_feasible, n_rows)
    out["top_unit_share"] = safe_div(agg.get("top_unit_n") or 0, n_rows)
    out["frac_low"] = safe_div(n_low, n_feasible)
    out["frac_normal"] = safe_div(n_normal, n_feasible)
    out["frac_high"] = safe_div(n_high, n_feasible)
    return out


# Representation-viability thresholds. Deliberately conservative and few: the tag
# is a reading aid over the raw fractions the survey already prints, not a
# decision. "Viable" means "populated often enough that a token built from it
# would carry signal for most of this concept's rows."
_RANGE_VIABLE = 0.50      # >=50% of rows codable as low/normal/high from a range
_VALCONCEPT_VIABLE = 0.50  # >=50% of rows carry a coded qualitative value
_NUMERIC_VIABLE = 0.50     # >=50% of rows carry a numeric value (binning candidate)


def classify_representation(summary: dict[str, Any]) -> str:
    """Tag a per-concept summary with the value-aware representation its
    coverage supports (see the four options in the arc design spec):

      "range-abnormality"  — range-derived low/normal/high is viable (preferred:
                             unit-robust, no external tables).
      "value-concept"      — OMOP's coded qualitative value is viable and a
                             range is not (unit-agnostic fallback).
      "numeric-needs-binning" — numeric values exist but neither a range nor a
                             coded value does; usable only via binning, which is
                             unit-exposed.
      "presence-only"      — no usable value signal; a bare presence token is all
                             this lab can contribute.

    Order encodes the cascade preference (range > coded value > binning >
    presence). Reads only fields ``derive_concept_summary`` produces.
    """
    if summary.get("pct_feasible", 0.0) >= _RANGE_VIABLE:
        return "range-abnormality"
    if summary.get("pct_val_concept", 0.0) >= _VALCONCEPT_VIABLE:
        return "value-concept"
    if summary.get("pct_val_number", 0.0) >= _NUMERIC_VIABLE:
        return "numeric-needs-binning"
    return "presence-only"


def summarize_representation_mix(
    summaries: list[dict[str, Any]], weight_key: str | None = "n_persons"
) -> dict[str, dict[str, Any]]:
    """Tally how the candidate vocabulary splits across representation tags.

    Returns ``{tag: {"n_concepts": int, "weight": int}}`` where weight sums
    ``weight_key`` (default distinct-patient count) over the concepts with that
    tag — so the digest can say "range-abnormality covers K concepts / M
    patient-concepts" rather than just counting concepts. ``weight_key=None``
    counts concepts only.
    """
    mix: dict[str, dict[str, Any]] = {}
    for s in summaries:
        tag = classify_representation(s)
        bucket = mix.setdefault(tag, {"n_concepts": 0, "weight": 0})
        bucket["n_concepts"] += 1
        if weight_key is not None:
            bucket["weight"] += int(s.get(weight_key) or 0)
    return mix
