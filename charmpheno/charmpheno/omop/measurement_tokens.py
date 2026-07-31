"""Value-aware token encoding for the OMOP measurement domain (measurement arc).

Pure (no pyspark) so it is the single source of truth for BOTH the Spark loader
(``bigquery.py`` ``source_table="measurement"``) and its unit tests. The loader
mirrors the cascade below with native Spark column expressions built FROM the
same allowlist dict, so the two cannot drift.

Why a synthetic token. The multi-domain engine keys every domain's vocabulary on
an integer ``concept_id`` (``topic_prep.to_bow_dataframe`` casts tokens to int).
A measurement's evidence is the *value*, not the bare concept, so we fold a small
value-*state* into the id:

    token = measurement_concept_id * TOKEN_BASE + state_code        (state_code < TOKEN_BASE)

This is injective (decodable via divmod) and stays within int64 for any real
OMOP concept_id. The token is not a real OMOP concept, so names are decoded here
for inspection ("Creatinine [high]") rather than by a `concept` join.

The representation cascade (insight 0077), applied per measurement row:

  1. range present   -> range-derived low / normal / high  (unit-robust: the
                        reference range is in the value's own unit)
  2. else coded value in the allowlist -> the coded qualitative state
                        (serologies / urinalysis; unit-free)
  3. else            -> bare presence ("measured")

Coded values are matched by NORMALIZED NAME, not concept_id — the same
vocabulary-agnostic strategy ``bigquery.decode_sex_from_name`` uses — because the
survey (0077) showed the useful codes are a small, stable name set while the
concept_id space is polluted with junk ("Null", "=", bare numerics).

Burstiness (≈924 rows/person, no measurement_era table) is handled OUTSIDE this
module: the domain is tokenized with per-document binary presence
(``to_bow_dataframe(..., binary=True)``), which reproduces for measurement what
the era rollup already does for the condition/drug domains.
"""
from __future__ import annotations

# ── state codes (the low digits of a synthetic token) ───────────────────────
TOKEN_BASE = 100  # supports up to 99 distinct states per concept

STATE_PRESENCE = 0
# range-derived (numeric value compared to the row's reference range)
STATE_RANGE_LOW = 1
STATE_RANGE_NORMAL = 2
STATE_RANGE_HIGH = 3
# coded qualitative (value_as_concept_id, allowlisted by name). Kept distinct
# from the range-derived codes so provenance is never conflated on inspection.
STATE_CODED_NEG = 10
STATE_CODED_POS = 11
STATE_CODED_NORMAL = 12
STATE_CODED_ABNORMAL = 13
STATE_CODED_LOW = 14
STATE_CODED_HIGH = 15

_STATE_LABELS = {
    STATE_PRESENCE: "measured",
    STATE_RANGE_LOW: "low",
    STATE_RANGE_NORMAL: "normal",
    STATE_RANGE_HIGH: "high",
    STATE_CODED_NEG: "neg",
    STATE_CODED_POS: "pos",
    STATE_CODED_NORMAL: "normal(coded)",
    STATE_CODED_ABNORMAL: "abnormal",
    STATE_CODED_LOW: "low(coded)",
    STATE_CODED_HIGH: "high(coded)",
}

# Coded-value allowlist: normalized (lower, stripped) value_as_concept name ->
# state code. Collapses synonyms (positive/detected/reactive/present/dipstick
# grades -> pos; negative/not-detected/nonreactive/absent -> neg) to reduce
# sparsity. Names NOT in this map fall through to presence (option 3) — that is
# how the junk codes ("null", "=", "0", "16", colors) are dropped without an
# explicit blocklist. Extend here (single source of truth for loader + tests).
_CODED_NAME_TO_STATE: dict[str, int] = {
    # negative-like
    "negative": STATE_CODED_NEG,
    "not detected": STATE_CODED_NEG,
    "not detected/negative": STATE_CODED_NEG,
    "nonreactive": STATE_CODED_NEG,
    "non-reactive": STATE_CODED_NEG,
    "absent": STATE_CODED_NEG,
    "none seen": STATE_CODED_NEG,
    # positive-like (including semiquantitative urine dipstick grades)
    "positive": STATE_CODED_POS,
    "detected": STATE_CODED_POS,
    "reactive": STATE_CODED_POS,
    "present": STATE_CODED_POS,
    "trace": STATE_CODED_POS,
    "1+": STATE_CODED_POS,
    "2+": STATE_CODED_POS,
    "3+": STATE_CODED_POS,
    "4+": STATE_CODED_POS,
    # qualitative normal/abnormal/high/low
    "normal": STATE_CODED_NORMAL,
    "abnormal": STATE_CODED_ABNORMAL,
    "high": STATE_CODED_HIGH,
    "low": STATE_CODED_LOW,
}


def coded_states() -> dict[str, int]:
    """The coded-value allowlist (normalized name -> state code). Returned as a
    copy so the loader can iterate it to build its Spark when-chain from the same
    source without risking mutation."""
    return dict(_CODED_NAME_TO_STATE)


def state_labels() -> dict[int, str]:
    """State code -> human label, as a copy (the loader iterates it to build its
    display-name Spark when-chain from the same source)."""
    return dict(_STATE_LABELS)


def normalize_name(name: str | None) -> str:
    """Lower/strip a value concept name for allowlist lookup. None -> ''."""
    return (name or "").strip().lower()


def coded_state_from_name(name: str | None) -> int | None:
    """State code for an allowlisted coded value name, else None (fall through)."""
    return _CODED_NAME_TO_STATE.get(normalize_name(name))


def range_state(value: float, range_low: float, range_high: float) -> int:
    """low / normal / high from a numeric value vs its reference range.

    Convention: value < range_low -> LOW; value > range_high -> HIGH; the closed
    interval [range_low, range_high] is NORMAL. Caller guarantees all three are
    non-null (the cascade's range branch is gated on that)."""
    if value < range_low:
        return STATE_RANGE_LOW
    if value > range_high:
        return STATE_RANGE_HIGH
    return STATE_RANGE_NORMAL


def classify_state(
    *,
    value_as_number: float | None,
    range_low: float | None,
    range_high: float | None,
    value_concept_name: str | None,
) -> int:
    """The full per-row cascade -> state code. Single source of truth; the Spark
    loader replicates this exactly with column expressions.

    1. numeric value AND a reference range -> range-derived low/normal/high
    2. else an allowlisted coded value name -> its coded state
    3. else -> presence
    """
    if value_as_number is not None and range_low is not None and range_high is not None:
        return range_state(value_as_number, range_low, range_high)
    coded = coded_state_from_name(value_concept_name)
    if coded is not None:
        return coded
    return STATE_PRESENCE


def encode_token(measurement_concept_id: int, state_code: int) -> int:
    """Fold (concept, state) into one synthetic integer token."""
    if not 0 <= state_code < TOKEN_BASE:
        raise ValueError(f"state_code {state_code} out of range [0,{TOKEN_BASE})")
    return int(measurement_concept_id) * TOKEN_BASE + state_code


def decode_token(token: int) -> tuple[int, int]:
    """Inverse of encode_token -> (measurement_concept_id, state_code)."""
    return divmod(int(token), TOKEN_BASE)


def state_label(state_code: int) -> str:
    """Human label for a state code (e.g. 3 -> 'high')."""
    return _STATE_LABELS.get(int(state_code), f"state{state_code}")


def display_name(real_concept_name: str | None, state_code: int) -> str:
    """Inspection name for a synthetic token, e.g. 'Creatinine ... [high]'."""
    return f"{real_concept_name or '?'} [{state_label(state_code)}]"


def decode_token_name(token: int, name_by_concept: dict[int, str]) -> str:
    """Full display name for a synthetic token given a real-concept name map."""
    concept_id, state_code = decode_token(token)
    return display_name(name_by_concept.get(concept_id), state_code)
