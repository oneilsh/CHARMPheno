"""Tests for the pure measurement value-aware token encoding (no pyspark).

Import-safe standalone (measurement_tokens imports nothing from pyspark), so it
runs under the charmpheno conftest in CI and via direct import locally.
"""
from charmpheno.omop import measurement_tokens as mt


def test_encode_decode_roundtrip():
    tok = mt.encode_token(3016723, mt.STATE_RANGE_HIGH)
    assert tok == 3016723 * 100 + 3
    assert mt.decode_token(tok) == (3016723, mt.STATE_RANGE_HIGH)


def test_encode_rejects_out_of_range_state():
    import pytest
    with pytest.raises(ValueError):
        mt.encode_token(1, 100)
    with pytest.raises(ValueError):
        mt.encode_token(1, -1)


def test_tokens_are_injective_across_concepts_and_states():
    seen = set()
    for cid in (1, 2, 999999, 40757494):
        for st in mt._STATE_LABELS:
            tok = mt.encode_token(cid, st)
            assert tok not in seen
            seen.add(tok)
            assert mt.decode_token(tok) == (cid, st)


def test_range_state_boundaries_are_inclusive_normal():
    assert mt.range_state(3.0, 3.5, 5.0) == mt.STATE_RANGE_LOW
    assert mt.range_state(3.5, 3.5, 5.0) == mt.STATE_RANGE_NORMAL   # == low bound
    assert mt.range_state(5.0, 3.5, 5.0) == mt.STATE_RANGE_NORMAL   # == high bound
    assert mt.range_state(6.8, 3.5, 5.0) == mt.STATE_RANGE_HIGH


def test_coded_state_from_name_normalizes_and_collapses_synonyms():
    assert mt.coded_state_from_name("Positive") == mt.STATE_CODED_POS
    assert mt.coded_state_from_name("  DETECTED ") == mt.STATE_CODED_POS
    assert mt.coded_state_from_name("Reactive") == mt.STATE_CODED_POS
    assert mt.coded_state_from_name("1+") == mt.STATE_CODED_POS
    assert mt.coded_state_from_name("Not detected") == mt.STATE_CODED_NEG
    assert mt.coded_state_from_name("Nonreactive") == mt.STATE_CODED_NEG
    assert mt.coded_state_from_name("Abnormal") == mt.STATE_CODED_ABNORMAL


def test_coded_state_from_name_rejects_junk_and_colors():
    # survey junk / non-meaningful codes fall through (None -> presence upstream)
    for junk in ("Null", "=", "0", "16", "Yellow", "Clear", "Indeterminate", None, ""):
        assert mt.coded_state_from_name(junk) is None


def test_classify_state_cascade_prefers_range_then_coded_then_presence():
    # 1. range present -> range-derived (even if a coded name is also present)
    assert mt.classify_state(value_as_number=6.8, range_low=3.5, range_high=5.0,
                             value_concept_name="High") == mt.STATE_RANGE_HIGH
    # 2. no range, allowlisted coded value -> coded state
    assert mt.classify_state(value_as_number=None, range_low=None, range_high=None,
                             value_concept_name="Positive") == mt.STATE_CODED_POS
    # numeric present but NO range -> not range branch; coded name decides
    assert mt.classify_state(value_as_number=12.0, range_low=None, range_high=None,
                             value_concept_name="Negative") == mt.STATE_CODED_NEG
    # 3. nothing usable -> presence
    assert mt.classify_state(value_as_number=12.0, range_low=None, range_high=None,
                             value_concept_name="Yellow") == mt.STATE_PRESENCE
    assert mt.classify_state(value_as_number=None, range_low=None, range_high=None,
                             value_concept_name=None) == mt.STATE_PRESENCE


def test_display_and_decode_names():
    tok = mt.encode_token(3016723, mt.STATE_RANGE_HIGH)
    assert mt.display_name("Creatinine", mt.STATE_RANGE_HIGH) == "Creatinine [high]"
    assert mt.decode_token_name(tok, {3016723: "Creatinine"}) == "Creatinine [high]"
    # missing name -> '?'
    assert mt.decode_token_name(tok, {}) == "? [high]"


def test_coded_states_returns_a_copy():
    d = mt.coded_states()
    d["positive"] = 999
    assert mt.coded_state_from_name("positive") == mt.STATE_CODED_POS
