"""Unit tests for the analysis/cloud corpus-cache key derivation.

Only the pure-Python hashing logic is tested here; the Spark-backed
try_load / save round-trip would require a live SparkSession and lives
on the cluster smoke path instead.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cloud"))

from _corpus_cache import compute_cache_key  # noqa: E402


_BASE = dict(
    source_table="condition_era",
    person_mod=10,
    vocab_size=10000,
    min_df=10,
    doc_spec_manifest={"name": "patient_year", "min_doc_length": 20,
                       "replicate_eras": True,
                       "date_start_col": "condition_era_start_date",
                       "date_end_col": "condition_era_end_date"},
)


def test_key_is_stable_across_calls():
    """Same inputs → same key, deterministically."""
    assert compute_cache_key(**_BASE) == compute_cache_key(**_BASE)


def test_key_changes_with_source_table():
    other = dict(_BASE, source_table="condition_occurrence")
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_person_mod():
    other = dict(_BASE, person_mod=20)
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_vocab_size():
    other = dict(_BASE, vocab_size=5000)
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_min_df():
    other = dict(_BASE, min_df=5)
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_changes_with_doc_spec():
    """min_doc_length living inside doc_spec_manifest must influence the key —
    same person+vocab settings on differently-binned docs must not alias."""
    other = dict(_BASE,
                 doc_spec_manifest=dict(_BASE["doc_spec_manifest"],
                                        min_doc_length=30))
    assert compute_cache_key(**_BASE) != compute_cache_key(**other)


def test_key_unchanged_by_dict_ordering():
    """Manifests assembled with different key insertion orders must still
    hash to the same value (sort_keys=True in the implementation)."""
    a = compute_cache_key(**_BASE)
    reordered = {
        "date_end_col": "condition_era_end_date",
        "min_doc_length": 20,
        "name": "patient_year",
        "replicate_eras": True,
        "date_start_col": "condition_era_start_date",
    }
    b = compute_cache_key(**dict(_BASE, doc_spec_manifest=reordered))
    assert a == b


def test_key_is_short_hex():
    """16 hex chars: long enough to be collision-resistant across a session,
    short enough to type into a path."""
    k = compute_cache_key(**_BASE)
    assert len(k) == 16
    int(k, 16)  # no exception → valid hex


def test_min_df_int_vs_float_alias():
    """min_df=10 and min_df=10.0 must hash identically — the driver may pass
    either depending on argparse/Makefile path."""
    a = compute_cache_key(**_BASE)
    b = compute_cache_key(**dict(_BASE, min_df=10.0))
    assert a == b
