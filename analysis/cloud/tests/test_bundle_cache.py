"""Tests for the corpus-bundle cache key + metadata round-trip (pure, no Spark)."""
from dataclasses import dataclass

from analysis.cloud.bundle_cache import (
    _bundle_meta,
    _meta_to_bundle_fields,
    bundle_dir,
    corpus_cache_key,
)


def test_key_is_deterministic_and_order_independent():
    a = corpus_cache_key({"disease": "rare", "domains": "drug", "min_n": 20})
    b = corpus_cache_key({"min_n": 20, "domains": "drug", "disease": "rare"})
    assert a == b
    assert len(a) == 16


def test_key_changes_on_corpus_param():
    base = {"disease": "rare", "min_n": 20, "hier_max_class_fraction": 0.6}
    changed = {**base, "hier_max_class_fraction": 1.0}
    assert corpus_cache_key(base) != corpus_cache_key(changed)


def test_key_stable_when_only_fit_params_differ():
    # The caller excludes fit params from the dict; same corpus dict -> same key,
    # regardless of what init/tpn/topo the run uses.
    corpus = {"disease": "rare", "domains": "drug,measurement", "cond_vocab_size": 5000}
    assert corpus_cache_key(corpus) == corpus_cache_key(dict(corpus))


def test_bundle_dir_joins_cleanly():
    assert bundle_dir("gs://b/cache", "abc123") == "gs://b/cache/abc123"
    assert bundle_dir("gs://b/cache/", "abc123") == "gs://b/cache/abc123"


@dataclass
class _FakeBundle:
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_maps: list
    name_by_id: dict
    ledger: dict


def test_meta_roundtrip_preserves_int_keys_and_structure():
    b = _FakeBundle(
        train_df=None, test_df=None,
        parent_int={1: [0], 2: [1, 0]},
        int2cid={1: 100, 2: 200},
        cid2int={100: 1, 200: 2},
        vocab_maps=[{100: 0, 101: 1}, {500: 0}],
        name_by_id={100: "Disease A", 200: "Disease B"},
        ledger={"kept": 158, "dropped": 1373},
    )
    fields = _meta_to_bundle_fields(_bundle_meta(b))
    assert fields["parent_int"] == {1: [0], 2: [1, 0]}
    assert fields["int2cid"] == {1: 100, 2: 200}
    assert fields["cid2int"] == {100: 1, 200: 2}
    assert fields["vocab_maps"] == [{100: 0, 101: 1}, {500: 0}]
    assert fields["name_by_id"] == {100: "Disease A", 200: "Disease B"}
    assert fields["ledger"] == {"kept": 158, "dropped": 1373}
    # keys really are ints, not strings, after the round-trip
    assert all(isinstance(k, int) for k in fields["parent_int"])
    assert all(isinstance(k, int) for k in fields["int2cid"])
