"""Unit tests for analysis/cloud/profile_eta.py — the WP-3 profile-eta boost
builder (plan 2026-09-06 D1-D4) and its layout contract.

Pure numpy/pandas; no Spark. The load-bearing claims:

  * D3 normalization — a node's positive boost vector sums to exactly
    strength * eta_base * V_condition;
  * the NOT (neg) delta math — neg-only concept -> -0.5 * eta_base; pos+neg
    concept -> the multiplicative-with-floor rule applied AFTER normalization;
  * the D4 coverage gate and the zero-positive-vector / unmapped-concept drops;
  * D1 topic targeting — the FIRST topic of the node's block (N=2: first two),
    on the SAME node_order/block layout inspect_topics.topic_labels documents
    (pinned against it on a fixture below — the anti-"boost leaked past its
    blocks" guard the plan's failure-reads name);
  * the emitted boost round-trips through the engine validator
    (gated_lda._resolve_eta_boost) with the same eta_base.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "cloud"))
import inspect_topics as it  # noqa: E402
import profile_eta as pe  # noqa: E402

pytest.importorskip("spark_vi")
from spark_vi.models.topic.dag_placement import DagLayout  # noqa: E402
from spark_vi.models.topic.gated_lda import _resolve_eta_boost  # noqa: E402

# Layout: n_bg=2, tpn=2 over nodes [1, 2, 3] -> blocks {1: [2,3], 2: [4,5],
# 3: [6,7]}, K=8. eta_base=0.1, V_cond=20, strength=1.0 -> target mass 2.0.
PARENT = {1: 0, 2: 0, 3: 1}
N_BG, TPN = 2, 2
ETA = 0.1
V_COND = 20
TARGET = 1.0 * ETA * V_COND                      # 2.0 added pseudo-mass

EID = {"MONDO:0000001": 1, "MONDO:0000002": 2, "MONDO:0000003": 3}
VOCAB = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}

# The hand fixture (mirrors the --emit-eta schema; a concept may be BOTH pos
# and neg for one node, zero-weight rows exist, 999 is not in the vocab, and
# MONDO:0000009 is not in this run's DAG).
ROWS = [
    # node 1: two mapped positives (3.0 + 1.0 -> scale 0.5), one unmapped
    # positive, a neg-only concept (103) and a pos+neg concept (101).
    ("MONDO:0000001", 101, 3.0, 0, 0.8),
    ("MONDO:0000001", 102, 1.0, 0, 0.8),
    ("MONDO:0000001", 999, 5.0, 0, 0.8),
    ("MONDO:0000001", 103, 0.7, 1, 0.8),
    ("MONDO:0000001", 101, 0.0, 1, 0.8),
    # node 2: zero-positive vector (all idf-0) + a zero-weight neg row (the
    # flag is the information; D2's downweight is not weight-scaled).
    ("MONDO:0000002", 104, 0.0, 0, 0.9),
    ("MONDO:0000002", 105, 0.0, 1, 0.9),
    # node 3: clean positive-only node, LOW coverage (min-coverage fodder).
    ("MONDO:0000003", 104, 2.0, 0, 0.05),
    ("MONDO:0000003", 105, 6.0, 0, 0.05),
    # a profiled node the label DAG dropped: skipped, counted, no error.
    ("MONDO:0000009", 101, 1.0, 0, 0.5),
]


def _df():
    return pd.DataFrame(
        ROWS, columns=["mondo_id", "concept_id", "weight", "neg", "coverage"])


def _lay():
    return DagLayout(PARENT, n_bg=N_BG, tpn=TPN)


def _build(**kw):
    args = dict(eid_by_mondo=EID, block_of=_lay().block,
                vocab_index_by_concept=VOCAB, eta_base=ETA,
                v_condition=V_COND, strength=1.0, topics=1, min_coverage=0.0)
    args.update(kw)
    return pe.build_profile_eta_boost(_df(), **args)


def test_d3_normalization_sums_to_strength_eta_v():
    """Node 3 (positive-only): its boost sums to exactly strength*eta*V, on the
    block's FIRST topic (6), with weights in the input's 2.0/6.0 proportion."""
    boost, stats = _build()
    idx, w = boost[6]
    np.testing.assert_array_equal(idx, [3, 4])
    np.testing.assert_allclose(w, [0.5, 1.5])            # scale = 2.0 / 8.0
    np.testing.assert_allclose(w.sum(), TARGET)
    # strength rescales the same vector linearly.
    boost3, _ = _build(strength=3.0)
    np.testing.assert_allclose(boost3[6][1], [1.5, 4.5])


def test_not_delta_math_neg_only_and_pos_plus_neg():
    """Node 1: neg-only concept 103 -> delta -0.5*eta; pos+neg concept 101 ->
    effective 0.5*(eta + pos_boost) so merged delta = effective - eta, with the
    positives normalized FIRST (pos_boost is the D3-scaled 1.5, not raw 3.0)."""
    boost, stats = _build()
    idx, w = boost[2]                                    # node 1's first topic
    np.testing.assert_array_equal(idx, [0, 1, 2])
    # 101 -> idx 0: pos 1.5 then NOT: eff = 0.5*(0.1+1.5)=0.8 -> 0.8-0.1 = 0.7
    # 102 -> idx 1: pos-only, normalized 0.5
    # 103 -> idx 2: neg-only: eff = 0.5*0.1 = 0.05 -> -0.05 = -0.5*eta
    np.testing.assert_allclose(w, [0.7, 0.5, -0.5 * ETA])
    assert stats["n_neg_concepts_applied"] == 3          # 101, 103, 105
    # every effective prior stays >= the 0.1*eta floor (garnish, never zero)
    for k, (ix, wk) in boost.items():
        assert (ETA + wk).min() >= 0.1 * ETA - 1e-15


def test_zero_positive_vector_node_keeps_only_its_neg_delta():
    """Node 2: mapped positives all weight 0 -> NO positive boost (counted),
    but the zero-weight neg row still applies the downweight (D2: the flag,
    not the weight, carries the NOT information)."""
    boost, stats = _build()
    idx, w = boost[4]                                    # node 2's first topic
    np.testing.assert_array_equal(idx, [4])              # concept 105 only
    np.testing.assert_allclose(w, [-0.5 * ETA])
    assert stats["n_nodes_zero_positive"] == 1


def test_min_coverage_drops_the_node_entirely():
    boost, stats = _build(min_coverage=0.2)
    assert 6 not in boost and 7 not in boost             # node 3 (coverage .05)
    assert 2 in boost and 4 in boost                     # nodes 1 (.8), 2 (.9)
    assert stats["n_nodes_dropped_min_coverage"] == 1
    off, off_stats = _build(min_coverage=0.0)            # 0 = gate off
    assert 6 in off and off_stats["n_nodes_dropped_min_coverage"] == 0


def test_unmapped_and_undagged_are_dropped_and_counted():
    boost, stats = _build()
    all_idx = np.concatenate([ix for ix, _ in boost.values()])
    assert set(all_idx) <= set(VOCAB.values())           # 999 never mapped
    assert stats["n_node_concepts_unmapped"] == 1
    assert stats["n_nodes_skipped_not_in_dag"] == 1      # MONDO:0000009
    assert stats["n_nodes_credited"] == 3


def test_floor_binds_when_the_multiplier_undershoots_it():
    """The 0.1*eta floor: with the default 0.5 multiplier and pos >= 0 it can
    never bind (0.5*(eta+p) >= 0.5*eta) — pinned below — so it is the guard
    against a sub-floor multiplier/stacked adjustment emitting an improper
    prior. mult=0.05 makes it bind."""
    assert pe.not_effective_prior(ETA, 0.0, mult=0.05) == 0.1 * ETA
    assert pe.not_effective_prior(ETA, 0.0) == 0.5 * ETA          # not binding
    for p in (0.0, 0.3, 5.0):
        assert pe.not_effective_prior(ETA, p) >= 0.1 * ETA
        assert pe.not_effective_prior(ETA, p) == 0.5 * (ETA + p)


def test_topic_targeting_first_topic_then_first_n():
    """D1: default topics=1 boosts ONLY each block's first topic; topics=2 puts
    the SAME vector on the first two (fresh arrays, not shared objects); the
    boost never lands outside the nodes' own blocks."""
    lay = _lay()
    b1, s1 = _build(topics=1)
    assert set(b1) == {2, 4, 6}                          # first topic per block
    assert s1["n_topics_boosted"] == 3
    b2, s2 = _build(topics=2)
    assert set(b2) == {2, 3, 4, 5, 6, 7}
    for u in lay.nodes:
        t0, t1 = lay.block[u]
        np.testing.assert_array_equal(b2[t0][0], b2[t1][0])
        np.testing.assert_array_equal(b2[t0][1], b2[t1][1])
        assert b2[t0][0] is not b2[t1][0]                # independent copies
    # no leak into the background block or across nodes
    fg = {t for u in lay.nodes for t in lay.block[u]}
    assert set(b1) <= fg and set(b2) <= fg
    with pytest.raises(ValueError, match="tpn"):
        _build(topics=3)                                 # > block width


def test_layout_agrees_with_inspect_topics_topic_labels():
    """The anti-leak contract: the builder targets DagLayout.block, and
    inspect_topics.topic_labels documents the SAME sorted-engine-id
    node_order/block layout — so topic t belongs to node u in one iff it does
    in the other, and the boosted first topic carries the node's own label."""
    lay = _lay()
    int2cid = {str(e): 1000 + e for e in [0] + lay.nodes}
    manifest = {
        "K": lay.K, "n_bg": N_BG, "tpn": TPN,
        "corpus_manifest": {
            "int2cid": int2cid,
            "name_by_id": {str(1000 + e): f"node{e}" for e in [0] + lay.nodes}},
    }
    labels, topic2engine = it.topic_labels(manifest)
    for u in lay.nodes:
        assert [t for t in range(lay.K) if topic2engine[t] == u] == lay.block[u]
    boost, _ = _build(topics=1)
    for u in lay.nodes:
        t = lay.block[u][0]
        if t in boost:
            assert labels[t] == f"node{u}" and topic2engine[t] == u


def test_output_round_trips_through_the_engine_validator():
    """End-to-end contract: everything the builder emits passes
    _resolve_eta_boost under the SAME eta_base (indices in-range/unique,
    weights finite, nonzero, eta + w > 0) — including the negative NOT deltas
    the WP-1 relaxation admits."""
    boost, _ = _build(topics=2)
    resolved = _resolve_eta_boost(boost, K=_lay().K, boost_v=V_COND, eta=ETA)
    assert set(resolved) == set(boost)
    assert any((w < 0).any() for _, w in resolved.values())


def test_stats_are_counts_only_no_coverage_values():
    """EGRESS: the stats dict (printed to the driver log / manifest) carries
    counts and the target mass — never the train-derived coverage column."""
    _, stats = _build(min_coverage=0.2)
    payload = json.dumps(stats)
    assert "coverage" not in payload.replace("min_coverage", "").replace(
        "dropped_min_coverage", "")
    for k, v in stats.items():
        assert isinstance(v, (int, float))
