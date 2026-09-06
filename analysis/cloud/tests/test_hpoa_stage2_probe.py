"""Pure-logic tests for the stage-2 HPOA support probe (no BQ, no Spark).

The BigQuery joins and the treeAggregate over the cached bundle are
cluster-covered (they need the live CDR and a cache HIT — same policy as
diag_sibling_support / diag_incident_census, whose Spark paths are exercised by
their production runs, and whose pure kernels are what gets unit tests). Four
groups here:

  1. **Token-set construction** from an emit-codes frame + a fake vocab map:
     membership in vocab_map IS the strip/min_df/cap survivorship test,
     negatives are excluded from the boost-side sets, and nodes outside the
     label DAG are skipped-and-counted, not errors.
  2. **The counting kernel** on hand-computable fake rows: an observed positive
     without a profile token counts in n_pos only; a masked-out positive counts
     nowhere; the BOW decode covers sparse- and dense-shaped cells; the
     reduction identity is a None sentinel (ADR 0047 addendum) so empty
     partitions produce no partial and the combiner handles None on either side.
  3. **Egress formatting**: a numerator at or under the disclosure floor (20)
     NEVER prints exact — `safe_coverage` bounds it, and the pooled summary
     runs its totals through `suppress_count`.
  4. **Closure collection end-to-end-ish**: the survey's emit-codes output
     (descendant xref pickup, no sibling leak, negatives carried) feeds
     straight into the token-set builder. The parse itself is covered in
     test_hpoa_profile_survey.py, next to the function.
"""
import numpy as np
import pandas as pd

import hpoa_profile_survey as s
import hpoa_stage2_probe as p


# --- fixtures ---------------------------------------------------------------

def _codes_df():
    # Two nodes; MONDO:0000001 has two positive terms (three codes, one shared
    # concept downstream) + one negative term; MONDO:0000002 is positive-only.
    return pd.DataFrame([
        ("MONDO:0000001", "HP:0000010", False, 0.9, "SNOMED", "111"),
        ("MONDO:0000001", "HP:0000010", False, 0.9, "SNOMED", "222"),
        ("MONDO:0000001", "HP:0000020", False, "",  "SNOMED", "333"),
        ("MONDO:0000001", "HP:0000030", True,  "",  "SNOMED", "444"),
        ("MONDO:0000002", "HP:0000010", False, "",  "SNOMED", "111"),
        ("MONDO:0000009", "HP:0000010", False, "",  "SNOMED", "111"),
    ], columns=["mondo_id", "hp_id", "neg", "freq", "vocab", "code"])


_CODE_TO_STD = {"111": {10}, "222": {20, 30}, "333": {40}, "444": {50}}
_VOCAB_MAP = {10: 0, 20: 5, 50: 7}      # 30/40 fell to min_df/cap; 50 in-vocab
_EID = {"MONDO:0000001": 3, "MONDO:0000002": 6}   # MONDO:0000009 not in DAG


# --- 1. token-set construction ----------------------------------------------

def test_token_sets_vocab_membership_is_the_survivorship_test():
    per_node, skipped = p.profile_token_sets(
        _codes_df(), _EID, _CODE_TO_STD, _VOCAB_MAP)
    a = per_node["MONDO:0000001"]
    # positive codes 111/222/333 -> concepts {10,20,30,40}; only 10,20 survive
    assert a["n_profile_concepts"] == 4
    assert a["n_in_vocab"] == 2
    assert a["tokens"] == frozenset({0, 5})
    assert a["eid"] == 3


def test_token_sets_negatives_excluded_from_boost_side():
    per_node, _ = p.profile_token_sets(
        _codes_df(), _EID, _CODE_TO_STD, _VOCAB_MAP)
    # concept 50 (token 7) is IN vocab but reached only via the NOT term 444:
    # it must not enter the boost-side token set.
    assert 7 not in per_node["MONDO:0000001"]["tokens"]


def test_token_sets_skips_nodes_outside_label_dag_and_counts_them():
    per_node, skipped = p.profile_token_sets(
        _codes_df(), _EID, _CODE_TO_STD, _VOCAB_MAP)
    assert skipped == ["MONDO:0000009"]
    assert "MONDO:0000009" not in per_node


def test_token_sets_tsv_roundtrip_stringified_bools(tmp_path):
    # A TSV round-trip turns the neg column into "True"/"False" strings; the
    # builder must still read polarity correctly.
    path = tmp_path / "codes.tsv"
    _codes_df().astype({"neg": str}).to_csv(path, sep="\t", index=False)
    back = pd.read_csv(path, sep="\t", dtype={"code": str})
    per_node, _ = p.profile_token_sets(back, _EID, _CODE_TO_STD, _VOCAB_MAP)
    assert 7 not in per_node["MONDO:0000001"]["tokens"]
    assert per_node["MONDO:0000001"]["tokens"] == frozenset({0, 5})


# --- 2. the counting kernel --------------------------------------------------

class _Sparse:
    """Duck-typed SparseVector: indices + values."""
    def __init__(self, idx, vals):
        self.indices, self.values = idx, vals


def _row(y, m, feat):
    return {"label": y, "labelMask": m, "features_0": feat}


def test_support_partial_counts_pos_and_hits():
    eids = [3, 6]
    tokens = [frozenset({0, 5}), frozenset({7})]
    rows = [
        # positive+observed for node 3, BOW carries token 5 -> pos AND hit
        _row([0, 0, 0, 1, 0, 0, 1], [1] * 7, _Sparse([2, 5], [1.0, 2.0])),
        # positive+observed for node 3, no profile token -> pos only
        _row([0, 0, 0, 1, 0, 0, 0], [1] * 7, [0, 1, 0, 0, 0, 0, 0]),
        # positive but MASKED OUT for node 3 -> counts nowhere
        _row([0, 0, 0, 1, 0, 0, 0], [0] * 7, _Sparse([0], [1.0])),
        # positive+observed for node 6, dense BOW carries token 7... which is
        # out of this 7-wide toy vector; use token 7 present via a wider list
        _row([0, 0, 0, 0, 0, 0, 1], [1] * 7, [0, 0, 0, 0, 0, 0, 0, 3.0]),
    ]
    (out,) = p.support_partial(rows, eids, tokens)
    n_pos, n_hit = out
    assert n_pos.tolist() == [2.0, 2.0]     # rows 1,2 for node 3; rows 1,4 for 6
    assert n_hit.tolist() == [1.0, 1.0]     # row 1 hits node 3; row 4 hits node 6


def test_support_partial_empty_partition_yields_no_partial():
    assert p.support_partial([], [3], [frozenset({0})]) == []


def test_support_combine_none_sentinel_identity():
    a = (np.array([1.0]), np.array([0.0]))
    assert p.support_combine(None, None) is None
    assert p.support_combine(a, None) is a
    assert p.support_combine(None, a) is a
    both = p.support_combine(a, (np.array([2.0]), np.array([1.0])))
    assert both[0].tolist() == [3.0] and both[1].tolist() == [1.0]


def test_bow_index_set_sparse_dense_and_explicit_zero():
    assert p._bow_index_set(_Sparse([1, 4], [2.0, 0.0])) == {1}  # stored zero
    assert p._bow_index_set([0.0, 3.0, 0.0, 1.0]) == {1, 3}


# --- 3. egress formatting -----------------------------------------------------

def test_safe_coverage_small_numerator_never_prints_exact():
    # n_hit=5 over n_pos=150: 5/150 = 0.03 would reconstruct the cell.
    got = p.safe_coverage(5, 150)
    assert "0.03" not in got
    assert got == "≤0.13"                   # the bound the floor implies: 20/150
    assert p.safe_coverage(0, 150) == "0.00"
    assert p.safe_coverage(75, 150) == "0.50"
    assert p.safe_coverage(5, 50) == "n/a"  # denominator under the bar
    assert p.safe_coverage(20, 150) == "≤0.13"   # floor itself is suppressed


def test_build_summary_suppresses_small_cells_and_omits_patient_counts():
    rows = [
        {"mondo_id": "MONDO:0000001", "name": "toy A",
         "n_profile_concepts": 4, "n_in_vocab": 2, "n_pos": 150, "n_hit": 5},
        {"mondo_id": "MONDO:0000002", "name": "toy B",
         "n_profile_concepts": 1, "n_in_vocab": 1, "n_pos": 400, "n_hit": 300},
        # under the fraction bar: must never appear with a fraction
        {"mondo_id": "MONDO:0000003", "name": "toy C",
         "n_profile_concepts": 1, "n_in_vocab": 1, "n_pos": 30, "n_hit": 12},
    ]
    md = p.build_summary(rows, {"run": "t", "C": 10, "profile_codes": "x.tsv",
                                "n_codes": 4, "n_std_concepts": 5,
                                "n_skipped_not_in_dag": 1})
    # toy A's exact small numerator/fraction never appear
    assert "0.03" not in md
    assert "≤0.13" in md
    # per-node patient counts never appear (150/400/30 as standalone counts);
    # the fraction table has no denominators at all
    assert " 150 " not in md and "| 150" not in md
    # the under-bar node gets no fraction row
    assert "toy C" not in md
    # pooled totals go through suppress_count; total_hit=317 prints exact,
    # but a small pooled cell would print as ≤20 (checked directly below)
    assert "317" in md


def test_build_summary_pooled_total_under_floor_is_suppressed():
    rows = [{"mondo_id": "MONDO:0000001", "name": "toy A",
             "n_profile_concepts": 1, "n_in_vocab": 1,
             "n_pos": 15, "n_hit": 5}]
    md = p.build_summary(rows, {})
    assert "≤20" in md
    assert ": 5\n" not in md and ": 15\n" not in md


# --- 4. survey emit-codes output feeds the token-set builder ------------------

def test_emit_codes_rows_feed_token_sets():
    """The stage-1 -> stage-2 handoff: profile_code_rows' frame is directly
    consumable by profile_token_sets (column names, polarity dtype)."""
    profiles = pd.DataFrame({
        "mondo_id": ["MONDO:0000001"], "hpo_id": ["HP:0001626"],
        "neg": [False], "freq": [0.9],
    })
    parents = {"HP:0001626": [], "HP:0001635": ["HP:0001626"]}
    xrefs = [("HP:0001635", "CHF", "SNOMED", "42343007")]
    codes = s.profile_code_rows(profiles, parents, xrefs)
    per_node, _ = p.profile_token_sets(
        codes, {"MONDO:0000001": 1}, {"42343007": {99}}, {99: 4})
    assert per_node["MONDO:0000001"]["tokens"] == frozenset({4})
