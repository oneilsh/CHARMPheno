"""The DUAL prevalent/incident readout (spec E2 / plan WP4) — the pure half.

Everything E2 adds is pure numpy sitting between two Spark calls, so all of it is
testable off-Spark, and it must be: the incident arm's whole job is to be a
DIFFERENT number from the prevalent one, which means nothing about it can be
checked by "the totals still add up".

Five groups:

  1. **The eligibility mask (D2/D3/D4).** `incident_eval_mask` on a hand-computed
     fixture, including the two asymmetries that are easy to get wrong: a prior
     carrier leaves BOTH classes (never the positives only), and a negative needs
     the observation mask while a positive does not.

  2. **AGREEMENT WITH THE CENSUS.** The same synthetic `(label, labelMask,
     preindexClosure)` triple is folded by `diag_incident_census.census_partial`
     and by `incident_eval_mask`, and the per-node positive/negative counts must
     match EXACTLY. This is the load-bearing test of the tranche: the census is
     the corpus probe that GATED this arm (2,222/2,714 nodes clear 20/20 on 0110's
     record corpus), and if the two disagreed about who is eligible, the gate would
     have been passed on a population the readout does not score.

  3. **R2.1's constant-column guard.** A column that is constant over its scored
     rows but has BOTH classes present — the C2.1 population, exactly what the
     prevalent mask hides and the incident mask exposes — is skipped with the
     third reason, not scored at 0.5. And the guard is OFF by default, because
     turning it on unconditionally would move every recorded prevalent macro.

  4. **R2.2's two node sets.** Shared-set and full-set macros differ, and both are
     emitted; a delta across different node sets is not a comparison.

  5. **The fourth CSR run.** `_lean_eval_kernel` quints round-trip through
     `_densify_lean_blocks` — dense, sparse and absent — and a legacy 6-tuple block
     still densifies (the eligibility-free path is byte-identical).

All pure; no Spark, no cluster.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import distributed_readout as dr  # noqa: E402
import gated_pc_cloud as gpc  # noqa: E402
from analysis.pc.evaluate import (SKIP_CONSTANT, SKIP_DEGENERATE, SKIP_SMALL,  # noqa: E402
                                  _macro, _score_label)

RT, FT = [0.9], [0.5]


# --------------------------------------------------------------------------- #
# 1. The eligibility mask (D2 / D3 / D4).                                      #
# --------------------------------------------------------------------------- #
def _fixture():
    """C=3, four documents, every cell readable by eye.

    node 0: doc 0 is a prior carrier; the other three are eligible.
    node 1: docs 0 and 1 are prior carriers.
    node 2: nobody is a prior carrier.
    """
    y = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=np.uint8)
    mask = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]], dtype=np.uint8)
    preindex = [[0, 1], [1], [], []]
    C = 3
    elig = np.ones((4, C), dtype=np.uint8)
    for d, ids in enumerate(preindex):
        for c in ids:
            elig[d, c] = 0
    return y, mask, elig, preindex, C




# --------------------------------------------------------------------------- #
# 2. Exact agreement with the census's eligibility semantics.                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 3. R2.1 — the constant-column guard on the RANKING axis.                     #
# --------------------------------------------------------------------------- #
def test_a_constant_column_with_both_classes_is_skipped_not_scored_at_half():
    """The C2.1 population, in one column: a train-degenerate node's constant
    fallback column that has ACQUIRED negatives. `roc_auc_score` is happy to
    return exactly 0.5 for it, `skipped=None`, straight into the macro."""
    y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    p = np.full(6, 0.83)                       # the head's constant fallback
    off = _score_label(y, p)
    assert off["skipped"] is None and off["auc"] == pytest.approx(0.5)
    on = _score_label(y, p, skip_constant=True)
    assert on["auc"] is None and on["skip_code"] == SKIP_CONSTANT
    assert "constant prediction column" in on["skipped"]


def test_the_guard_is_off_by_default_so_prevalent_macros_do_not_move():
    """Every recorded prevalent number in the repo was computed without it; the
    incident arm turns it on and reports the count instead."""
    y = np.array([1, 0, 1, 0], dtype=float)
    p = np.full(4, 0.5)
    assert _score_label(y, p)["skipped"] is None
    informative = np.array([0.9, 0.1, 0.8, 0.2])
    assert _score_label(y, informative, skip_constant=True)["skipped"] is None


def test_the_three_skip_reasons_are_counted_separately_and_never_summed():
    """Spec §8, standing rule 5. Three reasons, three counts."""
    per_label = {
        0: _score_label(np.array([1.0, 1.0, 1.0]), np.array([.1, .2, .3])),
        1: _score_label(np.array([1.0, 0.0, 1.0]), np.array([.1, .2, .3]),
                        min_count=20),
        2: _score_label(np.array([1.0, 0.0, 1.0]), np.full(3, 0.7),
                        skip_constant=True),
        3: _score_label(np.array([1.0, 0.0, 1.0]), np.array([.9, .1, .8])),
    }
    macro = _macro(per_label)
    assert macro["n_labels_scored"] == 1
    assert macro["skipped_by_reason"] == {SKIP_DEGENERATE: 1, SKIP_SMALL: 1,
                                          SKIP_CONSTANT: 1}
    # the lumped total is still there and still equals the sum, but the block is
    # what gets reported (never the sum of the three)
    assert macro["n_labels_skipped"] == 3



# --------------------------------------------------------------------------- #
# 4. R2.2 — macros on BOTH node sets.                                          #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 5. The fourth CSR run.                                                       #
# --------------------------------------------------------------------------- #
def _kernel_rows(y, mask, preindex, proba):
    return [(d, proba[d], y[d].astype(float), mask[d].astype(float),
             list(preindex[d])) for d in range(len(y))]


def test_the_fourth_run_round_trips_through_the_densifier():
    """Kernel packs `R_d` sparse, driver returns the COMPLEMENT — that inversion
    is the one place the wire format and the semantics could silently disagree."""
    y, mask, elig, preindex, C = _fixture()
    proba = np.arange(len(y) * C, dtype=np.float32).reshape(len(y), C)
    block = dr._lean_eval_kernel(iter(_kernel_rows(y, mask, preindex, proba)), C)
    assert len(block) == 8
    p, gy, gm, ids, gelig = gpc._densify_lean_blocks([block], C)
    assert np.array_equal(p, proba)
    assert np.array_equal(gy, y)
    assert np.array_equal(gm, mask)
    assert ids == [0, 1, 2, 3]
    assert np.array_equal(gelig, elig)


def test_quads_still_produce_an_eligibility_free_block():
    """The prevalent path is untouched: no fifth element in, `elig=None` out."""
    y, mask, _elig, _pre, C = _fixture()
    proba = np.zeros((len(y), C), dtype=np.float32)
    rows = [(d, proba[d], y[d].astype(float), mask[d].astype(float))
            for d in range(len(y))]
    block = dr._lean_eval_kernel(iter(rows), C)
    assert block[6] is None and block[7] is None
    _p, _gy, _gm, _ids, gelig = gpc._densify_lean_blocks([block], C)
    assert gelig is None


def test_a_legacy_six_tuple_block_still_densifies():
    """Blocks pickled by an older executor (or by the existing tests) carry six
    elements; the densifier must read them as 'no eligibility', not crash."""
    P = np.arange(6, dtype=np.float32).reshape(2, 3)
    blocks = [(np.array([7, 8], dtype=np.int64), P,
               np.array([0, 2], dtype=np.int32), np.array([0, 1, 2], dtype=np.int64),
               None, None)]
    proba, y, mask, ids, elig = gpc._densify_lean_blocks(blocks, 3)
    assert np.array_equal(proba, P) and ids == [7, 8]
    assert np.array_equal(mask, np.ones((2, 3), dtype=np.uint8))
    assert elig is None


def test_an_empty_preindex_list_means_eligible_everywhere_not_absent():
    """`R_d = []` (a document with no resolvable pre-index code) and 'the column
    was not collected' are different facts and must not collapse into each other."""
    C = 3
    y = np.array([[1, 0, 0]], dtype=np.uint8)
    mask = np.ones((1, C), dtype=np.uint8)
    rows = [(0, np.zeros(C, dtype=np.float32), y[0].astype(float),
             mask[0].astype(float), [])]
    block = dr._lean_eval_kernel(iter(rows), C)
    assert block[6] is not None                    # the run exists...
    assert block[6].size == 0                      # ...and is empty
    _p, _y, _m, _ids, elig = gpc._densify_lean_blocks([block], C)
    assert np.array_equal(elig, np.ones((1, C), dtype=np.uint8))


def test_multiple_partitions_concatenate_in_order():
    """Two blocks, two partitions: the densifier's row offsets have to carry the
    eligibility runs as well as the label ones."""
    y, mask, elig, preindex, C = _fixture()
    proba = np.arange(len(y) * C, dtype=np.float32).reshape(len(y), C)
    rows = _kernel_rows(y, mask, preindex, proba)
    b1 = dr._lean_eval_kernel(iter(rows[:2]), C)
    b2 = dr._lean_eval_kernel(iter(rows[2:]), C)
    p, gy, gm, ids, gelig = gpc._densify_lean_blocks([b1, b2], C)
    assert np.array_equal(p, proba) and ids == [0, 1, 2, 3]
    assert np.array_equal(gy, y) and np.array_equal(gm, mask)
    assert np.array_equal(gelig, elig)


def test_row_quints_passes_a_null_column_through_as_carried_nothing():
    """The left join's NULL is already coalesced to `[]` upstream; if one ever
    arrives anyway it must read as 'carried nothing', not explode."""
    row = {"person_id": 5, "topicDistribution": np.array([0.5, 0.5]),
           "label": np.array([1.0, 0.0]), "labelMask": np.array([1.0, 1.0]),
           "preindexClosure": None}
    got = list(dr._row_quints(iter([row]), "person_id", "topicDistribution",
                              "label", "labelMask", "preindexClosure"))
    assert got[0][4] == []
