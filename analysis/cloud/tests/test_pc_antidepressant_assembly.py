"""Pure numpy-level assembly tests for the PC antidepressant driver.

Covers the in-memory bridge + label/mask + column-ordering + split helpers of
``pc_antidepressant_cloud`` WITHOUT BigQuery and WITHOUT a live SparkSession.
SparseVector rows are built directly from ``pyspark.ml.linalg`` (no session
needed) so the bridge is exercised on exactly the ``features`` row shape
``to_bow_dataframe`` emits and ``build_test_bow`` reads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_CLOUD = str(Path(__file__).resolve().parents[1])
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import pc_antidepressant_cloud as drv  # noqa: E402


def _sv(size, mapping):
    """A pyspark ML SparseVector (constructed without a SparkSession)."""
    from pyspark.ml.linalg import SparseVector
    return SparseVector(size, mapping)


# --------------------------------------------------------------------------- #
# Bridge: SparseVector BOW rows -> dense X aligned to an explicit person order #
# --------------------------------------------------------------------------- #
def test_bow_rows_to_matrix_aligns_rows_to_person_order():
    V = 4
    feats = {
        "p1": _sv(V, {0: 2.0, 3: 1.0}),
        "p2": _sv(V, {1: 5.0}),
        "p3": _sv(V, {2: 1.0, 0: 4.0}),
    }
    # Deliberately NOT the insertion order — the matrix must follow person_order.
    person_order = ["p3", "p1", "p2"]
    X = drv.bow_rows_to_matrix(feats, person_order, V).toarray()

    assert X.shape == (3, V)
    np.testing.assert_array_equal(X[0], [4.0, 0.0, 1.0, 0.0])   # p3
    np.testing.assert_array_equal(X[1], [2.0, 0.0, 0.0, 1.0])   # p1
    np.testing.assert_array_equal(X[2], [0.0, 5.0, 0.0, 0.0])   # p2


def test_bow_rows_to_matrix_missing_person_is_zero_row():
    V = 3
    feats = {"a": _sv(V, {0: 1.0})}
    person_order = ["a", "ghost"]           # 'ghost' has no BOW row
    X = drv.bow_rows_to_matrix(feats, person_order, V).toarray()
    assert X.shape == (2, V)
    np.testing.assert_array_equal(X[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(X[1], [0.0, 0.0, 0.0])        # all-zero row


def test_bow_rows_to_matrix_empty_is_shaped_zero():
    X = drv.bow_rows_to_matrix({}, [], 5).toarray()
    assert X.shape == (0, 5)


# --------------------------------------------------------------------------- #
# Label / mask assembly: exactly one observed cell per row (the index drug)    #
# --------------------------------------------------------------------------- #
def test_assemble_labels_one_observed_cell_per_row_right_column_value():
    person_order = ["p1", "p2", "p3"]
    drug_order = ["sertraline", "bupropion"]
    outcome = {
        "p1": ("sertraline", True),    # column 0, positive
        "p2": ("bupropion", False),    # column 1, negative
        "p3": ("sertraline", False),   # column 0, negative
    }
    y, mask = drv.assemble_multitask_labels(outcome, person_order, drug_order)

    assert y.shape == (3, 2) and mask.shape == (3, 2)
    # Exactly one observed cell per row.
    np.testing.assert_array_equal(mask.sum(axis=1), [1, 1, 1])
    # Right column.
    np.testing.assert_array_equal(mask, [[1, 0], [0, 1], [1, 0]])
    # Right value at the observed cell (unobserved cells stay 0).
    np.testing.assert_array_equal(y, [[1, 0], [0, 0], [0, 0]])


def test_assemble_labels_unobserved_cells_have_mask_zero():
    person_order = ["p1"]
    drug_order = ["a", "b", "c"]
    y, mask = drv.assemble_multitask_labels({"p1": ("b", True)}, person_order, drug_order)
    # Only column 1 observed; the other two are mask=0 (their y value is ignored).
    np.testing.assert_array_equal(mask[0], [0, 1, 0])
    assert y[0, 1] == 1.0


def test_assemble_labels_person_without_outcome_is_all_unobserved():
    person_order = ["labeled", "unlabeled"]
    drug_order = ["a"]
    y, mask = drv.assemble_multitask_labels({"labeled": ("a", True)}, person_order, drug_order)
    np.testing.assert_array_equal(mask[0], [1])
    np.testing.assert_array_equal(mask[1], [0])          # no observed cell at all


def test_assemble_labels_index_drug_not_a_column_is_dropped():
    # Defensive: an outcome whose drug isn't in drug_order contributes no cell.
    person_order = ["p1"]
    y, mask = drv.assemble_multitask_labels({"p1": ("unknown", True)}, person_order, ["a", "b"])
    np.testing.assert_array_equal(mask[0], [0, 0])


# --------------------------------------------------------------------------- #
# Fully-observed label assembly (mdd_stable_treatment): all-ones mask, C=10     #
# --------------------------------------------------------------------------- #
def test_assemble_fullyobserved_single_drug_is_one_positive_mask_all_ones():
    person_order = ["p1"]
    drug_order = ["fluoxetine", "sertraline", "bupropion"]
    y, mask = drv.assemble_fullyobserved_labels(
        {"p1": ["fluoxetine"]}, person_order, drug_order,
    )
    assert y.shape == (1, 3) and mask.shape == (1, 3)
    # One positive at the drug's fixed column; the whole row is observed.
    np.testing.assert_array_equal(y, [[1, 0, 0]])
    np.testing.assert_array_equal(mask, [[1, 1, 1]])


def test_assemble_fullyobserved_combination_is_two_positives():
    person_order = ["p2"]
    drug_order = ["fluoxetine", "sertraline", "bupropion"]
    y, mask = drv.assemble_fullyobserved_labels(
        {"p2": ["fluoxetine", "sertraline"]}, person_order, drug_order,
    )
    # A held combination -> two positives, still a fully-observed (all-ones) row.
    np.testing.assert_array_equal(y, [[1, 1, 0]])
    np.testing.assert_array_equal(mask, [[1, 1, 1]])


def test_assemble_fullyobserved_uses_fixed_column_order():
    # Column c is drug_order[c] regardless of the subset's own ordering.
    person_order = ["p"]
    drug_order = ["a", "b", "c", "d"]
    y, _ = drv.assemble_fullyobserved_labels(
        {"p": ["d", "b"]}, person_order, drug_order,   # unsorted subset
    )
    np.testing.assert_array_equal(y, [[0, 1, 0, 1]])   # b -> col 1, d -> col 3


def test_assemble_fullyobserved_absent_person_is_all_zero_unobserved():
    person_order = ["present", "absent"]
    drug_order = ["a", "b"]
    y, mask = drv.assemble_fullyobserved_labels(
        {"present": ["a"]}, person_order, drug_order,
    )
    # present -> observed row (mask all-ones), positive at 'a'.
    np.testing.assert_array_equal(y[0], [1, 0])
    np.testing.assert_array_equal(mask[0], [1, 1])
    # absent -> all-zero AND all-UNOBSERVED (mask 0), a valid unlabeled row.
    np.testing.assert_array_equal(y[1], [0, 0])
    np.testing.assert_array_equal(mask[1], [0, 0])


def test_assemble_fullyobserved_row_aligned_to_person_order():
    # Rows follow person_order, not dict insertion order.
    person_order = ["z", "a"]
    drug_order = ["x", "y"]
    y, _ = drv.assemble_fullyobserved_labels(
        {"a": ["x"], "z": ["y"]}, person_order, drug_order,
    )
    np.testing.assert_array_equal(y[0], [0, 1])   # z -> 'y'
    np.testing.assert_array_equal(y[1], [1, 0])   # a -> 'x'


def test_assemble_fullyobserved_drug_not_a_column_is_ignored():
    # A subset name absent from drug_order contributes no positive (still observed).
    y, mask = drv.assemble_fullyobserved_labels(
        {"p": ["a", "unknown"]}, ["p"], ["a", "b"],
    )
    np.testing.assert_array_equal(y, [[1, 0]])
    np.testing.assert_array_equal(mask, [[1, 1]])


# --------------------------------------------------------------------------- #
# Stable drug-column ordering                                                  #
# --------------------------------------------------------------------------- #
def test_stable_drug_order_follows_reference_then_alpha_extras():
    reference = ("fluoxetine", "sertraline", "bupropion", "trazodone")
    present = ["trazodone", "fluoxetine", "zzz_custom", "bupropion"]
    order = drv.stable_drug_order(present, reference=reference)
    # Reference-ordered subset first, then non-reference extras alphabetically.
    assert order == ["fluoxetine", "bupropion", "trazodone", "zzz_custom"]


def test_stable_drug_order_is_deterministic_regardless_of_input_order():
    reference = ("a", "b", "c")
    assert drv.stable_drug_order(["c", "a"], reference=reference) == ["a", "c"]
    assert drv.stable_drug_order(["a", "c"], reference=reference) == ["a", "c"]


def test_stable_drug_order_empty_reference_is_alphabetical():
    assert drv.stable_drug_order(["c", "a", "b"]) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Seeded stratified split                                                      #
# --------------------------------------------------------------------------- #
def test_stratified_test_mask_is_seed_deterministic_and_fractional():
    groups = ["a"] * 8 + ["b"] * 4
    m1 = drv.stratified_test_mask(groups, test_frac=0.25, seed=7)
    m2 = drv.stratified_test_mask(groups, test_frac=0.25, seed=7)
    np.testing.assert_array_equal(m1, m2)                 # deterministic
    groups_arr = np.asarray(groups, dtype=object)
    # Per-group heldout counts: round(0.25 * 8)=2, round(0.25 * 4)=1.
    assert int(m1[groups_arr == "a"].sum()) == 2
    assert int(m1[groups_arr == "b"].sum()) == 1


def test_stratified_test_mask_different_seed_can_differ():
    groups = ["a"] * 10
    m1 = drv.stratified_test_mask(groups, test_frac=0.3, seed=1)
    m2 = drv.stratified_test_mask(groups, test_frac=0.3, seed=2)
    # Same count, but the chosen rows should not be forced identical.
    assert m1.sum() == m2.sum() == 3
    assert not np.array_equal(m1, m2)


def test_stratified_test_mask_singleton_group_stays_in_train():
    # round(0.25 * 1) == 0 -> the lone row is never held out.
    groups = ["solo"]
    m = drv.stratified_test_mask(groups, test_frac=0.25, seed=0)
    assert m.sum() == 0
