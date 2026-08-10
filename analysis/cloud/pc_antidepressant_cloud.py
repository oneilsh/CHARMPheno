"""Phase-C capstone driver: the Hughes antidepressant PC replication on BigQuery.

Ties the already-built, already-tested Phase-C components into ONE end-to-end
spark-submit driver that runs in the All-of-Us BigQuery/Spark workspace and
reproduces the Hughes "PC vs two-stage vs LR-on-codes" per-drug comparison for
antidepressant response. The pipeline is:

    1. Cohort + index  -- charmpheno.omop.cohorts.apply_mdd_antidepressant_cohort
       gives a per-person index table (person_id, index_date, index drug + class)
       for major-depression patients at their first antidepressant era across the
       15-drug set.
    2. Outcome         -- antidepressant_stability_label over the SAME drug_era
       read + the 15-drug concept map gives the per-person, per-drug
       (person_id, index_drug_name, worked) ">=90-day the drug worked" label.
    3. Features        -- load_omop_bigquery(concept_types=("condition","drug",
       "procedure")) emits a flat fused (person_id, concept_id, concept_name,
       event_date) stream; lookback_feature_label_events windows it to the
       pre-index lookback [index - lookback_days, index); to_bow_dataframe
       (PatientDocSpec) vectorizes it to a per-patient SparseVector BOW.
    4. Bridge + labels -- the BOW is collected to a dense count matrix ``X``
       aligned by an explicit person_id order (build_test_bow-style indptr
       accumulation), and the outcome is assembled into the multi-task
       ``(D, C)`` targets ``y`` / ``mask`` with the index-drug pattern (exactly
       one observed cell per row: the column of the drug that person initiated).
    5. Split + eval    -- persons are split into train/test (seeded, stratified
       by index drug), ``X/y/mask`` sliced, and analysis.pc.evaluate.
       evaluate_pc_multitask fits the shared PC + the two baselines and scores
       per-drug heldout AUC/AP. format_results_table renders the Hughes table.

CANNOT RUN HERE: every fact table (drug_era, condition_occurrence, concept,
concept_ancestor, observation_period) is read from the workspace CDR via the
spark-bigquery connector, so the end-to-end pipeline only executes inside the
All-of-Us Dataproc workspace. What IS covered by the repo's unit tests
(analysis/cloud/tests/test_pc_antidepressant_*.py) is the PURE in-memory
assembly logic that this container can exercise without BigQuery: the
Spark-BOW -> dense-``X`` bridge, the ``y``/``mask`` multi-task label assembly,
the stable drug-column ordering, the seeded stratified split, the pre-index
windowing (local-Spark synthetic frames), and the argparse/env validation gate.
Those pure helpers live at module scope below so they are importable and
testable without a live SparkSession.

Reads two environment variables (set by the workspace setup notebook), used as
the defaults for --cdr / --billing:
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from analysis/cloud on the Dataproc master):
    make pc-antidepressant  # (or spark-submit pc_antidepressant_cloud.py ...)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse as sp

from _driver_common import _phase, configure_logging, make_spark_session

# The fused-load common date column (bigquery._FUSED_EVENT_DATE) and the
# condition-only index date column. Kept as module constants so the driver and
# its tests name the same strings.
_FUSED_EVENT_DATE = "event_date"
_CONDITION_DATE = "condition_start_date"


# --------------------------------------------------------------------------- #
# Pure assembly helpers (no SparkSession required; unit-tested numpy-level).   #
# --------------------------------------------------------------------------- #
def bow_rows_to_matrix(
    features_by_person: Mapping[Any, Any],
    person_order: Sequence[Any],
    vocab_size: int,
) -> sp.csr_matrix:
    """Assemble a ``(len(person_order), vocab_size)`` CSR from per-person BOW rows.

    Mirrors ``analysis/cloud/lr_readout.py::build_test_bow``'s indptr
    accumulation, but aligned to an EXPLICIT ``person_order`` rather than the
    collected row order — so the feature matrix rows line up cell-for-cell with
    the label matrix built over the same order. ``features_by_person`` maps a
    person_id to that person's ``features`` SparseVector (anything exposing
    ``.indices`` / ``.values``, i.e. a pyspark ``SparseVector``). A person_id in
    ``person_order`` with no entry contributes an all-zero row (a patient with
    no in-vocab pre-index code), so the row count always equals
    ``len(person_order)``.
    """
    n = len(person_order)
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    for i, pid in enumerate(person_order):
        sv = features_by_person.get(pid)
        if sv is None:
            indptr[i + 1] = indptr[i]
            continue
        idx_chunks.append(np.asarray(sv.indices, dtype=np.int64))
        data_chunks.append(np.asarray(sv.values, dtype=np.float64))
        indptr[i + 1] = indptr[i] + len(sv.indices)
    indices = (
        np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=np.int64)
    )
    data = (
        np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float64)
    )
    return sp.csr_matrix((data, indices, indptr), shape=(n, vocab_size))


def stable_drug_order(
    present_drugs: Iterable[str],
    reference: Sequence[str] = (),
) -> list[str]:
    """A stable, reproducible column ordering for the distinct index drugs present.

    ``C`` — the multi-task label columns — is the set of index drugs that
    actually appear in the cohort. To make the column index deterministic across
    runs (so drug -> column is resume-stable and comparable between fits), the
    present drugs are ordered by their position in ``reference`` (the canonical
    15-drug ``_ANTIDEPRESSANT_INGREDIENTS`` list, in SSRI/SNRI/TCA/atypical
    order). Any present drug not in ``reference`` (defensive; not expected for
    the fixed 15-drug set) is appended in alphabetical order. With the default
    empty ``reference`` the whole set is returned alphabetically.
    """
    present = set(present_drugs)
    ordered = [d for d in reference if d in present]
    extra = sorted(present - set(reference))
    return ordered + extra


def assemble_multitask_labels(
    outcome_by_person: Mapping[Any, tuple[str, bool]],
    person_order: Sequence[Any],
    drug_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the index-drug ``(D, C)`` targets ``y`` and observed-mask ``mask``.

    The Hughes index-drug supervision pattern: each patient is labeled for
    EXACTLY the one antidepressant they initiated, and unobserved for every
    other drug. For person ``d`` (row) whose index drug is ``c`` (column) with
    outcome ``worked``, set ``y[d, c] = float(worked)`` and ``mask[d, c] = 1``;
    all other cells stay ``y = 0`` / ``mask = 0`` (unobserved). A person in
    ``person_order`` absent from ``outcome_by_person``, or whose index drug is
    not a column in ``drug_order``, contributes an all-unobserved row (``mask``
    all 0) — a valid unlabeled row for the shared PC, ignored by the baselines.

    ``outcome_by_person`` maps person_id -> ``(index_drug_name, worked)``.
    Returns ``(y, mask)``, both ``(D, C)`` float arrays with ``D =
    len(person_order)``, ``C = len(drug_order)``.
    """
    D, C = len(person_order), len(drug_order)
    col_of = {d: j for j, d in enumerate(drug_order)}
    y = np.zeros((D, C), dtype=np.float64)
    mask = np.zeros((D, C), dtype=np.float64)
    for i, pid in enumerate(person_order):
        rec = outcome_by_person.get(pid)
        if rec is None:
            continue
        drug, worked = rec
        j = col_of.get(drug)
        if j is None:
            continue
        y[i, j] = 1.0 if worked else 0.0
        mask[i, j] = 1.0
    return y, mask


def assemble_fullyobserved_labels(
    label_by_person: Mapping[Any, Iterable[str]],
    person_order: Sequence[Any],
    drug_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the FULLY-OBSERVED ``(D, C)`` targets ``y`` and all-ones ``mask``.

    The Hughes ``mdd_stable_treatment`` supervision pattern, contrasted with the
    per-index-drug :func:`assemble_multitask_labels`: here every present patient
    is labeled for EVERY drug column — ``y[d, c] = 1`` iff ``drug_order[c]`` is
    in that patient's stable drug subset, else ``0``, and the whole row is
    observed (``mask[d, :] = 1``). So a single-drug interval yields exactly one
    positive, a held combination (e.g. ``{fluoxetine, sertraline}``) yields two,
    and everything else is a true negative (the drug was NOT part of the stable
    regimen) rather than an unobserved cell. This is the mask-all-ones
    multi-label setup :func:`analysis.pc.evaluate.evaluate_pc_multitask` treats as
    standard ``C``-way multi-label.

    ``label_by_person`` maps person_id -> the stable drug subset (an iterable of
    ingredient names, i.e. the cohort index table's ``drug_subset``). A person in
    ``person_order`` absent from ``label_by_person`` contributes an all-zero,
    all-UNOBSERVED row (``mask`` all 0) — a valid unlabeled row, exactly as the
    per-drug assembler leaves a person with no outcome. ``drug_order`` is the
    FIXED length-``C`` column order (``_HUGHES_ANTIDEPRESSANTS``, length 10 — NOT
    a ``stable_drug_order`` over the drugs that happen to appear), so column c is
    the same Hughes drug across every run. Returns ``(y, mask)``, both ``(D, C)``
    with ``D = len(person_order)``, ``C = len(drug_order)``.
    """
    D, C = len(person_order), len(drug_order)
    col_of = {d: j for j, d in enumerate(drug_order)}
    y = np.zeros((D, C), dtype=np.float64)
    mask = np.zeros((D, C), dtype=np.float64)
    for i, pid in enumerate(person_order):
        subset = label_by_person.get(pid)
        if subset is None:
            continue                       # absent -> all-zero, all-unobserved
        mask[i, :] = 1.0                   # present -> the whole row is observed
        for name in subset:
            j = col_of.get(name)
            if j is not None:
                y[i, j] = 1.0
    return y, mask


def stratified_test_mask(
    groups: Sequence[Any],
    test_frac: float,
    seed: int,
) -> np.ndarray:
    """Seeded, per-group heldout mask — stratified by index drug where practical.

    Returns a boolean array (length ``len(groups)``) marking the heldout (test)
    rows. Within each distinct group value (the person's index drug, or ``None``
    for an unlabeled person) the rows are deterministically shuffled with a
    ``numpy`` Generator seeded from ``seed`` and the first ``round(test_frac *
    n_group)`` assigned to test. Stratifying per drug keeps every drug's
    positive/negative balance roughly matched across the split, which matters at
    the small per-drug counts the Hughes replication runs at; a singleton group
    (``round`` -> 0) simply keeps its lone row in train.
    """
    n = len(groups)
    is_test = np.zeros(n, dtype=bool)
    groups_arr = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    # Sorted group keys so the RNG-draw order is deterministic regardless of the
    # incoming person order (None sorts last via a stable key).
    keys = sorted({g for g in groups_arr.tolist()}, key=lambda g: (g is None, str(g)))
    for g in keys:
        idx = np.where(groups_arr == g)[0]
        rng.shuffle(idx)
        k = int(round(test_frac * len(idx)))
        is_test[idx[:k]] = True
    return is_test


def collect_bow_aligned(bow_df, vocab_size: int) -> tuple[np.ndarray, list[Any]]:
    """Collect a ``to_bow_dataframe`` BOW to a dense ``X`` + its person_id order.

    Collects ``bow_df[person_id, features]`` to the driver (as
    ``build_test_bow`` does for its heldout split), takes the collected row
    order as the canonical ``person_order``, and rebuilds a dense count matrix
    via :func:`bow_rows_to_matrix`. Under ``PatientDocSpec`` there is one BOW row
    per person, so ``person_order`` is a de-duplicated person list. Returns
    ``(X, person_order)`` with ``X`` dense ``(D, vocab_size)`` (evaluate_pc_*
    wants dense numpy) and ``person_order`` the row-aligned person_ids.
    """
    rows = bow_df.select("person_id", "features").collect()
    person_order = [r["person_id"] for r in rows]
    features_by_person = {r["person_id"]: r["features"] for r in rows}
    X = bow_rows_to_matrix(features_by_person, person_order, vocab_size).toarray()
    return X, person_order


# --------------------------------------------------------------------------- #
# VI backend helpers: attach the multi-task label/mask as Spark columns, split #
# by person at the DataFrame level, and collect a scored/labeled BOW to numpy. #
# --------------------------------------------------------------------------- #
def attach_multitask_label_columns(
    bow_df,
    outcome_by_person: Mapping[Any, tuple[str, bool]],
    drug_order: Sequence[str],
    spark,
    label_col: str = "y",
    mask_col: str = "label_mask",
):
    """Attach the per-patient multi-task label + mask to ``bow_df`` as columns.

    The Spark/column counterpart of :func:`assemble_multitask_labels`: it produces,
    per BOW row, the SAME length-``C`` label vector ``y`` (worked at the patient's
    index-drug column, else 0) and length-``C`` ``label_mask`` (1 at the index
    drug, 0 elsewhere) that the numpy assembler builds — keyed by index drug ->
    column via ``drug_order`` (identical order to :func:`stable_drug_order`), so the
    distributed VI-PC sees the exact same supervision as the in-memory path.

    IMPORTANT — column type. ``PCEstimator``'s shim (``_row_to_pc_document``) reads
    the label columns with ``isinstance(raw, (list, tuple, np.ndarray))`` and, on a
    miss, wraps the value as ``[raw]`` (promoting a scalar to length-1). A
    ``VectorUDT``/``DenseVector`` fails that check and would be wrapped to a spurious
    ``(1, C)`` shape, so the label/mask are emitted as Spark ``ArrayType(DoubleType)``
    columns — which deserialize to Python ``list`` and hit the ``isinstance`` fast
    path as a clean ``(C,)`` vector. (This differs from the STM covariate column,
    which the STM shim converts with a bare ``np.asarray(cov)`` and so can be a
    Vector.)

    The outcome dict is materialized to a tiny driver-side table and BROADCAST
    left-joined onto ``bow_df`` by ``person_id`` (mirroring the STM corpus+covariate
    broadcast join): a BOW person absent from ``outcome_by_person``, or whose index
    drug is not a column in ``drug_order``, gets all-zero ``y``/``label_mask`` — a
    valid unlabeled row (no observed cell), exactly as the numpy assembler leaves it.

    Returns ``bow_df`` with the two columns appended (the join keys ``_index_drug``/
    ``_worked`` are dropped); all original columns (``person_id``, ``features``, ...)
    are preserved.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType, BooleanType, DoubleType, StringType, StructField, StructType,
    )

    C = len(drug_order)
    col_of = {d: j for j, d in enumerate(drug_order)}

    person_dtype = bow_df.schema["person_id"].dataType
    schema = StructType([
        StructField("person_id", person_dtype, True),
        StructField("_index_drug", StringType(), True),
        StructField("_worked", BooleanType(), True),
    ])
    rows = [
        (pid, drug, bool(worked))
        for pid, (drug, worked) in outcome_by_person.items()
    ]
    outcome_df = spark.createDataFrame(rows, schema=schema)
    joined = bow_df.join(F.broadcast(outcome_df), on="person_id", how="left")

    def _y_vec(drug, worked):
        v = [0.0] * C
        j = col_of.get(drug)
        if j is not None and worked:
            v[j] = 1.0
        return v

    def _mask_vec(drug):
        v = [0.0] * C
        j = col_of.get(drug)
        if j is not None:
            v[j] = 1.0
        return v

    y_udf = F.udf(_y_vec, ArrayType(DoubleType()))
    mask_udf = F.udf(_mask_vec, ArrayType(DoubleType()))
    return (
        joined
        .withColumn(label_col, y_udf(F.col("_index_drug"), F.col("_worked")))
        .withColumn(mask_col, mask_udf(F.col("_index_drug")))
        .drop("_index_drug", "_worked")
    )


def attach_fullyobserved_label_columns(
    bow_df,
    label_by_person: Mapping[Any, Iterable[str]],
    drug_order: Sequence[str],
    spark,
    label_col: str = "y",
    mask_col: str = "label_mask",
):
    """Attach the fully-observed multi-label ``y`` + all-ones ``mask`` as columns.

    The Spark/column counterpart of :func:`assemble_fullyobserved_labels`, and the
    fully-observed sibling of :func:`attach_multitask_label_columns`: per BOW row
    it emits the SAME length-``C`` label vector ``y`` (1 at each column whose
    Hughes drug is in the patient's stable subset, else 0) and length-``C``
    ``label_mask`` (all-ones for a present patient, all-zero for one absent from
    ``label_by_person``) that the numpy assembler builds, keyed to columns by the
    FIXED ``drug_order`` (``_HUGHES_ANTIDEPRESSANTS``), so the distributed VI-PC
    sees the exact same supervision as the in-memory path.

    Same column-type contract as :func:`attach_multitask_label_columns`: the
    label/mask are Spark ``ArrayType(DoubleType)`` (NOT ``VectorUDT``) so
    ``PCEstimator``'s row shim deserializes them to Python ``list`` and reads a
    clean ``(C,)`` vector rather than wrapping a Vector to a spurious ``(1, C)``.

    ``label_by_person`` maps person_id -> the stable drug subset (list of
    ingredient names). It is materialized to a tiny driver-side table and
    BROADCAST left-joined by ``person_id``: a BOW person absent from
    ``label_by_person`` gets all-zero ``y`` AND all-zero ``label_mask`` (a valid
    unlabeled row), exactly as the numpy assembler leaves it. Returns ``bow_df``
    with the two columns appended (the join key ``_drug_subset`` is dropped).
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType, DoubleType, StringType, StructField, StructType,
    )

    C = len(drug_order)
    col_of = {d: j for j, d in enumerate(drug_order)}

    person_dtype = bow_df.schema["person_id"].dataType
    schema = StructType([
        StructField("person_id", person_dtype, True),
        StructField("_drug_subset", ArrayType(StringType()), True),
    ])
    rows = [
        (pid, [str(name) for name in subset])
        for pid, subset in label_by_person.items()
    ]
    label_df = spark.createDataFrame(rows, schema=schema)
    joined = bow_df.join(F.broadcast(label_df), on="person_id", how="left")

    def _y_vec(subset):
        v = [0.0] * C
        if subset is not None:
            for name in subset:
                j = col_of.get(name)
                if j is not None:
                    v[j] = 1.0
        return v

    def _mask_vec(subset):
        # All-ones for a present patient (fully observed), all-zero for absent.
        return [1.0] * C if subset is not None else [0.0] * C

    y_udf = F.udf(_y_vec, ArrayType(DoubleType()))
    mask_udf = F.udf(_mask_vec, ArrayType(DoubleType()))
    return (
        joined
        .withColumn(label_col, y_udf(F.col("_drug_subset")))
        .withColumn(mask_col, mask_udf(F.col("_drug_subset")))
        .drop("_drug_subset")
    )


def person_hash_split(
    df,
    test_frac: float,
    seed: int,
    key_col: str = "person_id",
    buckets: int = 10_000,
):
    """Seeded, DataFrame-level train/test split keyed on ``person_id``.

    Assigns each row deterministically by hashing ``(person_id, seed)`` into
    ``buckets`` buckets (Spark's ``F.hash``, made non-negative via ``pmod``) and
    holding out the buckets below ``round(test_frac * buckets)``. Under
    ``PatientDocSpec`` there is one row per person, so this is a per-person split;
    it is reproducible from ``(seed, test_frac)`` alone regardless of partitioning
    (unlike ``DataFrame.randomSplit``, whose result depends on partition layout).
    Not stratified by index drug — the in-memory path's
    :func:`stratified_test_mask` is; at the VI backend's corpus scale a hash split
    is adequate and keeps the split fully distributed. Returns ``(train_df,
    test_df)`` — disjoint, together the whole input.
    """
    from pyspark.sql import functions as F

    salted = F.concat_ws("_", F.col(key_col).cast("string"), F.lit(str(int(seed))))
    bucket = F.pmod(F.hash(salted), F.lit(int(buckets)))
    cut = int(round(float(test_frac) * buckets))
    df_h = df.withColumn("_split_bucket", bucket)
    test_df = df_h.where(F.col("_split_bucket") < F.lit(cut)).drop("_split_bucket")
    train_df = df_h.where(F.col("_split_bucket") >= F.lit(cut)).drop("_split_bucket")
    return train_df, test_df


def collect_labeled_bow(df, vocab_size: int, C: int, prob_col: str | None = None):
    """Collect a label-columned (optionally scored) BOW df to dense numpy arrays.

    The VI backend's bridge back to the shared numpy scoring/baseline helpers:
    collects ``person_id``, ``features``, ``y``, ``label_mask`` (and, when
    ``prob_col`` is set, the transform's per-label ``probabilityCol``) and returns
    ``(X, y_DC, mask_DC, proba_DC, person_order)`` — ``X`` dense ``(D, vocab_size)``
    via :func:`bow_rows_to_matrix`, ``y_DC``/``mask_DC`` the ``(D, C)`` arrays the
    ``ArrayType`` columns deserialize to, ``proba_DC`` the ``(D, C)`` head
    probabilities (or ``None`` when ``prob_col`` is None), all row-aligned to the
    collected ``person_order``. Empty df yields correctly-shaped ``(0, ...)``
    arrays.
    """
    cols = ["person_id", "features", "y", "label_mask"]
    if prob_col is not None:
        cols.append(prob_col)
    rows = df.select(*cols).collect()
    person_order = [r["person_id"] for r in rows]
    features_by_person = {r["person_id"]: r["features"] for r in rows}
    X = bow_rows_to_matrix(features_by_person, person_order, vocab_size).toarray()
    if rows:
        y_DC = np.asarray([[float(v) for v in r["y"]] for r in rows], dtype=np.float64)
        mask_DC = np.asarray(
            [[float(v) for v in r["label_mask"]] for r in rows], dtype=np.float64
        )
        proba_DC = (
            np.asarray([r[prob_col].toArray() for r in rows], dtype=np.float64)
            if prob_col is not None else None
        )
    else:
        y_DC = np.zeros((0, C), dtype=np.float64)
        mask_DC = np.zeros((0, C), dtype=np.float64)
        proba_DC = None if prob_col is None else np.zeros((0, C), dtype=np.float64)
    return X, y_DC, mask_DC, proba_DC, person_order


def _collect_topics_labels(df, C, topic_col="topicDistribution"):
    """Collect ONLY the K-dim per-doc topic vector + labels to numpy (no dense BOW).

    The lightweight collector for the distributed two-stage baseline: it pulls the
    ``topicDistribution`` (theta, K-dim) plus the ``y``/``label_mask`` arrays and
    NEVER the dense ``(D, V)`` counts, so it stays on the driver's memory budget at
    full-cohort scale. Returns ``(Pi (D, K), y_DC (D, C), mask_DC (D, C),
    person_order)`` row-aligned to the collected order. Empty df yields
    correctly-shaped zero arrays.
    """
    rows = df.select("person_id", topic_col, "y", "label_mask").collect()
    person_order = [r["person_id"] for r in rows]
    if rows:
        Pi = np.asarray([r[topic_col].toArray() for r in rows], dtype=np.float64)
        y_DC = np.asarray([[float(v) for v in r["y"]] for r in rows], dtype=np.float64)
        mask_DC = np.asarray(
            [[float(v) for v in r["label_mask"]] for r in rows], dtype=np.float64
        )
    else:
        Pi = np.zeros((0, 0), dtype=np.float64)
        y_DC = np.zeros((0, C), dtype=np.float64)
        mask_DC = np.zeros((0, C), dtype=np.float64)
    return Pi, y_DC, mask_DC, person_order


def _vi_two_stage_bundle(train_df, test_df, C, args):
    """Distributed-SVI two-stage baseline bundle (unsupervised topics -> per-label LR).

    The SVI-consistent replacement for the in-memory ``PCTopicModel(weight_y=0)``
    two-stage baseline: it fits a SECOND ``PCEstimator`` at ``weightY=0`` — the SAME
    distributed machinery as the VI-PC model (and the warm-start phase 1), so there
    is NO collect-to-driver and NO in-memory L-BFGS — transforms train/test to the
    K-dim ``topicDistribution`` (theta), collects only those small vectors
    (:func:`_collect_topics_labels`), and fits one masked
    :class:`~sklearn.linear_model.LogisticRegression` per outcome on the frozen
    topics. Same two-stage recipe as before, but its topics now come from the same
    SVI estimator (CAVI theta) as the model rather than the reference's NEF-MAP pi —
    a fidelity improvement for the VI path, and it removes the driver-side
    dense-collect + in-memory fit that OOM'd. ``--baseline-max-iter`` caps this
    extra fit (unsupervised topics converge faster than the supervised head;
    ``<= 0`` => use ``--max-iter``).

    Scored over the baseline's OWN collected test labels: per-column AUC/AP is
    row-order-invariant and the test membership is the same deterministic person
    split, so the bundle is directly comparable to the PC / LR-on-codes bundles.
    """
    from spark_vi.mllib.topic.pc import PCEstimator
    from analysis.pc.evaluate import _bundle_masked, _lr_proba_per_label_masked

    n_iter = (args.baseline_max_iter
              if getattr(args, "baseline_max_iter", 0) and args.baseline_max_iter > 0
              else args.max_iter)
    est = PCEstimator(
        featuresCol="features", labelCol="y", labelMaskCol="label_mask",
        numLabels=C, weightY=0.0, k=args.K, docConcentration=[float(args.alpha)],
        subsamplingRate=args.subsampling_rate,
        learningOffset=args.tau0, learningDecay=args.kappa,
        maxIter=int(n_iter), seed=args.seed,
        topicDistributionCol="topicDistribution",
    )
    model = est.fit(train_df)
    Pi_tr, y_tr, mask_tr, _ = _collect_topics_labels(model.transform(train_df), C)
    Pi_te, y_te, mask_te, _ = _collect_topics_labels(model.transform(test_df), C)
    ts_proba = _lr_proba_per_label_masked(Pi_tr, y_tr, mask_tr, Pi_te, C)
    return _bundle_masked(ts_proba, y_te, mask_te, C, args.min_label_count)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show defaults automatically, and preserve the docstring's paragraph layout."""


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (env-defaulted cdr/billing).

    Extracted so the argparse surface is inspectable without running ``main``.
    ``--cdr`` / ``--billing`` default from the workspace env vars, so an unset
    env leaves them ``None`` and ``main`` validates+exits before touching BQ.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    parser.add_argument(
        "--cdr", default=os.environ.get("WORKSPACE_CDR"),
        help="BQ CDR dataset '<project>.<dataset>' (default: $WORKSPACE_CDR)",
    )
    # --- Backend selector ----------------------------------------------------
    parser.add_argument(
        "--backend", choices=("inmem", "vi"), default="inmem",
        help=("PC fit backend. 'inmem' (default) = the in-memory L-BFGS "
              "PCTopicModel run on the driver via evaluate_pc_multitask (current "
              "behavior, unchanged). 'vi' = the distributed VI-native PCEstimator "
              "(SVI, no collect-to-memory for the fit); the two-stage / LR-on-codes "
              "baselines are still collected+fit in memory so the numbers are "
              "comparable to the same baselines."),
    )
    parser.add_argument(
        "--billing", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP billing/compute project (default: $GOOGLE_CLOUD_PROJECT)",
    )
    # --- PC / eval hyperparameters (passed through to evaluate_pc_multitask) --
    parser.add_argument("--K", type=int, default=20, help="number of PC topics")
    parser.add_argument(
        "--weight-y", type=float, default=1.0,
        help="PC prediction weight (> 0). The two-stage baseline refits weight_y=0.",
    )
    parser.add_argument("--alpha", type=float, default=1.1, help="PC theta Dirichlet alpha")
    parser.add_argument("--tau", type=float, default=1.1, help="PC beta Dirichlet tau")
    parser.add_argument("--pi-iters", type=int, default=100, help="per-doc pi CAVI iters")
    parser.add_argument("--max-iter", type=int, default=500, help="PC global max iters")
    parser.add_argument(
        "--doc-batch-size", type=int, default=2048,
        help=("document minibatch size for the PC full-batch gradient assembly; "
              "bounds driver autograd memory (~ doc_batch_size x pi_iters x V) at "
              "real-corpus scale without changing the objective or optimizer "
              "(inmem backend only)"),
    )
    # --- Distributed-SVI knobs (backend=vi only; ignored for inmem) -----------
    parser.add_argument(
        "--subsampling-rate", type=float, default=0.05,
        help=("VI backend: mini-batch fraction per SVI iteration. 1.0 = full-batch. "
              "Maps to PCEstimator.subsamplingRate. Ignored for --backend inmem."),
    )
    parser.add_argument(
        "--tau0", type=float, default=1024.0,
        help=("VI backend: Robbins-Monro learning offset tau0 in "
              "rho_t = (tau0 + t)^-kappa. Maps to PCEstimator.learningOffset. "
              "On smaller cohorts try ~10-64 so the head actually moves. Ignored "
              "for --backend inmem."),
    )
    parser.add_argument(
        "--kappa", type=float, default=0.51,
        help=("VI backend: Robbins-Monro learning decay kappa in "
              "rho_t = (tau0 + t)^-kappa (must be in (0.5, 1.0]). Maps to "
              "PCEstimator.learningDecay. Ignored for --backend inmem."),
    )
    parser.add_argument(
        "--head-lr-scale", type=float, default=1.0,
        help=("VI backend: extra multiplier on the LOGISTIC-HEAD SGD step ONLY "
              "(w_CK <- w_CK - rho*head_lr_scale*wy*grad); the topic/lambda step is "
              "untouched. Maps to PCEstimator.headLrScale. Use > 1 to converge the "
              "head in fewer iters when the topics are already stable but the head "
              "is under-moved (|w_CK| still climbing at max_iter) — the targeted "
              "alternative to lowering tau0, which speeds up everything. Pair a hot "
              "head with --weight-y-warmup-iters so it does not spike early. Ignored "
              "for --backend inmem."),
    )
    parser.add_argument(
        "--weight-y-warmup-iters", type=int, default=0,
        help=("VI backend: linearly ramp the effective weight_y from 0 to its full "
              "value over this many global SVI steps (0 = no warmup). Maps to "
              "PCEstimator.weightYWarmupIters. Softens the first supervised steps so "
              "a large weight_y and/or --head-lr-scale > 1 does not spike the head "
              "on early, high-variance minibatches. Ignored for --backend inmem."),
    )
    # --- Baseline controls (the two-stage / LR-on-codes comparison set) --------
    parser.add_argument(
        "--skip-two-stage", action="store_true",
        help=("skip the two-stage (unsupervised-topics -> per-label LR) baseline "
              "entirely; report only PC + LR-on-codes. For the VI backend the "
              "two-stage is a SECOND distributed SVI fit (weight_y=0), so skipping "
              "it roughly halves the wall-clock — handy for a fast --eval-only "
              "readout off a saved checkpoint. LR-on-codes still runs."),
    )
    parser.add_argument(
        "--baseline-max-iter", type=int, default=-1,
        help=("cap the two-stage baseline's unsupervised topic fit at this many "
              "iterations; <= 0 (default) reuses --max-iter. Unsupervised topics "
              "converge faster than the supervised head, so a smaller value (e.g. "
              "100) keeps the extra VI two-stage fit cheap. For --backend vi this "
              "caps the distributed weight_y=0 PCEstimator; for inmem it caps the "
              "in-memory PCTopicModel(weight_y=0) fit."),
    )
    parser.add_argument(
        "--warm-start-unsup-iters", type=int, default=0,
        help=("VI backend: unsupervised-warm-start protocol (Hughes et al.). "
              "0 (default) = cold start (single-phase supervised fit, byte-for-byte "
              "the prior behavior). N > 0 runs a two-phase fit: PHASE 1 fits the "
              "SAME PCEstimator machinery at weight_y=0 (unsupervised LDA-MAP) for "
              "N SVI iters to learn topics (the head stays at its zero init), then "
              "PHASE 2 warm-starts the supervised fit (real weight_y, --max-iter "
              "iters) from phase-1's topics with a FRESH Robbins-Monro schedule "
              "(rho restarts near rho_0) and a fresh zero head. Distinct from "
              "--resume-from (which continues the decayed counter). Ignored for "
              "--backend inmem; skipped on --resume-from (resuming continues an "
              "existing phase-2 fit)."),
    )
    # --- Feature window ------------------------------------------------------
    parser.add_argument(
        "--lookback-days", type=int, default=365,
        help="pre-index feature window: events in [index - lookback_days, index)",
    )
    # --- Cohort selector -----------------------------------------------------
    parser.add_argument(
        "--cohort", choices=("mdd_antidepressant", "mdd_stable_treatment"),
        default="mdd_antidepressant",
        help=("cohort + label shape. 'mdd_antidepressant' (default) = the "
              "per-index-drug incident-new-user cohort with the >=90-day "
              "'the drug worked' outcome (one observed cell per patient = the "
              "drug they initiated; current behavior, unchanged). "
              "'mdd_stable_treatment' = the Hughes-faithful stable-treatment "
              "cohort over the 10-drug set with a FULLY-OBSERVED length-10 "
              "indicator of the stable drug subset (every head trains on every "
              "patient; all-history pre-index features). The stable-treatment "
              "knobs below (--min-days/--max-gap-days/--min-history-events/"
              "--age-min/--age-max) apply only to this cohort; the antidepressant "
              "knobs (--window-days/--stability-days/--grace-gap-days/"
              "--lookback-days) apply only to the other."),
    )
    # --- Stable-treatment knobs (--cohort mdd_stable_treatment only) ----------
    parser.add_argument(
        "--min-days", type=int, default=90,
        help=("stable-treatment: minimum stable-interval length (days). A "
              "constant-subset antidepressant interval must last >= this to "
              "qualify. Ignored for --cohort mdd_antidepressant."),
    )
    parser.add_argument(
        "--max-gap-days", type=int, default=395,
        help=("stable-treatment: maximum permissible visit gap (days) bounding "
              "encounter regularity (Hughes: an encounter at least every ~13 "
              "months, both interval endpoints included). Ignored for "
              "--cohort mdd_antidepressant."),
    )
    parser.add_argument(
        "--min-history-events", type=int, default=2,
        help=("stable-treatment: minimum number of events strictly before the "
              "first antidepressant era (pre-treatment history requirement). "
              "Ignored for --cohort mdd_antidepressant."),
    )
    parser.add_argument(
        "--age-min", type=int, default=18,
        help=("stable-treatment: minimum age (inclusive) at the stable-interval "
              "start. Ignored for --cohort mdd_antidepressant."),
    )
    parser.add_argument(
        "--age-max", type=int, default=80,
        help=("stable-treatment: maximum age (inclusive) at the stable-interval "
              "start. Ignored for --cohort mdd_antidepressant."),
    )
    # --- Cohort + outcome ----------------------------------------------------
    parser.add_argument(
        "--window-days", type=int, default=365,
        help=("cohort follow-up observability requirement (days); must be >= "
              "--stability-days so the stability window is fully observed"),
    )
    parser.add_argument(
        "--stability-days", type=int, default=90,
        help="'the drug worked' continuation horizon (Hughes ~3 months)",
    )
    parser.add_argument(
        "--grace-gap-days", type=int, default=30,
        help="permissible same-ingredient refill gap when stitching coverage",
    )
    # --- BOW vocab -----------------------------------------------------------
    parser.add_argument("--vocab-size", type=int, default=2000, help="BOW vocabulary cap")
    parser.add_argument(
        "--min-df", type=int, default=20,
        help="minimum document (patient) frequency for vocab inclusion",
    )
    parser.add_argument(
        "--min-patient-count", type=int, default=20,
        help="minimum distinct-patient count for vocab inclusion (privacy guard)",
    )
    # --- Sampling / split / reproducibility ----------------------------------
    parser.add_argument(
        "--person-mod", type=int, default=1,
        help="whole-patient sampling: keep person_id %% M == 0 (1 = full cohort)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.25,
        help="heldout fraction, stratified by index drug",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (split + PC init)")
    parser.add_argument(
        "--min-label-count", type=int, default=20,
        help=("mask any drug label whose HELDOUT test column has fewer than this "
              "many cells of either class: its per-drug AUC/AP is dropped from "
              "the macro-average and its counts are suppressed in the printed "
              "table (an All-of-Us count < 20 is not disclosable and its AUC is "
              "too noisy to trust). Default 20 (the AoU small-cell floor); set 0 "
              "to disable and score every non-degenerate label."),
    )
    parser.add_argument(
        "--cache-uri", type=str, default=None,
        help=("optional cache root reserved for the BQ-load + BOW prep step "
              "(not yet wired; accepted so the CLI surface is stable)"),
    )
    parser.add_argument(
        "--out", type=str, default="pc_antidepressant_results.json",
        help="path to write the per-drug results (JSON)",
    )
    # --- Checkpoint / resume / eval (VI backend only) ------------------------
    # Mirror lda_bigquery_cloud.py's persistence flags. These are meaningful
    # ONLY for --backend vi: the VI-native PCEstimator checkpoints its VIResult
    # via the shim's saveDir/saveInterval and resumes via resumeFrom. The inmem
    # (L-BFGS) backend has NO interim state to checkpoint, so these are ignored
    # for --backend inmem.
    parser.add_argument(
        "--save-dir", default="",
        help=("VI backend only: directory for auto-saves and final result; "
              "empty (default) = no save. The directory becomes the "
              "authoritative post-fit artifact (manifest.json + params/), "
              "loadable via PCModel.load and usable as --resume-from. Ignored "
              "for --backend inmem (L-BFGS has no interim state)."),
    )
    parser.add_argument(
        "--save-interval", type=int, default=-1,
        help=("VI backend only: save every N iters during fit; -1 (default) = "
              "save only at end-of-fit if --save-dir is set. Ignored for "
              "--backend inmem."),
    )
    parser.add_argument(
        "--resume-from", default="",
        help=("VI backend only: path to a previously-written --save-dir; empty "
              "(default) = fresh start. When set, the fit loads the saved "
              "VIResult and continues from that iteration count (--max-iter is "
              "then ADDITIONAL iterations). Ignored for --backend inmem."),
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help=("VI backend only: skip training; load the checkpoint VIResult at "
              "--save-dir, wrap it in a PCModel, and run the per-drug eval "
              "(transform + score) so you can peek the AUC from a checkpoint "
              "without more fit. Requires --save-dir with a manifest.json."),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate the environment/CLI BEFORE importing charmpheno/analysis or
    # opening a Spark session, so a misconfigured invocation fails fast and the
    # driver stays importable+arg-checkable without the cluster deps (mirrors
    # lda_bigquery_cloud.py's env gate).
    cdr = args.cdr
    billing = args.billing
    if not (cdr and billing):
        print("ERROR: --cdr/--billing unset and WORKSPACE_CDR / "
              "GOOGLE_CLOUD_PROJECT missing from env. Run the workspace setup "
              "notebook (or `source ~/.bashrc`), or pass --cdr/--billing.",
              file=sys.stderr)
        return 1
    # The window-vs-stability bracket is a mdd_antidepressant-only constraint
    # (its follow-up window must fully observe the stability horizon). The
    # stable-treatment cohort has no such forward window — its observability gate
    # is "the stable interval falls within one observation period" — so the guard
    # is scoped to the antidepressant cohort. mdd_antidepressant is the default,
    # so the existing CLI behavior is unchanged.
    if args.cohort == "mdd_antidepressant" and args.window_days < args.stability_days:
        print(f"ERROR: --window-days ({args.window_days}) must be >= "
              f"--stability-days ({args.stability_days}) so the stability window "
              "is fully observed.", file=sys.stderr)
        return 1
    if args.cohort == "mdd_stable_treatment":
        if args.min_days <= 0:
            print(f"ERROR: --min-days must be > 0, got {args.min_days}.",
                  file=sys.stderr)
            return 1
        if args.age_min > args.age_max:
            print(f"ERROR: --age-min ({args.age_min}) must be <= --age-max "
                  f"({args.age_max}).", file=sys.stderr)
            return 1
    # Checkpoint/resume/eval are VI-native (the inmem L-BFGS backend has no
    # interim state). Validate the combinations BEFORE touching BQ so a
    # misconfigured invocation fails fast (mirrors the env gate above).
    if args.save_dir and args.backend != "vi":
        print(f"ERROR: --save-dir is VI-only (the inmem L-BFGS backend has no "
              f"interim state to checkpoint); got --backend {args.backend}.",
              file=sys.stderr)
        return 1
    if args.resume_from and args.backend != "vi":
        print(f"ERROR: --resume-from is VI-only; got --backend {args.backend}.",
              file=sys.stderr)
        return 1
    if args.warm_start_unsup_iters and args.backend != "vi":
        print(f"ERROR: --warm-start-unsup-iters is VI-only (the inmem L-BFGS "
              f"backend has no SVI phase-1 to warm-start from); got --backend "
              f"{args.backend}.", file=sys.stderr)
        return 1
    if args.warm_start_unsup_iters < 0:
        print(f"ERROR: --warm-start-unsup-iters must be >= 0, got "
              f"{args.warm_start_unsup_iters}.", file=sys.stderr)
        return 1
    if args.eval_only:
        if args.backend != "vi":
            print(f"ERROR: --eval-only is VI-only (only the VI backend writes a "
                  f"loadable checkpoint); got --backend {args.backend}.",
                  file=sys.stderr)
            return 1
        if not args.save_dir:
            print("ERROR: --eval-only requires --save-dir pointing at a "
                  "checkpoint directory.", file=sys.stderr)
            return 1
        if not (Path(args.save_dir) / "manifest.json").exists():
            print(f"ERROR: --eval-only: no checkpoint (manifest.json) at "
                  f"{args.save_dir}; nothing to evaluate.", file=sys.stderr)
            return 1

    # Driver-side imports proven first — fail fast if --py-files is misshapen.
    from charmpheno.omop import load_omop_bigquery, to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec
    from charmpheno.omop.cohorts import (
        _ANTIDEPRESSANT_INGREDIENTS,
        _HUGHES_ANTIDEPRESSANTS,
        _antidepressant_concept_map,
        all_history_feature_events,
        antidepressant_stability_label,
        apply_mdd_antidepressant_cohort,
        apply_mdd_stable_treatment_cohort,
        lookback_feature_label_events,
        stable_treatment_label,
    )
    from analysis.pc.evaluate import (
        _bundle_masked,
        _lr_proba_per_label_masked,
        evaluate_pc_multitask,
        format_results_table,
    )

    configure_logging()
    print(f"[driver] cdr={cdr}, billing_project={billing}, cohort={args.cohort}, "
          f"backend={args.backend}, K={args.K}, weight_y={args.weight_y}, "
          f"lookback_days={args.lookback_days}, window_days={args.window_days}, "
          f"stability_days={args.stability_days}, person_mod={args.person_mod}",
          flush=True)
    if args.backend == "vi" and (args.save_dir or args.resume_from or args.eval_only):
        print(f"[driver] checkpoint: save_dir={args.save_dir or '<none>'}, "
              f"save_interval={args.save_interval}, "
              f"resume_from={args.resume_from or '<none>'}, "
              f"eval_only={args.eval_only}", flush=True)
    if args.cache_uri:
        print(f"[driver] note: --cache-uri {args.cache_uri!r} accepted but the "
              "prep-cache path is not yet wired; prep runs fresh.", flush=True)

    spark = make_spark_session("pc_antidepressant_cloud")

    def _read(table: str):
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr}.{table}")
            .option("parentProject", billing)
            .load()
        )

    # ===================================================================== #
    # mdd_stable_treatment: the Hughes-faithful stable-treatment path.       #
    # Self-contained (cohort -> fully-observed 10-label -> all-history fused #
    # features -> eval), with its own report/JSON tail, so the per-index-    #
    # drug mdd_antidepressant path below stays byte-for-byte unchanged.      #
    # ===================================================================== #
    if args.cohort == "mdd_stable_treatment":
        drug_order_fixed = list(_HUGHES_ANTIDEPRESSANTS)

        # --- 1) Cohort index (per-person stable-treatment table) --------------
        with _phase("cohort index (apply_mdd_stable_treatment_cohort)"):
            cond_df = load_omop_bigquery(
                spark=spark, cdr_dataset=cdr, billing_project=billing,
                concept_types=("condition",), person_sample_mod=args.person_mod,
                cohort=None,
            )
            index_df = apply_mdd_stable_treatment_cohort(
                cond_df, spark=spark, cdr_dataset=cdr, billing_project=billing,
                date_col=_CONDITION_DATE,
                min_days=args.min_days, max_gap_days=args.max_gap_days,
                min_history_events=args.min_history_events,
                age_min=args.age_min, age_max=args.age_max,
            ).persist()
            n_index = index_df.count()
            print(f"[driver]   MDD stable-treatment persons: {n_index}", flush=True)

        # --- 2) Outcome (fully-observed length-10 Hughes indicator) -----------
        with _phase("outcome (stable_treatment_label, fully-observed 10-drug)"):
            # stable_treatment_label is the committed length-10 indicator over the
            # fixed Hughes order; the driver reconstructs each person's drug SUBSET
            # (names) from that indicator to feed the fully-observed assemblers.
            label_rows = stable_treatment_label(
                index_df, drug_order=_HUGHES_ANTIDEPRESSANTS,
            ).collect()
            label_by_person = {
                r["person_id"]: [
                    drug_order_fixed[i] for i, v in enumerate(r["y"])
                    if float(v) > 0.5
                ]
                for r in label_rows
            }
            pos = [0] * len(drug_order_fixed)
            for r in label_rows:
                for i, v in enumerate(r["y"]):
                    if float(v) > 0.5:
                        pos[i] += 1
            print(f"[driver]   stable-treatment labels: {len(label_by_person)} "
                  "persons; per-drug positives: "
                  + ", ".join(f"{drug_order_fixed[i]}={pos[i]}"
                              for i in range(len(pos))), flush=True)

        # --- 3) Fused ALL-HISTORY features -> BOW -----------------------------
        with _phase("fused features -> all-history window -> BOW"):
            fused = load_omop_bigquery(
                spark=spark, cdr_dataset=cdr, billing_project=billing,
                concept_types=("condition", "drug", "procedure"),
                person_sample_mod=args.person_mod, cohort=None,
            )
            feature_events = all_history_feature_events(
                fused, index_df.select("person_id", "index_date", "source_cohort"),
                date_col=_FUSED_EVENT_DATE,
            )
            bow_df, vocab_map = to_bow_dataframe(
                feature_events, doc_spec=PatientDocSpec(), token_col="concept_id",
                vocab_size=args.vocab_size, min_df=args.min_df,
                min_patient_count=args.min_patient_count,
            )
            bow_df = bow_df.persist()
            V = len(vocab_map)
            n_docs = bow_df.count()
            print(f"[driver]   vocab size: {V} (cap {args.vocab_size}), "
                  f"documents: {n_docs}", flush=True)

        # --- 4/5) Backend split: in-memory L-BFGS (default) vs distributed VI --
        if args.backend == "inmem":
            results, drug_order, n_train, n_test, n_persons = (
                _run_inmem_backend_fullyobserved(
                    bow_df, V, label_by_person, drug_order_fixed, args,
                    evaluate_pc_multitask,
                )
            )
        else:
            results, drug_order, n_train, n_test, n_persons = (
                _run_vi_backend_fullyobserved(
                    spark, bow_df, V, label_by_person, drug_order_fixed, args,
                    _lr_proba_per_label_masked, _bundle_masked, vocab_map,
                )
            )

        # --- Report + JSON (per-drug Hughes table, fully-observed) ------------
        print("\n[driver] per-drug results (column index -> Hughes drug):",
              flush=True)
        for c, name in enumerate(drug_order):
            print(f"[driver]   label {c} = {name}", flush=True)
        print(format_results_table(results), flush=True)
        _log_convergence(results["meta"])

        out_payload = {
            "results": results,
            "backend": args.backend,
            "cohort": args.cohort,
            "drug_order": drug_order,
            "column_drug_names": {c: name for c, name in enumerate(drug_order)},
            "vocab_size": V,
            "n_persons": n_persons,
            "n_train": n_train,
            "n_test": n_test,
            "params": {
                "backend": args.backend,
                "cohort": args.cohort,
                "K": args.K, "weight_y": args.weight_y, "alpha": args.alpha,
                "tau": args.tau, "pi_iters": args.pi_iters, "max_iter": args.max_iter,
                "doc_batch_size": args.doc_batch_size,
                "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
                "kappa": args.kappa,
                "head_lr_scale": args.head_lr_scale,
                "weight_y_warmup_iters": args.weight_y_warmup_iters,
                "warm_start_unsup_iters": args.warm_start_unsup_iters,
                # stable-treatment membership knobs (all-history features; no
                # lookback/window/stability bracket for this cohort).
                "min_days": args.min_days, "max_gap_days": args.max_gap_days,
                "min_history_events": args.min_history_events,
                "age_min": args.age_min, "age_max": args.age_max,
                "vocab_size": args.vocab_size, "min_df": args.min_df,
                "min_patient_count": args.min_patient_count,
                "person_mod": args.person_mod,
                "test_frac": args.test_frac, "seed": args.seed,
                "min_label_count": args.min_label_count,
                "skip_two_stage": args.skip_two_stage,
                "baseline_max_iter": args.baseline_max_iter,
            },
        }
        with open(args.out, "w") as f:
            json.dump(out_payload, f, indent=2)
        print(f"[driver] wrote results to {args.out}", flush=True)

        bow_df.unpersist()
        index_df.unpersist()
        print("[driver] done", flush=True)
        spark.stop()
        return 0

    # --- 1) Cohort index (per-person MDD antidepressant initiator table) ------
    with _phase("cohort index (apply_mdd_antidepressant_cohort)"):
        cond_df = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            concept_types=("condition",), person_sample_mod=args.person_mod,
            cohort=None,
        )
        index_df = apply_mdd_antidepressant_cohort(
            cond_df, spark=spark, cdr_dataset=cdr, billing_project=billing,
            date_col=_CONDITION_DATE,
            window_days=args.window_days, prior_obs_days=args.lookback_days,
        ).persist()
        n_index = index_df.count()
        print(f"[driver]   MDD antidepressant initiators: {n_index}", flush=True)

    # --- 2) Outcome (>=stability_days continuation) ---------------------------
    with _phase("outcome (antidepressant_stability_label)"):
        concept = _read("concept").select(
            "concept_id", "concept_name", "vocabulary_id", "concept_class_id",
        )
        ca = _read("concept_ancestor").select(
            "ancestor_concept_id", "descendant_concept_id",
        )
        drug_era = _read("drug_era").select(
            "person_id", "drug_concept_id", "drug_era_start_date",
            "drug_era_end_date", "gap_days",
        )
        # Same 15-drug concept map the cohort builds internally, rebuilt here so
        # the labeler can name each era's ingredient (switch detection). This is
        # the documented way to obtain drug_concept_sets for the labeler.
        concept_map = _antidepressant_concept_map(concept, ca)
        outcome_df = antidepressant_stability_label(
            drug_era, index_df, drug_concept_sets=concept_map,
            stability_days=args.stability_days, grace_gap_days=args.grace_gap_days,
        )
        outcome_rows = outcome_df.collect()
        outcome_by_person = {
            r["person_id"]: (r["index_drug_name"], bool(r["worked"]))
            for r in outcome_rows
        }
        n_worked = sum(1 for _, w in outcome_by_person.values() if w)
        print(f"[driver]   outcomes: {len(outcome_by_person)} persons, "
              f"{n_worked} worked=True", flush=True)

    # --- 3) Fused pre-index features -> BOW -----------------------------------
    with _phase("fused features -> pre-index window -> BOW"):
        fused = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            concept_types=("condition", "drug", "procedure"),
            person_sample_mod=args.person_mod, cohort=None,
        )
        feature_events, _ = lookback_feature_label_events(
            fused, index_df.select("person_id", "index_date", "source_cohort"),
            date_col=_FUSED_EVENT_DATE, lookback_days=args.lookback_days,
            label_window_days=args.window_days,
        )
        bow_df, vocab_map = to_bow_dataframe(
            feature_events, doc_spec=PatientDocSpec(), token_col="concept_id",
            vocab_size=args.vocab_size, min_df=args.min_df,
            min_patient_count=args.min_patient_count,
        )
        bow_df = bow_df.persist()
        V = len(vocab_map)
        n_docs = bow_df.count()
        print(f"[driver]   vocab size: {V} (cap {args.vocab_size}), "
              f"documents: {n_docs}", flush=True)

    # --- 4/5) Backend split: in-memory L-BFGS (default) vs distributed VI ------
    if args.backend == "inmem":
        results, drug_order, n_train, n_test, n_persons = _run_inmem_backend(
            bow_df, V, outcome_by_person, _ANTIDEPRESSANT_INGREDIENTS, args,
            evaluate_pc_multitask,
        )
    else:
        results, drug_order, n_train, n_test, n_persons = _run_vi_backend(
            spark, bow_df, V, outcome_by_person, _ANTIDEPRESSANT_INGREDIENTS, args,
            _lr_proba_per_label_masked, _bundle_masked, vocab_map,
        )

    # --- Report: the Hughes per-drug table (PC vs two-stage vs LR-on-codes) ---
    print("\n[driver] per-drug results (column index -> index drug):", flush=True)
    for c, name in enumerate(drug_order):
        print(f"[driver]   label {c} = {name}", flush=True)
    print(format_results_table(results), flush=True)
    _log_convergence(results["meta"])

    out_payload = {
        "results": results,
        "backend": args.backend,
        "drug_order": drug_order,
        "column_drug_names": {c: name for c, name in enumerate(drug_order)},
        "vocab_size": V,
        "n_persons": n_persons,
        "n_train": n_train,
        "n_test": n_test,
        "params": {
            "backend": args.backend,
            "K": args.K, "weight_y": args.weight_y, "alpha": args.alpha,
            "tau": args.tau, "pi_iters": args.pi_iters, "max_iter": args.max_iter,
            "doc_batch_size": args.doc_batch_size,
            "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
            "kappa": args.kappa,
            "head_lr_scale": args.head_lr_scale,
            "weight_y_warmup_iters": args.weight_y_warmup_iters,
            "warm_start_unsup_iters": args.warm_start_unsup_iters,
            "lookback_days": args.lookback_days, "window_days": args.window_days,
            "stability_days": args.stability_days, "grace_gap_days": args.grace_gap_days,
            "vocab_size": args.vocab_size, "min_df": args.min_df,
            "min_patient_count": args.min_patient_count, "person_mod": args.person_mod,
            "test_frac": args.test_frac, "seed": args.seed,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[driver] wrote results to {args.out}", flush=True)

    bow_df.unpersist()
    index_df.unpersist()
    print("[driver] done", flush=True)
    spark.stop()
    return 0


def _log_convergence(meta: Mapping[str, Any]) -> None:
    """Emit a one-line fit-health readout from a results ``meta`` block.

    Both backends record a convergence signal so a degenerate / untrained fit is
    obvious in the committed run log: ``|w_CK|max ~= 0`` means the logistic head
    never left its zero init (the failure that pinned every drug's AUC at 0.5).
    The in-memory path carries ``meta["pc_convergence"]`` (L-BFGS nit/success/obj);
    the VI path carries ``meta["vi_convergence"]`` (SVI n_iter/final ELBO).
    """
    if "pc_convergence" in meta:
        c = meta["pc_convergence"]
        print(
            f"[driver] PC fit: nit={c['n_iter']} success={c['success']} "
            f"obj init->final={c['init_obj']:.6g}->{c['final_obj']:.6g} "
            f"|w_CK|max={c['w_CK_absmax']:.4g}  "
            f"(|w_CK|max~=0 => head UNTRAINED)",
            flush=True,
        )
    elif "vi_convergence" in meta:
        c = meta["vi_convergence"]
        elbo = c.get("final_elbo")
        elbo_s = "n/a" if elbo is None else f"{elbo:.6g}"
        print(
            f"[driver] VI-PC fit: n_iter={c['n_iter']} converged={c['converged']} "
            f"final_elbo={elbo_s} |w_CK|max={c['w_CK_absmax']:.4g}  "
            f"(|w_CK|max~=0 => head UNTRAINED)",
            flush=True,
        )


def _run_inmem_backend(
    bow_df, V, outcome_by_person, reference_drugs, args, evaluate_pc_multitask,
):
    """In-memory L-BFGS PC backend (the default; unchanged behavior).

    Bridges the BOW to a dense ``X`` on the driver, assembles the ``(D, C)``
    multi-task ``y``/``mask``, does the seeded stratified split, and runs
    :func:`analysis.pc.evaluate.evaluate_pc_multitask` (PC + the two baselines).
    Returns ``(results, drug_order, n_train, n_test, n_persons)``.
    """
    # --- 4) Bridge to dense X + assemble multi-task y/mask --------------------
    with _phase("bridge (BOW -> dense X) + label/mask assembly"):
        X, person_order = collect_bow_aligned(bow_df, V)
        drugs_present = [
            outcome_by_person[p][0] for p in person_order if p in outcome_by_person
        ]
        drug_order = stable_drug_order(drugs_present, reference=reference_drugs)
        y, mask = assemble_multitask_labels(outcome_by_person, person_order, drug_order)
        C = len(drug_order)
        print(f"[driver]   X={X.shape}, C={C} drug columns: {drug_order}", flush=True)

    # --- 5) Split + evaluate --------------------------------------------------
    with _phase(f"split (test_frac={args.test_frac}) + evaluate_pc_multitask "
                f"(K={args.K}, weight_y={args.weight_y})"):
        groups = [outcome_by_person.get(p, (None,))[0] for p in person_order]
        is_test = stratified_test_mask(groups, args.test_frac, args.seed)
        tr, te = ~is_test, is_test
        print(f"[driver]   split: {int(tr.sum())} train / {int(te.sum())} test",
              flush=True)
        results = evaluate_pc_multitask(
            X[tr], y[tr], mask[tr], X[te], y[te], mask[te],
            K=args.K, weight_y=args.weight_y, alpha=args.alpha, tau=args.tau,
            pi_iters=args.pi_iters, max_iter=args.max_iter,
            doc_batch_size=args.doc_batch_size, seed=args.seed,
            min_label_count=args.min_label_count,
            skip_two_stage=args.skip_two_stage,
            baseline_max_iter=(args.baseline_max_iter
                               if args.baseline_max_iter > 0 else None),
        )
    return results, drug_order, int(tr.sum()), int(te.sum()), len(person_order)


def _run_inmem_backend_fullyobserved(
    bow_df, V, label_by_person, drug_order, args, evaluate_pc_multitask,
):
    """In-memory L-BFGS PC backend for the FULLY-OBSERVED stable-treatment path.

    The mask-all-ones sibling of :func:`_run_inmem_backend`. It bridges the BOW to
    a dense ``X``, assembles the fully-observed ``(D, 10)`` ``y``/``mask`` over the
    FIXED Hughes drug order (:func:`assemble_fullyobserved_labels`, so column c is
    the same Hughes drug every run — NOT a ``stable_drug_order`` over present
    drugs), does the seeded split, and runs the SAME
    :func:`analysis.pc.evaluate.evaluate_pc_multitask` (PC + the two baselines). An
    all-ones mask is exactly the standard C-way multi-label case that helper
    already handles, so nothing downstream changes. Returns ``(results,
    drug_order, n_train, n_test, n_persons)`` in the same shape as
    :func:`_run_inmem_backend`.
    """
    # --- 4) Bridge to dense X + assemble the fully-observed y/mask ------------
    with _phase("bridge (BOW -> dense X) + fully-observed label/mask assembly"):
        X, person_order = collect_bow_aligned(bow_df, V)
        y, mask = assemble_fullyobserved_labels(
            label_by_person, person_order, drug_order,
        )
        C = len(drug_order)
        n_labeled = int((mask.sum(axis=1) > 0).sum())
        print(f"[driver]   X={X.shape}, C={C} (fixed Hughes order), "
              f"{n_labeled} labeled persons", flush=True)

    # --- 5) Split + evaluate --------------------------------------------------
    with _phase(f"split (test_frac={args.test_frac}) + evaluate_pc_multitask "
                f"(K={args.K}, weight_y={args.weight_y})"):
        # Stratify on the stable drug-subset signature (a sorted tuple) so a rare
        # held combination stays balanced across train/test; an unlabeled person
        # (absent from label_by_person) forms its own empty-tuple group.
        groups = [tuple(sorted(label_by_person.get(p, ()))) for p in person_order]
        is_test = stratified_test_mask(groups, args.test_frac, args.seed)
        tr, te = ~is_test, is_test
        print(f"[driver]   split: {int(tr.sum())} train / {int(te.sum())} test",
              flush=True)
        results = evaluate_pc_multitask(
            X[tr], y[tr], mask[tr], X[te], y[te], mask[te],
            K=args.K, weight_y=args.weight_y, alpha=args.alpha, tau=args.tau,
            pi_iters=args.pi_iters, max_iter=args.max_iter,
            doc_batch_size=args.doc_batch_size, seed=args.seed,
            min_label_count=args.min_label_count,
            skip_two_stage=args.skip_two_stage,
            baseline_max_iter=(args.baseline_max_iter
                               if args.baseline_max_iter > 0 else None),
        )
    return results, drug_order, int(tr.sum()), int(te.sum()), len(person_order)


def _corpus_manifest_for(args) -> dict:
    """The corpus MEMBERSHIP fields for the checkpoint manifest, keyed by cohort.

    The resume-compat guard (``scripts/run_experiment.py::_resume_corpus_mismatches``)
    reads these to refuse a warm-start onto a checkpoint whose corpus differs. The
    membership knobs are cohort-specific: ``mdd_antidepressant``'s corpus is
    bracketed by the pre-index feature window + outcome-stability horizon, whereas
    ``mdd_stable_treatment`` uses an all-history feature window (no lookback /
    forward window / stability) and is instead defined by the stable-interval,
    encounter-regularity, pre-treatment-history and age knobs — so each records
    its own set. ``cohort`` + ``person_mod`` + the vocab-pruning knobs
    (``vocab_size``/``min_df``/``min_patient_count``) are common to both.
    """
    common = {
        "cohort": args.cohort,
        "person_mod": int(args.person_mod),
        "vocab_size": int(args.vocab_size),
        "min_df": int(args.min_df),
        "min_patient_count": int(args.min_patient_count),
    }
    if args.cohort == "mdd_stable_treatment":
        return {
            **common,
            "min_days": int(args.min_days),
            "max_gap_days": int(args.max_gap_days),
            "min_history_events": int(args.min_history_events),
            "age_min": int(args.age_min),
            "age_max": int(args.age_max),
        }
    return {
        **common,
        "lookback_days": int(args.lookback_days),
        "window_days": int(args.window_days),
        "stability_days": int(args.stability_days),
        "grace_gap_days": int(args.grace_gap_days),
    }


def _augmented_resave_pc(model, drug_order, V, vocab_map, args) -> None:
    """Overwrite the shim's final checkpoint with self-describing metadata.

    The PC counterpart of lda_bigquery_cloud.py's augmented re-save: the shim's
    mid-fit / end-of-fit auto-saves carry only the trained VIResult, enough for
    resume continuity. This re-save bundles what PC needs to be self-describing
    on resume/eval — the drug->column order (``stable_drug_order``, so the loaded
    head columns line up), the BOW ``vocab``, the SVI config, and a ``corpus_manifest``
    (the resume-compat guard's membership fields) — and writes ``manifest.json``
    marking the directory as an authoritative, loadable checkpoint.
    """
    from spark_vi.core.result import VIResult
    from spark_vi.io.export import save_result

    vocab_list: list = [None] * (len(vocab_map) if vocab_map else 0)
    if vocab_map:
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid

    augmented = VIResult(
        global_params=model.result.global_params,
        elbo_trace=model.result.elbo_trace,
        n_iterations=model.result.n_iterations,
        converged=model.result.converged,
        diagnostic_traces=model.result.diagnostic_traces,
        metadata={
            **model.result.metadata,
            "vocab": vocab_list,
            "stable_drug_order": list(drug_order),
            "column_drug_names": {c: name for c, name in enumerate(drug_order)},
            "K": int(args.K),
            "svi": {
                "subsampling_rate": float(args.subsampling_rate),
                "tau0": float(args.tau0),
                "kappa": float(args.kappa),
                "head_lr_scale": float(args.head_lr_scale),
                "weight_y_warmup_iters": int(args.weight_y_warmup_iters),
                "max_iter": int(args.max_iter),
                "weight_y": float(args.weight_y),
                "alpha": float(args.alpha),
            },
            # corpus MEMBERSHIP fields the resume-compat guard reads
            # (scripts/run_experiment.py::_resume_corpus_mismatches). Only the
            # fields that actually determine which patients/features the fit saw
            # go here; a mismatch on any refuses to warm-start. The membership
            # knobs differ by cohort: mdd_antidepressant is bracketed by the
            # pre-index window + outcome-stability horizon, while
            # mdd_stable_treatment is defined by the stable-interval / encounter-
            # regularity / age knobs with an all-history feature window (no
            # lookback/window/stability), so each records its OWN knob set.
            "corpus_manifest": _corpus_manifest_for(args),
        },
    )
    save_result(augmented, args.save_dir)
    print(f"[driver] re-saved augmented VIResult (manifest.json) to "
          f"{args.save_dir}", flush=True)


def _run_vi_backend(
    spark, bow_df, V, outcome_by_person, reference_drugs, args,
    _lr_proba_per_label_masked, _bundle_masked, vocab_map=None,
):
    """Distributed VI-native PC backend (``--backend vi``).

    Reuses the SAME cohort/outcome/feature pipeline, then: attaches the multi-task
    label + mask to ``bow_df`` as Spark ``ArrayType`` columns
    (:func:`attach_multitask_label_columns`), splits by person at the DataFrame
    level (:func:`person_hash_split`), fits the distributed
    :class:`~spark_vi.mllib.topic.pc.PCEstimator` (SVI, no collect-to-memory for the
    fit), scores per-drug heldout AUC/AP from the transform's ``probabilityCol``,
    and computes the identical two-stage / LR-on-codes baselines by collecting
    train+test BOW to memory (:func:`multitask_baseline_probas`). Returns
    ``(results, drug_order, n_train, n_test, n_persons)`` in the same shape as the
    in-memory backend so :func:`format_results_table` and the JSON payload are
    backend-agnostic.

    Checkpoint / resume / eval (mirrors lda_bigquery_cloud.py):
      * ``--save-dir`` + ``--save-interval`` are threaded into the estimator
        (``saveDir``/``saveInterval``); the shim auto-checkpoints the VIResult
        during fit and writes the authoritative final result at end-of-fit.
      * ``--resume-from`` is threaded as ``resumeFrom`` so a re-run continues the
        prior checkpoint (``--max-iter`` = ADDITIONAL iters).
      * ``--eval-only`` skips training entirely: the checkpoint VIResult at
        ``--save-dir`` is loaded via ``PCModel.load`` and scored, so a user can
        peek the AUC from a checkpoint without more fit. The drug->column order
        is read from the checkpoint metadata (``stable_drug_order``) so the
        loaded head columns line up with the eval labels.
      * On a real fit with ``--save-dir`` set, an augmented re-save overwrites the
        shim's final checkpoint with self-describing metadata (``stable_drug_order``,
        ``vocab``, the SVI config, and a corpus manifest) so eval/resume don't
        depend on re-deriving state.
    """
    from spark_vi.mllib.topic.pc import PCEstimator, PCModel

    # --- 4) Attach labels as Spark columns + person split --------------------
    with _phase(f"attach multi-task label/mask columns + person split "
                f"(test_frac={args.test_frac})"):
        # On eval-only, take the drug->column order from the checkpoint so the
        # loaded head (C, K) columns align with the eval labels exactly; else
        # derive it from the cohort as usual (deterministic given the data).
        drug_order = None
        if args.eval_only:
            from spark_vi.io.export import load_result
            ck_meta = load_result(args.save_dir).metadata
            drug_order = ck_meta.get("stable_drug_order")
            if drug_order is not None:
                drug_order = list(drug_order)
                print(f"[driver]   eval-only: drug_order from checkpoint "
                      f"({len(drug_order)} columns)", flush=True)
        if drug_order is None:
            drugs_present = [d for (d, _w) in outcome_by_person.values()]
            drug_order = stable_drug_order(drugs_present, reference=reference_drugs)
        C = len(drug_order)
        labeled = attach_multitask_label_columns(
            bow_df, outcome_by_person, drug_order, spark,
        )
        train_df, test_df = person_hash_split(labeled, args.test_frac, args.seed)
        train_df = train_df.persist()
        test_df = test_df.persist()
        n_train = train_df.count()
        n_test = test_df.count()
        print(f"[driver]   C={C} drug columns: {drug_order}", flush=True)
        print(f"[driver]   split: {n_train} train / {n_test} test", flush=True)

    # --- 5a) Distributed VI-PC fit (SVI; no collect-to-memory) ----------------
    #         ... or, on --eval-only, load the checkpoint model (no training).
    if args.eval_only:
        with _phase(f"VI-PC eval-only: load checkpoint from {args.save_dir}"):
            model = PCModel.load(args.save_dir)
            # PCModel.load restores only the wrapped VIResult (Params aren't
            # persisted, ADR 0009/0012); set the ones transform reads so it
            # emits the head-derived probabilityCol (needs weightY > 0 + C).
            model._set(
                featuresCol="features", numLabels=C,
                weightY=float(args.weight_y), probabilityCol="probability",
            )
            vres = model.result
            w_ck_absmax = float(np.abs(model.headWeights()).max())
            print(f"[driver]   VI-PC checkpoint loaded: n_iter={vres.n_iterations}, "
                  f"converged={vres.converged}, final_elbo={vres.final_elbo}, "
                  f"|w_CK|max={w_ck_absmax:.4g}", flush=True)
    else:
        # Unsupervised-warm-start protocol (Hughes et al.): when
        # --warm-start-unsup-iters N > 0 and we are NOT resuming, run PHASE 1
        # (weight_y=0, N iters) to learn topics, then warm-init PHASE 2 (the real
        # supervised fit) from phase-1's topics with a FRESH Robbins-Monro
        # schedule. On --resume-from we skip phase 1 entirely: resuming continues
        # an existing phase-2 fit (warm-start is a fresh-start init, not a resume).
        warm_iters = int(args.warm_start_unsup_iters)
        warm_start_dir = ""
        if warm_iters > 0 and args.resume_from:
            print("[driver] note: --warm-start-unsup-iters ignored on "
                  "--resume-from (resuming continues the existing phase-2 fit; "
                  "warm-start is a fresh-start init).", flush=True)
            warm_iters = 0
        if warm_iters > 0:
            with _phase(f"VI-PC warm-start PHASE 1 (unsupervised LDA-MAP, "
                        f"weight_y=0, {warm_iters} iters, K={args.K}, "
                        f"subsamplingRate={args.subsampling_rate}, "
                        f"tau0={args.tau0}, kappa={args.kappa})"):
                # Phase 1 is a warm-up: it writes ONLY to a driver-local temp dir
                # (loaded back on the driver by phase 2's warm-init), NOT to
                # --save-dir. --save-dir checkpoints the real (phase-2) fit.
                warm_start_dir = tempfile.mkdtemp(prefix="pc_warmup_")
                phase1 = PCEstimator(
                    featuresCol="features", labelCol="y", labelMaskCol="label_mask",
                    numLabels=C, weightY=0.0, k=args.K,
                    docConcentration=[float(args.alpha)],
                    subsamplingRate=args.subsampling_rate,
                    learningOffset=args.tau0, learningDecay=args.kappa,
                    maxIter=warm_iters, seed=args.seed,
                    probabilityCol="probability",
                )
                model_p1 = phase1.fit(train_df)
                model_p1.save(warm_start_dir)
                w1 = float(np.abs(model_p1.headWeights()).max())
                print(f"[driver]   warm-start phase 1 done: "
                      f"n_iter={model_p1.result.n_iterations}, "
                      f"final_elbo={model_p1.result.final_elbo}, "
                      f"|w_CK|max={w1:.4g} (head at zero init, as expected for "
                      f"weight_y=0)", flush=True)

        with _phase(f"VI-PC fit{' PHASE 2 (supervised, warm-started)' if warm_start_dir else ''} "
                    f"(SVI, K={args.K}, weight_y={args.weight_y}, "
                    f"subsamplingRate={args.subsampling_rate}, tau0={args.tau0}, "
                    f"kappa={args.kappa}, maxIter={args.max_iter}, "
                    f"saveDir={args.save_dir or '<none>'}, "
                    f"resumeFrom={args.resume_from or '<none>'}, "
                    f"warmStartFrom={'<phase1>' if warm_start_dir else '<none>'})"):
            estimator = PCEstimator(
                featuresCol="features", labelCol="y", labelMaskCol="label_mask",
                numLabels=C, weightY=float(args.weight_y), k=args.K,
                docConcentration=[float(args.alpha)],
                subsamplingRate=args.subsampling_rate,
                learningOffset=args.tau0, learningDecay=args.kappa,
                headLrScale=args.head_lr_scale,
                weightYWarmupIters=args.weight_y_warmup_iters,
                maxIter=args.max_iter, seed=args.seed,
                probabilityCol="probability",
                saveDir=args.save_dir, saveInterval=args.save_interval,
                resumeFrom=args.resume_from,
                warmStartFrom=warm_start_dir,
            )
            try:
                model = estimator.fit(train_df)
            finally:
                # Phase-1 temp checkpoint is consumed by phase 2's warm-init;
                # drop it once the fit has loaded it.
                if warm_start_dir:
                    shutil.rmtree(warm_start_dir, ignore_errors=True)
            vres = model.result
            w_ck_absmax = float(np.abs(model.headWeights()).max())
            print(f"[driver]   VI-PC fit done: n_iter={vres.n_iterations}, "
                  f"converged={vres.converged}, final_elbo={vres.final_elbo}, "
                  f"|w_CK|max={w_ck_absmax:.4g}", flush=True)

        # Augmented re-save: overwrite the shim's final checkpoint with
        # self-describing metadata so eval/resume don't re-derive state. Mirrors
        # lda_bigquery_cloud.py's augmented re-save. Only on a real fit (eval-only
        # never mutates the checkpoint).
        if args.save_dir:
            _augmented_resave_pc(model, drug_order, V, vocab_map, args)

    # --- 5b) Transform + score PC, then the shared baselines ------------------
    with _phase("VI-PC transform + collect + baselines + per-drug scoring"):
        scored = model.transform(test_df)

        # Two-stage baseline: distributed SVI (unsupervised PCEstimator weight_y=0
        # -> per-label LR on the K-dim topics). Fit BEFORE the collect/unpersist,
        # while train/test_df are persisted; no dense collect + no in-memory fit,
        # so it is SVI-consistent with the VI-PC model above and does not OOM the
        # driver. Skipped by --skip-two-stage (e.g. a fast --eval-only readout).
        two_stage_bundle = None
        if not args.skip_two_stage:
            with _phase("two-stage baseline (distributed SVI unsup, weight_y=0)"):
                two_stage_bundle = _vi_two_stage_bundle(train_df, test_df, C, args)

        X_te, y_te_DC, mask_te_DC, proba_DC, _ = collect_labeled_bow(
            scored, V, C, prob_col="probability",
        )
        X_tr, y_tr_DC, mask_tr_DC, _, _ = collect_labeled_bow(train_df, V, C)
        train_df.unpersist()
        test_df.unpersist()

        # LR-on-codes baseline: per-label LR on the raw dense counts (inherently a
        # code-space model, so the dense X collect above is its irreducible cost).
        lrc_proba = _lr_proba_per_label_masked(X_tr, y_tr_DC, mask_tr_DC, X_te, C)

        n_obs_train = [int(mask_tr_DC[:, c].sum()) for c in range(C)]
        n_obs_test = [int(mask_te_DC[:, c].sum()) for c in range(C)]
        results = {
            "PC": _bundle_masked(proba_DC, y_te_DC, mask_te_DC, C, args.min_label_count),
            "lr_codes": _bundle_masked(lrc_proba, y_te_DC, mask_te_DC, C, args.min_label_count),
            "meta": {
                "C": C,
                "K": int(args.K),
                "weight_y": float(args.weight_y),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n_labeled": int(sum(n_obs_train)),
                "n_obs_train": n_obs_train,
                "n_obs_test": n_obs_test,
                "min_label_count": int(args.min_label_count),
                "two_stage_skipped": bool(args.skip_two_stage),
                "backend": "vi",
                "svi": {
                    "subsampling_rate": float(args.subsampling_rate),
                    "tau0": float(args.tau0),
                    "kappa": float(args.kappa),
                    "head_lr_scale": float(args.head_lr_scale),
                    "weight_y_warmup_iters": int(args.weight_y_warmup_iters),
                    "max_iter": int(args.max_iter),
                },
                # Distributed-fit convergence signal (the untrained-head tell).
                "vi_convergence": {
                    "n_iter": int(vres.n_iterations),
                    "final_elbo": (None if vres.final_elbo is None
                                   else float(vres.final_elbo)),
                    "converged": bool(vres.converged),
                    "w_CK_absmax": w_ck_absmax,
                },
                "baseline_max_iter": int(args.baseline_max_iter),
                "model_names": {
                    "PC": "VI-PC (SVI, joint)",
                    "two_stage": "two-stage (unsup+LR)",
                    "lr_codes": "LR-on-codes",
                },
            },
        }
        if two_stage_bundle is not None:
            results["two_stage"] = two_stage_bundle
    return results, drug_order, int(n_train), int(n_test), int(n_train + n_test)


def _run_vi_backend_fullyobserved(
    spark, bow_df, V, label_by_person, drug_order, args,
    _lr_proba_per_label_masked, _bundle_masked, vocab_map=None,
):
    """Distributed VI-native PC backend for the FULLY-OBSERVED stable-treatment path.

    The mask-all-ones sibling of :func:`_run_vi_backend`. Identical machinery —
    attach label/mask as Spark ``ArrayType`` columns, ``person_hash_split``, the
    distributed :class:`~spark_vi.mllib.topic.pc.PCEstimator` SVI fit (with the SAME
    unsupervised warm-start + checkpoint/resume/eval-only paths), then
    :func:`collect_labeled_bow` + the shared two-stage / LR-on-codes baselines —
    with exactly TWO differences from the per-drug sibling:

    1. The labels come from :func:`attach_fullyobserved_label_columns` over the
       stable drug subset (all-ones mask) rather than
       :func:`attach_multitask_label_columns` over the index drug (one observed
       cell). ``evaluate_pc_multitask`` / ``_bundle_masked`` read the all-ones mask
       as standard C-way multi-label, so scoring is unchanged.
    2. ``drug_order`` is the FIXED length-10 Hughes order passed in (column c is
       the same Hughes drug every run), NOT a ``stable_drug_order`` over present
       drugs — so it is used as-is off a non-eval fit; on ``--eval-only`` it is
       still taken from the checkpoint metadata so the loaded head columns line up.

    Returns ``(results, drug_order, n_train, n_test, n_persons)`` in the same shape
    as the per-drug backend so the report + JSON payload are backend-agnostic.
    """
    from spark_vi.mllib.topic.pc import PCEstimator, PCModel

    # --- 4) Attach labels as Spark columns + person split --------------------
    with _phase(f"attach fully-observed label/mask columns + person split "
                f"(test_frac={args.test_frac})"):
        # On eval-only, take the drug->column order from the checkpoint so the
        # loaded head (C, K) columns align with the eval labels; else use the
        # FIXED Hughes order passed in (deterministic, independent of the data).
        if args.eval_only:
            from spark_vi.io.export import load_result
            ck_meta = load_result(args.save_dir).metadata
            ck_order = ck_meta.get("stable_drug_order")
            if ck_order is not None:
                drug_order = list(ck_order)
                print(f"[driver]   eval-only: drug_order from checkpoint "
                      f"({len(drug_order)} columns)", flush=True)
        drug_order = list(drug_order)
        C = len(drug_order)
        labeled = attach_fullyobserved_label_columns(
            bow_df, label_by_person, drug_order, spark,
        )
        train_df, test_df = person_hash_split(labeled, args.test_frac, args.seed)
        train_df = train_df.persist()
        test_df = test_df.persist()
        n_train = train_df.count()
        n_test = test_df.count()
        print(f"[driver]   C={C} (fixed Hughes order): {drug_order}", flush=True)
        print(f"[driver]   split: {n_train} train / {n_test} test", flush=True)

    # --- 5a) Distributed VI-PC fit (SVI; no collect-to-memory) ----------------
    #         ... or, on --eval-only, load the checkpoint model (no training).
    if args.eval_only:
        with _phase(f"VI-PC eval-only: load checkpoint from {args.save_dir}"):
            model = PCModel.load(args.save_dir)
            model._set(
                featuresCol="features", numLabels=C,
                weightY=float(args.weight_y), probabilityCol="probability",
            )
            vres = model.result
            w_ck_absmax = float(np.abs(model.headWeights()).max())
            print(f"[driver]   VI-PC checkpoint loaded: n_iter={vres.n_iterations}, "
                  f"converged={vres.converged}, final_elbo={vres.final_elbo}, "
                  f"|w_CK|max={w_ck_absmax:.4g}", flush=True)
    else:
        # Unsupervised-warm-start protocol (Hughes et al.), identical to the
        # per-drug backend: phase 1 (weight_y=0, N iters) learns topics, then a
        # fresh-Robbins-Monro supervised phase 2 warm-starts from them. Skipped on
        # --resume-from (resuming continues an existing phase-2 fit).
        warm_iters = int(args.warm_start_unsup_iters)
        warm_start_dir = ""
        if warm_iters > 0 and args.resume_from:
            print("[driver] note: --warm-start-unsup-iters ignored on "
                  "--resume-from (resuming continues the existing phase-2 fit; "
                  "warm-start is a fresh-start init).", flush=True)
            warm_iters = 0
        if warm_iters > 0:
            with _phase(f"VI-PC warm-start PHASE 1 (unsupervised LDA-MAP, "
                        f"weight_y=0, {warm_iters} iters, K={args.K}, "
                        f"subsamplingRate={args.subsampling_rate}, "
                        f"tau0={args.tau0}, kappa={args.kappa})"):
                warm_start_dir = tempfile.mkdtemp(prefix="pc_warmup_")
                phase1 = PCEstimator(
                    featuresCol="features", labelCol="y", labelMaskCol="label_mask",
                    numLabels=C, weightY=0.0, k=args.K,
                    docConcentration=[float(args.alpha)],
                    subsamplingRate=args.subsampling_rate,
                    learningOffset=args.tau0, learningDecay=args.kappa,
                    maxIter=warm_iters, seed=args.seed,
                    probabilityCol="probability",
                )
                model_p1 = phase1.fit(train_df)
                model_p1.save(warm_start_dir)
                w1 = float(np.abs(model_p1.headWeights()).max())
                print(f"[driver]   warm-start phase 1 done: "
                      f"n_iter={model_p1.result.n_iterations}, "
                      f"final_elbo={model_p1.result.final_elbo}, "
                      f"|w_CK|max={w1:.4g} (head at zero init, as expected for "
                      f"weight_y=0)", flush=True)

        with _phase(f"VI-PC fit{' PHASE 2 (supervised, warm-started)' if warm_start_dir else ''} "
                    f"(SVI, K={args.K}, weight_y={args.weight_y}, "
                    f"subsamplingRate={args.subsampling_rate}, tau0={args.tau0}, "
                    f"kappa={args.kappa}, maxIter={args.max_iter}, "
                    f"saveDir={args.save_dir or '<none>'}, "
                    f"resumeFrom={args.resume_from or '<none>'}, "
                    f"warmStartFrom={'<phase1>' if warm_start_dir else '<none>'})"):
            estimator = PCEstimator(
                featuresCol="features", labelCol="y", labelMaskCol="label_mask",
                numLabels=C, weightY=float(args.weight_y), k=args.K,
                docConcentration=[float(args.alpha)],
                subsamplingRate=args.subsampling_rate,
                learningOffset=args.tau0, learningDecay=args.kappa,
                headLrScale=args.head_lr_scale,
                weightYWarmupIters=args.weight_y_warmup_iters,
                maxIter=args.max_iter, seed=args.seed,
                probabilityCol="probability",
                saveDir=args.save_dir, saveInterval=args.save_interval,
                resumeFrom=args.resume_from,
                warmStartFrom=warm_start_dir,
            )
            try:
                model = estimator.fit(train_df)
            finally:
                if warm_start_dir:
                    shutil.rmtree(warm_start_dir, ignore_errors=True)
            vres = model.result
            w_ck_absmax = float(np.abs(model.headWeights()).max())
            print(f"[driver]   VI-PC fit done: n_iter={vres.n_iterations}, "
                  f"converged={vres.converged}, final_elbo={vres.final_elbo}, "
                  f"|w_CK|max={w_ck_absmax:.4g}", flush=True)

        # Augmented re-save with the stable-treatment corpus manifest
        # (_corpus_manifest_for records cohort + the stable knobs).
        if args.save_dir:
            _augmented_resave_pc(model, drug_order, V, vocab_map, args)

    # --- 5b) Transform + score PC, then the shared baselines ------------------
    with _phase("VI-PC transform + collect + baselines + per-drug scoring"):
        scored = model.transform(test_df)

        # Two-stage baseline: distributed SVI (unsupervised PCEstimator weight_y=0
        # -> per-label LR on the K-dim topics). Fit BEFORE the collect/unpersist,
        # while train/test_df are persisted; no dense collect + no in-memory fit,
        # so it is SVI-consistent with the VI-PC model above and does not OOM the
        # driver. Skipped by --skip-two-stage (e.g. a fast --eval-only readout).
        two_stage_bundle = None
        if not args.skip_two_stage:
            with _phase("two-stage baseline (distributed SVI unsup, weight_y=0)"):
                two_stage_bundle = _vi_two_stage_bundle(train_df, test_df, C, args)

        X_te, y_te_DC, mask_te_DC, proba_DC, _ = collect_labeled_bow(
            scored, V, C, prob_col="probability",
        )
        X_tr, y_tr_DC, mask_tr_DC, _, _ = collect_labeled_bow(train_df, V, C)
        train_df.unpersist()
        test_df.unpersist()

        # LR-on-codes baseline: per-label LR on the raw dense counts (inherently a
        # code-space model, so the dense X collect above is its irreducible cost).
        lrc_proba = _lr_proba_per_label_masked(X_tr, y_tr_DC, mask_tr_DC, X_te, C)

        n_obs_train = [int(mask_tr_DC[:, c].sum()) for c in range(C)]
        n_obs_test = [int(mask_te_DC[:, c].sum()) for c in range(C)]
        results = {
            "PC": _bundle_masked(proba_DC, y_te_DC, mask_te_DC, C, args.min_label_count),
            "lr_codes": _bundle_masked(lrc_proba, y_te_DC, mask_te_DC, C, args.min_label_count),
            "meta": {
                "C": C,
                "K": int(args.K),
                "weight_y": float(args.weight_y),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n_labeled": int(n_train + n_test),
                "n_obs_train": n_obs_train,
                "n_obs_test": n_obs_test,
                "min_label_count": int(args.min_label_count),
                "two_stage_skipped": bool(args.skip_two_stage),
                "backend": "vi",
                "svi": {
                    "subsampling_rate": float(args.subsampling_rate),
                    "tau0": float(args.tau0),
                    "kappa": float(args.kappa),
                    "head_lr_scale": float(args.head_lr_scale),
                    "weight_y_warmup_iters": int(args.weight_y_warmup_iters),
                    "max_iter": int(args.max_iter),
                },
                "vi_convergence": {
                    "n_iter": int(vres.n_iterations),
                    "final_elbo": (None if vres.final_elbo is None
                                   else float(vres.final_elbo)),
                    "converged": bool(vres.converged),
                    "w_CK_absmax": w_ck_absmax,
                },
                "baseline_max_iter": int(args.baseline_max_iter),
                "model_names": {
                    "PC": "VI-PC (SVI, joint)",
                    "two_stage": "two-stage (unsup+LR)",
                    "lr_codes": "LR-on-codes",
                },
            },
        }
        if two_stage_bundle is not None:
            results["two_stage"] = two_stage_bundle
    return results, drug_order, int(n_train), int(n_test), int(n_train + n_test)


if __name__ == "__main__":
    sys.exit(main())
