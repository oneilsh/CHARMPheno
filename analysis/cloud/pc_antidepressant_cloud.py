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
import sys
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
    # --- Feature window ------------------------------------------------------
    parser.add_argument(
        "--lookback-days", type=int, default=365,
        help="pre-index feature window: events in [index - lookback_days, index)",
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
    if args.window_days < args.stability_days:
        print(f"ERROR: --window-days ({args.window_days}) must be >= "
              f"--stability-days ({args.stability_days}) so the stability window "
              "is fully observed.", file=sys.stderr)
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
        _antidepressant_concept_map,
        antidepressant_stability_label,
        apply_mdd_antidepressant_cohort,
        lookback_feature_label_events,
    )
    from analysis.pc.evaluate import (
        _bundle_masked,
        evaluate_pc_multitask,
        format_results_table,
        multitask_baseline_probas,
    )

    configure_logging()
    print(f"[driver] cdr={cdr}, billing_project={billing}, backend={args.backend}, "
          f"K={args.K}, weight_y={args.weight_y}, lookback_days={args.lookback_days}, "
          f"window_days={args.window_days}, stability_days={args.stability_days}, "
          f"person_mod={args.person_mod}", flush=True)
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
            multitask_baseline_probas, _bundle_masked, vocab_map,
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
        )
    return results, drug_order, int(tr.sum()), int(te.sum()), len(person_order)


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
                "max_iter": int(args.max_iter),
                "weight_y": float(args.weight_y),
                "alpha": float(args.alpha),
            },
            # corpus MEMBERSHIP fields the resume-compat guard reads
            # (scripts/run_experiment.py::_resume_corpus_mismatches). Only the
            # fields that actually determine which patients/features the fit saw
            # go here; a mismatch on any refuses to warm-start.
            "corpus_manifest": {
                "cohort": "mdd_antidepressant",
                "person_mod": int(args.person_mod),
                "lookback_days": int(args.lookback_days),
                "window_days": int(args.window_days),
                "stability_days": int(args.stability_days),
                "grace_gap_days": int(args.grace_gap_days),
                "vocab_size": int(args.vocab_size),
                "min_df": int(args.min_df),
                "min_patient_count": int(args.min_patient_count),
            },
        },
    )
    save_result(augmented, args.save_dir)
    print(f"[driver] re-saved augmented VIResult (manifest.json) to "
          f"{args.save_dir}", flush=True)


def _run_vi_backend(
    spark, bow_df, V, outcome_by_person, reference_drugs, args,
    multitask_baseline_probas, _bundle_masked, vocab_map=None,
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
        with _phase(f"VI-PC fit (SVI, K={args.K}, weight_y={args.weight_y}, "
                    f"subsamplingRate={args.subsampling_rate}, tau0={args.tau0}, "
                    f"kappa={args.kappa}, maxIter={args.max_iter}, "
                    f"saveDir={args.save_dir or '<none>'}, "
                    f"resumeFrom={args.resume_from or '<none>'})"):
            estimator = PCEstimator(
                featuresCol="features", labelCol="y", labelMaskCol="label_mask",
                numLabels=C, weightY=float(args.weight_y), k=args.K,
                docConcentration=[float(args.alpha)],
                subsamplingRate=args.subsampling_rate,
                learningOffset=args.tau0, learningDecay=args.kappa,
                maxIter=args.max_iter, seed=args.seed,
                probabilityCol="probability",
                saveDir=args.save_dir, saveInterval=args.save_interval,
                resumeFrom=args.resume_from,
            )
            model = estimator.fit(train_df)
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
        X_te, y_te_DC, mask_te_DC, proba_DC, _ = collect_labeled_bow(
            scored, V, C, prob_col="probability",
        )
        X_tr, y_tr_DC, mask_tr_DC, _, _ = collect_labeled_bow(train_df, V, C)
        train_df.unpersist()
        test_df.unpersist()

        shared = dict(
            K=args.K, C=C, alpha=args.alpha, tau=args.tau, pi_iters=args.pi_iters,
            max_iter=args.max_iter, doc_batch_size=args.doc_batch_size, seed=args.seed,
        )
        ts_proba, lrc_proba = multitask_baseline_probas(
            X_tr, y_tr_DC, mask_tr_DC, X_te, C, shared,
        )

        n_obs_train = [int(mask_tr_DC[:, c].sum()) for c in range(C)]
        n_obs_test = [int(mask_te_DC[:, c].sum()) for c in range(C)]
        results = {
            "PC": _bundle_masked(proba_DC, y_te_DC, mask_te_DC, C),
            "two_stage": _bundle_masked(ts_proba, y_te_DC, mask_te_DC, C),
            "lr_codes": _bundle_masked(lrc_proba, y_te_DC, mask_te_DC, C),
            "meta": {
                "C": C,
                "K": int(args.K),
                "weight_y": float(args.weight_y),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n_labeled": int(sum(n_obs_train)),
                "n_obs_train": n_obs_train,
                "n_obs_test": n_obs_test,
                "backend": "vi",
                "svi": {
                    "subsampling_rate": float(args.subsampling_rate),
                    "tau0": float(args.tau0),
                    "kappa": float(args.kappa),
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
                "model_names": {
                    "PC": "VI-PC (SVI, joint)",
                    "two_stage": "two-stage (unsup+LR)",
                    "lr_codes": "LR-on-codes",
                },
            },
        }
    return results, drug_order, int(n_train), int(n_test), int(n_train + n_test)


if __name__ == "__main__":
    sys.exit(main())
