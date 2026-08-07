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
              "real-corpus scale without changing the objective or optimizer"),
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
    from analysis.pc.evaluate import evaluate_pc_multitask, format_results_table

    configure_logging()
    print(f"[driver] cdr={cdr}, billing_project={billing}, K={args.K}, "
          f"weight_y={args.weight_y}, lookback_days={args.lookback_days}, "
          f"window_days={args.window_days}, stability_days={args.stability_days}, "
          f"person_mod={args.person_mod}", flush=True)
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

    # --- 4) Bridge to dense X + assemble multi-task y/mask --------------------
    with _phase("bridge (BOW -> dense X) + label/mask assembly"):
        X, person_order = collect_bow_aligned(bow_df, V)
        drugs_present = [
            outcome_by_person[p][0] for p in person_order if p in outcome_by_person
        ]
        drug_order = stable_drug_order(
            drugs_present, reference=_ANTIDEPRESSANT_INGREDIENTS,
        )
        y, mask = assemble_multitask_labels(outcome_by_person, person_order, drug_order)
        C = len(drug_order)
        print(f"[driver]   X={X.shape}, C={C} drug columns: {drug_order}", flush=True)

    # --- 5) Split + evaluate --------------------------------------------------
    with _phase(f"split (test_frac={args.test_frac}) + evaluate_pc_multitask "
                f"(K={args.K}, weight_y={args.weight_y})"):
        groups = [
            outcome_by_person.get(p, (None,))[0] for p in person_order
        ]
        is_test = stratified_test_mask(groups, args.test_frac, args.seed)
        tr, te = ~is_test, is_test
        print(f"[driver]   split: {int(tr.sum())} train / {int(te.sum())} test", flush=True)
        results = evaluate_pc_multitask(
            X[tr], y[tr], mask[tr], X[te], y[te], mask[te],
            K=args.K, weight_y=args.weight_y, alpha=args.alpha, tau=args.tau,
            pi_iters=args.pi_iters, max_iter=args.max_iter,
            doc_batch_size=args.doc_batch_size, seed=args.seed,
        )

    # --- Report: the Hughes per-drug table (PC vs two-stage vs LR-on-codes) ---
    print("\n[driver] per-drug results (column index -> index drug):", flush=True)
    for c, name in enumerate(drug_order):
        print(f"[driver]   label {c} = {name}", flush=True)
    print(format_results_table(results), flush=True)

    out_payload = {
        "results": results,
        "drug_order": drug_order,
        "column_drug_names": {c: name for c, name in enumerate(drug_order)},
        "vocab_size": V,
        "n_persons": len(person_order),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "params": {
            "K": args.K, "weight_y": args.weight_y, "alpha": args.alpha,
            "tau": args.tau, "pi_iters": args.pi_iters, "max_iter": args.max_iter,
            "doc_batch_size": args.doc_batch_size,
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


if __name__ == "__main__":
    sys.exit(main())
