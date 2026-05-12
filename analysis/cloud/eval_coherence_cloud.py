"""Cloud-side held-out NPMI coherence for a saved OnlineLDA / OnlineHDP checkpoint.

Mirrors `analysis/local/eval_coherence.py` but rebuilds the corpus from
BigQuery using parameters stamped at fit time. Loads the augmented
VIResult written by `lda_bigquery_cloud.py` / `hdp_bigquery_cloud.py`,
reproduces the BQ -> BOW pipeline from `metadata['corpus_manifest']`,
applies the matching person-keyed split from `metadata['split']`, and
computes per-topic NPMI on the holdout.

For HDP, only the top-K topics by E[β_t] are scored (the corpus stick
shrinks unused topics toward 0; scoring them would add noise rather than
signal). K is set by `--hdp-k`.

Two environment variables must be set (the same ones the fit driver
reads):
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from analysis/cloud on the Dataproc master):
    make eval-bq-coherence CHECKPOINT=/mnt/gcs/$BUCKET/runs/<id>
    make eval-bq-coherence CHECKPOINT=... EVAL_ARGS='--model-class hdp --hdp-k 50'
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[driver]   %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("spark_vi").setLevel(logging.INFO)


@contextmanager
def _phase(name: str):
    print(f"[driver] >>> {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[driver] <<< {name}: {time.perf_counter() - t0:.1f}s",
              flush=True)


def _print_ranked_report(report, name_by_idx, *,
                         npmi_reference: str = "full") -> None:
    """Print per-topic NPMI ranked from best to worst.

    `name_by_idx[term_idx]` maps a vocab index to a "concept_id (name)"
    label. We accept a dict (or list) keyed by integer vocab index rather
    than a (vocab_list, concept_names) pair the local driver uses, because
    the cloud-side concept-name lookup is its own BQ query rather than a
    metadata sidecar.

    Coverage (cov=NN%) is the fraction of top-N pairs that cleared the
    min_pair_count threshold. Topics with cov=0% are shown last with
    NPMI=NaN.
    """
    rows = list(zip(
        report.topic_indices, report.per_topic_npmi,
        report.per_topic_scored_pairs, report.top_term_indices,
    ))
    rows.sort(key=lambda r: (np.isnan(r[1]), -r[1] if not np.isnan(r[1]) else 0.0))
    total_pairs = report.per_topic_total_pairs
    print(f"\n  per-topic NPMI (reference={npmi_reference}, "
          f"reference_size={report.reference_size}, "
          f"top_n={report.top_n}, min_pair_count={report.min_pair_count}, "
          f"unrated={report.n_topics_unrated}/{len(report.per_topic_npmi)}):")
    print(f"  mean={report.mean:+.4f}  median={report.median:+.4f}  "
          f"stdev={report.stdev:.4f}  min={report.min:+.4f}  "
          f"max={report.max:+.4f}\n")
    for topic_idx, npmi, scored_pairs, term_idx in rows:
        labels = [name_by_idx.get(int(i), f"#{int(i)}") for i in term_idx]
        cov_pct = int(round(100 * scored_pairs / total_pairs)) if total_pairs else 0
        npmi_str = "  NaN   " if np.isnan(npmi) else f"{npmi:+.4f}"
        print(f"  topic {int(topic_idx):3d}  NPMI={npmi_str}  "
              f"cov={cov_pct:>3d}%  top: {', '.join(labels[:8])}")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show defaults automatically, and preserve the docstring's paragraph layout."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=_HelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True,
        help=("path to the saved VIResult directory (local or GCS-mounted, "
              "e.g. /mnt/gcs/<bucket>/runs/<id>)"),
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="number of top terms per topic used for the NPMI computation",
    )
    parser.add_argument(
        "--model-class", choices=["lda", "hdp"], default="lda",
        help="lda scores all K rows of lambda; hdp scores the top-K topics "
             "by E[beta_t] (see --hdp-k).",
    )
    parser.add_argument(
        "--hdp-k", type=int, default=50,
        help="Top-K HDP topics by E[beta_t] to score (ignored for LDA).",
    )
    parser.add_argument(
        "--npmi-reference", choices=["holdout", "full"], default="full",
        help=("Which docs serve as the co-occurrence reference for NPMI. "
              "'full' (default) uses train ∪ holdout — 5× the data, "
              "fewer rare-pair zeros; methodologically sound because "
              "topic-word distributions are fixed post-fit (no "
              "overfitting concern). 'holdout' reproduces pre-2026-05-12 "
              "behavior."),
    )
    parser.add_argument(
        "--npmi-min-pair-count", type=int, default=3,
        help=("Skip pairs with joint count below this threshold in the "
              "reference corpus. Default 3 — pairs in {0, 1, 2} contribute "
              "nothing to their topic mean (vs. the pre-2026-05-12 -1 "
              "floor that biased rare-phenotype topics toward maximally "
              "incoherent scores). Set to 1 to reproduce the historical "
              "behavior."),
    )
    args = parser.parse_args(argv)

    cdr_env = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr_env and billing):
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    # Driver-side imports proven first.
    from charmpheno.omop import (
        DocSpec, load_omop_bigquery, to_bow_dataframe,
    )
    from charmpheno.omop.split import split_bow_by_person
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic import compute_npmi_coherence, top_k_used_topics
    from spark_vi.io import load_result

    # The shared verify_split_contract lives under the top-level `analysis`
    # package. spark-submit runs this file with cwd=analysis/cloud (after
    # `cd` in the Makefile), so add the repo root (two levels up) to sys.path
    # so `from analysis._eval_common import ...` resolves.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from analysis._eval_common import verify_split_contract  # noqa: E402

    _configure_logging()

    print(f"[driver] checkpoint={args.checkpoint}, top_n={args.top_n}, "
          f"model_class={args.model_class}", flush=True)

    with _phase("load checkpoint"):
        result = load_result(args.checkpoint)
        split = result.metadata.get("split")
        corpus = result.metadata.get("corpus_manifest")
        if split is None:
            raise SystemExit(
                "checkpoint metadata is missing 'split'. The cloud eval driver "
                "requires the augmented VIResult shape written by the post-"
                "split-contract lda_bigquery_cloud.py. Re-run the fit driver "
                "to regenerate the checkpoint with augmented metadata."
            )
        if corpus is None:
            raise SystemExit(
                "checkpoint metadata is missing 'corpus_manifest'. The cloud "
                "eval driver needs the (cdr, person_mod, vocab_size, min_df) "
                "to reproduce the exact BOW. Re-run the fit driver to "
                "regenerate the checkpoint."
            )
        if not split.get("applied", False):
            raise SystemExit(
                "checkpoint records split.applied=False — the fit driver was "
                "run without --holdout-fraction. There is no held-out partition "
                "to evaluate against. Re-fit with --holdout-fraction > 0."
            )
        holdout_fraction = float(split["holdout_fraction"])
        holdout_seed = int(split["holdout_seed"])
        # Reconstruct the fit-time DocSpec from the manifest. Pre-ADR-0018
        # checkpoints lack the doc_spec field; default to PatientDocSpec to
        # match their actual fit behavior. source_table is similarly defaulted
        # to condition_occurrence for back-compat.
        doc_spec_manifest = corpus.get("doc_spec", {"name": "patient"})
        doc_spec = DocSpec.from_manifest(doc_spec_manifest)
        source_table = corpus.get("source_table", "condition_occurrence")
        print(f"[driver]   split: holdout_fraction={holdout_fraction}, "
              f"holdout_seed={holdout_seed}", flush=True)
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"source_table={source_table}, "
              f"person_mod={corpus['person_mod']}, "
              f"vocab_size={corpus['vocab_size']}, "
              f"min_df={corpus['min_df']}", flush=True)
        print(f"[driver]   doc_spec: {doc_spec_manifest}", flush=True)

        # Exercise the shared contract check. The fit-stamped values are
        # tautologically equal to the ones we just read out of metadata, so
        # this matches silently — but it keeps the cloud driver on the same
        # validation code path as the local one (future failure modes,
        # e.g. checkpoint corruption, surface identically).
        verify_split_contract(
            result, holdout_fraction=holdout_fraction, seed=holdout_seed,
        )

        if corpus["cdr"] != cdr_env:
            log = logging.getLogger(__name__)
            log.warning(
                "WORKSPACE_CDR (%s) differs from corpus_manifest['cdr'] (%s); "
                "using the checkpoint's cdr so the BOW is reproducible.",
                cdr_env, corpus["cdr"],
            )

    spark = SparkSession.builder.appName("eval_coherence_cloud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    # Silence the spot-reclamation flood (BlockManager cascades, FetchFailed
    # stack traces from TaskSetManager, etc.) without losing other WARN.
    from _log_utils import quiet_spot_reclamation
    quiet_spot_reclamation(spark)
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    try:
        with _phase("BQ load + summary"):
            omop = load_omop_bigquery(
                spark=spark,
                cdr_dataset=corpus["cdr"],
                billing_project=billing,
                person_sample_mod=corpus["person_mod"],
                source_table=source_table,
            ).persist()
            summary = omop.agg(
                F.count(F.lit(1)).alias("rows"),
                F.countDistinct("person_id").alias("persons"),
            ).collect()[0]
            print(f"[driver]   OMOP: {summary['rows']} rows, "
                  f"{summary['persons']} distinct persons", flush=True)

        with _phase(f"vectorize (CountVectorizer, doc_spec={doc_spec.name})"):
            bow_df, vocab_map = to_bow_dataframe(
                omop,
                doc_spec=doc_spec,
                vocab_size=corpus["vocab_size"],
                min_df=corpus["min_df"],
            )
            print(f"[driver]   vocab size: {len(vocab_map)}", flush=True)

        if args.npmi_reference == "full":
            # NPMI doesn't have a predictive-overfitting concern (topic-
            # word distributions are fixed post-fit). Using train ∪
            # holdout as the co-occurrence reference gives 5× more
            # statistical evidence per pair, dramatically reducing
            # rare-pair sparsity. See ADR 0017 revisions (2026-05-12).
            with _phase("npmi reference = full corpus"):
                reference_df = bow_df.persist()
                n_ref = reference_df.count()
                print(f"[driver]   reference: {n_ref} docs (train + holdout)",
                      flush=True)
        else:
            with _phase(f"npmi reference = holdout split "
                         f"(holdout_fraction={holdout_fraction}, "
                         f"holdout_seed={holdout_seed})"):
                _train_df, reference_df = split_bow_by_person(
                    bow_df,
                    holdout_fraction=holdout_fraction,
                    seed=holdout_seed,
                )
                reference_df = reference_df.persist()
                n_ref = reference_df.count()
                print(f"[driver]   reference: {n_ref} docs (holdout)",
                      flush=True)

        with _phase("concept-name lookup"):
            name_rows = (
                omop.where(F.col("concept_id").isin(list(vocab_map.keys())))
                    .select("concept_id", "concept_name")
                    .dropDuplicates(["concept_id"])
                    .collect()
            )
            name_by_id = {
                int(r["concept_id"]): r["concept_name"] for r in name_rows
            }
            # Build idx -> "cid (name)" once so the rank report below can
            # look up by vocab index without re-walking vocab_map per term.
            idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
            name_by_idx = {
                idx: f"{cid} ({name_by_id.get(cid, '?')})"
                for idx, cid in idx_to_cid.items()
            }
            print(f"[driver]   resolved {len(name_by_id)} concept names",
                  flush=True)

        with _phase(f"NPMI coherence (top_n={args.top_n}, "
                     f"min_pair_count={args.npmi_min_pair_count})"):
            lambda_ = result.global_params["lambda"]
            topic_term = lambda_ / lambda_.sum(axis=1, keepdims=True)
            if args.model_class == "hdp":
                u = result.global_params["u"]
                v = result.global_params["v"]
                mask = top_k_used_topics(u=u, v=v, k=args.hdp_k)
                print(f"[driver]   HDP top-K mask: scoring "
                      f"{int(mask.sum())}/{len(mask)} topics by E[β_t]",
                      flush=True)
            else:
                mask = None
            reference_rdd = reference_df.rdd.map(BOWDocument.from_spark_row)
            report = compute_npmi_coherence(
                topic_term, reference_rdd, top_n=args.top_n,
                hdp_topic_mask=mask,
                min_pair_count=args.npmi_min_pair_count,
            )

        # Property checks — NaN entries (unrated topics) bypass the
        # bound check intentionally; they're sentinels, not NPMI values.
        rated = ~np.isnan(report.per_topic_npmi)
        assert (report.per_topic_npmi[rated] >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi[rated] <= 1.0).all(), "NPMI > 1 found"

        _print_ranked_report(report, name_by_idx,
                             npmi_reference=args.npmi_reference)

        reference_df.unpersist()
        omop.unpersist()
        print("[driver] EVAL COHERENCE CLOUD PASSED", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
