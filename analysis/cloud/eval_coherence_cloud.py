"""Cloud-side NPMI coherence for a saved OnlineLDA / OnlineHDP checkpoint.

Mirrors `analysis/local/eval_coherence.py` for the BigQuery-sourced cloud
setting. Loads the augmented VIResult written by `lda_bigquery_cloud.py`
/ `hdp_bigquery_cloud.py`, reproduces the BQ -> BOW pipeline from
`metadata['corpus_manifest']` with the *frozen* vocab from
`metadata['vocab']`, and computes per-topic NPMI over the full corpus.

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


# Ranked-report printing lives in analysis._eval_common.print_ranked_report —
# imported lazily inside main() so the spark-submit path-munging there
# (adding the repo root to sys.path) runs first.


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
        "--npmi-min-pair-count", type=int, default=3,
        help=("Skip pairs with joint count below this threshold in the "
              "reference corpus. Default 3 — pairs in {0, 1, 2} contribute "
              "nothing to their topic mean (vs. the pre-2026-05-12 -1 "
              "floor that biased rare-phenotype topics toward maximally "
              "incoherent scores). Set to 1 to reproduce the historical "
              "behavior."),
    )
    parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto",
        help=("ANSI dimming of unused (NaN-NPMI) topics in the per-topic "
              "printout. 'auto' enables when stdout is a TTY."),
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
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic import compute_npmi_coherence, top_k_used_topics
    from spark_vi.io import load_result

    # The shared print_ranked_report lives under the top-level `analysis`
    # package. spark-submit runs this file with cwd=analysis/cloud (after
    # `cd` in the Makefile), so add the repo root (two levels up) to sys.path
    # so `from analysis._eval_common import ...` resolves.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from analysis._eval_common import print_ranked_report  # noqa: E402

    _configure_logging()

    print(f"[driver] checkpoint={args.checkpoint}, top_n={args.top_n}, "
          f"model_class={args.model_class}", flush=True)

    with _phase("load checkpoint"):
        result = load_result(args.checkpoint)
        corpus = result.metadata.get("corpus_manifest")
        if corpus is None:
            raise SystemExit(
                "checkpoint metadata is missing 'corpus_manifest'. The cloud "
                "eval driver needs the (cdr, person_mod) to reproduce the "
                "OMOP fetch. Re-run the fit driver to regenerate the "
                "checkpoint."
            )
        vocab_list = result.metadata.get("vocab")
        if not vocab_list:
            raise SystemExit(
                "checkpoint metadata has no 'vocab'; cannot freeze BOW build. "
                "This checkpoint predates the vocab-in-metadata convention; "
                "re-fit to use this eval driver."
            )
        # Reconstruct the fit-time DocSpec from the manifest. Pre-ADR-0018
        # checkpoints lack the doc_spec field; default to PatientDocSpec to
        # match their actual fit behavior. source_table is similarly defaulted
        # to condition_occurrence for back-compat.
        doc_spec_manifest = corpus.get("doc_spec", {"name": "patient"})
        doc_spec = DocSpec.from_manifest(doc_spec_manifest)
        source_table = corpus.get("source_table", "condition_occurrence")
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"source_table={source_table}, "
              f"person_mod={corpus['person_mod']}", flush=True)
        print(f"[driver]   doc_spec: {doc_spec_manifest}", flush=True)
        print(f"[driver]   frozen vocab: {len(vocab_list)} terms", flush=True)

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

        with _phase(f"vectorize (frozen vocab, doc_spec={doc_spec.name})"):
            bow_df, vocab_map = to_bow_dataframe(
                omop,
                doc_spec=doc_spec,
                vocab=vocab_list,
            )
            print(f"[driver]   vocab size: {len(vocab_map)}", flush=True)

        with _phase("npmi reference (full corpus)"):
            reference_df = bow_df.persist()
            n_ref = reference_df.count()
            print(f"[driver]   reference: {n_ref} docs", flush=True)

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
                topic_mask=mask,
                min_pair_count=args.npmi_min_pair_count,
            )

        # Property checks — NaN entries (unrated topics) bypass the
        # bound check intentionally; they're sentinels, not NPMI values.
        rated = ~np.isnan(report.per_topic_npmi)
        assert (report.per_topic_npmi[rated] >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi[rated] <= 1.0).all(), "NPMI > 1 found"

        alpha_param = (
            result.global_params.get("alpha")
            if args.model_class == "lda" else None
        )
        print_ranked_report(
            report, name_by_idx, lambda_,
            alpha=alpha_param,
            color=args.color,
        )

        reference_df.unpersist()
        omop.unpersist()
        print("[driver] EVAL COHERENCE CLOUD PASSED", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
