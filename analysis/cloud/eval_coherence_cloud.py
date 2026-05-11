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


def _print_ranked_report(report, name_by_idx) -> None:
    """Print per-topic NPMI ranked from best to worst.

    `name_by_idx[term_idx]` maps a vocab index to a "concept_id (name)"
    label. We accept a dict (or list) keyed by integer vocab index rather
    than a (vocab_list, concept_names) pair the local driver uses, because
    the cloud-side concept-name lookup is its own BQ query rather than a
    metadata sidecar.
    """
    rows = sorted(
        zip(report.topic_indices, report.per_topic_npmi,
            report.top_term_indices),
        key=lambda r: -r[1],
    )
    print(f"\n  per-topic NPMI (n_holdout_docs={report.n_holdout_docs}, "
          f"top_n={report.top_n}):")
    print(f"  mean={report.mean:+.4f}  median={report.median:+.4f}  "
          f"stdev={report.stdev:.4f}  min={report.min:+.4f}  "
          f"max={report.max:+.4f}\n")
    for topic_idx, npmi, term_idx in rows:
        labels = [name_by_idx.get(int(i), f"#{int(i)}") for i in term_idx]
        print(f"  topic {int(topic_idx):3d}  NPMI={npmi:+.4f}  "
              f"top: {', '.join(labels[:8])}")


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
    args = parser.parse_args(argv)

    cdr_env = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr_env and billing):
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    # Driver-side imports proven first.
    from charmpheno.omop import load_omop_bigquery, to_bow_dataframe
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
        print(f"[driver]   split: holdout_fraction={holdout_fraction}, "
              f"holdout_seed={holdout_seed}", flush=True)
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"person_mod={corpus['person_mod']}, "
              f"vocab_size={corpus['vocab_size']}, "
              f"min_df={corpus['min_df']}", flush=True)

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
            ).persist()
            summary = omop.agg(
                F.count(F.lit(1)).alias("rows"),
                F.countDistinct("person_id").alias("persons"),
            ).collect()[0]
            print(f"[driver]   OMOP: {summary['rows']} rows, "
                  f"{summary['persons']} distinct persons", flush=True)

        with _phase("vectorize (CountVectorizer)"):
            bow_df, vocab_map = to_bow_dataframe(
                omop,
                vocab_size=corpus["vocab_size"],
                min_df=corpus["min_df"],
            )
            print(f"[driver]   vocab size: {len(vocab_map)}", flush=True)

        with _phase(f"person-keyed split "
                     f"(holdout_fraction={holdout_fraction}, "
                     f"holdout_seed={holdout_seed})"):
            _train_df, holdout_df = split_bow_by_person(
                bow_df,
                holdout_fraction=holdout_fraction,
                seed=holdout_seed,
            )
            holdout_df = holdout_df.persist()
            n_holdout = holdout_df.count()
            print(f"[driver]   holdout: {n_holdout} docs", flush=True)

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

        with _phase(f"NPMI coherence (top_n={args.top_n})"):
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
            holdout_rdd = holdout_df.rdd.map(BOWDocument.from_spark_row)
            report = compute_npmi_coherence(
                topic_term, holdout_rdd, top_n=args.top_n,
                hdp_topic_mask=mask,
            )

        # Property checks — match the local driver's assertions.
        assert (report.per_topic_npmi >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi <= 1.0).all(), "NPMI > 1 found"

        _print_ranked_report(report, name_by_idx)

        holdout_df.unpersist()
        omop.unpersist()
        print("[driver] EVAL COHERENCE CLOUD PASSED", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
