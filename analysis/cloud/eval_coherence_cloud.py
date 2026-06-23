"""Cloud-side NPMI coherence for a saved OnlineLDA / OnlineHDP / STM checkpoint.

Mirrors `analysis/local/eval_coherence.py` for the BigQuery-sourced cloud
setting. Loads the augmented VIResult written by `lda_bigquery_cloud.py`
/ `hdp_bigquery_cloud.py` / `stm_bigquery_cloud.py`, reproduces the BQ -> BOW
pipeline from `metadata['corpus_manifest']` (including the fit cohort and its
prior_obs_days lookback) with the *frozen* vocab from `metadata['vocab']`, and
computes per-topic NPMI over that fit corpus. The cohort is reproduced, not
skipped: cohort-derived columns (e.g. source_cohort for the combined cohort's
patient_cohort doc_spec) only exist once the cohort filter runs, and a
cohort-matched reference scores coherence over the corpus the topics saw.

STM is scored exactly like LDA: its topic-term β (global_params['lambda']) is
K×V with fixed K and no Dirichlet alpha, so all K topics are scored.

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

import numpy as np
from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session


# Ranked-report printing lives in analysis._eval_common.print_ranked_report —
# imported lazily inside main() so the spark-submit path-munging there
# (adding the repo root to sys.path) runs first.


def foreground_reference_groups(topic_block_spec):
    """Map topic index -> group label (None = background / full-corpus reference).

    Foreground topics are scored against their group's sub-corpus rather than the
    full corpus: scoring a rare phenotype against majority docs that can never
    contain it triggers the NPMI zero-pair penalty (docs/insights/0007). Returns
    {} when there is no gating partition (all topics scored on the full corpus).
    """
    if not topic_block_spec:
        return {}
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition.from_dict(topic_block_spec)
    labels = part.topic_labels()
    return {k: (None if lbl == "background" else lbl)
            for k, lbl in enumerate(labels)}


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
        "--model-class", choices=["lda", "hdp", "stm"], default="lda",
        help="lda/stm score all K rows of lambda (STM's topic-term beta has "
             "the same K×V shape, fixed K, and no Dirichlet alpha); hdp scores "
             "the top-K topics by E[beta_t] (see --hdp-k).",
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

    configure_logging()

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
        # Reproduce the fit COHORT, not just the base load: cohort-derived
        # columns (e.g. source_cohort for the combined cohort's patient_cohort
        # doc_spec) only exist once the cohort filter is applied, and a
        # cohort-matched NPMI reference is the corpus the topics were fit on.
        cohort = corpus.get("cohort")
        if cohort == "none":
            cohort = None
        # prior_obs_days entered the manifest after the cohort feature; older
        # checkpoints default to the historical 365-day lookback.
        prior_obs_days = corpus.get("prior_obs_days", 365)
        topic_block_spec = corpus.get("topic_block_spec")
        fg_groups = foreground_reference_groups(topic_block_spec)
        if fg_groups:
            print(f"[driver]   gating: {sum(v is not None for v in fg_groups.values())} "
                  f"foreground topics scored on group sub-corpora", flush=True)
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"source_table={source_table}, "
              f"person_mod={corpus['person_mod']}, cohort={cohort}, "
              f"prior_obs_days={prior_obs_days}", flush=True)
        print(f"[driver]   doc_spec: {doc_spec_manifest}", flush=True)
        print(f"[driver]   frozen vocab: {len(vocab_list)} terms", flush=True)

        if corpus["cdr"] != cdr_env:
            log = logging.getLogger(__name__)
            log.warning(
                "WORKSPACE_CDR (%s) differs from corpus_manifest['cdr'] (%s); "
                "using the checkpoint's cdr so the BOW is reproducible.",
                cdr_env, corpus["cdr"],
            )

    spark = make_spark_session("eval_coherence_cloud")

    try:
        with _phase("BQ load + summary"):
            omop = load_omop_bigquery(
                spark=spark,
                cdr_dataset=corpus["cdr"],
                billing_project=billing,
                person_sample_mod=corpus["person_mod"],
                source_table=source_table,
                cohort=cohort,
                prior_obs_days=prior_obs_days,
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

        with _phase("npmi reference (fit corpus)"):
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
            if fg_groups:
                from pyspark.sql import functions as F
                bow_g = bow_df.withColumn(
                    "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))
                per_topic = np.full(topic_term.shape[0], np.nan)
                # Background topics: full-corpus reference.
                bg_idx = [k for k, g in fg_groups.items() if g is None]
                bg_rep = compute_npmi_coherence(
                    topic_term[bg_idx], reference_rdd, top_n=args.top_n,
                    min_pair_count=args.npmi_min_pair_count)
                for j, k in enumerate(bg_idx):
                    per_topic[k] = bg_rep.per_topic_npmi[j]
                # Foreground topics: per-group sub-corpus reference.
                for g in sorted({v for v in fg_groups.values() if v is not None}):
                    g_idx = [k for k, gg in fg_groups.items() if gg == g]
                    g_ref = (bow_g.where(F.col("source_cohort") == g)
                             .rdd.map(BOWDocument.from_spark_row))
                    g_rep = compute_npmi_coherence(
                        topic_term[g_idx], g_ref, top_n=args.top_n,
                        min_pair_count=args.npmi_min_pair_count)
                    for j, k in enumerate(g_idx):
                        per_topic[k] = g_rep.per_topic_npmi[j]
                print("[driver]   per-topic NPMI (block-aware reference):", flush=True)
                labels = foreground_reference_groups(topic_block_spec)
                for k in range(topic_term.shape[0]):
                    blk = "background" if labels[k] is None else labels[k]
                    print(f"[driver]     topic {k:3d} [{blk}] NPMI={per_topic[k]:+.4f}",
                          flush=True)
                print("[driver] EVAL COHERENCE CLOUD PASSED", flush=True)
                return 0
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
