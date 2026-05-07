"""End-to-end Online HDP on BigQuery-resident OMOP condition data.

Sibling of lda_bigquery_cloud.py. Reads condition_occurrence from the
workspace CDR (full-patient sampled), joins to concept for human-readable
names, vectorizes via CountVectorizer, fits OnlineHDPEstimator (the
MLlib-shaped shim around our SVI/CAVI HDP), and prints top concept names
per *active* topic. Print-only; no artifact persistence in v2.

Reads two environment variables (set by the workspace setup notebook,
exported in ~/.bashrc on the Dataproc master):
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from this directory on the Dataproc master):
    make hdp-bq-smoke
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
    """Surface spark_vi.core.runner per-iter INFO lines with [driver] prefix."""
    logging.basicConfig(
        level=logging.WARNING,
        format="[driver]   %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("spark_vi").setLevel(logging.INFO)


@contextmanager
def _phase(name: str):
    """Bracket a driver phase with start/end markers and elapsed wall time."""
    print(f"[driver] >>> {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[driver] <<< {name}: {time.perf_counter() - t0:.1f}s",
              flush=True)


def _make_topic_evolution_logger(
    top_n, every_n, idx_to_cid, name_by_id, T, mass_threshold,
):
    """Build an on_iteration callback that prints top-N tokens per *active*
    HDP topic.

    Same factory pattern as lda_bigquery_cloud._make_topic_evolution_logger,
    but:
      * Sorts topics by E[β_t] (corpus stick mean), not λ row-sum, to get
        the HDP-native usage ranking.
      * Filters to the smallest set of topics whose E[β_t] sum to ≥
        mass_threshold (truncation-independent; matches the shim's
        OnlineHDPModel.activeTopicCount and the model's iteration_summary).
      * Per-topic prefix is the corpus stick weight E[β_t] plus Σλ and
        peakedness, mirroring the LDA driver's diagnostic shape.

    OnlineHDP's global_params has keys "lambda" (T, V), "u" (T-1,), and
    "v" (T-1,). Stick means feed E[β_t]; row-normalized λ is the
    topic-word distribution.
    """
    # Import deferred to factory body — same fail-fast-on-misshapen-py-files
    # discipline as the spark_vi imports inside main().
    from spark_vi.models.online_hdp import (
        expected_corpus_betas,
        topic_count_at_mass,
    )

    def _on_iter(iter_num: int, global_params: dict,
                 _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        lam = global_params["lambda"]                         # (T, V)
        u = global_params["u"]                                # (T-1,)
        v = global_params["v"]                                # (T-1,)
        E_beta = expected_corpus_betas(u, v, T=T)
        n_active = topic_count_at_mass(E_beta, mass_threshold)
        order = [int(t) for t in np.argsort(E_beta)[::-1][:n_active]]
        lam_row_sums = lam.sum(axis=1)
        peak = lam.max(axis=1) / np.maximum(lam_row_sums, 1e-12)
        topics = lam / np.maximum(lam_row_sums[:, None], 1e-12)
        print(
            f"[driver]   --- topics @ iter {iter_num} "
            f"({n_active}/{T} active) ---",
            flush=True,
        )
        for t in order:
            top = topics[t].argsort()[::-1][:top_n]
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}"
                f"({topics[t, int(j)]:.3f})"
                for j in top
            )
            print(
                f"[driver]    topic {t:>3}  "
                f"E[β]={E_beta[t]:.4f}  Σλ={lam_row_sums[t]:.3g}  "
                f"peak={peak[t]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show defaults automatically, and preserve the docstring's paragraph layout."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=_HelpFormatter)
    parser.add_argument(
        "--T", type=int, default=150,
        help=("corpus truncation: upper bound on discoverable topics. The "
              "effective active count is typically much smaller (HDP shrinks "
              "the inactive sticks toward 0). Mapped to the shim's `k` Param."),
    )
    parser.add_argument(
        "--K", type=int, default=15,
        help=("doc-level truncation: upper bound on topics per visit. "
              "Should be much smaller than T — clinical visits typically span "
              "a handful of phenotypes."),
    )
    parser.add_argument(
        "--max-iter", type=int, default=50,
        help="maximum SVI iterations on the global parameters",
    )
    parser.add_argument(
        "--person-mod", type=int, default=1000,
        help=("whole-patient sampling: keep rows where MOD(person_id, M) == 0. "
              "Larger M => smaller cohort. Set to 1 for the full corpus."),
    )
    parser.add_argument(
        "--vocab-size", type=int, default=2000,
        help="CountVectorizer vocabulary cap",
    )
    parser.add_argument(
        "--min-df", type=int, default=5,
        help="minimum document frequency for a concept to enter the vocabulary",
    )
    parser.add_argument(
        "--top-n-tokens", type=int, default=10,
        help="tokens shown per topic in the post-fit summary table",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--subsampling-rate", type=float, default=0.05,
        help=("mini-batch fraction per SVI iteration. 1.0 = full-batch. "
              "On small corpora, prefer larger fractions; on real OMOP "
              "scale, 0.05 is a good default."),
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="doc-stick concentration α",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help=("corpus-stick concentration γ. Higher → more topics get "
              "discovered. ADR 0011 keeps this fixed in v1."),
    )
    parser.add_argument(
        "--eta", type=float, default=0.01,
        help="topic-word Dirichlet concentration η",
    )
    parser.add_argument(
        "--print-topics-every", type=int, default=10,
        help=("during fit, emit a topic summary (top tokens per active topic) "
              "every N iterations. 0 disables."),
    )
    parser.add_argument(
        "--active-mass-threshold", type=float, default=0.95,
        help=("cumulative-mass threshold for the 'active' filter — count and "
              "show the smallest set of topics whose top-ranked E[β_t] sum "
              "to ≥ this value. Default 0.95 (PCA's explained-variance "
              "analog). Used by mid-fit snapshots and the post-fit summary."),
    )
    args = parser.parse_args(argv)

    cdr = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr and billing):
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    # Driver-side imports proven first — fail fast if --py-files is misshapen.
    from charmpheno.omop import load_omop_bigquery, to_bow_dataframe
    from spark_vi.diagnostics.persist import assert_persisted
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    _configure_logging()

    print(f"[driver] cdr={cdr}, billing_project={billing}, "
          f"T={args.T}, K={args.K}, max_iter={args.max_iter}, "
          f"person_mod={args.person_mod}", flush=True)

    spark = SparkSession.builder.appName("hdp_bigquery_cloud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    with _phase("BQ load + summary"):
        omop = load_omop_bigquery(
            spark=spark,
            cdr_dataset=cdr,
            billing_project=billing,
            person_sample_mod=args.person_mod,
        ).persist()
        summary = omop.agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("person_id").alias("persons"),
        ).collect()[0]
        assert_persisted(omop, name="omop")
        print(f"[driver]   OMOP: {summary['rows']} rows, "
              f"{summary['persons']} distinct persons", flush=True)

    with _phase("vectorize (CountVectorizer)"):
        bow_df, vocab_map = to_bow_dataframe(
            omop, vocab_size=args.vocab_size, min_df=args.min_df,
        )
        bow_df = bow_df.persist()
        n_docs = bow_df.count()
        assert_persisted(bow_df, name="bow_df")
        idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
        print(f"[driver]   vocab size: {len(vocab_map)} (cap {args.vocab_size}, "
              f"minDF {args.min_df}), documents: {n_docs}", flush=True)

    with _phase("concept-name lookup"):
        name_rows = (
            omop.where(F.col("concept_id").isin(list(vocab_map.keys())))
                .select("concept_id", "concept_name")
                .dropDuplicates(["concept_id"])
                .collect()
        )
        name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}
        print(f"[driver]   resolved {len(name_by_id)} concept names",
              flush=True)

    with _phase(f"fit (T={args.T}, K={args.K}, maxIter={args.max_iter}, "
                 f"subsamplingRate={args.subsampling_rate}, "
                 f"α={args.alpha}, γ={args.gamma}, η={args.eta})"):
        estimator = OnlineHDPEstimator(
            k=args.T, docTruncation=args.K, maxIter=args.max_iter,
            seed=args.seed, subsamplingRate=args.subsampling_rate,
            docConcentration=[args.alpha],
            corpusConcentration=args.gamma,
            topicConcentration=args.eta,
        )
        if args.print_topics_every > 0:
            estimator.setOnIteration(_make_topic_evolution_logger(
                top_n=6, every_n=args.print_topics_every,
                idx_to_cid=idx_to_cid, name_by_id=name_by_id, T=args.T,
                mass_threshold=args.active_mass_threshold,
            ))
        model = estimator.fit(bow_df)
        print(f"[driver]   elbo trace tail: {model.result.elbo_trace[-3:]}",
              flush=True)
        n_active = model.activeTopicCount(
            mass_threshold=args.active_mass_threshold,
        )
        print(
            f"[driver]   active topics: {n_active}/{args.T} "
            f"(covering ≥{args.active_mass_threshold:.0%} of corpus mass)",
            flush=True,
        )

    # topicsMatrix is (V, T); pick the active topics by E[β_t] descending,
    # cumulative-mass threshold (matches the shim's activeTopicCount).
    tm = model.topicsMatrix().toArray()
    E_beta = model.corpusStickWeights()
    n_active_t = model.activeTopicCount(mass_threshold=args.active_mass_threshold)
    active_t = [int(t) for t in np.argsort(E_beta)[::-1][:n_active_t]]
    print(f"\n[driver] top-{args.top_n_tokens} tokens per active topic "
          f"(concept_id  concept_name  weight), ordered by E[β]:",
          flush=True)
    for t in active_t:
        col = tm[:, t]
        top_idx = col.argsort()[::-1][:args.top_n_tokens]
        print(f"\n  Topic {t} (E[β]={E_beta[t]:.4f}):", flush=True)
        for j in top_idx:
            cid = idx_to_cid[int(j)]
            name = name_by_id.get(cid, "<unknown>")
            print(f"    {cid:>10}  {name[:60]:<60}  {col[j]:.4f}", flush=True)

    with _phase("transform sample"):
        (model.transform(bow_df)
              .withColumn("person_hash",
                          F.substring(
                              F.sha2(F.col("person_id").cast("string"), 256),
                              1, 12))
              .select("person_hash", "topicDistribution")
              .show(3, truncate=False))

    omop.unpersist()
    bow_df.unpersist()
    print("[driver] HDP BQ SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
