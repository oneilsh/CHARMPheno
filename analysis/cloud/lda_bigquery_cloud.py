"""End-to-end LDA on BigQuery-resident OMOP condition data.

Reads condition_occurrence from the workspace CDR (full-patient sampled),
joins to concept for human-readable names, vectorizes via CountVectorizer,
fits VanillaLDAEstimator (the MLlib-shaped shim around our SVI), and
prints top concept names per topic. Print-only; no artifact persistence
in v1.

Reads two environment variables (set by the workspace setup notebook,
exported in ~/.bashrc on the Dataproc master):
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from this directory on the Dataproc master):
    make lda-bq-smoke
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
    """Surface spark_vi.core.runner per-iter INFO lines with [driver] prefix.

    Root stays at WARNING so PySpark / numpy / etc don't spam. spark_vi is
    bumped to INFO so the runner's iteration progress lines come through.
    `force=True` overrides any handler PySpark may have installed.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="[driver]   %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("spark_vi").setLevel(logging.INFO)


@contextmanager
def _phase(name: str):
    """Bracket a driver phase with start/end markers and elapsed wall time.

    Lets us tell where wall time goes (BQ read vs vectorize vs fit vs
    transform) when comparing slow runs against fast ones.
    """
    print(f"[driver] >>> {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[driver] <<< {name}: {time.perf_counter() - t0:.1f}s",
              flush=True)


def _make_topic_evolution_logger(top_n, every_n, idx_to_cid, name_by_id):
    """Build an on_iteration callback that prints top-N tokens per topic.

    Factory rather than a bare def: the framework's on_iteration contract
    is `(iter_num, global_params, elbo_trace)`, so domain context (vocab
    map, concept names, throttle cadence) rides in via closure capture
    instead of widening the framework signature.

    VanillaLDA's global_params has key "lambda" with shape (K, V) — each
    row is the unnormalized Dirichlet variational parameter for one topic
    over the full vocabulary. Row-normalizing gives the topic-word
    distribution; argsort + slice gives the top tokens. Each topic line
    is prefixed with per-topic stats:
        α_k    — current empirical-Bayes prior on θ_·k (length-K vector
                 in global_params["alpha"]). Diverges from 1/K under
                 optimize_alpha; corpus-popular topics get larger α_k.
        Σλ_k   — total topic mass over the vocabulary. Proxy for how much
                 corpus evidence has accreted to topic k.
        peak   — max_v(λ_kv) / Σλ_k. Peakedness of the topic-word
                 posterior. ~1/V means uniform / undifferentiated; rising
                 toward 1.0 means the topic specialized (or, at small K,
                 collapsed onto a single term).
    """
    def _on_iter(iter_num: int, global_params: dict,
                 _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        lam = global_params["lambda"]                         # (K, V)
        alpha = global_params["alpha"]                        # (K,)
        lam_row_sums = lam.sum(axis=1)                        # (K,)
        peak = lam.max(axis=1) / np.maximum(lam_row_sums, 1e-12)
        topics = lam / lam_row_sums[:, None]                  # row-stochastic
        # Sort topics by Σλ_k descending so the heaviest topics are listed
        # first. The k label printed on each line is the topic's native
        # index (stable across iterations), so a topic moving up or down
        # the ranking is a meaningful signal.
        order = np.argsort(lam_row_sums)[::-1]
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in order:
            top = topics[k].argsort()[::-1][:top_n]
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}"
                f"({topics[k, int(j)]:.3f})"
                for j in top
            )
            print(
                f"[driver]    topic {k:>2}  "
                f"α={alpha[k]:.4g}  Σλ={lam_row_sums[k]:.3g}  "
                f"peak={peak[k]:.3f}  | {terms}",
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
        "--K", type=int, default=20,
        help="number of topics to fit",
    )
    parser.add_argument(
        "--max-iter", type=int, default=50,
        help="maximum SVI iterations on the global parameters",
    )
    parser.add_argument(
        "--person-mod", type=int, default=1000,
        help=("whole-patient sampling: keep rows where MOD(person_id, M) == 0. "
              "Larger M => smaller cohort. Whole-patient (vs row-level) "
              "sampling preserves each retained patient's full condition list, "
              "which matters for LDA's per-document token bag. Set to 1 for "
              "the full corpus."),
    )
    parser.add_argument(
        "--vocab-size", type=int, default=2000,
        help=("CountVectorizer vocabulary cap; concepts beyond this rank "
              "(by document frequency) are dropped before fit"),
    )
    parser.add_argument(
        "--min-df", type=int, default=5,
        help=("minimum document frequency for a concept to enter the "
              "vocabulary; filters singletons / typos before the vocab-size "
              "cap is applied"),
    )
    parser.add_argument(
        "--top-n-tokens", type=int, default=10,
        help="tokens shown per topic in the post-fit summary table",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help=("RNG seed for reproducibility (lambda initialization + "
              "mini-batch sampling)"),
    )
    parser.add_argument(
        "--subsampling-rate", type=float, default=0.05,
        help=("mini-batch fraction per SVI iteration. 1.0 = full-batch, "
              "recommended on small corpora where Spark coordination "
              "dominates per-iter time. Tiny rates (<0.1) on small corpora "
              "yield tiny batches whose per-iter overhead exceeds the actual "
              "work"),
    )
    parser.add_argument(
        "--print-topics-every", type=int, default=10,
        help=("during fit, emit a topic summary (top tokens per topic with "
              "weights) every N iterations. 0 disables. Cheap — runs on the "
              "driver, reads from the broadcast lambda"),
    )
    parser.add_argument(
        "--optimize-doc-concentration",
        action=argparse.BooleanOptionalAction, default=True,
        help=("learn an asymmetric α (length K) via Newton-Raphson empirical "
              "Bayes during fit (Blei 2003 App. A.4.2). After training the fitted "
              "α is on model.alpha — a global prior over per-doc topic "
              "distributions, biased toward whatever topics are corpus-"
              "popular. Wallach 2009 found this is the high-value asymmetry. "
              "Negate with --no-optimize-doc-concentration to keep α static."),
    )
    parser.add_argument(
        "--optimize-topic-concentration",
        action=argparse.BooleanOptionalAction, default=False,
        help=("learn a symmetric scalar η via Newton-Raphson during fit "
              "(Hoffman 2010 §3.4). Off by default — η optimization is the "
              "less stable of the two on small corpora; Wallach 2009 also "
              "argued symmetric η over vocabulary is the right default. "
              "Enable with --optimize-topic-concentration if you want it."),
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
    from spark_vi.mllib.lda import VanillaLDAEstimator

    _configure_logging()

    print(f"[driver] cdr={cdr}, billing_project={billing}, "
          f"K={args.K}, max_iter={args.max_iter}, "
          f"person_mod={args.person_mod}", flush=True)

    spark = SparkSession.builder.appName("lda_bigquery_cloud").getOrCreate()
    # Silence the GCS connector chatter (RequestTracker / hflush rate-limit
    # noise from event-log writes). Our [driver] prints are stdout, not
    # log4j, so they survive. Set BEFORE any actions.
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
        # One agg pass instead of two — distinct-counting person_id is cheap
        # piggybacked on the row count that has to scan everything anyway.
        # This action also forces the persist to materialize.
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
        n_docs = bow_df.count()  # forces materialization
        assert_persisted(bow_df, name="bow_df")
        idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
        print(f"[driver]   vocab size: {len(vocab_map)} (cap {args.vocab_size}, "
              f"minDF {args.min_df}), documents: {n_docs}", flush=True)

    with _phase("concept-name lookup"):
        # Vocabulary-only lookup; small enough for the driver. dropDuplicates
        # because OMOP occasionally has multiple name variants per concept_id.
        name_rows = (
            omop.where(F.col("concept_id").isin(list(vocab_map.keys())))
                .select("concept_id", "concept_name")
                .dropDuplicates(["concept_id"])
                .collect()
        )
        name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}
        print(f"[driver]   resolved {len(name_by_id)} concept names",
              flush=True)

    with _phase(f"fit (K={args.K}, maxIter={args.max_iter}, "
                 f"subsamplingRate={args.subsampling_rate}, "
                 f"optimizeDocConc={args.optimize_doc_concentration}, "
                 f"optimizeTopicConc={args.optimize_topic_concentration})"):
        estimator = VanillaLDAEstimator(
            k=args.K, maxIter=args.max_iter, seed=args.seed,
            subsamplingRate=args.subsampling_rate,
            optimizeDocConcentration=args.optimize_doc_concentration,
            optimizeTopicConcentration=args.optimize_topic_concentration,
        )
        if args.print_topics_every > 0:
            estimator.setOnIteration(_make_topic_evolution_logger(
                top_n=6, every_n=args.print_topics_every,
                idx_to_cid=idx_to_cid, name_by_id=name_by_id,
            ))
        model = estimator.fit(bow_df)
        print(f"[driver]   elbo trace tail: {model.result.elbo_trace[-3:]}",
              flush=True)

    # topicsMatrix is (V, K); columns are topics. Print top-N tokens per topic.
    tm = model.topicsMatrix().toArray()
    print(f"\n[driver] top-{args.top_n_tokens} tokens per topic "
          f"(concept_id  concept_name  weight):", flush=True)
    for k in range(args.K):
        col = tm[:, k]
        top_idx = col.argsort()[::-1][:args.top_n_tokens]
        print(f"\n  Topic {k}:", flush=True)
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
    print("[driver] LDA BQ SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
