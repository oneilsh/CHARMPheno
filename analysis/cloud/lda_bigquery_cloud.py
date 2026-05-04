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


def _make_topic_evolution_logger(K, top_n, every_n, idx_to_cid, name_by_id):
    """Build an on_iteration callback that prints top-N tokens per topic.

    VanillaLDA's global_params has key "lambda" with shape (K, V) — each
    row is the unnormalized Dirichlet variational parameter for one topic
    over the full vocabulary. Row-normalizing gives the topic-word
    distribution; argsort + slice gives the top tokens. Compact one-line
    summary per topic so a K=10, every-N=10, 100-iter run produces 100
    lines of evolution output total.
    """
    def _on_iter(iter_num: int, global_params: dict,
                 _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        lam = global_params["lambda"]                         # (K, V)
        topics = lam / lam.sum(axis=1, keepdims=True)         # row-stochastic
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in range(K):
            top = topics[k].argsort()[::-1][:top_n]
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}"
                f"({topics[k, int(j)]:.3f})"
                for j in top
            )
            print(f"[driver]    topic {k}: {terms}", flush=True)
    return _on_iter


def _log_persist(df, name: str) -> None:
    """Confirm a DataFrame's persist hint stuck in cache.

    storageLevel reflects the requested-and-honored cache shape. A NONE
    here after .persist() + an action means caching silently failed —
    the prime suspect when re-execution shows up as per-iteration slowdown.
    """
    sl = df.storageLevel
    if sl.useMemory or sl.useDisk:
        bits = []
        if sl.useMemory:
            bits.append("MEM")
        if sl.useDisk:
            bits.append("DISK")
        if sl.deserialized:
            bits.append("DESER")
        print(f"[driver]   persist({name}) -> {'+'.join(bits)} x{sl.replication}",
              flush=True)
    else:
        print(f"[driver]   *** persist({name}) NOT IN CACHE: {sl}",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--person-mod", type=int, default=1000,
                         help="MOD(person_id, M) == 0 sampling factor")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--top-n-tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsampling-rate", type=float, default=0.05,
                         help="mini-batch fraction; use 1.0 for full-batch "
                              "(recommended on small corpora where Spark "
                              "coordination dominates per-iter time)")
    parser.add_argument("--print-topics-every", type=int, default=10,
                         help="emit top-3 tokens per topic every N iterations "
                              "(0 disables; cheap, runs on driver)")
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
        _log_persist(omop, "omop")
        print(f"[driver]   OMOP: {summary['rows']} rows, "
              f"{summary['persons']} distinct persons", flush=True)

    with _phase("vectorize (CountVectorizer)"):
        bow_df, vocab_map = to_bow_dataframe(
            omop, vocab_size=args.vocab_size, min_df=args.min_df,
        )
        bow_df = bow_df.persist()
        n_docs = bow_df.count()  # forces materialization
        _log_persist(bow_df, "bow_df")
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
                 f"subsamplingRate={args.subsampling_rate})"):
        estimator = VanillaLDAEstimator(
            k=args.K, maxIter=args.max_iter, seed=args.seed,
            subsamplingRate=args.subsampling_rate,
        )
        if args.print_topics_every > 0:
            estimator.setOnIteration(_make_topic_evolution_logger(
                K=args.K, top_n=8, every_n=args.print_topics_every,
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
