"""End-to-end LDA on BigQuery-resident OMOP condition data.

Reads condition_occurrence from the workspace CDR (full-patient sampled),
joins to concept for human-readable names, vectorizes via CountVectorizer,
fits OnlineLDAEstimator (the MLlib-shaped shim around our SVI), and
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

    OnlineLDA's global_params has key "lambda" with shape (K, V) — each
    row is the unnormalized Dirichlet variational parameter for one topic
    over the full vocabulary. Row-normalizing gives the topic-word
    distribution; argsort + slice gives the top tokens. Each topic line
    is prefixed with per-topic stats:
        α_k    — current empirical-Bayes prior on θ_·k (length-K vector
                 in global_params["alpha"]). Diverges from 1/K under
                 optimize_alpha; corpus-popular topics get larger α_k.
        Σλ_k   — total topic mass over the vocabulary. Proxy for how much
                 corpus evidence has accreted to topic k.
        E[β_k] — Σλ_k / Σ_j Σλ_j. Fraction of corpus mass assigned to
                 topic k (the LDA analogue of HDP's corpus-level stick
                 weight; distinct from α_k, which is the prior on per-doc
                 topic distributions, not the empirical topic prevalence).
                 ~1/K means evenly used; values >> 1/K mark catch-alls,
                 << 1/K mark sparsely-used / vestigial slots.
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
        E_beta = lam_row_sums / max(lam_row_sums.sum(), 1e-12)  # (K,)
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
                f"α={alpha[k]:.4g}  E[β]={E_beta[k]:.4f}  "
                f"Σλ={lam_row_sums[k]:.3g}  peak={peak[k]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    """Show defaults automatically, and preserve the docstring's paragraph layout."""


def _load_or_build_corpus(spark, args, doc_spec, cdr, billing):
    """Return (bow_df, vocab_map, name_by_id) with optional cache short-circuit.

    On a cache hit, skips BQ load + CountVectorizer fit entirely. On a miss
    (or when --corpus-cache-uri is unset), runs the original BQ+vectorize
    pipeline, resolves concept names for the vocab, and writes through to
    the cache for the next run.

    Concept-name resolution lives here (not in main) so the cached entry
    is self-sufficient — eval drivers and downstream printing don't need
    a live omop DataFrame on the hit path.
    """
    from charmpheno.omop import load_omop_bigquery, to_bow_dataframe
    from spark_vi.diagnostics.persist import assert_persisted
    from _corpus_cache import compute_cache_key, save, try_load

    cache_uri = args.corpus_cache_uri
    cache_key = None
    if cache_uri:
        cache_key = compute_cache_key(
            source_table=args.source_table,
            person_mod=args.person_mod,
            vocab_size=args.vocab_size,
            min_df=args.min_df,
            doc_spec_manifest=doc_spec.manifest(),
        )
        with _phase(f"corpus-cache lookup ({cache_uri}/{cache_key})"):
            cached = try_load(spark, cache_uri, cache_key)
        if cached is not None:
            print(f"[driver]   corpus-cache HIT", flush=True)
            return cached
        print(f"[driver]   corpus-cache MISS, building...", flush=True)

    with _phase("BQ load + summary"):
        omop = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            person_sample_mod=args.person_mod, source_table=args.source_table,
        ).persist()
        summary = omop.agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("person_id").alias("persons"),
        ).collect()[0]
        assert_persisted(omop, name="omop")
        print(f"[driver]   OMOP: {summary['rows']} rows, "
              f"{summary['persons']} distinct persons", flush=True)

    with _phase(f"vectorize (CountVectorizer, doc_spec={doc_spec.name}, "
                f"min_doc_length={doc_spec.min_doc_length})"):
        bow_df, vocab_map = to_bow_dataframe(
            omop, doc_spec=doc_spec,
            vocab_size=args.vocab_size, min_df=args.min_df,
        )

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
        print(f"[driver]   resolved {len(name_by_id)} concept names", flush=True)

    omop.unpersist()

    if cache_uri:
        with _phase(f"corpus-cache write ({cache_uri}/{cache_key})"):
            save(spark, bow_df, vocab_map, name_by_id, cache_uri, cache_key)

    return bow_df, vocab_map, name_by_id


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
        "--save-dir", default="",
        help=("directory for auto-saves and final result; empty (default) = "
              "no save (the directory becomes the authoritative post-fit "
              "artifact, loadable via OnlineLDAModel.load)"),
    )
    parser.add_argument(
        "--save-interval", type=int, default=-1,
        help=("save every N iters during fit; -1 (default) = save only at "
              "end-of-fit if --save-dir is set"),
    )
    parser.add_argument(
        "--resume-from", default="",
        help=("path to a previously-written save dir; empty (default) = "
              "fresh start (when set, fit loads the saved VIResult and "
              "continues from that iteration count)"),
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
        "--tau0", type=float, default=1024.0,
        help=("Robbins-Monro learning offset τ₀ in ρ_t = (τ₀ + t + 1)^(-κ). "
              "Larger τ₀ ⇒ smaller initial step (slower start). Default "
              "1024.0 mirrors MLlib's onlineLDAOptimizer; on smaller "
              "corpora try ~10-64 to make α/η optimization actually move "
              "(default ρ_0 ≈ 0.029 is glacial)."),
    )
    parser.add_argument(
        "--kappa", type=float, default=0.51,
        help=("Robbins-Monro learning decay κ in ρ_t = (τ₀ + t + 1)^(-κ). "
              "Must be in (0.5, 1.0] for SVI convergence guarantees "
              "(Hoffman 2013 §2.3). Larger κ ⇒ faster decay. Default 0.51 "
              "matches MLlib."),
    )
    parser.add_argument(
        "--optimize-doc-concentration",
        action=argparse.BooleanOptionalAction, default=True,
        help=("learn an asymmetric α (length K) via Newton-Raphson empirical "
              "Bayes during fit (Blei 2003 App. A.4.2). After training the fitted "
              "α is on model.trainedAlpha() — a global prior over per-doc topic "
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
    parser.add_argument(
        "--doc-unit", choices=["patient", "patient_year"], default="patient",
        help=("How OMOP event rows become documents (see ADR 0018). "
              "'patient' = one doc per person over full history (legacy "
              "default). 'patient_year' = one doc per (person, year-active), "
              "requires --source-table condition_era for era replication."),
    )
    parser.add_argument(
        "--doc-min-length", type=int, default=None,
        help=("Minimum token count per doc before it enters the BOW; shorter "
              "docs are dropped. None uses the DocSpec's own default "
              "(0 for patient, 30 for patient_year)."),
    )
    parser.add_argument(
        "--source-table", choices=["condition_occurrence", "condition_era"],
        default="condition_occurrence",
        help=("Which OMOP fact table to read. condition_era is required for "
              "PatientYearDocSpec's era-replication semantics."),
    )
    parser.add_argument(
        "--corpus-cache-uri", type=str, default=None,
        help=("Optional cache root for the (BQ load + BOW build) prep step. "
              "When set, the prep result is keyed by a hash of "
              "(source_table, person_mod, vocab_size, min_df, doc_spec) and "
              "loaded from {uri}/{key}/ on a hit, built+saved on a miss. "
              "Typical values: hdfs:///user/dataproc/charm/corpus_cache "
              "(session-only, fast); gs://<bucket>/charm/corpus_cache "
              "(persistent). Omit to disable caching."),
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
    from charmpheno.omop import (
        doc_spec_from_cli,
        load_omop_bigquery,
        to_bow_dataframe,
    )
    from spark_vi.core import VIResult
    from spark_vi.diagnostics.persist import assert_persisted
    from spark_vi.io import save_result
    from spark_vi.mllib.topic.lda import OnlineLDAEstimator

    doc_spec = doc_spec_from_cli(args.doc_unit, min_doc_length=args.doc_min_length)
    # Sanity check: patient_year semantics need condition_era for era spans.
    if args.doc_unit == "patient_year" and args.source_table != "condition_era":
        print("ERROR: --doc-unit patient_year requires --source-table "
              "condition_era for era replication semantics.", file=sys.stderr)
        return 1

    _configure_logging()

    print(f"[driver] cdr={cdr}, billing_project={billing}, "
          f"K={args.K}, max_iter={args.max_iter}, "
          f"person_mod={args.person_mod}", flush=True)

    spark = SparkSession.builder.appName("lda_bigquery_cloud").getOrCreate()
    # Silence the GCS connector chatter (RequestTracker / hflush rate-limit
    # noise from event-log writes). Our [driver] prints are stdout, not
    # log4j, so they survive. Set BEFORE any actions.
    spark.sparkContext.setLogLevel("WARN")
    # Additionally silence the spot-reclamation flood (BlockManager cascades,
    # FetchFailed stack traces from TaskSetManager, etc.) without losing
    # other WARN messages.
    from _log_utils import quiet_spot_reclamation
    quiet_spot_reclamation(spark)
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    bow_df, vocab_map, name_by_id = _load_or_build_corpus(
        spark, args, doc_spec, cdr, billing,
    )
    bow_df = bow_df.persist()
    n_docs = bow_df.count()  # forces materialization
    assert_persisted(bow_df, name="bow_df")
    idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
    print(f"[driver]   vocab size: {len(vocab_map)} (cap {args.vocab_size}, "
          f"minDF {args.min_df}), documents: {n_docs}", flush=True)

    fit_df = bow_df

    with _phase(f"fit (K={args.K}, maxIter={args.max_iter}, "
                 f"subsamplingRate={args.subsampling_rate}, "
                 f"τ₀={args.tau0}, κ={args.kappa}, "
                 f"optimizeDocConc={args.optimize_doc_concentration}, "
                 f"optimizeTopicConc={args.optimize_topic_concentration})"):
        estimator = OnlineLDAEstimator(
            k=args.K, maxIter=args.max_iter, seed=args.seed,
            subsamplingRate=args.subsampling_rate,
            learningOffset=args.tau0,
            learningDecay=args.kappa,
            optimizeDocConcentration=args.optimize_doc_concentration,
            optimizeTopicConcentration=args.optimize_topic_concentration,
            saveDir=args.save_dir,
            saveInterval=args.save_interval,
            resumeFrom=args.resume_from,
        )
        if args.print_topics_every > 0:
            estimator.setOnIteration(_make_topic_evolution_logger(
                top_n=6, every_n=args.print_topics_every,
                idx_to_cid=idx_to_cid, name_by_id=name_by_id,
            ))
        model = estimator.fit(fit_df)
        print(f"[driver]   elbo trace tail: {model.result.elbo_trace[-3:]}",
              flush=True)

    # Augmented re-save: bundle vocab + corpus_manifest into
    # VIResult.metadata and overwrite the shim's final auto-save. The shim's
    # mid-fit auto-saves are for resume continuity (ADR 0015); overwriting at
    # end-of-fit with this richer metadata gives eval_coherence_cloud.py
    # everything it needs to freeze the BOW vocab and reproduce the OMOP
    # fetch from the checkpoint alone.
    if args.save_dir:
        vocab_list: list = [None] * len(vocab_map)
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
                "K": args.K,
                "corpus_manifest": {
                    "source": "bigquery",
                    "source_table": args.source_table,
                    "cdr": cdr,
                    "person_mod": args.person_mod,
                    "vocab_size": args.vocab_size,
                    "min_df": args.min_df,
                    "doc_spec": doc_spec.manifest(),
                },
            },
        )
        save_result(augmented, args.save_dir)
        print(f"[driver] re-saved augmented VIResult to {args.save_dir}",
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
        # Transform against the un-split bow_df so the sample shows topic
        # distributions for some patients regardless of holdout assignment.
        (model.transform(bow_df)
              .withColumn("person_hash",
                          F.substring(
                              F.sha2(F.col("person_id").cast("string"), 256),
                              1, 12))
              .select("person_hash", "topicDistribution")
              .show(3, truncate=False))

    bow_df.unpersist()
    print("[driver] LDA BQ SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
