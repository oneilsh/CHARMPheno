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
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


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

    print(f"[driver] cdr={cdr}, billing_project={billing}, "
          f"K={args.K}, max_iter={args.max_iter}, "
          f"person_mod={args.person_mod}", flush=True)

    spark = SparkSession.builder.appName("lda_bigquery_cloud").getOrCreate()
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    print("[driver] reading OMOP from BigQuery...", flush=True)
    omop = load_omop_bigquery(
        spark=spark,
        cdr_dataset=cdr,
        billing_project=billing,
        person_sample_mod=args.person_mod,
    ).persist()
    n_rows = omop.count()
    n_persons = omop.select("person_id").distinct().count()
    print(f"[driver] OMOP: {n_rows} rows, {n_persons} distinct persons", flush=True)

    print("[driver] vectorizing into bag-of-words documents...", flush=True)
    bow_df, vocab_map = to_bow_dataframe(
        omop, vocab_size=args.vocab_size, min_df=args.min_df,
    )
    bow_df = bow_df.persist()
    idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
    print(f"[driver] vocab size: {len(vocab_map)} (cap {args.vocab_size}, "
          f"minDF {args.min_df})", flush=True)
    print(f"[driver] documents: {bow_df.count()}", flush=True)

    # Vocabulary-only concept-name lookup; small enough for the driver.
    # dropDuplicates because OMOP occasionally has multiple name variants
    # for one concept_id.
    print("[driver] resolving concept names for vocabulary...", flush=True)
    name_rows = (
        omop.where(F.col("concept_id").isin(list(vocab_map.keys())))
            .select("concept_id", "concept_name")
            .dropDuplicates(["concept_id"])
            .collect()
    )
    name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}

    print(f"[driver] fitting VanillaLDAEstimator (K={args.K}, maxIter={args.max_iter})...",
          flush=True)
    t0 = time.perf_counter()
    model = VanillaLDAEstimator(
        k=args.K, maxIter=args.max_iter, seed=args.seed,
    ).fit(bow_df)
    t_fit = time.perf_counter() - t0
    print(f"[driver] fit wall time: {t_fit:.1f}s", flush=True)
    print(f"[driver] elbo trace tail: {model.result.elbo_trace[-3:]}", flush=True)

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

    print("\n[driver] transform sample (executor UDF):", flush=True)
    model.transform(bow_df).select("person_id", "topicDistribution").show(3, truncate=False)

    omop.unpersist()
    bow_df.unpersist()
    print("[driver] LDA BQ SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
