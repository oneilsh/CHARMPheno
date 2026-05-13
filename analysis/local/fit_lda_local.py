"""End-to-end local: simulator parquet -> OnlineLDA via shim -> saved VIResult.

Sibling of fit_hdp_local.py. Builds a SparkSession in local mode, loads
the OMOP parquet, builds the bag-of-words DataFrame, fits OnlineLDA
through the MLlib shim (`OnlineLDAEstimator`), and saves the result +
vocab sidecar.

Going through the shim (rather than VIRunner directly) is deliberate:
this is the local proxy for what analysis/cloud/lda_bigquery_cloud.py
runs on Dataproc. Catching shim issues on a 10-doc parquet is cheaper
than catching them on a billed cloud submit.

The vocab map is recorded under VIResult.metadata["vocab"] as a list[int]
ordered by index, so a downstream load_result + lambda inspection can
re-attach concept_ids without baking data-shape knowledge into spark_vi.

Usage:
    poetry run python analysis/local/fit_lda_local.py \\
        --input data/simulated/omop_N1000_seed42.parquet \\
        --K 10 \\
        --max-iterations 200 \\
        --mini-batch-fraction 0.1 \\
        --seed 42 \\
        --output data/runs/lda_<timestamp>
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession

from charmpheno.omop import (
    doc_spec_from_cli,
    load_omop_parquet,
    to_bow_dataframe,
)
from spark_vi.core import VIResult
from spark_vi.io import save_result
from spark_vi.mllib.topic.lda import OnlineLDAEstimator

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_lda_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to OMOP-shaped parquet")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for the saved VIResult")
    parser.add_argument("--K", type=int, required=True, help="Number of topics")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--mini-batch-fraction", type=float, default=0.1)
    parser.add_argument("--tau0", type=float, default=1024.0)
    parser.add_argument("--kappa", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimize-doc-concentration",
        action=argparse.BooleanOptionalAction, default=True,
        help="empirical-Bayes Newton-Raphson on asymmetric α (Blei 2003 App. A.4.2)",
    )
    parser.add_argument(
        "--optimize-topic-concentration",
        action=argparse.BooleanOptionalAction, default=False,
        help="empirical-Bayes Newton-Raphson on symmetric scalar η (Hoffman 2010 §3.4)",
    )
    parser.add_argument(
        "--doc-unit", choices=["patient", "patient_year"], default="patient",
        help=("How OMOP event rows become documents (ADR 0018). "
              "patient_year requires the parquet to carry "
              "condition_era_start_date / condition_era_end_date columns."),
    )
    parser.add_argument(
        "--doc-min-length", type=int, default=None,
        help="Minimum tokens per doc before it enters the BOW.",
    )
    args = parser.parse_args(argv)

    doc_spec = doc_spec_from_cli(args.doc_unit, min_doc_length=args.doc_min_length)
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        df = load_omop_parquet(str(args.input), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(df, doc_spec=doc_spec)
        bow_df = bow_df.persist()
        fit_df = bow_df
        n_docs = fit_df.count()
        log.info("vocab=%d docs=%d", len(vocab_map), n_docs)

        estimator = OnlineLDAEstimator(
            k=args.K,
            maxIter=args.max_iterations,
            seed=args.seed,
            subsamplingRate=args.mini_batch_fraction,
            learningOffset=args.tau0,
            learningDecay=args.kappa,
            optimizeDocConcentration=args.optimize_doc_concentration,
            optimizeTopicConcentration=args.optimize_topic_concentration,
        )
        model = estimator.fit(fit_df)

        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid
        result_with_vocab = VIResult(
            global_params=model.result.global_params,
            elbo_trace=model.result.elbo_trace,
            n_iterations=model.result.n_iterations,
            converged=model.result.converged,
            metadata={
                **model.result.metadata,
                "vocab": vocab_list,
                "K": args.K,
                "corpus_manifest": {
                    "source": "parquet",
                    "input_path": str(args.input),
                    "doc_spec": doc_spec.manifest(),
                },
            },
        )
        save_result(result_with_vocab, args.output)
        log.info(
            "wrote %s (K=%d, V=%d, n_iterations=%d, converged=%s)",
            args.output, args.K, len(vocab_map),
            model.result.n_iterations, model.result.converged,
        )
        fit_df.unpersist()
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
