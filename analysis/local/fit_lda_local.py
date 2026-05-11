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

from charmpheno.omop import load_omop_parquet, to_bow_dataframe
from charmpheno.omop.split import split_bow_by_person
from spark_vi.core import VIResult
from spark_vi.io import save_result
from spark_vi.mllib.lda import OnlineLDAEstimator

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
        "--holdout-fraction", type=float, default=0.0,
        help="If >0, deterministically hold out this fraction of patients before "
             "fitting (eval-driver companion). 0 means fit on full corpus.",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=None,
        help="Seed for the holdout hash. Defaults to --seed.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    holdout_seed = args.holdout_seed if args.holdout_seed is not None else args.seed
    holdout_fraction = float(args.holdout_fraction)
    apply_split = holdout_fraction > 0.0
    if not (0.0 <= holdout_fraction < 1.0):
        raise SystemExit(
            f"--holdout-fraction must be in [0, 1), got {holdout_fraction}"
        )

    spark = _build_spark()
    try:
        df = load_omop_parquet(str(args.input), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(df)

        if apply_split:
            train_df, _holdout_df = split_bow_by_person(
                bow_df,
                holdout_fraction=holdout_fraction,
                seed=holdout_seed,
            )
            train_df = train_df.persist()
            fit_df = train_df
            n_docs = fit_df.count()
            log.info("train split: %d docs (holdout_fraction=%.3f, seed=%d)",
                     n_docs, holdout_fraction, holdout_seed)
        else:
            bow_df = bow_df.persist()
            fit_df = bow_df
            n_docs = fit_df.count()
            log.info("no holdout split applied")
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
        split_metadata: dict = {"applied": apply_split}
        if apply_split:
            split_metadata.update(
                holdout_fraction=holdout_fraction,
                holdout_seed=holdout_seed,
                person_id_col="person_id",
                splitter="charmpheno.omop.split.split_bow_by_person",
            )
        result_with_vocab = VIResult(
            global_params=model.result.global_params,
            elbo_trace=model.result.elbo_trace,
            n_iterations=model.result.n_iterations,
            converged=model.result.converged,
            metadata={
                **model.result.metadata,
                "vocab": vocab_list,
                "K": args.K,
                "split": split_metadata,
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
