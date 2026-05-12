"""End-to-end local: simulator parquet -> OnlineHDP via shim -> saved VIResult.

Sibling of fit_lda_local.py — both go through their respective MLlib
shims (`OnlineLDAEstimator` / `OnlineHDPEstimator`) so the local driver
exercises the same code path as the cloud driver. Catching shim issues
on a 10-doc parquet is cheaper than catching them on a billed cloud submit.

Builds a SparkSession in local mode, loads the OMOP parquet, builds the
bag-of-words DataFrame, fits OnlineHDP through the shim, and saves the
result + vocab sidecar.

Usage:
    poetry run python analysis/local/fit_hdp_local.py \\
        --input data/simulated/omop_N1000_seed42.parquet \\
        --T 30 --K 6 \\
        --max-iterations 50 \\
        --mini-batch-fraction 0.5 \\
        --seed 42 \\
        --output data/runs/hdp_<timestamp>
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
from charmpheno.omop.split import split_bow_by_person
from spark_vi.core import VIResult
from spark_vi.io import save_result
from spark_vi.mllib.hdp import OnlineHDPEstimator

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_hdp_local")
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
    parser.add_argument("--T", type=int, default=50,
                        help="corpus truncation (upper bound on discovered topics)")
    parser.add_argument("--K", type=int, default=8,
                        help="doc-level truncation (max topics per doc)")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--mini-batch-fraction", type=float, default=0.5)
    parser.add_argument("--tau0", type=float, default=1024.0)
    parser.add_argument("--kappa", type=float, default=0.51)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="doc-stick concentration")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="corpus-stick concentration")
    parser.add_argument("--eta", type=float, default=0.01,
                        help="topic-word Dirichlet concentration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.0,
        help="If >0, deterministically hold out this fraction of patients before "
             "fitting (eval-driver companion). 0 means fit on full corpus.",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=None,
        help="Seed for the holdout hash. Defaults to --seed.",
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

    holdout_seed = args.holdout_seed if args.holdout_seed is not None else args.seed
    holdout_fraction = float(args.holdout_fraction)
    apply_split = holdout_fraction > 0.0
    if not (0.0 <= holdout_fraction < 1.0):
        raise SystemExit(
            f"--holdout-fraction must be in [0, 1), got {holdout_fraction}"
        )
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        df = load_omop_parquet(str(args.input), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(df, doc_spec=doc_spec)

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

        estimator = OnlineHDPEstimator(
            k=args.T, docTruncation=args.K,
            maxIter=args.max_iterations,
            seed=args.seed,
            subsamplingRate=args.mini_batch_fraction,
            learningOffset=args.tau0,
            learningDecay=args.kappa,
            docConcentration=[args.alpha],
            corpusConcentration=args.gamma,
            topicConcentration=args.eta,
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
                "T": args.T,
                "K": args.K,
                "split": split_metadata,
                "corpus_manifest": {
                    "source": "parquet",
                    "input_path": str(args.input),
                    "doc_spec": doc_spec.manifest(),
                },
            },
        )
        save_result(result_with_vocab, args.output)

        n_active = model.activeTopicCount()
        log.info(
            "wrote %s (T=%d, K=%d, V=%d, n_iterations=%d, converged=%s, "
            "active_topics=%d/%d)",
            args.output, args.T, args.K, len(vocab_map),
            model.result.n_iterations, model.result.converged,
            n_active, args.T,
        )
        fit_df.unpersist()
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
