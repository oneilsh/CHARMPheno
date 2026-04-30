"""End-to-end local: simulator parquet -> VanillaLDA -> saved VIResult.

Sibling of fit_charmpheno_local.py. Builds a SparkSession in local mode,
loads the OMOP parquet, builds the bag-of-words DataFrame, fits VanillaLDA
via VIRunner, and saves the result + vocab sidecar.

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
from spark_vi.core import BOWDocument, VIConfig, VIRunner
from spark_vi.io import save_result
from spark_vi.models.lda import VanillaLDA

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_lda_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
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
    parser.add_argument("--kappa", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        df = load_omop_parquet(str(args.input), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(df)
        rdd = bow_df.rdd.map(BOWDocument.from_spark_row)
        rdd.persist()

        cfg = VIConfig(
            max_iterations=args.max_iterations,
            learning_rate_tau0=args.tau0,
            learning_rate_kappa=args.kappa,
            mini_batch_fraction=args.mini_batch_fraction,
            random_seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            convergence_tol=1e-6,
        )
        model = VanillaLDA(K=args.K, vocab_size=len(vocab_map))
        result = VIRunner(model, config=cfg).fit(rdd)

        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid
        from spark_vi.core import VIResult
        result_with_vocab = VIResult(
            global_params=result.global_params,
            elbo_trace=result.elbo_trace,
            n_iterations=result.n_iterations,
            converged=result.converged,
            metadata={**result.metadata, "vocab": vocab_list},
        )
        save_result(result_with_vocab, args.output)
        log.info("Wrote %s (K=%d, V=%d, n_iterations=%d, converged=%s)",
                 args.output, args.K, len(vocab_map),
                 result.n_iterations, result.converged)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
