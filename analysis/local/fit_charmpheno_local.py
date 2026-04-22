"""End-to-end local smoke: simulator parquet → VIRunner → saved artifact.

Bootstrap scope: uses CountingModel (coin-flip posterior) to exercise the
plumbing because the real OnlineHDP is still stubbed. When the real HDP
lands, this script gets an option to select between CountingModel and
CharmPhenoHDP — or is split into two entrypoints.

Expected input schema (OMOP-shaped parquet):
    person_id, visit_occurrence_id, concept_id, concept_name[, true_topic_id]

For the smoke: rows are treated as 0/1 based on concept_id parity. Real
topic modeling lives in the follow-on HDP spec.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from spark_vi.core import VIConfig, VIRunner
from spark_vi.io import save_result
from spark_vi.models import CountingModel

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_charmpheno_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to OMOP-shaped parquet (e.g. from scripts/simulate_lda_omop.py)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for the saved VIResult")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--spark", type=str, default=None,
                        help="Optional externally-provided SparkSession (unused from CLI)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        from charmpheno.omop import load_omop_parquet, validate
        df = load_omop_parquet(str(args.input), spark=spark)
        validate(df)

        # Coerce to 0/1 for the smoke's CountingModel: even concept_ids → 1,
        # odd → 0. This is nonsensical for clinical data but exercises the
        # end-to-end pipeline with real Spark distribution.
        rdd = df.select("concept_id").rdd.map(lambda row: 1 if row["concept_id"] % 2 == 0 else 0)

        runner = VIRunner(
            CountingModel(prior_alpha=1.0, prior_beta=1.0),
            config=VIConfig(max_iterations=args.max_iterations, convergence_tol=1e-12),
        )
        result = runner.fit(rdd)
        save_result(result, args.output)
        log.info("Wrote %s (n_iterations=%d, converged=%s)",
                 args.output, result.n_iterations, result.converged)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
