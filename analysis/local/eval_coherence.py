"""Local driver: held-out NPMI coherence for a saved OnlineLDA or OnlineHDP checkpoint.

Loads a VIResult, rebuilds the BOW from the same OMOP parquet, applies the same
deterministic person-keyed split, and computes per-topic NPMI on the holdout
partition. Prints a ranked report.

The fit driver (analysis/local/fit_lda_local.py / fit_hdp_local.py) and this
eval driver must agree on (holdout_fraction, seed). v1 documents this as a
human contract; v2 may stamp it into VIResult.metadata for verification.

Usage:
    poetry run python analysis/local/eval_coherence.py \\
        --checkpoint data/runs/lda_<timestamp> \\
        --input data/simulated/omop_N1000_seed42.parquet \\
        --holdout-fraction 0.1 --seed 42 --top-n 20 \\
        --model-class lda
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from charmpheno.omop import load_omop_parquet, to_bow_dataframe
from charmpheno.omop.split import split_bow_by_person
from spark_vi.core.types import BOWDocument
from spark_vi.eval.topic import (
    CoherenceReport,
    compute_npmi_coherence,
    top_k_used_topics,
)
from spark_vi.io import load_result

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("eval_coherence")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def _verify_split_contract(result, *, holdout_fraction: float, seed: int) -> None:
    """Verify the eval CLI args match the split provenance stamped at fit time.

    The fit driver (fit_lda_local.py / fit_hdp_local.py) stamps split parameters
    under VIResult.metadata['split']. If absent, the model was fit on the full
    corpus and the eval is optimistically biased (the held-out patients were
    seen during training); we warn loudly. If present but mismatched, we abort.
    """
    split_meta = result.metadata.get("split")
    if split_meta is None or not split_meta.get("applied", False):
        log.warning(
            "checkpoint has no split provenance; the fit driver was likely run "
            "without --holdout-fraction. NPMI on the hashed holdout will be "
            "OPTIMISTICALLY BIASED because the model saw those patients during "
            "fitting. Re-fit with matching --holdout-fraction and --holdout-seed "
            "for an honest benchmark."
        )
        return
    fit_frac = float(split_meta.get("holdout_fraction", -1.0))
    fit_seed = int(split_meta.get("holdout_seed", -1))
    if (abs(fit_frac - holdout_fraction) > 1e-9) or (fit_seed != seed):
        raise SystemExit(
            "split mismatch: checkpoint was fit with "
            f"holdout_fraction={fit_frac}, seed={fit_seed} but eval was invoked "
            f"with holdout_fraction={holdout_fraction}, seed={seed}. "
            "Re-run with matching values (the eval holdout must be the held-out "
            "portion the model did NOT see)."
        )


def run_eval(
    *,
    checkpoint: Path,
    input_parquet: Path,
    holdout_fraction: float,
    seed: int,
    top_n: int,
    model_class: str,
    hdp_k: int,
    spark: SparkSession,
) -> CoherenceReport:
    """Run the eval and return the report. Importable for tests."""
    result = load_result(checkpoint)
    _verify_split_contract(result, holdout_fraction=holdout_fraction, seed=seed)
    lambda_ = result.global_params["lambda"]
    topic_term = lambda_ / lambda_.sum(axis=1, keepdims=True)

    if model_class == "hdp":
        u = result.global_params["u"]
        v = result.global_params["v"]
        mask = top_k_used_topics(u=u, v=v, k=hdp_k)
    else:
        mask = None

    df = load_omop_parquet(str(input_parquet), spark=spark)
    bow_df, _vocab_map = to_bow_dataframe(df)
    _train, holdout_df = split_bow_by_person(
        bow_df, holdout_fraction=holdout_fraction, seed=seed
    )
    holdout_df = holdout_df.persist()
    n_holdout = holdout_df.count()
    log.info("holdout: %d docs", n_holdout)

    holdout_rdd = holdout_df.rdd.map(BOWDocument.from_spark_row)

    report = compute_npmi_coherence(
        topic_term, holdout_rdd, top_n=top_n, hdp_topic_mask=mask
    )

    holdout_df.unpersist()
    return report


def _print_ranked_report(
    report: CoherenceReport,
    vocab: list[int],
    concept_names: dict[int, str] | None = None,
) -> None:
    rows = sorted(
        zip(report.topic_indices, report.per_topic_npmi, report.top_term_indices),
        key=lambda r: -r[1],
    )
    print(f"\n  per-topic NPMI (n_holdout_docs={report.n_holdout_docs}, top_n={report.top_n}):")
    print(f"  mean={report.mean:+.4f}  median={report.median:+.4f}  stdev={report.stdev:.4f}  "
          f"min={report.min:+.4f}  max={report.max:+.4f}\n")
    for topic_idx, npmi, term_idx in rows:
        terms = [vocab[i] if i < len(vocab) else f"#{i}" for i in term_idx]
        if concept_names:
            terms = [f"{t} ({concept_names.get(t, '?')})" for t in terms]
        print(f"  topic {int(topic_idx):3d}  NPMI={npmi:+.4f}  top: {', '.join(map(str, terms[:8]))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--model-class", choices=["lda", "hdp"], required=True)
    parser.add_argument("--hdp-k", type=int, default=50,
                        help="Top-K HDP topics by E[beta] to score (ignored for LDA)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        report = run_eval(
            checkpoint=args.checkpoint,
            input_parquet=args.input,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            top_n=args.top_n,
            model_class=args.model_class,
            hdp_k=args.hdp_k,
            spark=spark,
        )

        # Property checks (the spec calls for these to be assertion-style in the driver).
        assert (report.per_topic_npmi >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi <= 1.0).all(), "NPMI > 1 found"

        result = load_result(args.checkpoint)
        vocab = result.metadata.get("vocab", [])
        _print_ranked_report(report, vocab)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
