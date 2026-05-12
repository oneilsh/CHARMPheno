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
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

# Make the repo root importable so `analysis._eval_common` resolves when this
# script is invoked as `poetry run python analysis/local/eval_coherence.py`
# (which puts analysis/local/ on sys.path, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis._eval_common import verify_split_contract  # noqa: E402
from charmpheno.omop import DocSpec, load_omop_parquet, to_bow_dataframe
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
    verify_split_contract(result, holdout_fraction=holdout_fraction, seed=seed)
    lambda_ = result.global_params["lambda"]
    topic_term = lambda_ / lambda_.sum(axis=1, keepdims=True)

    if model_class == "hdp":
        u = result.global_params["u"]
        v = result.global_params["v"]
        mask = top_k_used_topics(u=u, v=v, k=hdp_k)
    else:
        mask = None

    # Reconstruct the fit-time DocSpec from corpus_manifest so eval BOWs
    # match the fit BOWs exactly. Pre-ADR-0018 checkpoints lack the field;
    # default to PatientDocSpec, which matches their actual behavior.
    corpus = result.metadata.get("corpus_manifest", {})
    doc_spec_manifest = corpus.get("doc_spec", {"name": "patient"})
    doc_spec = DocSpec.from_manifest(doc_spec_manifest)
    log.info("doc_spec from checkpoint manifest: %s", doc_spec_manifest)

    df = load_omop_parquet(str(input_parquet), spark=spark)
    bow_df, _vocab_map = to_bow_dataframe(df, doc_spec=doc_spec)
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
