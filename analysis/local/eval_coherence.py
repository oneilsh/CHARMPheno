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

from analysis._eval_common import (  # noqa: E402
    print_ranked_report,
    verify_split_contract,
)
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
    npmi_reference: str,
    npmi_min_pair_count: int,
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
    if npmi_reference == "full":
        # NPMI doesn't have a predictive-overfitting concern (topic-word
        # distributions are fixed post-fit). Using train ∪ holdout as the
        # co-occurrence reference gives 5× more statistical evidence per
        # pair, dramatically reducing rare-pair sparsity. See ADR 0017
        # revisions (2026-05-12).
        reference_df = bow_df.persist()
    else:
        _train, reference_df = split_bow_by_person(
            bow_df, holdout_fraction=holdout_fraction, seed=seed
        )
        reference_df = reference_df.persist()
    n_ref = reference_df.count()
    log.info("npmi reference (%s): %d docs", npmi_reference, n_ref)

    reference_rdd = reference_df.rdd.map(BOWDocument.from_spark_row)

    report = compute_npmi_coherence(
        topic_term, reference_rdd, top_n=top_n, hdp_topic_mask=mask,
        min_pair_count=npmi_min_pair_count,
    )

    reference_df.unpersist()
    return report


def _build_name_by_idx(
    vocab: list[int],
    concept_names: dict[int, str] | None = None,
) -> dict[int, str]:
    """Adapter from (vocab list, optional concept_names) -> idx -> label.

    The local checkpoint exposes vocab as a positional list of concept_ids
    (vocab[idx] = cid); the shared printer wants a dict keyed by vocab idx.
    When `concept_names` is provided we append the name in parentheses,
    matching the pre-refactor format.
    """
    out: dict[int, str] = {}
    for idx, cid in enumerate(vocab):
        if concept_names and cid in concept_names:
            out[idx] = f"{cid} ({concept_names[cid]})"
        else:
            out[idx] = str(cid)
    return out


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
    parser.add_argument(
        "--npmi-reference", choices=["holdout", "full"], default="full",
        help=("Which docs serve as the co-occurrence reference for NPMI. "
              "`full` (default) uses train ∪ holdout — 5× the data, "
              "fewer rare-pair zeros; methodologically sound because "
              "topic-word distributions are fixed post-fit (no "
              "overfitting concern). `holdout` reproduces pre-2026-05-12 "
              "behavior."),
    )
    parser.add_argument(
        "--npmi-min-pair-count", type=int, default=3,
        help=("Skip pairs with joint count below this threshold in the "
              "reference corpus. Default 3 — pairs in {0, 1, 2} contribute "
              "nothing to their topic mean (vs. the pre-2026-05-12 -1 "
              "floor that biased rare-phenotype topics toward maximally "
              "incoherent scores). Set to 1 to reproduce the historical "
              "behavior (only skip true zeros)."),
    )
    parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto",
        help=("ANSI dimming of unused (NaN-NPMI) topics in the per-topic "
              "printout. 'auto' enables when stdout is a TTY."),
    )
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
            npmi_reference=args.npmi_reference,
            npmi_min_pair_count=args.npmi_min_pair_count,
            spark=spark,
        )

        # Property checks (the spec calls for these to be assertion-style
        # in the driver). NaN entries (unrated topics) bypass the bound
        # check intentionally — they aren't NPMI values, they're sentinels.
        rated = ~np.isnan(report.per_topic_npmi)
        assert (report.per_topic_npmi[rated] >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi[rated] <= 1.0).all(), "NPMI > 1 found"

        result = load_result(args.checkpoint)
        vocab = result.metadata.get("vocab", [])
        concept_names = result.metadata.get("concept_names")
        lambda_ = result.global_params["lambda"]
        alpha = (
            result.global_params.get("alpha")
            if args.model_class == "lda" else None
        )
        print_ranked_report(
            report,
            _build_name_by_idx(vocab, concept_names),
            lambda_,
            alpha=alpha,
            npmi_reference=args.npmi_reference,
            color=args.color,
        )
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
