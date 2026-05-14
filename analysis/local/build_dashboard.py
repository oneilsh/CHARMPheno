"""Build the dashboard data bundle from a saved VIResult (LDA or HDP).

Outputs four JSON files into the target directory:
  model.json, phenotypes.json, vocab.json, corpus_stats.json

Model-class normalization happens in charmpheno.export.model_adapter.
Synthetic cohorts and topic-map MDS are computed client-side.

Usage:
    poetry run python analysis/local/build_dashboard.py \\
        --checkpoint data/runs/<run> \\
        --input data/simulated/omop_N10000_seed42.parquet \\
        --out-dir dashboard/public/data \\
        --vocab-top-n 5000 \\
        --hdp-top-k 50
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

_REPO_ROOT = Path(__file__).resolve().parents[2]
# charmpheno package lives in charmpheno/charmpheno/ (one level below repo root)
_CHARMPHENO_PKG = _REPO_ROOT / "charmpheno"
for _p in [str(_CHARMPHENO_PKG), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# spark_vi.core.result must be loaded before spark_vi.io to avoid a circular
# import: io.export → core.result → core.__init__ → runner → io.export.
import spark_vi.core.result as _  # noqa: F401,E402  (side-effect: seeds sys.modules)

from charmpheno.export.corpus_stats import (
    compute_corpus_stats_from_bow_df,
    write_corpus_stats_sidecar,
)
from charmpheno.export.dashboard import (
    write_model_and_vocab_bundles,
    write_phenotypes_bundle,
)
from charmpheno.export.model_adapter import adapt
from charmpheno.omop import DocSpec, load_omop_parquet, to_bow_dataframe
from spark_vi.io import load_result
from spark_vi.models.topic.types import BOWDocument
from spark_vi.eval.topic import compute_npmi_coherence
from pyspark.sql import functions as F

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("build_dashboard")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vocab-top-n", type=int, default=5000)
    parser.add_argument("--hdp-top-k", type=int, default=50,
                        help="Top-K used HDP topics (ignored for LDA)")
    parser.add_argument("--top-n-codes-for-npmi", type=int, default=20)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = load_result(args.checkpoint)
    model_class = result.metadata.get("model_class", "lda")
    log.info("model_class=%s", model_class)

    # Adapter normalizes LDA/HDP/etc. to a uniform DashboardExport
    export = adapt(result, hdp_top_k=args.hdp_top_k)
    K_disp, V_full = export.beta.shape
    log.info("K_display=%d V_full=%d (model_class=%s)", K_disp, V_full, model_class)

    vocab_ids = result.metadata.get("vocab")
    if not vocab_ids:
        raise SystemExit("checkpoint metadata has no 'vocab'; re-fit needed.")
    descriptions = result.metadata.get("concept_names", {}) or {}
    domains = result.metadata.get("concept_domains", {}) or {}

    # Stats from the input parquet
    corpus_manifest = result.metadata.get("corpus_manifest", {})
    doc_spec_manifest = corpus_manifest.get("doc_spec", {"name": "patient"})
    doc_spec = DocSpec.from_manifest(doc_spec_manifest)
    spark = _build_spark()
    df = load_omop_parquet(str(args.input), spark=spark)
    bow_df, _ = to_bow_dataframe(df, doc_spec=doc_spec, vocab=vocab_ids)
    # compute_corpus_stats_from_bow_df needs 'indices' and 'counts' array columns.
    # to_bow_dataframe returns a SparseVector 'features' column; extract them.
    _sv_indices = F.udf(lambda sv: sv.indices.tolist() if sv is not None else [],
                        "array<int>")
    _sv_counts = F.udf(lambda sv: [float(x) for x in sv.values] if sv is not None else [],
                       "array<double>")
    bow_df_stats = bow_df.select(
        _sv_indices(F.col("features")).alias("indices"),
        _sv_counts(F.col("features")).alias("counts"),
    )
    bow_df = bow_df.persist()
    bow_df_stats = bow_df_stats.persist()
    stats = compute_corpus_stats_from_bow_df(bow_df_stats, vocab_size=V_full, k=K_disp)
    log.info("corpus stats: n_docs=%d mean_codes=%.2f",
             stats.corpus_size_docs, stats.mean_codes_per_doc)

    # NPMI on the adapter's displayed-topic β (already filtered for HDP)
    # Cap top_n to the vocab size so small-vocab smoke fixtures don't error.
    top_n_npmi = min(args.top_n_codes_for_npmi, V_full)
    holdout_bow = bow_df.rdd.map(BOWDocument.from_spark_row)
    report = compute_npmi_coherence(export.beta, holdout_bow, top_n=top_n_npmi)
    npmi = report.per_topic_npmi.tolist()
    # Fraction of top-N pairs that contributed to each topic's mean NPMI.
    # Zero means "unrated" — no pairs cleared min_pair_count.
    pair_coverage = (
        report.per_topic_scored_pairs.astype(float)
        / float(report.per_topic_total_pairs)
    ).tolist()
    bow_df.unpersist()
    bow_df_stats.unpersist()
    spark.stop()

    # write_model_and_vocab_bundles expects (lambda_, alpha) pre-trim.
    # The adapter's β is already row-stochastic; reconstruct a faux-lambda
    # that produces the same beta after the normalize step inside the writer.
    # (Simplest: multiply by a big scalar so renormalize is identity.)
    pseudo_lambda = export.beta * 1.0e6  # any positive scalar; row-norm is identity
    v_disp = write_model_and_vocab_bundles(
        out_dir=args.out_dir,
        lambda_=pseudo_lambda, alpha=export.alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=stats.code_marginals, top_n=args.vocab_top_n,
    )
    write_phenotypes_bundle(
        args.out_dir / "phenotypes.json",
        npmi=npmi,
        pair_coverage=pair_coverage,
        corpus_prevalence=export.corpus_prevalence.tolist(),
        topic_indices=export.topic_indices.tolist(),
        labels=None,
    )
    write_corpus_stats_sidecar(stats, args.out_dir / "corpus_stats.json", v_displayed=v_disp)

    log.info("wrote 4 files to %s (V_disp=%d K_disp=%d)", args.out_dir, v_disp, K_disp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
