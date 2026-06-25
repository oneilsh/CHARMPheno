"""Build the dashboard data bundle from a saved VIResult (LDA, HDP, or STM).

Outputs four JSON files into the target directory:
  model.json, phenotypes.json, vocab.json, corpus_stats.json

For gated STM checkpoints (model_class=="stm" + topic_block_spec in
corpus_manifest) three additional files are written:
  gating.json, covariate_effects.json, covariate_schema.json

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
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

_REPO_ROOT = Path(__file__).resolve().parents[2]
# charmpheno package lives in charmpheno/charmpheno/ (one level below repo root)
_CHARMPHENO_PKG = _REPO_ROOT / "charmpheno"
_CLOUD_DIR = _REPO_ROOT / "analysis" / "cloud"
for _p in [str(_CHARMPHENO_PKG), str(_REPO_ROOT), str(_CLOUD_DIR)]:
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


def _corpus_manifest_covariate_names(result) -> list[str]:
    """Return the ordered covariate name list from the checkpoint metadata."""
    return result.metadata["covariate_manifest"]["covariate_names"]


def _write_local_covariate_schema(out_dir, result, cov_pdf, X, k):
    """Derive and write covariate_schema.json from the local pandas covariate matrix.

    Mirrors build_dashboard_cloud._write_covariate_schema but reads from the
    already-materialized pandas DataFrame (cov_pdf) and numpy design matrix (X)
    instead of a Spark DataFrame, so no Spark session is needed at build time.

    Dummy-column sums (per-level patient counts) are column sums of the binary
    indicator columns in X. Continuous percentiles (p5, p50, p95) are computed
    via numpy. Categorical levels and references are read from
    covariate_manifest["categorical_levels"] (persisted at fit time) when
    available, with a fallback to formulaic model_spec introspection for older
    or cloud checkpoints.
    """
    import re as _re
    from charmpheno.export.covariate_schema import build_covariate_schema

    cov_manifest = result.metadata["covariate_manifest"]
    covariate_names = cov_manifest["covariate_names"]
    continuous_cols = list(cov_manifest.get("continuous_cols", []))
    name_idx = {n: i for i, n in enumerate(covariate_names)}

    # Per-dummy-column sums (= approximate per-level patient counts).
    # Each C(var)[T.level] column is a 0/1 indicator; its sum is the
    # number of documents (patients) with that level.
    dummy_pat = _re.compile(r"^C\(.+\)\[T\..+\]$")
    dummy_names = [n for n in covariate_names if dummy_pat.match(n)]
    if dummy_names:
        level_counts = {n: int(X[:, name_idx[n]].sum()) for n in dummy_names}
    else:
        level_counts = {}

    # Coarse percentiles for continuous columns (p5, p50, p95), rounded.
    continuous_stats = {}
    for var in continuous_cols:
        if var not in name_idx:
            log.warning("STM: continuous covariate %r absent from design vector; "
                        "skipping its control.", var)
            continue
        idx = name_idx[var]
        q = np.percentile(X[:, idx], [5.0, 50.0, 95.0])
        continuous_stats[var] = tuple(round(float(v)) for v in q)

    # Categorical levels + reference: prefer the value persisted at fit time
    # under covariate_manifest["categorical_levels"] (populated by fit_stm_local
    # via _extract_categorical_levels). Fall back to the formulaic model_spec
    # introspection path (cloud checkpoints) or an empty dict (older checkpoints
    # or non-formulaic fit paths) so the schema write degrades gracefully.
    categorical_levels = cov_manifest.get("categorical_levels")
    if not categorical_levels:
        from build_dashboard_cloud import _categorical_levels_from_spec
        model_spec = getattr(result, "model_spec", None)
        categorical_levels = (
            _categorical_levels_from_spec(model_spec, covariate_names=covariate_names)
            if model_spec is not None
            else {}
        )

    schema = build_covariate_schema(
        covariate_names=covariate_names, continuous_cols=continuous_cols,
        categorical_levels=categorical_levels, level_counts=level_counts,
        continuous_stats=continuous_stats, k=k,
        n_total=int(X.shape[0]),
    )
    (out_dir / "covariate_schema.json").write_text(json.dumps(schema, indent=2))
    log.info("STM: wrote covariate_schema.json (controls=%d, unsupported=%d)",
             len(schema["controls"]), len(schema["unsupported"]))


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

    # --- STM gating: masked prevalence + covariate + gating.json (offline) ---
    corpus = result.metadata.get("corpus_manifest", {})
    tbs = corpus.get("topic_block_spec") if model_class == "stm" else None
    if tbs:
        import pandas as pd
        from spark_vi.models.topic.partition import TopicBlockPartition
        from spark_vi.models.topic.stm import corpus_mean_topic_proportions_gated
        from charmpheno.export.gating import suppressed_topic_ids, build_gating_json
        from charmpheno.export.dashboard import write_covariate_effects
        from charmpheno.export.model_adapter import adapt_stm as adapt_stm_export

        partition = TopicBlockPartition.from_dict(tbs)
        cov_path = Path(args.checkpoint) / "covariates.parquet"
        cov = pd.read_parquet(cov_path)
        X = np.vstack(cov["covariates"].to_numpy())          # (D, P)
        groups_per_doc = [frozenset({g}) for g in cov["source_cohort"]]
        Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)

        # per-group patient counts (distinct person_id) for k-anon
        gc = cov.groupby("source_cohort")["person_id"].nunique().to_dict()
        k = int(corpus.get("min_patient_count", 20))
        suppressed = suppressed_topic_ids(partition, gc, k)

        masked_prev = corpus_mean_topic_proportions_gated(
            Gamma, X, groups_per_doc, partition)

        export = adapt_stm_export(result, corpus_prevalence=masked_prev,
                                  partition=partition, suppressed=suppressed)
        kept_ids = [int(i) for i in export.topic_indices]
        gating = build_gating_json(partition, gc, k, kept_ids)
        (args.out_dir / "gating.json").write_text(json.dumps(gating, indent=2))
        log.info("STM: wrote gating.json (groups=%s, kept_topics=%d)",
                 gating["groups"], len(kept_ids))

        # covariate_effects.json for the KEPT topics (Gamma columns subset)
        P = Gamma.shape[0]
        write_covariate_effects(out_dir=args.out_dir,
                                Gamma=Gamma[:, kept_ids],
                                covariate_names=_corpus_manifest_covariate_names(result),
                                K=len(kept_ids), P=P)
        log.info("STM: wrote covariate_effects.json (K=%d, P=%d)", len(kept_ids), P)

        # covariate_schema.json from the local covariate matrix
        _write_local_covariate_schema(args.out_dir, result, cov, X, k)

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

    v_disp = write_model_and_vocab_bundles(
        out_dir=args.out_dir,
        beta=export.beta, alpha=export.alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=stats.code_marginals,
        top_n=args.vocab_top_n,
        sigma=export.sigma,
    )
    if export.theta_histogram is not None:
        # NaN-suppressed bins → None for JSON serialization
        hist = [
            [None if np.isnan(v) else float(v) for v in row]
            for row in export.theta_histogram.tolist()
        ]
    else:
        hist = None

    if export.theta_percentiles is not None:
        # Columns are in [p5, p25, p50, p75, p95] order per DashboardExport
        pct = [
            {"p5": float(row[0]), "p25": float(row[1]),
             "p50": float(row[2]), "p75": float(row[3]), "p95": float(row[4])}
            for row in export.theta_percentiles
        ]
    else:
        pct = None

    write_phenotypes_bundle(
        args.out_dir / "phenotypes.json",
        npmi=npmi,
        pair_coverage=pair_coverage,
        corpus_prevalence=export.corpus_prevalence.tolist(),
        theta_histogram=hist,
        theta_percentiles=pct,
        topic_indices=export.topic_indices.tolist(),
        labels=None,
    )
    write_corpus_stats_sidecar(stats, args.out_dir / "corpus_stats.json", v_displayed=v_disp)

    log.info("wrote 4 files to %s (V_disp=%d K_disp=%d)", args.out_dir, v_disp, K_disp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
