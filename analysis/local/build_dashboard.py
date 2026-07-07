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
    # Suppression threshold to REPORT as theta_histogram_min_count. Defaults to
    # the LDA fit-time value (fit_lda's compute_theta_aggregates default = 20);
    # the STM build-step below overrides it to this run's k-anon k_thresh, since
    # that path suppresses the histogram at k_thresh here.
    theta_hist_min_count = 20

    # Predictive-gain bundle-level diagnostics (set by the STM build-step
    # below on success; stay None on any non-STM/non-gated build or an
    # enhancement-only failure, in which case write_phenotypes_bundle omits
    # the whole "predictive_gain" object). Bound here (like eta_scale) so
    # they are always defined for the unconditional write-bundle call below.
    pg_prominence_bin_edges = None
    pg_null_band = None
    pg_observed_delta_range = None
    pg_downdate_audit = None
    pg_scale = None
    pg_n_docs = None
    pg_smoothing = None

    # Load the vocab + BOW (Spark) up front: the STM correlation block below
    # reuses bow_df for the eta_scale E-step join, and corpus-stats / NPMI reuse it
    # again later (single load, single Spark session for the whole build).
    vocab_ids = result.metadata.get("vocab")
    if not vocab_ids:
        raise SystemExit("checkpoint metadata has no 'vocab'; re-fit needed.")
    corpus_manifest = result.metadata.get("corpus_manifest", {})
    doc_spec_manifest = corpus_manifest.get("doc_spec", {"name": "patient"})
    doc_spec = DocSpec.from_manifest(doc_spec_manifest)
    spark = _build_spark()
    df = load_omop_parquet(str(args.input), spark=spark)
    bow_df, _ = to_bow_dataframe(df, doc_spec=doc_spec, vocab=vocab_ids)
    bow_df = bow_df.persist()

    # Corpus stats (code_marginals) computed up front -- moved ahead of the STM
    # gating block (Task S2) because the predictive_gain build-step below needs
    # the corpus unigram for its background-smoothed predictive score, mirroring
    # build_dashboard_cloud.py's earlier "corpus stats" phase. V_full/K_disp only
    # depend on export.beta (already set by adapt() above), so this is safe to
    # hoist ahead of the tbs block without any other reordering.
    K_disp, V_full = export.beta.shape
    _sv_indices = F.udf(lambda sv: sv.indices.tolist() if sv is not None else [],
                        "array<int>")
    _sv_counts = F.udf(lambda sv: [float(x) for x in sv.values] if sv is not None else [],
                       "array<double>")
    bow_df_stats = bow_df.select(
        _sv_indices(F.col("features")).alias("indices"),
        _sv_counts(F.col("features")).alias("counts"),
    )
    bow_df_stats = bow_df_stats.persist()
    stats = compute_corpus_stats_from_bow_df(bow_df_stats, vocab_size=V_full, k=K_disp)
    log.info("corpus stats: n_docs=%d mean_codes=%.2f",
             stats.corpus_size_docs, stats.mean_codes_per_doc)

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

        # Held-out generative scale c (set by the correlation block below when
        # calibration succeeds); the theta_histogram phase infers at this scale.
        # Bound here so it is always defined even on the pre-Task-1 (no n_pairs)
        # path, where the histogram falls back to scale=1.0.
        eta_scale = None

        # correlation.json: logistic-normal topic correlation R + identified mask
        if "n_pairs" in result.global_params:
            from spark_vi.models.topic._linalg import topic_correlation_identified
            from charmpheno.export.correlation import build_correlation_json

            Sigma_corr = result.global_params["Sigma"]
            n_pairs = result.global_params["n_pairs"]
            # min_pair_support may be at top level (newer get_metadata) or
            # nested under stm_hardening (models saved before that change);
            # check both so the export mask floor always matches the fit.
            stm_hardening = result.metadata.get("stm_hardening", {}) or {}
            mps = int(result.metadata.get("min_pair_support")
                      or stm_hardening.get("min_pair_support", 1))
            reference_id = 0 if stm_hardening.get("reference_topic") else None
            R, ident = topic_correlation_identified(Sigma_corr, n_pairs, mps)

            # Generative variance scale c (held-out predictive-LL calibration,
            # corpus_heldout_scale_sweep_gated_rdd, HS-1/commit e22bcae): recovers the single
            # concentration scale that the unit-diagonal fitted R (ADR 0034) discards, so the
            # dashboard can rescale R -> Sigma_gen = eta_scale*R. This bounded grid sweep
            # SUPERSEDES the iterated pooled EM (corpus_eta_scale_gated_rdd), which is biased
            # and unstable: it has positive feedback in the scale direction (no trust region)
            # and ran away on the cluster (c: 3.6 -> 1116 -> 770918). The sweep scores each
            # grid c's held-out predictive LL; the raw grid ARGMAX over that curve is a
            # quantized, jittery point estimate (the curve is a broad, flat shelf -- LL
            # differences ~0.001-0.01 nats across c in roughly [2,12], within resampling
            # noise -- so argmax over a coarse grid drove a 5 -> 12 -> 8 refit wander). We
            # SHIP a smoothed reducer instead: a local quadratic fit in log c
            # (smooth_scale_log_quadratic) that recovers a sub-grid, noise-averaged c* plus
            # a curvature-based SE, honestly large when the shelf is flat -- see that
            # function's docstring for the algorithm and its Numerical-Recipes/Brent +
            # delta-method citations. The grid is GEOMETRIC (even resolution in log c, since
            # c is a multiplicative scale). We sweep 3 holdout fractions (0.5 shipped; 0.8/
            # 0.95 probe robustness as the visible token set shrinks toward the small-seed
            # regime) and SHIP the smoothed c* at holdout=0.5. ENHANCEMENT only: any failure
            # leaves eta_scale=None and the key is omitted (dashboard falls back to eta_var,
            # then unit R). Reuses the loaded bow_df (frozen fit vocab) joined with the local
            # covariate sidecar.
            eta_scale = None
            eta_scale_diag = None
            try:
                from spark_vi.mllib.topic.stm import (
                    corpus_heldout_scale_sweep_gated_rdd,
                    smooth_scale_log_quadratic,
                )
                from spark_vi.mllib.topic._common import _vector_to_stm_document
                from pyspark.ml.linalg import Vectors, VectorUDT
                from pyspark.sql.types import (
                    StructType, StructField, LongType, StringType,
                )
                # Spark DataFrame from the pandas covariate sidecar
                # (person_id, source_cohort, covariates DenseVector), joined to
                # bow_df on person_id -> STMDocument RDD (same assembly the fit
                # uses: features + covariates + source_cohort gating group).
                cov_schema = StructType([
                    StructField("person_id", LongType(), False),
                    StructField("source_cohort", StringType(), False),
                    StructField("covariates", VectorUDT(), False),
                ])
                cov_rows = [
                    (int(pid), str(sc), Vectors.dense(np.asarray(vec, dtype=float)))
                    for pid, sc, vec in zip(
                        cov["person_id"], cov["source_cohort"], cov["covariates"])
                ]
                cov_sdf = spark.createDataFrame(cov_rows, schema=cov_schema)
                doc_df = bow_df.select("person_id", "features").join(
                    cov_sdf, on="person_id", how="inner")
                doc_rdd = doc_df.rdd.map(
                    lambda row: _vector_to_stm_document(
                        row, features_col="features",
                        covariates_col="covariates",
                        group_col="source_cohort",
                    )
                )
                # Geometric (log-uniform) grid, ~13 points over [0.5, 32] --
                # NOT literature values, a heuristic bracket wide enough to
                # bound the c's observed on the cluster so far with even
                # resolution in log c (c is a multiplicative scale).
                C_GRID = [round(x, 2) for x in np.geomspace(0.5, 32.0, num=13)]
                HOLDOUTS = [0.5, 0.8, 0.95]
                robustness = {}
                robustness_argmax = {}
                lls_shipped = None
                smoothed_shipped = None
                for hf in HOLDOUTS:
                    sweep = corpus_heldout_scale_sweep_gated_rdd(
                        doc_rdd, result.global_params, partition,
                        c_grid=C_GRID, holdout_frac=hf,
                        reference=reference_id, seed=0,
                    )
                    smoothed_hf = smooth_scale_log_quadratic(sweep["lls"])
                    robustness[str(hf)] = smoothed_hf["c_star"]
                    robustness_argmax[str(hf)] = sweep["argmax_c"]
                    if hf == 0.5:
                        lls_shipped = {
                            str(k): float(v) for k, v in sweep["lls"].items()
                        }
                        smoothed_shipped = smoothed_hf
                c_star = smoothed_shipped["c_star"]
                grid_argmax_c = robustness_argmax["0.5"]
                not_interior = not smoothed_shipped["interior"]
                if not_interior:
                    log.warning(
                        "STM: smoothed held-out c*=%s is NOT an interior "
                        "maximum (grid %s) -- widen the grid and re-run "
                        "(calibrated scale is not identified from this "
                        "grid).", c_star, C_GRID)
                eta_scale = float(c_star)   # SHIP the smoothed held-out estimate
                eta_scale_diag = {
                    "method": "heldout_ll_gated",
                    "c_star": float(c_star),
                    "grid_argmax_c": float(grid_argmax_c),
                    "holdout_frac_shipped": 0.5,
                    "c_grid": C_GRID,
                    "robustness_argmax_by_holdout": robustness,
                    "robustness_grid_argmax_by_holdout": robustness_argmax,
                    "lls_at_shipped_holdout": lls_shipped,
                    "argmax_at_grid_boundary": bool(
                        grid_argmax_c in (C_GRID[0], C_GRID[-1])),
                    "smoothed": smoothed_shipped,
                }
                log.info(
                    "STM: smoothed held-out c*=%.4f (holdout 0.5, "
                    "grid_argmax=%.4f); robustness=%s.",
                    c_star, grid_argmax_c, robustness)
            except Exception as exc:  # enhancement-only: never fatal
                log.warning("STM: eta_scale held-out calibration failed (%s); "
                            "correlation.json omits eta_scale (dashboard falls "
                            "back to eta_var/unit R).", exc)
                eta_scale = None
                eta_scale_diag = None

            corr = build_correlation_json(R, ident, n_pairs, partition, kept_ids,
                                           reference_id=reference_id,
                                           eta_scale=eta_scale,
                                           eta_scale_diagnostic=eta_scale_diag)
            (args.out_dir / "correlation.json").write_text(json.dumps(corr, indent=2))
            log.info("STM: wrote correlation.json (topics=%d, min_pair_support=%d)",
                     len(kept_ids), mps)
        else:
            log.warning("STM: saved model lacks 'n_pairs' in global_params "
                        "(pre-Task-1 checkpoint); skipping correlation.json.")

        # covariate_effects.json for the KEPT topics (Gamma columns subset)
        P = Gamma.shape[0]
        write_covariate_effects(out_dir=args.out_dir,
                                Gamma=Gamma[:, kept_ids],
                                covariate_names=_corpus_manifest_covariate_names(result),
                                K=len(kept_ids), P=P)
        log.info("STM: wrote covariate_effects.json (K=%d, P=%d)", len(kept_ids), P)

        # covariate_schema.json from the local covariate matrix
        _write_local_covariate_schema(args.out_dir, result, cov, X, k)

        # theta_histogram: per-doc gated MAP theta distribution (the dashboard's
        # "topic mass distribution" panel). Plain-LDA writes per-doc theta
        # aggregates at fit time; the STM fit does not, so we compute them here
        # from the fitted checkpoint + corpus (a BUILD-STEP, like corpus_prevalence
        # / eta_scale). Guarded on STM + gating (partition present); a non-gated or
        # LDA build simply never enters this branch and export.theta_histogram
        # stays None. Enhancement-only: any failure logs and leaves the two fields
        # None (dashboard hides the panel). Builds its OWN doc_rdd here (a second
        # per-doc pass) mirroring the eta_scale block; it could later be fused with
        # that block's doc_rdd -- kept independent for now.
        try:
            from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd
            from spark_vi.mllib.topic._common import _vector_to_stm_document
            from charmpheno.export.theta_aggregates import compute_theta_aggregates
            from charmpheno.export.model_adapter import (
                _parse_theta_histogram, _parse_theta_percentiles,
            )
            from pyspark.ml.linalg import Vectors, VectorUDT
            from pyspark.sql.types import (
                StructType, StructField, LongType, StringType,
            )

            stm_hardening = result.metadata.get("stm_hardening", {}) or {}
            reference_id = 0 if stm_hardening.get("reference_topic") else None
            k_thresh = int(corpus.get("min_patient_count", 20))

            # Self-contained STMDocument RDD (same assembly as the eta_scale
            # block: bow features + covariate sidecar + source_cohort gating group).
            cov_schema = StructType([
                StructField("person_id", LongType(), False),
                StructField("source_cohort", StringType(), False),
                StructField("covariates", VectorUDT(), False),
            ])
            cov_rows = [
                (int(pid), str(sc), Vectors.dense(np.asarray(vec, dtype=float)))
                for pid, sc, vec in zip(
                    cov["person_id"], cov["source_cohort"], cov["covariates"])
            ]
            cov_sdf = spark.createDataFrame(cov_rows, schema=cov_schema)
            doc_df = bow_df.select("person_id", "features").join(
                cov_sdf, on="person_id", how="inner")
            doc_rdd = doc_df.rdd.map(
                lambda row: _vector_to_stm_document(
                    row, features_col="features",
                    covariates_col="covariates",
                    group_col="source_cohort",
                )
            )
            # Infer the display histogram at the CALIBRATED generation scale
            # (eta_scale = c ~ 4.6), not the over-diffuse unit fit scale: the
            # calibrated prior concentrates each patient's theta_hat onto the
            # topics they actually express (honest prevalence). Falls back to
            # scale=1.0 when calibration failed / eta_scale is None.
            hist_scale = float(eta_scale) if eta_scale else 1.0
            log.info("STM: theta histogram inferred at scale=%.4f (%s).",
                     hist_scale,
                     "calibrated eta_scale" if eta_scale else "unit fallback")
            # 200k sample_cap is a heuristic driver-memory bound, not a literature
            # value; corpus_theta_gated_rdd logs the sampled N / fraction.
            theta_arr = corpus_theta_gated_rdd(
                doc_rdd, result.global_params, partition,
                reference=reference_id, scale=hist_scale,
                sample_cap=200_000, seed=0)
            agg = compute_theta_aggregates(theta_arr, min_count=k_thresh)
            kept = export.topic_indices.tolist()
            # DashboardExport is a frozen dataclass; replace() returns a new
            # instance with the two theta fields set (do NOT touch
            # corpus_prevalence -- keep the faithful masked covariate-mean).
            from dataclasses import replace as _dc_replace
            export = _dc_replace(
                export,
                theta_histogram=_parse_theta_histogram(
                    agg["theta_histogram"])[kept],
                theta_percentiles=_parse_theta_percentiles(
                    agg["theta_percentiles"])[kept],
            )
            # Report the threshold the histogram was actually suppressed at.
            theta_hist_min_count = k_thresh
            log.info(
                "STM: computed per-doc theta histogram (sampled_docs=%d, "
                "kept_topics=%d).", theta_arr.shape[0], len(kept))
        except Exception as exc:  # enhancement-only: never fatal
            log.warning("STM: per-doc theta histogram failed (%s); phenotypes.json "
                        "omits theta_histogram/theta_percentiles (panel hidden).",
                        exc)

        # predictive_gain: per-topic presence/depth/prominence aggregates (the
        # dashboard's predictive-gain view, spark_vi.mllib.topic.predictive_gain
        # Phase 2). Leave-one-topic-out held-out predictive gain Delta_k
        # answers "how much held-out predictive power does topic k actually
        # contribute", complementing theta_histogram's "how much MASS does a
        # patient put on topic k" view. Runs AFTER theta_histogram (same
        # self-contained doc_rdd assembly) and BEFORE the write-bundle phase
        # (shared, unconditional code below). Enhancement-only: any failure
        # leaves the six new DashboardExport fields (and the pg_* diagnostic
        # locals bound above) None (dashboard hides the panel). PROVISIONAL
        # schema -- see write_phenotypes_bundle's docstring; Phase-2 will
        # recalibrate prominence_range from observed_delta_range, logged here.
        # Uses fast=True (the warm-start Newton downdate, ~2x a single
        # inference pass) validated against the COLD oracle via a small-sample
        # audit (predictive_gain_downdate_audit) whose max_abs_overall is the
        # headline cold-reliability number, logged prominently below. Parity
        # with analysis/cloud/build_dashboard_cloud.py.
        try:
            from spark_vi.mllib.topic.predictive_gain import (
                corpus_predictive_gain_gated_rdd,
                predictive_gain_downdate_audit,
            )
            from spark_vi.mllib.topic._common import _vector_to_stm_document
            from pyspark.ml.linalg import Vectors, VectorUDT
            from pyspark.sql.types import (
                StructType, StructField, LongType, StringType,
            )
            from dataclasses import replace as _dc_replace

            stm_hardening = result.metadata.get("stm_hardening", {}) or {}
            pg_reference = 0 if stm_hardening.get("reference_topic") else None

            # Self-contained STMDocument RDD (same assembly as the eta_scale /
            # theta_histogram blocks: bow features + covariate sidecar +
            # source_cohort gating group).
            pg_cov_schema = StructType([
                StructField("person_id", LongType(), False),
                StructField("source_cohort", StringType(), False),
                StructField("covariates", VectorUDT(), False),
            ])
            pg_cov_rows = [
                (int(pid), str(sc), Vectors.dense(np.asarray(vec, dtype=float)))
                for pid, sc, vec in zip(
                    cov["person_id"], cov["source_cohort"], cov["covariates"])
            ]
            pg_cov_sdf = spark.createDataFrame(pg_cov_rows, schema=pg_cov_schema)
            pg_doc_df = bow_df.select("person_id", "features").join(
                pg_cov_sdf, on="person_id", how="inner")
            pg_doc_rdd = pg_doc_df.rdd.map(
                lambda row: _vector_to_stm_document(
                    row, features_col="features",
                    covariates_col="covariates",
                    group_col="source_cohort",
                )
            )
            # Same calibrated-scale-or-unit-fallback convention as the
            # theta_histogram block's hist_scale: Sigma_gen = c*R uses the
            # held-out-LL calibrated eta_scale when available.
            pg_scale = float(eta_scale) if eta_scale else 1.0
            log.info("STM: predictive gain computed at scale=%.4f (%s).",
                      pg_scale,
                      "calibrated eta_scale" if eta_scale else "unit fallback")

            # Corpus unigram (Task S2): activates predictive_gain's S1
            # background-smoothed predictive score p_S(w) =
            # (1-eps)*(theta@beta)(w) + eps*m_w in place of the historical
            # unsmoothed 1e-12-floor path. stats.code_marginals (computed up
            # front, ahead of the STM gating block) is already the length-
            # V_full token-frequency vector -- normalize defensively (it
            # should already sum to ~1) and length-check against the fitted
            # beta's vocab width before trusting it.
            pg_marginal_raw = np.asarray(stats.code_marginals, dtype=float)
            lam_v = np.asarray(result.global_params["lambda"]).shape[1]
            if pg_marginal_raw.shape[0] != lam_v:
                log.warning(
                    "predictive_gain: marginal unavailable, unsmoothed "
                    "fallback (code_marginals length=%d != lambda V=%d).",
                    pg_marginal_raw.shape[0], lam_v)
                pg_marginal = None
            else:
                pg_marginal_sum = float(pg_marginal_raw.sum())
                if (not np.isfinite(pg_marginal_sum) or pg_marginal_sum <= 0
                        or np.any(np.isnan(pg_marginal_raw))):
                    log.warning(
                        "predictive_gain: marginal unavailable, unsmoothed "
                        "fallback (code_marginals sum=%s is degenerate).",
                        pg_marginal_sum)
                    pg_marginal = None
                else:
                    pg_marginal = pg_marginal_raw / pg_marginal_sum
                    # Floor with a small uniform mass so NO vocab token has zero
                    # backoff probability (a zero m_w defeats the smoother:
                    # log(eps*0) hits the safety floor and Delta spikes like the
                    # old 1e-12 problem). Cloud parity; standard backoff smoothing.
                    _pgV = pg_marginal.shape[0]
                    pg_marginal = 0.99 * pg_marginal + 0.01 / _pgV
            # print (not log.info) so the smoother status is ALWAYS visible
            # regardless of the driver's logger level (cloud parity).
            if pg_marginal is not None:
                print("[driver]   predictive_gain: smoothed score active "
                      "(lambda=1.0)", flush=True)
            else:
                print("[driver]   predictive_gain: marginal unavailable, "
                      "unsmoothed fallback", flush=True)
            pg_smoothing = {
                "active": pg_marginal is not None,
                "lambda": 1.0,
                "marginal_floor": 0.01,
            }

            pg = corpus_predictive_gain_gated_rdd(
                pg_doc_rdd, result.global_params, partition,
                c=pg_scale, reference=pg_reference, fast=True,
                sample_cap=200_000, seed=0,
                marginal=pg_marginal, smoothing_lambda=1.0,
            )

            # Cold-vs-fast downdate reliability audit on a small in-memory
            # sample -- own try/except so an audit failure cannot drop the
            # (already computed) main aggregates.
            try:
                pg_audit_docs = pg_doc_rdd.takeSample(False, 50, seed=0)
                pg_audit_raw = predictive_gain_downdate_audit(
                    pg_audit_docs, result.global_params, partition,
                    c=pg_scale, reference=pg_reference,
                    marginal=pg_marginal, smoothing_lambda=1.0,
                )
                _pg_mad = np.asarray(
                    pg_audit_raw["mean_abs_discrepancy"], dtype=float)
                pg_mean_abs_overall = (
                    float(np.nanmean(_pg_mad))
                    if np.isfinite(_pg_mad).any() else float("nan"))
                pg_downdate_audit = {
                    "max_abs_overall": float(pg_audit_raw["max_abs_overall"]),
                    "mean_abs_overall": pg_mean_abs_overall,
                    "n_docs_audited": int(pg_audit_raw["n_docs_audited"]),
                }
                log.info(
                    "STM: predictive-gain downdate audit max_abs_overall=%.6f "
                    "(n_docs_audited=%d) -- cold-vs-fast (fast=True) "
                    "reliability gate.",
                    pg_downdate_audit["max_abs_overall"],
                    pg_downdate_audit["n_docs_audited"])
            except Exception as audit_exc:
                log.warning(
                    "STM: predictive-gain downdate audit failed (%s); "
                    "phenotypes.json omits the downdate_audit diagnostic "
                    "(main aggregates unaffected).", audit_exc)
                pg_downdate_audit = None

            kept = export.topic_indices.tolist()
            export = _dc_replace(
                export,
                presence=pg["presence"][kept],
                mean_gain=pg["mean_gain"][kept],
                depth=pg["depth"][kept],
                prominence_hist=pg["prominence_hist"][kept],
                length_corr=pg["length_corr"][kept],
                dedup_gain=pg["dedup_mean_gain"][kept],
            )
            pg_prominence_bin_edges = pg["prominence_bin_edges"].tolist()
            pg_null_band = pg["null_band"]
            pg_observed_delta_range = list(pg["observed_delta_range"])
            pg_n_docs = int(pg["n_docs"])
            log.info(
                "STM: computed predictive-gain aggregates (n_docs=%d, "
                "kept_topics=%d, observed_delta_range=%s).",
                pg_n_docs, len(kept), pg_observed_delta_range)
        except Exception as exc:  # enhancement-only: never fatal
            log.warning("STM: predictive-gain aggregation failed (%s); "
                        "phenotypes.json omits the predictive_gain object "
                        "(panel hidden).", exc)
            pg_prominence_bin_edges = None
            pg_null_band = None
            pg_observed_delta_range = None
            pg_downdate_audit = None
            pg_scale = None
            pg_n_docs = None
            pg_smoothing = None

    log.info("K_display=%d V_full=%d (model_class=%s)", K_disp, V_full, model_class)

    descriptions = result.metadata.get("concept_names", {}) or {}
    domains = result.metadata.get("concept_domains", {}) or {}

    # stats (code_marginals) was computed up front, ahead of the STM gating
    # block, so the predictive_gain build-step above could thread the corpus
    # unigram into it (Task S2); bow_df/bow_df_stats are reused here for NPMI.

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

    # Predictive-gain per-topic arrays (PROVISIONAL — see
    # write_phenotypes_bundle's docstring): None when the phase above never
    # ran or failed (export.presence etc. stay unset), in which case
    # write_phenotypes_bundle omits the whole "predictive_gain" key
    # (byte-unchanged bundle). NaN -> None, same convention as theta_histogram.
    def _nan_to_none(arr):
        return [None if np.isnan(v) else float(v) for v in arr.tolist()]

    if export.presence is not None:
        pg_presence = _nan_to_none(export.presence)
        pg_mean_gain = _nan_to_none(export.mean_gain)
        pg_depth = _nan_to_none(export.depth)
        pg_length_corr = _nan_to_none(export.length_corr)
        pg_dedup_gain = _nan_to_none(export.dedup_gain)
        pg_prominence_hist_json = [
            [None if np.isnan(v) else float(v) for v in row]
            for row in export.prominence_hist.tolist()
        ]
    else:
        pg_presence = None
        pg_mean_gain = None
        pg_depth = None
        pg_length_corr = None
        pg_dedup_gain = None
        pg_prominence_hist_json = None

    write_phenotypes_bundle(
        args.out_dir / "phenotypes.json",
        npmi=npmi,
        pair_coverage=pair_coverage,
        corpus_prevalence=export.corpus_prevalence.tolist(),
        theta_histogram=hist,
        theta_percentiles=pct,
        topic_indices=export.topic_indices.tolist(),
        min_count=theta_hist_min_count,
        labels=None,
        presence=pg_presence,
        mean_gain=pg_mean_gain,
        depth=pg_depth,
        prominence_hist=pg_prominence_hist_json,
        length_corr=pg_length_corr,
        dedup_gain=pg_dedup_gain,
        prominence_bin_edges=pg_prominence_bin_edges,
        null_band=pg_null_band,
        observed_delta_range=pg_observed_delta_range,
        predictive_gain_downdate_audit=pg_downdate_audit,
        predictive_gain_scale=pg_scale,
        predictive_gain_n_docs=pg_n_docs,
        predictive_gain_smoothing=pg_smoothing,
    )
    write_corpus_stats_sidecar(stats, args.out_dir / "corpus_stats.json", v_displayed=v_disp)

    log.info("wrote 4 files to %s (V_disp=%d K_disp=%d)", args.out_dir, v_disp, K_disp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
