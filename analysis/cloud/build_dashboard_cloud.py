"""Cloud-side dashboard bundle builder for a saved OnlineLDA / OnlineHDP checkpoint.

Mirrors `analysis/local/build_dashboard.py` for the BigQuery-sourced cloud
setting. Loads the augmented VIResult written by `lda_bigquery_cloud.py`
/ `hdp_bigquery_cloud.py`, reproduces the BQ -> BOW pipeline from
`metadata['corpus_manifest']` with the *frozen* vocab from
`metadata['vocab']`, looks up concept_name/domain_id from the OMOP
`concept` table, and writes the four-file dashboard bundle. Output dir
defaults to `<checkpoint>/dashboard_bundle/`; a sibling `.zip` is also
written for easy download via `gsutil cp`.

Env (same as the fit + eval drivers):
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Submit (from analysis/cloud on the Dataproc master):
    make build-dashboard-bundle CHECKPOINT=/mnt/gcs/$BUCKET/runs/<id>
    make build-dashboard-bundle CHECKPOINT=... \\
        BUNDLE_ARGS='--model-class hdp --hdp-top-k 50'

The 4 base files (model.json, vocab.json, phenotypes.json, corpus_stats.json)
land in --out-dir and are zipped into <out-dir>.zip. Any optional gated /
correlation outputs that were written (gating.json, covariate_schema.json,
covariate_effects.json, correlation.json) are included in the zip too, so the
downloadable artifact is the complete bundle.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session

COVARIATE_CACHE_MISS_EXIT = 42  # distinct exit code so run_experiment.py --build-only can auto-rebuild the covariate cache and retry (see run_experiment._build_only_with_auto_covariates)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    pass


def _quant_col_df(arr, idx: int):
    """Return (df_with_col, col_name) where col_name is the element column."""
    col_name = f"_q_{idx}"
    from pyspark.sql import functions as _F
    return arr.withColumn(col_name, _F.col("x")[idx]), col_name


def _categorical_levels_from_spec(model_spec, covariate_names=()):
    """Extract {var: {"levels": [...], "reference": "..."}} from a formulaic ModelSpec.

    Tries model_spec.structure (formulaic >= 0.5) which exposes Factor records
    with .levels/.reference attributes.  If that path is unavailable (different
    formulaic version or unexpected layout), falls back to parsing C(var)[T.level]
    strings from covariate_names and reading the reference from whatever the
    spec exposes via .encoder_state or .factors.  If every path fails, returns
    {} for that variable so the whole schema write degrades gracefully.

    NOTE: formulaic introspection is cluster-validated — the exact attribute
    names depend on the installed formulaic version on the cluster.
    """
    import re as _re

    result = {}

    # --- primary path: model_spec.structure (formulaic >= 0.5) ---
    try:
        for factor in model_spec.structure:
            # Each factor record has .name, .levels, .reference (or similar).
            # Try the most common attribute shapes.
            var = getattr(factor, "name", None)
            if var is None:
                continue
            levels = None
            reference = None
            # formulaic >= 0.5 stores these directly:
            for levels_attr in ("levels", "categories", "codes"):
                if hasattr(factor, levels_attr):
                    levels = list(getattr(factor, levels_attr))
                    break
            for ref_attr in ("reference", "base", "reference_level", "drop_field"):
                if hasattr(factor, ref_attr):
                    reference = str(getattr(factor, ref_attr))
                    break
            if levels is not None and reference is not None:
                result[var] = {"levels": levels, "reference": reference}
        if result:
            return result
    except Exception:
        pass  # fall through to fallback

    # --- fallback: parse C(var)[T.level] strings from covariate_names ---
    try:
        dummy_pat = _re.compile(r"^C\((?P<var>[^)]+)\)\[T\.(?P<lvl>.+)\]$")
        from collections import defaultdict
        parsed: dict[str, list[str]] = defaultdict(list)
        for name in covariate_names:
            m = dummy_pat.match(name)
            if m:
                parsed[m.group("var")].append(m.group("lvl"))

        if not parsed:
            return result  # nothing to do

        # Try to read references from encoder_state or factors on the spec.
        ref_map: dict[str, str] = {}
        try:
            # encoder_state is a dict keyed by factor name in some formulaic versions
            for var, enc in model_spec.encoder_state.items():
                # enc may be a dict with "reference" or a CategorizationEncoder
                if isinstance(enc, dict) and "reference" in enc:
                    ref_map[var] = str(enc["reference"])
                elif hasattr(enc, "reference"):
                    ref_map[var] = str(enc.reference)
        except Exception:
            pass
        try:
            for factor in model_spec.factors:
                var = getattr(factor, "name", None)
                for ref_attr in ("reference", "base", "reference_level"):
                    if hasattr(factor, ref_attr):
                        ref_map[var] = str(getattr(factor, ref_attr))
                        break
        except Exception:
            pass

        for var, t_levels in parsed.items():
            reference = ref_map.get(var, "")
            all_levels = ([reference] if reference else []) + t_levels
            result[var] = {"levels": all_levels, "reference": reference}
        return result
    except Exception:
        return result  # return whatever we managed to accumulate


def _covariate_cache_key(*, corpus, cov_manifest, source_table, cohort):
    """Covariate-cache key for the dashboard build (single source of truth).

    Both build-side lookups go through here so they cannot drift, and — the bug
    this closes — so ``prior_obs_days`` is part of the key. The producers (the
    fit via ``_covariates_load`` and ``build-covariates``) key on it; it is
    load-bearing in composite cohorts (see ``_covariates_cache.compute_cache_key``).
    Consumers that omitted it defaulted to 365 and missed the cache for every
    experiment with a non-default lookback (e.g. exp 0027, ``prior_obs_days=0``),
    silently dropping gating.json / covariate_schema.json and forcing the
    intercept stand-in for corpus_prevalence. ``.get(..., 365)`` preserves the
    old value for pre-record checkpoints that never stamped it.
    """
    from _covariates_cache import compute_cache_key
    return compute_cache_key(
        covariate_formula=cov_manifest["covariate_formula"],
        person_mod=corpus["person_mod"],
        cdr=corpus["cdr"],
        source_table=source_table,
        cohort=cohort,
        prior_obs_days=int(corpus.get("prior_obs_days", 365)),
    )


def assert_covariate_sidecar_matches_model(*, sidecar_names, model_covariate_names):
    """Fail loud when the loaded covariate sidecar's design != the model's.

    The covariate cache key is content-BLIND (formula/person_mod/cdr/cohort/
    prior_obs_days — see _covariate_cache_key), so a changed design that keeps
    those fixed reuses a stale sidecar under the same key. exp 0028 hit this:
    known_sex_only dropped the 'Unknown' sex level (P 4 -> 3), but the cached
    sidecar was still P=4, so Gamma (P=3) @ x (P=4) crashed corpus_prevalence —
    which was caught and degraded to the intercept stand-in, also dropping
    gating.json. That silent degradation is exactly what we refuse here: a
    dimension/name mismatch is a stale-cache bug, never cosmetic.
    """
    sidecar = list(sidecar_names)
    model = list(model_covariate_names)
    if sidecar != model:
        raise SystemExit(
            "STM covariate sidecar is STALE relative to the model: sidecar has "
            f"P={len(sidecar)} design columns {sidecar}, but the model's Gamma "
            f"has P={len(model)} {model}. The covariate cache key is "
            "content-blind, so a changed design (e.g. known_sex_only dropping a "
            "sex level) reuses a stale sidecar. Rebuild it before exporting: "
            "`make build-covariates EXP=<id> FORCE=1`, then re-run the build."
        )


def build_marginalized_scale_diagnostic(
    *, map_cstar_by_holdout, marg_cstar_by_holdout, n_samples, n_docs_sampled,
    c_grid, holdouts,
):
    """Assemble the MAP-vs-marginalized held-out scale comparison for the
    eta_scale diagnostic. Pure/deterministic (no Spark): given the smoothed c*
    per holdout fraction for each estimator, return a dict recording both
    curves and each estimator's residual drift (max - min c* across holdout
    fractions -- the quantity that should be ~0 for a well-behaved,
    prefix-independent scale). map_/marg_cstar_by_holdout are dicts keyed by
    str(holdout_fraction) -> smoothed c* (float)."""
    def _drift(d):
        vals = [float(d[str(h)]) for h in holdouts]
        return float(max(vals) - min(vals))
    return {
        "n_samples": int(n_samples),
        "n_docs_sampled": int(n_docs_sampled),
        "c_grid": list(c_grid),
        "holdouts": [float(h) for h in holdouts],
        "map_cstar_by_holdout": {str(h): float(map_cstar_by_holdout[str(h)]) for h in holdouts},
        "marg_cstar_by_holdout": {str(h): float(marg_cstar_by_holdout[str(h)]) for h in holdouts},
        "map_residual_drift": _drift(map_cstar_by_holdout),
        "marg_residual_drift": _drift(marg_cstar_by_holdout),
    }


def _required_stm_outputs(*, gated: bool) -> list[str]:
    """Covariate-dependent bundle files an STM export must produce."""
    req = ["covariate_effects.json", "covariate_schema.json"]
    if gated:
        req = ["gating.json", *req]
    return req


def assert_stm_bundle_complete(out_dir, *, gated, allow_incomplete, log=None):
    """Fail loud when a gated/STM bundle is missing its covariate outputs.

    A covariate-cache MISS (or absent --cache-uri) silently skips gating.json /
    covariate_schema.json, yielding a bundle that *looks* complete but renders
    ungated with no covariate panel. Rather than warn-and-continue, abort so the
    operator rebuilds the covariate cache before export. --allow-incomplete-bundle
    downgrades to a warning when a degraded ungated bundle is genuinely intended.
    """
    out_dir = Path(out_dir)
    missing = [f for f in _required_stm_outputs(gated=gated)
               if not (out_dir / f).exists()]
    if not missing:
        return
    detail = (
        f"STM bundle is INCOMPLETE — missing {missing}. Most likely the "
        "covariate cache missed during the build (no --cache-uri, or a stale/"
        "unbuilt sidecar), so gating.json / covariate_schema.json were skipped "
        "and the dashboard would render ungated with no covariate panel. Fix: "
        "`make build-covariates EXP=<id> FORCE=1` so the export hits the cache, "
        "then rebuild."
    )
    if allow_incomplete:
        if log is not None:
            log.warning("%s (--allow-incomplete-bundle set; continuing)", detail)
        return
    raise SystemExit(
        detail + " Pass --allow-incomplete-bundle to accept the degraded bundle."
    )


def assert_covariate_sidecar_present(
    *, is_stm, gated, sidecar_present, allow_incomplete, exp_hint=None, log=None,
):
    """Fail loud EARLY when a gated STM build has no covariate sidecar.

    This is the up-front counterpart of ``assert_stm_bundle_complete``. That
    end-of-build guard checks whether the covariate output FILES exist in
    ``out_dir`` -- which a reused ``out_dir`` silently defeats: a prior build's
    ``gating.json`` / ``covariate_schema.json`` linger and satisfy the existence
    check even when THIS run wrote nothing (covariate-cache MISS). This guard
    instead checks the SIDECAR itself the moment we know it is absent (right
    after the covariate-cache lookup, before the expensive corpus-stats / NPMI
    phases), so a miss aborts fast and is immune to stale files.

    Without the sidecar a gated build skips ALL per-document STM outputs at once
    -- gating.json, covariate_schema.json, eta_scale, the theta histogram, and
    predictive_gain -- yielding a bundle that renders ungated. The cache is
    cluster-local (HDFS), so a fresh cluster needs ``make build-covariates``
    re-run once. ``--allow-incomplete-bundle`` downgrades to a warning when a
    degraded ungated bundle is genuinely intended.
    """
    if not (is_stm and gated) or sidecar_present:
        return
    fix = "make build-covariates EXP={}".format(exp_hint) if exp_hint else \
        "make build-covariates EXP=<id>"
    detail = (
        "STM covariate sidecar MISSING for a gated build (covariate-cache MISS). "
        "Without it gating.json, covariate_schema.json, eta_scale, the theta "
        "histogram, and predictive_gain are ALL skipped and the dashboard renders "
        f"ungated. Fix: `{fix}` on this cluster, then rebuild. (The covariate "
        "cache is cluster-local HDFS, so a fresh cluster needs it rebuilt once.)"
    )
    if allow_incomplete:
        if log is not None:
            log.warning("%s (--allow-incomplete-bundle set; continuing)", detail)
        return
    msg = detail + " Pass --allow-incomplete-bundle to accept the degraded bundle."
    if log is not None:
        log.error(msg)
    else:
        print(msg, flush=True)
    raise SystemExit(COVARIATE_CACHE_MISS_EXIT)


def _write_covariate_schema(spark, *, result, corpus, source_table, cohort_name,
                            cache_uri, out_dir, log):
    """Derive + write covariate_schema.json from the covariate sidecar.

    No-op (logs a warning) when the sidecar is unavailable, so the Atlas panel
    simply hides. All stats are in-enclave aggregates (dummy-column sums,
    coarse percentiles) — nothing single-patient leaves.
    """
    if not cache_uri:
        log.warning("STM: no --cache-uri; covariate_schema.json not written.")
        return
    try:
        import json
        import math  # noqa: F401  (available for callers of helpers)
        import re as _re
        from pyspark.sql import functions as F
        from pyspark.ml.functions import vector_to_array
        from _covariates_cache import try_load
        from charmpheno.export.covariate_schema import build_covariate_schema

        cov_manifest = result.metadata["covariate_manifest"]
        key = _covariate_cache_key(
            corpus=corpus, cov_manifest=cov_manifest,
            source_table=source_table, cohort=cohort_name,
        )
        cached = try_load(spark, cache_uri, key)
        if cached is None:
            log.warning("STM: covariate-cache MISS; covariate_schema.json not written.")
            return
        cov_df, model_spec, covariate_names = cached
        continuous_cols = list(cov_manifest.get("continuous_cols", []))
        k = int(corpus.get("min_patient_count", 20))
        n_total = int(cov_df.count())

        # Project the design vector to an array column once.
        arr = cov_df.select(vector_to_array("covariates").alias("x"))
        name_idx = {n: i for i, n in enumerate(covariate_names)}

        # Dummy-column sums (= per-level patient counts) for every C(var)[T.level].
        dummy_names = [n for n in covariate_names
                       if _re.match(r"^C\(.+\)\[T\..+\]$", n)]
        if dummy_names:
            sums = arr.agg(*[
                F.sum(F.col("x")[name_idx[n]]).alias(n) for n in dummy_names
            ]).collect()[0].asDict()
            level_counts = {n: int(sums[n]) for n in dummy_names}
        else:
            level_counts = {}

        # Coarse percentiles for continuous columns (p5, p50, p95), rounded.
        continuous_stats = {}
        for var in continuous_cols:
            if var not in name_idx:
                log.warning("STM: continuous covariate %r absent from design vector; skipping its control.", var)
                continue
            idx = name_idx[var]
            arr_with_col, col_name = _quant_col_df(arr, idx)
            q = arr_with_col.approxQuantile(col_name, [0.05, 0.5, 0.95], 0.01)
            continuous_stats[var] = tuple(round(v) for v in q)

        # Levels + reference: prefer the value persisted at fit time under
        # covariate_manifest["categorical_levels"] (populated by the STM cloud
        # fitter via _extract_categorical_levels). Fall back to the formulaic
        # model_spec introspection path for older checkpoints or cache-loaded
        # specs. Mirrors the local build_dashboard._write_local_covariate_schema.
        categorical_levels = cov_manifest.get("categorical_levels")
        if not categorical_levels:
            categorical_levels = _categorical_levels_from_spec(
                model_spec, covariate_names=covariate_names,
            )

        schema = build_covariate_schema(
            covariate_names=covariate_names, continuous_cols=continuous_cols,
            categorical_levels=categorical_levels, level_counts=level_counts,
            continuous_stats=continuous_stats, k=k,
            n_total=n_total,
        )
        (out_dir / "covariate_schema.json").write_text(json.dumps(schema, indent=2))
        log.info("STM: wrote covariate_schema.json (controls=%d, unsupported=%d)",
                 len(schema["controls"]), len(schema["unsupported"]))
        print("[driver]   covariate_schema:", json.dumps(schema, indent=2), flush=True)
    except Exception as exc:  # cosmetic-only; never fail the bundle build
        log.warning("STM: covariate_schema derivation failed (%s); skipping.", exc)


def _stm_corpus_prevalence(spark, *, result, corpus, source_table,
                           cohort_name, cache_uri, log,
                           partition=None):
    """Faithful corpus-mean alpha-equivalent for STM, or None to fall back.

    For NON-GATED checkpoints (partition is None):
    Reloads the covariate sidecar from its cache and reduces it with
    corpus_mean_proportions_from_covariate_df (distributed treeReduce; only a
    K-vector reaches the driver).

    For GATED checkpoints (partition is not None):
    Reduces the sidecar with corpus_mean_proportions_gated_from_covariate_df
    (distributed mapPartitions+treeReduce; masked before softmax so each
    foreground topic's prevalence reflects only its group's share). Only a
    K-vector reaches the driver — no full-corpus collect.

    Returns None — leaving adapt_stm on its softmax(Gamma[intercept]) stand-in
    — when no cache_uri is supplied, the cache misses, or anything raises.
    The quantity is cosmetic (the dashboard's "default topic proportion" widget),
    so it must never abort the bundle build.

    Returns a 3-tuple (prev, gc, cov_df). gc is the per-group patient counts
    (gated only; None otherwise) used for k-anon suppression; cov_df is the
    loaded covariate sidecar DataFrame (person_id, source_cohort, covariates)
    when the cache hits, else None. cov_df is returned so the main flow can
    reuse it (join with bow_df) for the eta_scale E-step pass without a second
    cache load. Returns (None, None, None) on any failure path.
    """
    if not cache_uri:
        log.warning("STM: no --cache-uri; corpus_prevalence uses the "
                    "softmax(Gamma[intercept]) stand-in.")
        return None, None, None
    try:
        from _covariates_cache import try_load

        cov_manifest = result.metadata["covariate_manifest"]
        key = _covariate_cache_key(
            corpus=corpus, cov_manifest=cov_manifest,
            source_table=source_table, cohort=cohort_name,
        )
        with _phase(f"covariates-cache lookup ({cache_uri}/{key})"):
            cached = try_load(spark, cache_uri, key)
        if cached is None:
            log.warning("STM: covariate-cache MISS (%s/%s); corpus_prevalence "
                        "uses the intercept stand-in.", cache_uri, key)
            return None, None, None
        cov_df, _spec, _names = cached
        # Fail loud on a stale sidecar (content-blind cache key) BEFORE the
        # matmul: a P mismatch here otherwise crashes into the except below and
        # degrades silently. SystemExit is a BaseException, so it propagates
        # past `except Exception` and aborts the build with a rebuild hint.
        assert_covariate_sidecar_matches_model(
            sidecar_names=_names,
            model_covariate_names=cov_manifest["covariate_names"],
        )
        Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)

        if partition is not None:
            # --- Gated path: distributed, masked-before-softmax (no collect) ---
            # Assumption: cov_df for a gated combined-cohort fit is keyed
            # (person_id, source_cohort); source_cohort is always present.
            from charmpheno.omop.covariates import (
                corpus_mean_proportions_gated_from_covariate_df,
            )
            with _phase("gated corpus-mean prevalence (distributed, masked-before-softmax)"):
                prev = corpus_mean_proportions_gated_from_covariate_df(
                    cov_df, Gamma, partition)
            # Per-group patient counts for k-anon suppression.
            gc = (
                cov_df.groupBy("source_cohort")
                .agg(F.countDistinct("person_id").alias("n"))
                .collect()
            )
            gc = {r["source_cohort"]: int(r["n"]) for r in gc}
            log.info("STM: gated corpus_prevalence computed (K=%d, groups=%s).",
                     prev.shape[0], list(gc.keys()))
            return prev, gc, cov_df
        else:
            # --- Non-gated path: distributed treeReduce ---
            from charmpheno.omop.covariates import (
                corpus_mean_proportions_from_covariate_df,
            )
            with _phase("corpus-mean prior proportions (treeReduce)"):
                prev = corpus_mean_proportions_from_covariate_df(cov_df, Gamma)
            log.info("STM: faithful corpus_prevalence computed (K=%d).", prev.shape[0])
            return prev, None, None
    except Exception as exc:  # cosmetic-only: never fatal to the bundle build
        log.warning("STM: corpus_prevalence computation failed (%s); "
                    "using the intercept stand-in.", exc)
        return None, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=_HelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="path to the saved VIResult directory")
    parser.add_argument("--out-dir", default=None,
                        help="output dir for the 4 JSON files "
                             "(default: <checkpoint>/dashboard_bundle)")
    parser.add_argument("--model-class", choices=["lda", "hdp", "stm"], default="lda")
    parser.add_argument("--hdp-top-k", type=int, default=50,
                        help="top-K used HDP topics (ignored for LDA)")
    parser.add_argument("--cache-uri", default=None,
                        help="GCS/HDFS URI prefix for the covariate cache "
                             "(STM only). Enables the faithful corpus-mean "
                             "corpus_prevalence; without it the dashboard falls "
                             "back to the softmax(Gamma[intercept]) stand-in.")
    parser.add_argument("--allow-incomplete-bundle", action="store_true",
                        help="Downgrade the STM bundle-completeness check to a "
                             "warning: permit exporting a gated bundle missing "
                             "gating.json / covariate_schema.json (covariate-"
                             "cache miss) as a degraded ungated bundle.")
    parser.add_argument("--vocab-top-n", type=int, default=5000,
                        help="trim vocab to top-N codes by corpus_freq")
    parser.add_argument("--top-n-codes-for-npmi", type=int, default=20)
    parser.add_argument("--zip-name", default=None,
                        help="basename for the zip artifact (written as sibling "
                             "of --out-dir). Default: <out_dir_name>.zip "
                             "(e.g. dashboard_bundle.zip).")
    args = parser.parse_args(argv)

    cdr_env = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr_env and billing):
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    # spark_vi.core.result must be loaded before spark_vi.io to avoid a
    # circular import: spark_vi.io.__init__ -> io.export -> core.__init__
    # -> core.runner -> io.export (which is still initializing).
    import spark_vi.core.result as _spark_vi_core_result  # noqa: F401

    # Driver-side imports proven first.
    from charmpheno.omop import (
        DocSpec, cohort_metadata, load_omop_bigquery, to_bow_dataframe,
    )
    from charmpheno.export.corpus_stats import (
        compute_corpus_stats_from_bow_df,
        write_corpus_stats_sidecar,
    )
    from charmpheno.export.dashboard import (
        write_model_and_vocab_bundles,
        write_phenotypes_bundle,
        write_covariate_effects,
    )
    from charmpheno.export.model_adapter import adapt
    from spark_vi.io import load_result
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic import compute_npmi_coherence

    configure_logging(extra_loggers={"charmpheno": logging.INFO})
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint) / "dashboard_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A reused out_dir must not let a PRIOR build's JSONs masquerade as this run's
    # output: a covariate-cache miss that skips gating.json / covariate_schema.json
    # would otherwise leave the previous build's copies in place, silently defeating
    # the completeness guard and shipping a stale/mixed bundle (observed 2026-07-06).
    # Every build regenerates the bundle from scratch, so clear stale JSON outputs.
    for _stale_json in out_dir.glob("*.json"):
        _stale_json.unlink()
    print(f"[driver] checkpoint={args.checkpoint}", flush=True)
    print(f"[driver] out_dir={out_dir}", flush=True)
    print(f"[driver] model_class={args.model_class}", flush=True)

    with _phase("load checkpoint"):
        result = load_result(args.checkpoint)
        corpus = result.metadata.get("corpus_manifest")
        if corpus is None:
            raise SystemExit(
                "checkpoint metadata missing 'corpus_manifest'; re-fit "
                "with a current driver to regenerate."
            )
        vocab_list = result.metadata.get("vocab")
        if not vocab_list:
            raise SystemExit(
                "checkpoint metadata has no 'vocab'; re-fit needed."
            )
        doc_spec_manifest = corpus.get("doc_spec", {"name": "patient"})
        doc_spec = DocSpec.from_manifest(doc_spec_manifest)
        source_table = corpus.get("source_table", "condition_occurrence")
        cohort_name = corpus.get("cohort")
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"source_table={source_table}, "
              f"person_mod={corpus['person_mod']}, "
              f"cohort={cohort_name!r}", flush=True)
        print(f"[driver]   doc_spec: {doc_spec_manifest}", flush=True)
        print(f"[driver]   frozen vocab: {len(vocab_list)} terms", flush=True)

        if corpus["cdr"] != cdr_env:
            log.warning(
                "WORKSPACE_CDR (%s) differs from corpus_manifest['cdr'] (%s); "
                "using the checkpoint's cdr so the BOW is reproducible.",
                cdr_env, corpus["cdr"],
            )

    spark = make_spark_session("build_dashboard_cloud")

    try:
        with _phase("BQ load (OMOP)"):
            omop = load_omop_bigquery(
                spark=spark,
                cdr_dataset=corpus["cdr"],
                billing_project=billing,
                person_sample_mod=corpus["person_mod"],
                source_table=source_table,
                cohort=cohort_name,
            ).persist()
            n_rows = omop.count()
            print(f"[driver]   OMOP: {n_rows} rows", flush=True)

        with _phase(f"vectorize (frozen vocab, doc_spec={doc_spec.name})"):
            bow_df, vocab_map = to_bow_dataframe(
                omop, doc_spec=doc_spec, vocab=vocab_list,
            )
            print(f"[driver]   vocab size: {len(vocab_map)}", flush=True)

        is_stm = (args.model_class == "stm"
                  or result.metadata.get("model_class") == "stm")

        # Build the gating partition (None for non-gated STM and all LDA/HDP).
        # Guarded on topic_block_spec so non-gated checkpoints are unchanged.
        stm_partition = None
        stm_suppressed = frozenset()
        stm_gc = None
        tbs = corpus.get("topic_block_spec") if is_stm else None
        if tbs:
            from spark_vi.models.topic.partition import TopicBlockPartition
            stm_partition = TopicBlockPartition.from_dict(tbs)

        stm_corpus_prev = None
        stm_cov_df = None
        if is_stm:
            stm_corpus_prev, stm_gc, stm_cov_df = _stm_corpus_prevalence(
                spark, result=result, corpus=corpus,
                source_table=source_table, cohort_name=cohort_name,
                cache_uri=args.cache_uri, log=log,
                partition=stm_partition,
            )

        # Fail loud EARLY on a covariate-cache miss for a gated build -- before the
        # expensive corpus-stats / NPMI phases and immune to stale out_dir files
        # (unlike the end-of-build assert_stm_bundle_complete). A missing sidecar
        # skips every per-doc STM output (gating / covariate_schema / eta_scale /
        # theta_histogram / predictive_gain); aborting here prevents a silent
        # ungated bundle. --allow-incomplete-bundle is the intentional escape.
        assert_covariate_sidecar_present(
            is_stm=is_stm,
            gated=bool(tbs and stm_partition is not None),
            sidecar_present=stm_cov_df is not None,
            allow_incomplete=args.allow_incomplete_bundle,
            exp_hint=Path(args.checkpoint).name.split("-")[0] or None,
            log=log,
        )

        # For gated STM, compute suppression from per-group counts.
        if tbs and stm_gc is not None:
            from charmpheno.export.gating import suppressed_topic_ids
            k_thresh = int(corpus.get("min_patient_count", 20))
            stm_suppressed = suppressed_topic_ids(stm_partition, stm_gc, k_thresh)

        with _phase("adapter (model-class normalize)"):
            if is_stm:
                from charmpheno.export.model_adapter import adapt_stm as _adapt_stm
                export = _adapt_stm(result, corpus_prevalence=stm_corpus_prev,
                                    partition=stm_partition,
                                    suppressed=stm_suppressed)
            else:
                export = adapt(result, hdp_top_k=args.hdp_top_k)
            K_disp, V_full = export.beta.shape
            print(f"[driver]   K_display={K_disp} V_full={V_full}", flush=True)
            # Suppression threshold to REPORT as theta_histogram_min_count.
            # Defaults to the LDA fit-time value (compute_theta_aggregates
            # default = 20); the STM theta-histogram phase below overrides it to
            # this run's k-anon k_thresh, the threshold it suppresses at here.
            theta_hist_min_count = 20

        with _phase("concept name + domain lookup"):
            vocab_ids_int = [int(c) for c in vocab_list]
            concept_tbl = (
                spark.read.format("bigquery")
                .option("table", f"{corpus['cdr']}.concept")
                .option("parentProject", billing)
                .load()
                .where(F.col("concept_id").isin(vocab_ids_int))
                .select("concept_id", "concept_name", "domain_id")
                .dropDuplicates(["concept_id"])
                .collect()
            )
            descriptions: dict[int, str] = {
                int(r["concept_id"]): (r["concept_name"] or "") for r in concept_tbl
            }
            domains: dict[int, str] = {
                int(r["concept_id"]): (r["domain_id"] or "unknown").lower()
                for r in concept_tbl
            }
            print(f"[driver]   resolved {len(descriptions)} concept names, "
                  f"{len(domains)} domains", flush=True)

        with _phase("corpus stats"):
            # to_bow_dataframe emits 'features: SparseVector'; the corpus_stats
            # helper expects 'indices' + 'counts' array columns. Extract them
            # with small UDFs (same pattern as analysis/local/build_dashboard.py).
            _sv_indices = F.udf(
                lambda sv: sv.indices.tolist() if sv is not None else [],
                "array<int>",
            )
            _sv_counts = F.udf(
                lambda sv: [float(x) for x in sv.values] if sv is not None else [],
                "array<double>",
            )
            bow_df_stats = bow_df.select(
                _sv_indices(F.col("features")).alias("indices"),
                _sv_counts(F.col("features")).alias("counts"),
            ).persist()
            bow_df_kept = bow_df.persist()
            stats = compute_corpus_stats_from_bow_df(
                bow_df_stats, vocab_size=V_full, k=K_disp,
            )
            print(f"[driver]   n_docs={stats.corpus_size_docs} "
                  f"mean_codes={stats.mean_codes_per_doc:.2f}", flush=True)

        with _phase(f"NPMI (top_n={args.top_n_codes_for_npmi})"):
            top_n_npmi = min(args.top_n_codes_for_npmi, V_full)
            holdout_bow = bow_df_kept.rdd.map(BOWDocument.from_spark_row)
            report = compute_npmi_coherence(
                export.beta, holdout_bow, top_n=top_n_npmi,
            )
            npmi = report.per_topic_npmi.tolist()
            rated = ~np.isnan(report.per_topic_npmi)
            assert (report.per_topic_npmi[rated] >= -1.0).all(), "NPMI < -1"
            assert (report.per_topic_npmi[rated] <= 1.0).all(), "NPMI > 1"
            # Fraction of top-N pairs that contributed to each topic's mean
            # NPMI. Zero means "unrated" — no pairs cleared min_pair_count.
            pair_coverage = (
                report.per_topic_scored_pairs.astype(float)
                / float(report.per_topic_total_pairs)
            ).tolist()

        # bow_df_stats is no longer needed. bow_df_kept (== bow_df) and omop
        # stay persisted through the write-bundle phase: the eta_scale E-step and
        # the correlation.json write both reuse bow_df (no new corpus scan).
        bow_df_stats.unpersist()

        # eta_scale: held-out generative-variance scale c (calibration). HOISTED
        # here so it runs BEFORE the theta_histogram phase (which now infers the
        # display histogram at this calibrated scale) and before the write-bundle
        # correlation.json write (which SHIPS eta_scale). eta_scale / eta_scale_diag
        # are outer-scope vars (like stm_corpus_prev): the histogram phase consumes
        # eta_scale as `scale`, and the correlation.json write consumes both
        # precomputed vars instead of recomputing them.
        #
        # Held-out predictive-LL calibration (corpus_heldout_scale_sweep_gated_rdd,
        # HS-1/commit e22bcae): recovers the single concentration scale that the
        # unit-diagonal fitted R (ADR 0034) discards, so the dashboard can rescale
        # R -> Sigma_gen = eta_scale*R. This bounded grid sweep SUPERSEDES the
        # iterated pooled EM (corpus_eta_scale_gated_rdd), which is biased and
        # unstable: it has positive feedback in the scale direction (no trust region)
        # and ran away on the cluster (c: 3.6 -> 1116 -> 770918). The sweep scores
        # each grid c's held-out predictive LL; the raw grid ARGMAX is a quantized,
        # jittery point estimate (the curve is a broad, flat shelf -- LL differences
        # ~0.001-0.01 nats across c in roughly [2,12], within resampling noise -- so
        # argmax over a coarse grid drove a 5 -> 12 -> 8 refit wander). We SHIP a
        # smoothed reducer instead: a local quadratic fit in log c
        # (smooth_scale_log_quadratic) recovering a sub-grid, noise-averaged c* plus
        # a curvature-based SE, honestly large when the shelf is flat -- see that
        # function's docstring for the algorithm and its Numerical-Recipes/Brent +
        # delta-method citations. The grid is GEOMETRIC (even resolution in log c,
        # since c is a multiplicative scale). We sweep 3 holdout fractions (0.5
        # shipped; 0.8/0.95 probe robustness as the visible token set shrinks toward
        # the small-seed regime) and SHIP the smoothed c* at holdout=0.5. ENHANCEMENT
        # only: any failure (pre-Task-1 checkpoint, cache miss -> stm_cov_df is None,
        # E-step error) leaves eta_scale=None (dashboard falls back to unit R) and
        # the histogram falls back to scale=1.0. Reuses the already-loaded
        # bow_df (frozen fit vocab) + the sidecar cov_df -> no new scan.
        eta_scale = None
        eta_scale_diag = None
        # Iteration fast-path: BUILD_ETA_SCALE_OVERRIDE=<float> PINS c* and SKIPS
        # the ~13-min held-out-LL sweep. c* is stable (~4.6) across refits, and the
        # downstream phases (theta_histogram, predictive_gain, correlation.json)
        # consume only its value -- so pinning is exact for them and matches the
        # "calibrate once, then pin" pattern. Leave UNSET for a real calibration.
        _eta_override = os.environ.get("BUILD_ETA_SCALE_OVERRIDE")
        if _eta_override:
            try:
                eta_scale = float(_eta_override)
                eta_scale_diag = {"override": True, "value": eta_scale}
                print(f"[driver]   eta_scale: OVERRIDE={eta_scale} "
                      "(BUILD_ETA_SCALE_OVERRIDE set; skipping the held-out sweep)",
                      flush=True)
            except ValueError:
                print("[driver]   eta_scale: ignoring invalid "
                      f"BUILD_ETA_SCALE_OVERRIDE={_eta_override!r}", flush=True)
        if eta_scale is None and (is_stm and tbs and stm_partition is not None
                and "n_pairs" in result.global_params
                and stm_cov_df is not None):
            try:
                from spark_vi.mllib.topic.stm import (
                    corpus_heldout_scale_sweep_gated_rdd,
                    smooth_scale_log_quadratic,
                )
                from spark_vi.mllib.topic._common import (
                    _vector_to_stm_document,
                )
                stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                reference_id = 0 if stm_hardening.get("reference_topic") else None
                doc_df = bow_df.select("person_id", "features").join(
                    stm_cov_df.select(
                        "person_id", "source_cohort", "covariates"),
                    on="person_id", how="inner",
                )
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
                with _phase("eta_scale (held-out-LL calibration)"):
                    for hf in HOLDOUTS:
                        sweep = corpus_heldout_scale_sweep_gated_rdd(
                            doc_rdd, result.global_params, stm_partition,
                            c_grid=C_GRID, holdout_frac=hf,
                            reference=reference_id, seed=0,
                        )
                        smoothed_hf = smooth_scale_log_quadratic(sweep["lls"])
                        robustness[str(hf)] = smoothed_hf["c_star"]
                        robustness_argmax[str(hf)] = sweep["argmax_c"]
                        if hf == 0.5:
                            lls_shipped = {
                                str(k): float(v)
                                for k, v in sweep["lls"].items()
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

                # FLAGGED diagnostic (Task 6', off by default): re-runs BOTH the
                # MAP-plug-in sweep above and the Laplace-MC MARGINALIZED sweep
                # (Task 5, corpus_heldout_scale_sweep_gated_rdd(marginalize=True))
                # on the SAME sampled docs at the SAME 3 holdout fractions, and
                # records both curves + each estimator's residual drift (max-min
                # c* across holdouts -- ~0 for a well-behaved, prefix-independent
                # scale). This is the real-corpus decisive test for whether
                # marginalization's synthetic-K=60 bias inversion (exp 0046)
                # also inverts on the misspecified real corpus, where the MAP
                # estimate itself already drifts (4.58->3.65). Entirely
                # diagnostic: it does NOT change eta_scale or eta_scale_diag's
                # other fields, and its own try/except below ensures a failure
                # here can never blank the shipped scale (unlike the outer
                # except, which does exactly that on a real calibration
                # failure). Own try/except is required for that isolation.
                if os.environ.get("BUILD_MARGINALIZE_SCALE_DIAGNOSTIC"):
                    try:
                        _n_samp = int(os.environ.get(
                            "BUILD_MARGINALIZE_SCALE_SAMPLES", "64"))
                        _doc_frac = float(os.environ.get(
                            "BUILD_MARGINALIZE_SCALE_DOC_FRAC", "0.02"))
                        # Cost control: the marginalized sweep is ~n_samples x the
                        # MAP cost, so it runs on a SAMPLE of the corpus (both
                        # estimators on the SAME sampled docs for an
                        # apples-to-apples comparison). The full-corpus MAP
                        # robustness above is unaffected and still ships.
                        _sampled = doc_rdd.sample(
                            withReplacement=False, fraction=_doc_frac,
                            seed=0).cache()
                        _n_docs = _sampled.count()
                        _map_by, _marg_by = {}, {}
                        with _phase("eta_scale marginalized diagnostic "
                                    "(cluster test)"):
                            for hf in HOLDOUTS:
                                _m = corpus_heldout_scale_sweep_gated_rdd(
                                    _sampled, result.global_params,
                                    stm_partition, c_grid=C_GRID,
                                    holdout_frac=hf, reference=reference_id,
                                    seed=0, marginalize=False)
                                _g = corpus_heldout_scale_sweep_gated_rdd(
                                    _sampled, result.global_params,
                                    stm_partition, c_grid=C_GRID,
                                    holdout_frac=hf, reference=reference_id,
                                    seed=0, marginalize=True,
                                    n_samples=_n_samp)
                                _map_by[str(hf)] = smooth_scale_log_quadratic(
                                    _m["lls"])["c_star"]
                                _marg_by[str(hf)] = smooth_scale_log_quadratic(
                                    _g["lls"])["c_star"]
                        _sampled.unpersist()
                        eta_scale_diag["marginalized_diagnostic"] = (
                            build_marginalized_scale_diagnostic(
                                map_cstar_by_holdout=_map_by,
                                marg_cstar_by_holdout=_marg_by,
                                n_samples=_n_samp, n_docs_sampled=_n_docs,
                                c_grid=C_GRID, holdouts=HOLDOUTS))
                        _md = eta_scale_diag["marginalized_diagnostic"]
                        log.info(
                            "STM marginalized-scale diagnostic (sample n=%d, "
                            "S=%d): MAP drift=%.4f marg drift=%.4f; MAP "
                            "c*=%s marg c*=%s",
                            _n_docs, _n_samp, _md["map_residual_drift"],
                            _md["marg_residual_drift"],
                            _md["map_cstar_by_holdout"],
                            _md["marg_cstar_by_holdout"])
                    except Exception as _dexc:
                        # diagnostic-only: NEVER affects the shipped eta_scale
                        log.warning(
                            "STM marginalized-scale diagnostic failed (%s); "
                            "shipped eta_scale and bundle are UNAFFECTED.",
                            _dexc)
            except Exception as exc:  # enhancement-only: never fatal
                log.warning("STM: eta_scale held-out calibration failed (%s); "
                            "correlation.json omits eta_scale (dashboard falls "
                            "back to unit-diagonal R) and the theta "
                            "histogram falls back to scale=1.0.", exc)
                eta_scale = None
                eta_scale_diag = None
        elif (eta_scale is None and is_stm and tbs and stm_partition is not None
                and "n_pairs" in result.global_params):
            # Reached only because stm_cov_df is None (covariate cache miss / no
            # --cache-uri): eta_scale cannot be calibrated. Warn explicitly --
            # otherwise the cause is silent (eta_scale is omitted from
            # correlation.json and the theta histogram falls back to the unit
            # fit scale). Restores the diagnostic the pre-hoist try/except emitted.
            log.warning(
                "STM: covariate sidecar unavailable (cache miss / no --cache-uri); "
                "skipping eta_scale calibration -- correlation.json omits eta_scale "
                "and the theta histogram uses the unit fit scale. Run "
                "build-covariates first.")

        # concentration_heterogeneity diagnostic (FLAGGED, off by default):
        # runs the raw-vs-dedup concentration/burstiness gate
        # (corpus_concentration_heterogeneity_rdd, built in 00bc3e7/5b5cd1b)
        # on the real gated-STM corpus. This is a standalone diagnostic, NOT
        # part of the shipped bundle -- it is intentionally its OWN top-level
        # block (not nested inside the eta_scale try/except above) so it can
        # still run when BUILD_ETA_SCALE_OVERRIDE is set and skips that whole
        # section. Guarded on the same gated-STM + covariate-sidecar
        # preconditions the eta_scale block checks. Builds its OWN doc_rdd
        # (same assembly as the eta_scale / theta_histogram / predictive_gain
        # blocks) since the eta_scale block's doc_rdd may not exist under an
        # override. Never fatal: any failure only logs a warning and writes
        # nothing -- the shipped bundle and assert_stm_bundle_complete's
        # required-file check (which only checks for MISSING files) are
        # unaffected by the extra concentration_heterogeneity.json file this
        # writes when it succeeds.
        if os.environ.get("BUILD_CONCENTRATION_HETEROGENEITY_DIAGNOSTIC") and (
                is_stm and tbs and stm_partition is not None
                and stm_cov_df is not None):
            try:
                from spark_vi.mllib.topic.stm import (
                    corpus_concentration_heterogeneity_rdd,
                )
                from spark_vi.mllib.topic._common import _vector_to_stm_document

                _ch_frac = float(os.environ.get(
                    "BUILD_CONCENTRATION_HETEROGENEITY_DOC_FRAC", "0.05"))
                # Inference scale: use the deployed/calibrated eta_scale when
                # available (matches the scale downstream panels infer at);
                # fall back to the unit scale (c=1.0) when calibration did not
                # run/failed, same convention as theta_histogram's hist_scale
                # and predictive_gain's pg_scale above -- log which was used.
                _ch_c = float(eta_scale) if eta_scale else 1.0
                _stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                _ch_ref = 0 if _stm_hardening.get("reference_topic") else None

                # Self-contained STMDocument RDD (same assembly as the
                # eta_scale / theta_histogram / predictive_gain blocks): the
                # eta_scale block's doc_rdd is not guaranteed to exist here
                # (BUILD_ETA_SCALE_OVERRIDE skips building it), so rebuild
                # independently rather than depend on it.
                _ch_doc_df = bow_df.select("person_id", "features").join(
                    stm_cov_df.select(
                        "person_id", "source_cohort", "covariates"),
                    on="person_id", how="inner",
                )
                _ch_doc_rdd = _ch_doc_df.rdd.map(
                    lambda row: _vector_to_stm_document(
                        row, features_col="features",
                        covariates_col="covariates",
                        group_col="source_cohort",
                    )
                )
                log.info(
                    "STM: concentration-heterogeneity diagnostic inference "
                    "scale c=%.4f (%s).",
                    _ch_c, "calibrated eta_scale" if eta_scale else "unit fallback")
                with _phase("concentration-heterogeneity diagnostic"):
                    _ch = corpus_concentration_heterogeneity_rdd(
                        _ch_doc_rdd, result.global_params, stm_partition,
                        c=_ch_c, reference=_ch_ref, sample_frac=_ch_frac, seed=0,
                    )
                import json as _json
                (out_dir / "concentration_heterogeneity.json").write_text(
                    _json.dumps(_ch, indent=2))
                log.info(
                    "concentration-heterogeneity diagnostic (n=%d, "
                    "skipped=%d, c=%.3f, frac=%s): spread_ratio=%.3f "
                    "rank_corr=%.3f burstiness_corr=%.3f",
                    _ch["n_docs"], _ch["n_skipped"], _ch_c, _ch_frac,
                    _ch["spread_ratio_top_mass"], _ch["rank_corr_top_mass"],
                    _ch["burstiness_corr_top_mass"])
            except Exception as _chexc:  # diagnostic-only: never fatal
                log.warning(
                    "concentration-heterogeneity diagnostic failed (%s); "
                    "bundle UNAFFECTED.", _chexc)

        # theta_histogram: per-doc gated MAP theta distribution (the dashboard's
        # "topic mass distribution" panel). Plain-LDA writes per-doc theta
        # aggregates at fit time; the STM fit does not, so we compute them here
        # from the fitted checkpoint + corpus (a BUILD-STEP, like corpus_prevalence
        # / eta_scale). Runs AFTER the hoisted eta_scale phase (so it can infer at
        # the calibrated scale) and BEFORE the write-bundle phase (where
        # phenotypes.json is written). Builds its OWN doc_rdd -- a second per-doc
        # pass, adjacent to the eta_scale block's doc_rdd; they could be fused, kept
        # independent for now. Guarded on STM + gating (partition + covariate
        # sidecar present); enhancement-only: any failure leaves the two fields None
        # (dashboard hides the panel). Parity with analysis/local/build_dashboard.py.
        if is_stm and tbs and stm_partition is not None and stm_cov_df is not None:
            with _phase("theta_histogram (per-doc θ distribution)"):
                try:
                    from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd
                    from spark_vi.mllib.topic._common import _vector_to_stm_document
                    from charmpheno.export.theta_aggregates import (
                        compute_theta_aggregates,
                    )
                    from charmpheno.export.model_adapter import (
                        _parse_theta_histogram, _parse_theta_percentiles,
                    )
                    from dataclasses import replace as _dc_replace

                    stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                    theta_reference = (
                        0 if stm_hardening.get("reference_topic") else None)
                    k_thresh = int(corpus.get("min_patient_count", 20))

                    # Self-contained STMDocument RDD (same assembly as the
                    # eta_scale block: bow features + covariate sidecar +
                    # source_cohort gating group).
                    theta_doc_df = bow_df.select("person_id", "features").join(
                        stm_cov_df.select(
                            "person_id", "source_cohort", "covariates"),
                        on="person_id", how="inner",
                    )
                    theta_doc_rdd = theta_doc_df.rdd.map(
                        lambda row: _vector_to_stm_document(
                            row, features_col="features",
                            covariates_col="covariates",
                            group_col="source_cohort",
                        )
                    )
                    # Infer the display histogram at the CALIBRATED generation
                    # scale (eta_scale = c ~ 4.6) computed by the hoisted phase
                    # above, not the over-diffuse unit fit scale: the calibrated
                    # prior concentrates each patient's theta_hat onto the topics
                    # they actually express (honest prevalence). Falls back to
                    # scale=1.0 when calibration failed / eta_scale is None.
                    hist_scale = float(eta_scale) if eta_scale else 1.0
                    log.info(
                        "STM: theta histogram inferred at scale=%.4f (%s).",
                        hist_scale,
                        "calibrated eta_scale" if eta_scale else "unit fallback")
                    # 200k sample_cap is a heuristic driver-memory bound, not a
                    # literature value; corpus_theta_gated_rdd logs sampled N/frac.
                    theta_arr = corpus_theta_gated_rdd(
                        theta_doc_rdd, result.global_params, stm_partition,
                        reference=theta_reference, scale=hist_scale,
                        sample_cap=200_000, seed=0)
                    agg = compute_theta_aggregates(theta_arr, min_count=k_thresh)
                    kept = export.topic_indices.tolist()
                    # frozen dataclass -> replace(); do NOT touch corpus_prevalence.
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
                        "STM: computed per-doc theta histogram "
                        "(sampled_docs=%d, kept_topics=%d).",
                        theta_arr.shape[0], len(kept))
                    print(f"[driver]   theta_histogram sampled_docs="
                          f"{theta_arr.shape[0]} kept_topics={len(kept)}",
                          flush=True)
                except Exception as exc:  # enhancement-only: never fatal
                    log.warning(
                        "STM: per-doc theta histogram failed (%s); "
                        "phenotypes.json omits theta_histogram/theta_percentiles "
                        "(panel hidden).", exc)

        # predictive_gain: per-topic presence/depth/prominence aggregates (the
        # dashboard's predictive-gain view, spark_vi.mllib.topic.predictive_gain
        # Phase 2). Leave-one-topic-out held-out predictive gain Delta_k
        # answers "how much held-out predictive power does topic k actually
        # contribute", complementing theta_histogram's "how much MASS does a
        # patient put on topic k" view. Runs AFTER theta_histogram (same guard,
        # same self-contained doc_rdd assembly) and BEFORE the write-bundle
        # phase. Guarded on STM + gating (partition + covariate sidecar
        # present); enhancement-only: any failure leaves the six new
        # DashboardExport fields None (dashboard hides the panel). PROVISIONAL
        # schema -- see write_phenotypes_bundle's docstring; Phase-2 will
        # recalibrate prominence_range from observed_delta_range, logged here.
        # Uses fast=True (the warm-start Newton downdate, ~2x a single
        # inference pass) validated against the COLD oracle via a small-sample
        # audit (predictive_gain_downdate_audit) whose max_abs_overall is the
        # headline cold-reliability number, logged prominently below. Parity
        # with analysis/local/build_dashboard.py.
        pg_prominence_bin_edges = None
        pg_null_band = None
        pg_observed_delta_range = None
        pg_downdate_audit = None
        pg_scale = None
        pg_n_docs = None
        pg_smoothing = None
        if is_stm and tbs and stm_partition is not None and stm_cov_df is not None:
            with _phase("predictive_gain (presence/depth/prominence)"):
                try:
                    from spark_vi.mllib.topic.predictive_gain import (
                        corpus_predictive_gain_gated_rdd,
                        predictive_gain_downdate_audit,
                    )
                    from spark_vi.mllib.topic._common import _vector_to_stm_document
                    from dataclasses import replace as _dc_replace

                    stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                    pg_reference = (
                        0 if stm_hardening.get("reference_topic") else None)

                    # Self-contained STMDocument RDD (same assembly as the
                    # eta_scale / theta_histogram blocks: bow features +
                    # covariate sidecar + source_cohort gating group).
                    pg_doc_df = bow_df.select("person_id", "features").join(
                        stm_cov_df.select(
                            "person_id", "source_cohort", "covariates"),
                        on="person_id", how="inner",
                    )
                    pg_doc_rdd = pg_doc_df.rdd.map(
                        lambda row: _vector_to_stm_document(
                            row, features_col="features",
                            covariates_col="covariates",
                            group_col="source_cohort",
                        )
                    )
                    # Same calibrated-scale-or-unit-fallback convention as the
                    # theta_histogram block's hist_scale: Sigma_gen = c*R uses
                    # the held-out-LL calibrated eta_scale when available.
                    pg_scale = float(eta_scale) if eta_scale else 1.0
                    log.info(
                        "STM: predictive gain computed at scale=%.4f (%s).",
                        pg_scale,
                        "calibrated eta_scale" if eta_scale else "unit fallback")

                    # Corpus unigram (Task S2): activates predictive_gain's
                    # S1 background-smoothed predictive score p_S(w) =
                    # (1-eps)*(theta@beta)(w) + eps*m_w in place of the
                    # historical unsmoothed 1e-12-floor path. stats.code_marginals
                    # (computed above in the "corpus stats" phase) is already the
                    # length-V_full token-frequency vector -- normalize defensively
                    # (it should already sum to ~1) and length-check against the
                    # fitted beta's vocab width before trusting it.
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
                            # Floor with a small uniform mass so NO vocab token has
                            # zero backoff probability. A zero m_w defeats the
                            # smoother: log(eps*0) hits the numeric safety floor and
                            # Delta spikes exactly like the old 1e-12 problem. Any
                            # frozen-vocab code absent from THIS build's doc set has
                            # code_marginals == 0. Standard smoothing of the backoff:
                            # barely touches common tokens, bounds the rare ones.
                            _pgV = pg_marginal.shape[0]
                            pg_marginal = 0.99 * pg_marginal + 0.01 / _pgV
                    # print (not log.info) so the smoother status is ALWAYS
                    # visible: the builder's own logger sits at WARNING, so a
                    # log.info here is silently dropped -- which left us unable to
                    # tell whether the smoother engaged on the cluster.
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
                        pg_doc_rdd, result.global_params, stm_partition,
                        c=pg_scale, reference=pg_reference, fast=True,
                        sample_cap=200_000, seed=0,
                        marginal=pg_marginal, smoothing_lambda=1.0,
                    )

                    # Cold-vs-fast downdate reliability audit on a small
                    # in-memory sample -- own try/except so an audit failure
                    # cannot drop the (already computed) main aggregates.
                    try:
                        pg_audit_docs = pg_doc_rdd.takeSample(False, 50, seed=0)
                        pg_audit_raw = predictive_gain_downdate_audit(
                            pg_audit_docs, result.global_params, stm_partition,
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
                            "STM: predictive-gain downdate audit "
                            "max_abs_overall=%.6f (n_docs_audited=%d) -- "
                            "cold-vs-fast (fast=True) reliability gate.",
                            pg_downdate_audit["max_abs_overall"],
                            pg_downdate_audit["n_docs_audited"])
                    except Exception as audit_exc:
                        log.warning(
                            "STM: predictive-gain downdate audit failed (%s); "
                            "phenotypes.json omits the downdate_audit "
                            "diagnostic (main aggregates unaffected).",
                            audit_exc)
                        pg_downdate_audit = None

                    kept = export.topic_indices.tolist()
                    # Export-boundary headline choice: the dashboard's `presence`
                    # is the permuted-null test (library `presence_vs_null`,
                    # "statistically present"); the looser beats-zero fraction
                    # (library `presence`) ships as the `presence_beats_zero`
                    # diagnostic so the UI can show both and reveal whether the
                    # null actually collapsed to ~0 on the real corpus.
                    export = _dc_replace(
                        export,
                        presence=pg["presence_vs_null"][kept],
                        presence_beats_zero=pg["presence"][kept],
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
                        "STM: computed predictive-gain aggregates "
                        "(n_docs=%d, kept_topics=%d, observed_delta_range=%s).",
                        pg_n_docs, len(kept), pg_observed_delta_range)
                    print(f"[driver]   predictive_gain n_docs={pg_n_docs} "
                          f"kept_topics={len(kept)}", flush=True)
                except Exception as exc:  # enhancement-only: never fatal
                    log.warning(
                        "STM: predictive-gain aggregation failed (%s); "
                        "phenotypes.json omits the predictive_gain object "
                        "(panel hidden).", exc)
                    pg_prominence_bin_edges = None
                    pg_null_band = None
                    pg_observed_delta_range = None
                    pg_downdate_audit = None
                    pg_scale = None
                    pg_n_docs = None
                    pg_smoothing = None

        with _phase("write bundle"):
            v_disp = write_model_and_vocab_bundles(
                out_dir=out_dir,
                beta=export.beta, alpha=export.alpha,
                vocab_ids=vocab_list,
                descriptions=descriptions, domains=domains,
                code_marginals=stats.code_marginals,
                top_n=args.vocab_top_n,
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
            # write_phenotypes_bundle's docstring): None when the phase above
            # never ran or failed (export.presence etc. stay unset), in which
            # case write_phenotypes_bundle omits the whole "predictive_gain"
            # key (byte-unchanged bundle). NaN -> None, same convention as
            # theta_histogram.
            def _nan_to_none(arr):
                return [None if np.isnan(v) else float(v) for v in arr.tolist()]

            if export.presence is not None:
                pg_presence = _nan_to_none(export.presence)
                pg_presence_beats_zero = (
                    _nan_to_none(export.presence_beats_zero)
                    if export.presence_beats_zero is not None else None)
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
                pg_presence_beats_zero = None
                pg_mean_gain = None
                pg_depth = None
                pg_length_corr = None
                pg_dedup_gain = None
                pg_prominence_hist_json = None

            write_phenotypes_bundle(
                out_dir / "phenotypes.json",
                npmi=npmi,
                pair_coverage=pair_coverage,
                corpus_prevalence=export.corpus_prevalence.tolist(),
                theta_histogram=hist,
                theta_percentiles=pct,
                topic_indices=export.topic_indices.tolist(),
                min_count=theta_hist_min_count,
                labels=None,
                presence=pg_presence,
                presence_beats_zero=pg_presence_beats_zero,
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
            if is_stm:
                import json as _json
                Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)
                covariate_manifest = result.metadata["covariate_manifest"]
                covariate_names = covariate_manifest["covariate_names"]
                kept_ids = [int(i) for i in export.topic_indices]

                # --- Gating outputs (guarded on topic_block_spec) ---
                if tbs and stm_gc is not None:
                    from charmpheno.export.gating import build_gating_json
                    k_thresh = int(corpus.get("min_patient_count", 20))
                    gating = build_gating_json(
                        stm_partition, stm_gc, k_thresh, kept_ids)
                    (out_dir / "gating.json").write_text(
                        _json.dumps(gating, indent=2))
                    log.info("STM: wrote gating.json (groups=%s, kept_topics=%d)",
                             gating["groups"], len(kept_ids))
                    print(f"[driver]   wrote gating.json "
                          f"(groups={gating['groups']}, "
                          f"kept_topics={len(kept_ids)})", flush=True)
                elif tbs:
                    # Gated checkpoint but no covariate data (cache miss / no
                    # --cache-uri): gating.json is deliberately NOT written. A
                    # gating.json without covariate_effects would render a group
                    # selector that cannot drive masked prevalence, so the
                    # dashboard degrades to the ungated view instead. Warn so the
                    # operator knows gating outputs were skipped, not lost.
                    log.warning("STM: topic_block_spec present but no covariate "
                                "data (cache miss); gating.json/covariate outputs "
                                "skipped. Dashboard renders ungated. Provide "
                                "--cache-uri to enable gating.")

                # correlation.json: logistic-normal topic correlation R + identified mask
                if tbs and stm_partition is not None:
                    if "n_pairs" in result.global_params:
                        from spark_vi.models.topic._linalg import topic_correlation_identified
                        from charmpheno.export.correlation import build_correlation_json

                        Sigma_corr = result.global_params["Sigma"]
                        n_pairs = result.global_params["n_pairs"]
                        # min_pair_support may be at top level (newer
                        # get_metadata) or nested under stm_hardening (models
                        # saved before that change); check both so the export
                        # mask floor always matches the fit.
                        stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                        mps = int(result.metadata.get("min_pair_support")
                                  or stm_hardening.get("min_pair_support", 1))
                        reference_id = 0 if stm_hardening.get("reference_topic") else None
                        R, ident = topic_correlation_identified(Sigma_corr, n_pairs, mps)

                        # eta_scale / eta_scale_diag were computed by the HOISTED
                        # eta_scale phase (before the theta_histogram phase, so the
                        # display histogram could infer at the calibrated scale).
                        # We simply SHIP the precomputed values here -- no recompute.
                        # They are None if that phase's guard failed or its
                        # calibration raised, in which case the key is omitted and
                        # the dashboard falls back to unit R.
                        corr = build_correlation_json(
                            R, ident, n_pairs, stm_partition, kept_ids,
                            reference_id=reference_id, eta_scale=eta_scale,
                            eta_scale_diagnostic=eta_scale_diag)
                        (out_dir / "correlation.json").write_text(
                            _json.dumps(corr, indent=2))
                        log.info("STM: wrote correlation.json (topics=%d, "
                                 "min_pair_support=%d)", len(kept_ids), mps)
                        print(f"[driver]   wrote correlation.json "
                              f"(topics={len(kept_ids)})", flush=True)
                    else:
                        log.warning("STM: saved model lacks 'n_pairs' in "
                                    "global_params (pre-Task-1 checkpoint); "
                                    "skipping correlation.json.")

                # covariate_effects.json: subset Gamma columns to kept topics.
                write_covariate_effects(
                    out_dir=out_dir, Gamma=Gamma[:, kept_ids],
                    covariate_names=covariate_names,
                    K=len(kept_ids), P=Gamma.shape[0],
                )
                print(f"[driver]   wrote covariate_effects.json (K={len(kept_ids)}, "
                      f"P={Gamma.shape[0]})", flush=True)
                _write_covariate_schema(
                    spark, result=result, corpus=corpus,
                    source_table=source_table, cohort_name=cohort_name,
                    cache_uri=args.cache_uri, out_dir=out_dir, log=log,
                )
                # Fail loud if the covariate-dependent outputs weren't written
                # (cache miss / no --cache-uri): a gated bundle without
                # gating.json + covariate_schema.json renders ungated and is a
                # silent degradation, not a complete bundle.
                assert_stm_bundle_complete(
                    out_dir, gated=bool(tbs),
                    allow_incomplete=args.allow_incomplete_bundle, log=log,
                )
            write_corpus_stats_sidecar(
                stats, out_dir / "corpus_stats.json", v_displayed=v_disp,
                cohort=cohort_metadata(cohort_name),
            )
            print(f"[driver]   wrote 4 files to {out_dir} "
                  f"(V_disp={v_disp} K_disp={K_disp})", flush=True)

        # Deferred from before the write-bundle phase so the eta_scale E-step join
        # could reuse the persisted bow_df / omop (no second full-corpus scan).
        bow_df_kept.unpersist()
        omop.unpersist()

        with _phase("zip bundle"):
            zip_path = (
                out_dir.parent / args.zip_name if args.zip_name
                else out_dir.with_suffix(".zip")
            )
            # Local path required: zipfile can't write through the GCS FUSE
            # mount layer reliably (random-access writes). Stage in /tmp.
            tmp_zip = Path("/tmp") / zip_path.name
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                # Base 4-file bundle (always written).
                for f in ("model.json", "vocab.json",
                          "phenotypes.json", "corpus_stats.json"):
                    zf.write(out_dir / f, arcname=f)
                # Optional gated / correlation outputs: include whichever were
                # written so the downloadable zip is the COMPLETE bundle.
                # gating.json + covariate_* need the covariate cache;
                # correlation.json needs a gated fit with persisted n_pairs;
                # concentration_heterogeneity.json is the off-by-default
                # dedup/burstiness diagnostic (BUILD_CONCENTRATION_HETEROGENEITY_DIAGNOSTIC).
                for f in ("gating.json", "covariate_schema.json",
                          "covariate_effects.json", "correlation.json",
                          "concentration_heterogeneity.json"):
                    p = out_dir / f
                    if p.exists():
                        zf.write(p, arcname=f)
                        print(f"[driver]   zip: +{f}", flush=True)
            # Now copy the staged zip to the final destination (which may be
            # a GCS-mounted path that accepts sequential writes).
            import shutil
            shutil.copyfile(tmp_zip, zip_path)
            tmp_zip.unlink()
            print(f"[driver]   zipped -> {zip_path}", flush=True)

        print("[driver] BUILD DASHBOARD CLOUD PASSED", flush=True)
        # The bundle + zip are fully written above. Mark THIS run's output
        # explicitly (path + mtime) so a captured log is unambiguous about which
        # run produced the artifact, then hard-exit: SparkContext teardown on
        # YARN can hang for minutes AFTER the bundle is done, which previously
        # looked like a failed build and led to a stale download. Once this line
        # prints, the zip is safe to download.
        import os as _os
        import time as _t
        try:
            _mt = _t.strftime("%Y-%m-%d %H:%M:%S",
                              _t.localtime(_os.path.getmtime(zip_path)))
            print(f"[driver] BUNDLE WRITTEN: {zip_path} (mtime {_mt})", flush=True)
        except Exception:
            pass
        _os._exit(0)   # skip the hang-prone YARN teardown; bundle already written
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
