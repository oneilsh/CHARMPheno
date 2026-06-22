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

The 4 files (model.json, vocab.json, phenotypes.json, corpus_stats.json)
land in --out-dir and are zipped into <out-dir>.zip alongside.
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


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    pass


def _quant_col(arr, idx: int):
    """Return a named temp-column name for use with approxQuantile.

    approxQuantile requires a column name string, not a column expression, so
    we materialise the array element into a named column on a cached DataFrame
    and return that name. Callers must call this on the *same* arr DataFrame
    they pass to approxQuantile.
    """
    col_name = f"_q_{idx}"
    # Re-use the column if already present (idempotent per idx).
    if col_name not in arr.columns:
        arr.__class__ = arr.__class__  # no-op; real add happens below
    return col_name


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
        from _covariates_cache import compute_cache_key, try_load
        from charmpheno.export.covariate_schema import build_covariate_schema

        cov_manifest = result.metadata["covariate_manifest"]
        key = compute_cache_key(
            covariate_formula=cov_manifest["covariate_formula"],
            person_mod=corpus["person_mod"], cdr=corpus["cdr"],
            source_table=source_table, cohort=cohort_name,
        )
        cached = try_load(spark, cache_uri, key)
        if cached is None:
            log.warning("STM: covariate-cache MISS; covariate_schema.json not written.")
            return
        cov_df, model_spec, covariate_names = cached
        continuous_cols = list(cov_manifest.get("continuous_cols", []))
        k = int(corpus.get("min_patient_count", 20))

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
            idx = name_idx[var]
            arr_with_col, col_name = _quant_col_df(arr, idx)
            q = arr_with_col.approxQuantile(col_name, [0.05, 0.5, 0.95], 0.01)
            continuous_stats[var] = tuple(round(v) for v in q)

        # Levels + reference from the fitted ModelSpec.
        categorical_levels = _categorical_levels_from_spec(
            model_spec, covariate_names=covariate_names,
        )

        schema = build_covariate_schema(
            covariate_names=covariate_names, continuous_cols=continuous_cols,
            categorical_levels=categorical_levels, level_counts=level_counts,
            continuous_stats=continuous_stats, k=k,
        )
        (out_dir / "covariate_schema.json").write_text(json.dumps(schema, indent=2))
        log.info("STM: wrote covariate_schema.json (controls=%d, unsupported=%d)",
                 len(schema["controls"]), len(schema["unsupported"]))
        print("[driver]   covariate_schema:", json.dumps(schema, indent=2), flush=True)
    except Exception as exc:  # cosmetic-only; never fail the bundle build
        log.warning("STM: covariate_schema derivation failed (%s); skipping.", exc)


def _stm_corpus_prevalence(spark, *, result, corpus, source_table,
                           cohort_name, cache_uri, log):
    """Faithful corpus-mean alpha-equivalent for STM, or None to fall back.

    Reloads the covariate sidecar from its cache (key recomputed from the
    checkpoint's covariate_manifest + corpus_manifest) and reduces it with
    corpus_mean_proportions_from_covariate_df (distributed treeReduce; only a
    K-vector reaches the driver). Returns None — leaving adapt_stm on its
    softmax(Gamma[intercept]) stand-in — when no cache_uri is supplied, the
    cache misses, or anything raises. The quantity is cosmetic (the dashboard's
    "default topic proportion" widget), so it must never abort the bundle build.
    """
    if not cache_uri:
        log.warning("STM: no --cache-uri; corpus_prevalence uses the "
                    "softmax(Gamma[intercept]) stand-in.")
        return None
    try:
        from _covariates_cache import compute_cache_key, try_load
        from charmpheno.omop.covariates import (
            corpus_mean_proportions_from_covariate_df,
        )

        cov_manifest = result.metadata["covariate_manifest"]
        key = compute_cache_key(
            covariate_formula=cov_manifest["covariate_formula"],
            person_mod=corpus["person_mod"],
            cdr=corpus["cdr"],
            source_table=source_table,
            cohort=cohort_name,
        )
        with _phase(f"covariates-cache lookup ({cache_uri}/{key})"):
            cached = try_load(spark, cache_uri, key)
        if cached is None:
            log.warning("STM: covariate-cache MISS (%s/%s); corpus_prevalence "
                        "uses the intercept stand-in.", cache_uri, key)
            return None
        cov_df, _spec, _names = cached
        Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)
        with _phase("corpus-mean prior proportions (treeReduce)"):
            prev = corpus_mean_proportions_from_covariate_df(cov_df, Gamma)
        log.info("STM: faithful corpus_prevalence computed (K=%d).", prev.shape[0])
        return prev
    except Exception as exc:  # cosmetic-only: never fatal to the bundle build
        log.warning("STM: corpus_prevalence computation failed (%s); "
                    "using the intercept stand-in.", exc)
        return None


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
        adapt_stm as dashboard_adapt_stm,
    )
    from charmpheno.export.model_adapter import adapt
    from spark_vi.io import load_result
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic import compute_npmi_coherence

    configure_logging(extra_loggers={"charmpheno": logging.INFO})
    log = logging.getLogger(__name__)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint) / "dashboard_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
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
        stm_corpus_prev = (
            _stm_corpus_prevalence(
                spark, result=result, corpus=corpus,
                source_table=source_table, cohort_name=cohort_name,
                cache_uri=args.cache_uri, log=log,
            )
            if is_stm else None
        )

        with _phase("adapter (model-class normalize)"):
            export = adapt(result, hdp_top_k=args.hdp_top_k,
                           stm_corpus_prevalence=stm_corpus_prev)
            K_disp, V_full = export.beta.shape
            print(f"[driver]   K_display={K_disp} V_full={V_full}", flush=True)

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

        bow_df_kept.unpersist()
        bow_df_stats.unpersist()
        omop.unpersist()

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

            write_phenotypes_bundle(
                out_dir / "phenotypes.json",
                npmi=npmi,
                pair_coverage=pair_coverage,
                corpus_prevalence=export.corpus_prevalence.tolist(),
                theta_histogram=hist,
                theta_percentiles=pct,
                topic_indices=export.topic_indices.tolist(),
                labels=None,
            )
            if is_stm:
                Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)
                covariate_manifest = result.metadata["covariate_manifest"]
                covariate_names = covariate_manifest["covariate_names"]
                dashboard_adapt_stm(
                    out_dir=out_dir, Gamma=Gamma,
                    covariate_names=covariate_names,
                    K=Gamma.shape[1], P=Gamma.shape[0],
                )
                print(f"[driver]   wrote covariate_effects.json (K={Gamma.shape[1]}, "
                      f"P={Gamma.shape[0]})", flush=True)
                _write_covariate_schema(
                    spark, result=result, corpus=corpus,
                    source_table=source_table, cohort_name=cohort_name,
                    cache_uri=args.cache_uri, out_dir=out_dir, log=log,
                )
            write_corpus_stats_sidecar(
                stats, out_dir / "corpus_stats.json", v_displayed=v_disp,
                cohort=cohort_metadata(cohort_name),
            )
            print(f"[driver]   wrote 4 files to {out_dir} "
                  f"(V_disp={v_disp} K_disp={K_disp})", flush=True)

        with _phase("zip bundle"):
            zip_path = (
                out_dir.parent / args.zip_name if args.zip_name
                else out_dir.with_suffix(".zip")
            )
            # Local path required: zipfile can't write through the GCS FUSE
            # mount layer reliably (random-access writes). Stage in /tmp.
            tmp_zip = Path("/tmp") / zip_path.name
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in ("model.json", "vocab.json",
                          "phenotypes.json", "corpus_stats.json"):
                    zf.write(out_dir / f, arcname=f)
            # Now copy the staged zip to the final destination (which may be
            # a GCS-mounted path that accepts sequential writes).
            import shutil
            shutil.copyfile(tmp_zip, zip_path)
            tmp_zip.unlink()
            print(f"[driver]   zipped -> {zip_path}", flush=True)

        print("[driver] BUILD DASHBOARD CLOUD PASSED", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
