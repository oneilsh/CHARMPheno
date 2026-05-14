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
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[driver]   %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("spark_vi").setLevel(logging.INFO)
    logging.getLogger("charmpheno").setLevel(logging.INFO)


@contextmanager
def _phase(name: str):
    print(f"[driver] >>> {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[driver] <<< {name}: {time.perf_counter() - t0:.1f}s",
              flush=True)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=_HelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="path to the saved VIResult directory")
    parser.add_argument("--out-dir", default=None,
                        help="output dir for the 4 JSON files "
                             "(default: <checkpoint>/dashboard_bundle)")
    parser.add_argument("--model-class", choices=["lda", "hdp"], default="lda")
    parser.add_argument("--hdp-top-k", type=int, default=50,
                        help="top-K used HDP topics (ignored for LDA)")
    parser.add_argument("--vocab-top-n", type=int, default=5000,
                        help="trim vocab to top-N codes by corpus_freq")
    parser.add_argument("--top-n-codes-for-npmi", type=int, default=20)
    parser.add_argument("--junk-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    cdr_env = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr_env and billing):
        print("ERROR: WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set in env. "
              "Run the workspace setup notebook (or `source ~/.bashrc`) first.",
              file=sys.stderr)
        return 1

    # Driver-side imports proven first.
    from charmpheno.omop import DocSpec, load_omop_bigquery, to_bow_dataframe
    from charmpheno.export.corpus_stats import (
        compute_corpus_stats_from_bow_df,
        write_corpus_stats_sidecar,
    )
    from charmpheno.export.dashboard import (
        write_model_and_vocab_bundles,
        write_phenotypes_bundle,
    )
    from charmpheno.export.model_adapter import adapt
    from spark_vi.io import load_result
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic import compute_npmi_coherence

    _configure_logging()
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
        print(f"[driver]   corpus_manifest: cdr={corpus['cdr']}, "
              f"source_table={source_table}, "
              f"person_mod={corpus['person_mod']}", flush=True)
        print(f"[driver]   doc_spec: {doc_spec_manifest}", flush=True)
        print(f"[driver]   frozen vocab: {len(vocab_list)} terms", flush=True)

        if corpus["cdr"] != cdr_env:
            log.warning(
                "WORKSPACE_CDR (%s) differs from corpus_manifest['cdr'] (%s); "
                "using the checkpoint's cdr so the BOW is reproducible.",
                cdr_env, corpus["cdr"],
            )

    spark = SparkSession.builder.appName("build_dashboard_cloud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    from _log_utils import quiet_spot_reclamation
    quiet_spot_reclamation(spark)
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    try:
        with _phase("BQ load (OMOP)"):
            omop = load_omop_bigquery(
                spark=spark,
                cdr_dataset=corpus["cdr"],
                billing_project=billing,
                person_sample_mod=corpus["person_mod"],
                source_table=source_table,
            ).persist()
            n_rows = omop.count()
            print(f"[driver]   OMOP: {n_rows} rows", flush=True)

        with _phase(f"vectorize (frozen vocab, doc_spec={doc_spec.name})"):
            bow_df, vocab_map = to_bow_dataframe(
                omop, doc_spec=doc_spec, vocab=vocab_list,
            )
            print(f"[driver]   vocab size: {len(vocab_map)}", flush=True)

        with _phase("adapter (model-class normalize)"):
            export = adapt(result, hdp_top_k=args.hdp_top_k)
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

        bow_df_kept.unpersist()
        bow_df_stats.unpersist()
        omop.unpersist()

        with _phase("write bundle"):
            # The writer normalizes lambda by row sum; the adapter's beta is
            # already row-stochastic, so any positive scalar cancels.
            pseudo_lambda = export.beta * 1.0e6
            v_disp = write_model_and_vocab_bundles(
                out_dir=out_dir,
                lambda_=pseudo_lambda, alpha=export.alpha,
                vocab_ids=vocab_list,
                descriptions=descriptions, domains=domains,
                code_marginals=stats.code_marginals,
                top_n=args.vocab_top_n,
            )
            write_phenotypes_bundle(
                out_dir / "phenotypes.json",
                npmi=npmi,
                corpus_prevalence=export.corpus_prevalence.tolist(),
                topic_indices=export.topic_indices.tolist(),
                labels=None,
                junk_threshold=args.junk_threshold,
            )
            write_corpus_stats_sidecar(
                stats, out_dir / "corpus_stats.json", v_displayed=v_disp,
            )
            print(f"[driver]   wrote 4 files to {out_dir} "
                  f"(V_disp={v_disp} K_disp={K_disp})", flush=True)

        with _phase("zip bundle"):
            zip_path = out_dir.with_suffix(".zip")
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
