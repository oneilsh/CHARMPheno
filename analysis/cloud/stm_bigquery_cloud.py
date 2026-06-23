"""STM (prevalence-only) fit driver — analogous to lda_bigquery_cloud.py.

Loads corpus + covariates from caches (or rebuilds from BQ), broadcast-joins
by person_id, constructs StreamingSTM (Path A), runs VIRunner.fit, and
saves the augmented STMModel with corpus_manifest + covariate_manifest in
metadata.

Decision context: docs/superpowers/specs/2026-05-29-stm-prevalence-design.md
                  docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session
from _corpus_load import load_or_build_corpus
from _covariates_load import load_or_build_covariates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STM fit driver (prevalence-only)")
    # Mirror LDA driver flags for shared params.
    p.add_argument("--cdr", required=True,
                   help="Fully-qualified BQ dataset '<project>.<dataset>'.")
    p.add_argument("--billing", required=True,
                   help="GCP billing project for BQ jobs.")
    p.add_argument("--source-table", default="condition_era",
                   choices=("condition_occurrence", "condition_era"),
                   help="OMOP fact table to read (default: condition_era).")
    p.add_argument("--cohort", default=None,
                   help="Optional cohort filter name (e.g. first_cancer_year).")
    p.add_argument("--prior-obs-days", type=int, default=365,
                   help="Prior-observation lookback (days) for the cohort index "
                        "date. 365 (default) requires a year of pre-index "
                        "coverage; 0 drops the lookback, admitting prevalent "
                        "cases. Keys the corpus + covariate caches.")
    p.add_argument("--person-mod", type=int, default=10,
                   help="Deterministic person sample: keep MOD(person_id, N)==0.")
    p.add_argument("--vocab-size", type=int, default=10_000,
                   help="Maximum vocabulary size after top-K pruning.")
    p.add_argument("--min-df", type=int, default=20,
                   help="Minimum document frequency for vocabulary inclusion.")
    p.add_argument("--min-patient-count", type=int, default=20,
                   help="Minimum patient count for vocabulary inclusion.")
    p.add_argument("--doc-spec", default="patient_year",
                   help="Document granularity spec name (default: patient_year).")
    p.add_argument("--doc-min-length", type=int, default=20,
                   help="Minimum document token count (shorter docs dropped).")
    p.add_argument("--cache-uri", default=None,
                   help="GCS/HDFS URI prefix for the corpus + covariates caches.")
    # STM-specific flags.
    p.add_argument("--K", type=int, default=40,
                   help="Number of latent topics.")
    p.add_argument("--max-iter", type=int, default=20,
                   help="Maximum number of SVI iterations.")
    p.add_argument("--save-interval", type=int, default=5,
                   help="Checkpoint interval in iterations (requires --cache-uri).")
    p.add_argument("--covariate-formula", required=True,
                   help="R-style prevalence formula, e.g. '~ C(sex) + age'. Required.")
    p.add_argument("--categorical-cols", required=True,
                   help="Comma-separated categorical column names in the formula.")
    p.add_argument("--continuous-cols", required=True,
                   help="Comma-separated continuous column names in the formula.")
    p.add_argument("--sigma-init", type=float, default=1.0,
                   help="Initial diagonal Sigma for the topic-covariate prior.")
    p.add_argument("--sigma-ridge", type=float, default=1e-6,
                   help="Ridge regularisation added to Sigma diagonal.")
    p.add_argument("--lbfgs-max-iter", type=int, default=50,
                   help="Max iterations for the per-doc L-BFGS optimiser.")
    p.add_argument("--lbfgs-tol", type=float, default=1e-4,
                   help="Convergence tolerance for the per-doc L-BFGS optimiser.")
    # Mini-batch SVI step-size flags (Robbins-Monro schedule).
    p.add_argument("--subsampling-rate", type=float, default=0.2,
                   help="Mini-batch fraction per SVI iteration (0, 1].")
    p.add_argument("--tau0", type=float, default=64.0,
                   help="Robbins-Monro delay parameter rho_t=(tau0+t)^-kappa.")
    p.add_argument("--kappa", type=float, default=0.7,
                   help="Robbins-Monro decay exponent (0.5, 1] for convergence.")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Optional random seed for reproducibility (best-effort).")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the saved STMModel.")
    p.add_argument("--resume-from", default="",
                   help="Path to a previously-saved STMModel dir; empty "
                        "(default) = fresh fit. When set, the fit loads its "
                        "global_params + n_iterations and continues the SVI "
                        "schedule, so --max-iter is ADDITIONAL iterations on "
                        "top of the loaded count. Resume only with the same "
                        "corpus + covariate formula (shapes must match).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.continuous_cols.split(",") if c.strip()]

    with make_spark_session(app_name="stm-fit") as spark:
        # --- Corpus load ---
        from charmpheno.omop import doc_spec_from_cli

        doc_spec = doc_spec_from_cli(args.doc_spec, min_doc_length=args.doc_min_length)

        with _phase("corpus load"):
            bow_df, vocab_map, name_by_id = load_or_build_corpus(
                spark,
                doc_spec=doc_spec, cdr=args.cdr, billing=args.billing,
                source_table=args.source_table, person_mod=args.person_mod,
                vocab_size=args.vocab_size, min_df=args.min_df,
                min_patient_count=args.min_patient_count,
                cache_uri=args.cache_uri, cohort=args.cohort,
                prior_obs_days=args.prior_obs_days,
            )

        composite = "source_cohort" in cat_cols

        if composite:
            # doc_id == "{source_cohort}:{person_id}"; recover the label.
            bow_df = bow_df.withColumn(
                "source_cohort",
                F.split(F.col("doc_id"), ":").getItem(0),
            )
            labels = bow_df.select("person_id", "source_cohort").distinct()
            cov_key_cols = ["person_id", "source_cohort"]
            join_on = ["person_id", "source_cohort"]
        else:
            labels = None
            cov_key_cols = ["person_id"]
            join_on = "person_id"

        # --- Diagnostics: corpus doc counts + per-cohort breakdown ---
        # Answers "is the doc count right, or is the join dropping docs?" The
        # per-source_cohort split is the combined-cohort validation readout.
        with _phase("corpus doc-count diagnostics"):
            n_bow = bow_df.count()
            print(f"[driver]   corpus docs (pre-join) = {n_bow}", flush=True)
            if composite:
                for r in (bow_df.groupBy("source_cohort").count()
                          .orderBy("source_cohort").collect()):
                    print(f"[driver]     source_cohort={r['source_cohort']}: "
                          f"{r['count']} docs", flush=True)
                pc = bow_df.select("person_id", "source_cohort").distinct()
                n_persons = pc.select("person_id").distinct().count()
                n_comorbid = (pc.groupBy("person_id").count()
                              .where(F.col("count") > 1).count())
                print(f"[driver]   distinct persons = {n_persons}, "
                      f"comorbid (in both cohorts) = {n_comorbid}", flush=True)

        # --- Person table load (source of covariates) ---
        from charmpheno.omop import load_person_table

        with _phase("person table load"):
            person_df = load_person_table(
                spark=spark,
                cdr_dataset=args.cdr,
                billing_project=args.billing,
                person_sample_mod=args.person_mod,
                cohort=args.cohort,
            )
            if composite:
                person_df = person_df.join(labels, on="person_id", how="inner")

        # --- Covariates load ---
        with _phase("covariates load"):
            cov_df, model_spec, covariate_names = load_or_build_covariates(
                spark,
                person_df=person_df,
                covariate_formula=args.covariate_formula,
                categorical_cols=cat_cols,
                continuous_cols=cont_cols,
                cdr=args.cdr,
                source_table=args.source_table,
                cohort=args.cohort,
                person_mod=args.person_mod,
                cache_uri=args.cache_uri,
                key_cols=cov_key_cols,
                prior_obs_days=args.prior_obs_days,
            )

        # --- Broadcast join: corpus + covariates by person_id ---
        with _phase("corpus + covariates join"):
            n_cov = cov_df.count()
            joined = bow_df.join(F.broadcast(cov_df), on=join_on, how="inner")
            n_joined = joined.count()
            dropped = n_bow - n_joined
            print(f"[driver]   covariate rows = {n_cov}", flush=True)
            if dropped > 0:
                print(f"[driver]   joined docs = {n_joined} (dropped {dropped} of "
                      f"{n_bow} corpus docs with no covariate match — usually a "
                      f"null sex/age that the formula NaN-drops)", flush=True)
            else:
                print(f"[driver]   joined docs = {n_joined} "
                      f"(all {n_bow} corpus docs matched)", flush=True)

        # --- Construct StreamingSTM (Path A) and fit ---
        from spark_vi.mllib.topic.stm import StreamingSTM

        with _phase("STM fit"):
            est = StreamingSTM(
                K=args.K,
                features_col="features",
                covariates_col="covariates",
                covariate_names=covariate_names,
                sigma_init=args.sigma_init,
                sigma_ridge=args.sigma_ridge,
                lbfgs_max_iter=args.lbfgs_max_iter,
                lbfgs_tol=args.lbfgs_tol,
                random_seed=args.random_seed,
            )
            # StreamingSTM.fit returns an STMModel.
            # The fit method builds a VIConfig internally from the supplied
            # max_iter, subsampling_rate, tau0, kappa parameters.
            stm_model = est.fit(
                joined,
                max_iter=args.max_iter,
                subsampling_rate=args.subsampling_rate,
                tau0=args.tau0,
                kappa=args.kappa,
                save_interval=args.save_interval,
                resume_from=(args.resume_from or None),
            )

        # --- Augment STMModel metadata with corpus_manifest + covariate_manifest ---
        with _phase("augment metadata"):
            # Build an index-ordered vocab list (matching LDA's layout so that
            # build_dashboard_cloud.py can read result.metadata["vocab"] uniformly).
            vocab_list: list = [None] * len(vocab_map)
            for cid, idx in vocab_map.items():
                vocab_list[idx] = cid
            stm_model.metadata.setdefault("corpus_manifest", {
                "cdr": args.cdr,
                "source_table": args.source_table,
                "cohort": args.cohort,
                "prior_obs_days": args.prior_obs_days,
                "person_mod": args.person_mod,
                "doc_spec": doc_spec.manifest(),
                "vocab_size": len(vocab_map),
                # Keep nested copies for manifest self-containment (same dual
                # storage as lda_bigquery_cloud.py).
                "vocab": vocab_list,
                "name_by_id": name_by_id,
            })
            # Top-level vocab + name_by_id mirror LDA's VIResult metadata layout
            # so build_dashboard_cloud.py can read them uniformly via
            # result.metadata.get("vocab") without knowing the model class.
            stm_model.metadata["vocab"] = vocab_list
            stm_model.metadata["name_by_id"] = name_by_id
            stm_model.metadata["covariate_manifest"] = {
                "covariate_formula": args.covariate_formula,
                "categorical_cols": cat_cols,
                "continuous_cols": cont_cols,
                "covariate_names": covariate_names,
                # The fitted ModelSpec is saved alongside as model_spec.pkl.
                # Bundle consumers prefer covariate_names for display labels.
            }
            stm_model.metadata["model_class"] = "stm"

        # --- Save ---
        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            stm_model.save(out)
            print(f"[driver]   saved STMModel to {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
