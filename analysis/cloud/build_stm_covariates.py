"""Rebuild only the STM covariate cache.

Useful when iterating on `covariate_formula` for an experiment: rerunning
the full fit driver to refresh covariates is wasteful when the corpus
hasn't changed. This script runs load_or_build_covariates against the
configured cache and exits.
"""
from __future__ import annotations

import argparse
import sys

from _driver_common import configure_logging, make_spark_session, _phase
from _covariates_load import load_or_build_covariates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STM covariate-cache builder")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--cohort", default=None)
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--cache-uri", required=True,
                   help="GCS/HDFS URI hosting the covariate cache.")
    p.add_argument("--covariate-formula", required=True)
    p.add_argument("--categorical-cols", required=True,
                   help="Comma-separated list of categorical column names.")
    p.add_argument("--continuous-cols", required=True,
                   help="Comma-separated list of continuous column names.")
    p.add_argument("--force", action="store_true",
                   help="Delete the cached covariate dir before building.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.continuous_cols.split(",") if c.strip()]
    with make_spark_session(app_name="stm-build-covariates") as spark:
        # If --force, delete cache key dir so load_or_build_covariates rebuilds.
        if args.force:
            from _covariates_cache import compute_cache_key
            key = compute_cache_key(
                covariate_formula=args.covariate_formula,
                person_mod=args.person_mod, cdr=args.cdr,
                source_table=args.source_table, cohort=args.cohort,
            )
            base = f"{args.cache_uri.rstrip('/')}/{key}"
            with _phase(f"force-delete {base}"):
                sc = spark.sparkContext
                jvm = sc._jvm
                fs_path = jvm.org.apache.hadoop.fs.Path(base)
                fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
                if fs.exists(fs_path):
                    fs.delete(fs_path, True)
                    print(f"[driver]   deleted cache dir {base}", flush=True)
                else:
                    print(f"[driver]   no cache dir at {base} (nothing to delete)",
                          flush=True)

        from charmpheno.omop import load_person_table
        with _phase("person table load"):
            person_df = load_person_table(
                spark=spark, cdr_dataset=args.cdr,
                billing_project=args.billing,
                person_sample_mod=args.person_mod,
                cohort=args.cohort,
            )
        cov_df, _, names = load_or_build_covariates(
            spark, person_df=person_df,
            covariate_formula=args.covariate_formula,
            categorical_cols=cat_cols, continuous_cols=cont_cols,
            cdr=args.cdr, source_table=args.source_table,
            cohort=args.cohort, person_mod=args.person_mod,
            cache_uri=args.cache_uri,
        )
        n_rows = cov_df.count()
        print(
            f"[driver]   covariate cache ready: {n_rows} rows, P={len(names)}, "
            f"names={names}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
