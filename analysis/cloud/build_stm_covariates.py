"""Rebuild only the STM covariate cache.

Useful when iterating on `covariate_formula` for an experiment: rerunning
the full fit driver to refresh covariates is wasteful when the corpus
hasn't changed. This script runs load_or_build_covariates against the
configured cache and exits.

For a gated fit (``--group-var source_cohort``), the script loads the
tagged OMOP events to derive the per-person group labels, expands the
person table with those labels, and passes
``key_cols=["person_id", "source_cohort"]`` so the sidecar carries each
document's group — the gated dashboard prevalence groups by it, and the
schema matches the one the fit driver produces. The group key is a pure
label, never a covariate (see validate_label_not_covariate).
"""
from __future__ import annotations

import argparse
import sys

from _driver_common import configure_logging, make_spark_session, _phase
from _covariates_load import (
    load_or_build_covariates,
    covariate_key_cols,
    validate_label_not_covariate,
    DEFAULT_GROUP_COL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STM covariate-cache builder")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--cohort", default=None)
    p.add_argument("--prior-obs-days", type=int, default=365,
                   help="Prior-observation lookback (days) for the cohort "
                        "index date; 0 drops the lookback. Must match the fit "
                        "driver so the shared covariate cache key is identical.")
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--cache-uri", required=True,
                   help="GCS/HDFS URI hosting the covariate cache.")
    p.add_argument("--covariate-formula", required=True)
    p.add_argument("--categorical-cols", required=True,
                   help="Comma-separated list of categorical column names.")
    p.add_argument("--continuous-cols", required=True,
                   help="Comma-separated list of continuous column names.")
    p.add_argument("--group-var", default=None,
                   help="Gating/label key (e.g. source_cohort). When set, the "
                        "covariate cache is keyed on (person_id, group-var) so "
                        "the gated dashboard prevalence can group by it — this "
                        "must match the fit driver's gating. Omit for an "
                        "ungated fit.")
    p.add_argument("--force", action="store_true",
                   help="Delete the cached covariate dir before building.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.continuous_cols.split(",") if c.strip()]

    # The gating/label key is a pure per-document label, never a covariate.
    validate_label_not_covariate(
        cat_cols, cont_cols, label=args.group_var or DEFAULT_GROUP_COL)

    gated = bool(args.group_var)
    if gated and args.group_var != "source_cohort":
        raise SystemExit(
            f"--group-var {args.group_var!r} is not a materializable group "
            f"column; only 'source_cohort' is supported today. It must match "
            f"the fit driver's --group-var.")
    # A gated fit needs the per-(person, label) sidecar so the dashboard's
    # masked prevalence can group by the label; an ungated fit keys on
    # person_id alone.
    composite = gated
    key_cols = covariate_key_cols(gated=gated, label=args.group_var or DEFAULT_GROUP_COL)

    with make_spark_session(app_name="stm-build-covariates") as spark:
        # If --force, delete cache key dir so load_or_build_covariates rebuilds.
        if args.force:
            from _covariates_cache import compute_cache_key
            key = compute_cache_key(
                covariate_formula=args.covariate_formula,
                person_mod=args.person_mod, cdr=args.cdr,
                source_table=args.source_table, cohort=args.cohort,
                prior_obs_days=args.prior_obs_days,
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
        from charmpheno.omop import load_omop_bigquery

        if composite:
            # A gated fit keys the sidecar on (person_id, group-var); the
            # person table alone has no group column. Load the tagged events —
            # load_omop_bigquery returns the group column for the combined
            # cohort via apply_cohort — and derive the per-(person, group)
            # label set. This mirrors the fit driver's gated branch so the
            # shared cache key (and now the sidecar schema) is identical.
            group_var = args.group_var
            with _phase("events load (composite cohort label derivation)"):
                events = load_omop_bigquery(
                    spark=spark, cdr_dataset=args.cdr,
                    billing_project=args.billing,
                    person_sample_mod=args.person_mod,
                    source_table=args.source_table,
                    cohort=args.cohort, prior_obs_days=args.prior_obs_days,
                )
                labels = events.select("person_id", group_var).distinct()

            with _phase("person table load"):
                person_df = load_person_table(
                    spark=spark, cdr_dataset=args.cdr,
                    billing_project=args.billing,
                    person_sample_mod=args.person_mod,
                    cohort=args.cohort,
                )
                person_df = person_df.join(labels, on="person_id", how="inner")
        else:
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
            key_cols=key_cols,
            prior_obs_days=args.prior_obs_days,
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
