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

import numpy as np
from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session
from _corpus_load import load_or_build_corpus
from _covariates_load import load_or_build_covariates


def _make_topic_evolution_logger(top_n, every_n, idx_to_cid, name_by_id,
                                 topic_labels=None):
    """Build an on_iteration callback that prints top-N tokens per topic.

    STM analogue of the LDA driver's `_make_topic_evolution_logger`. STM's
    global_params carries the same (K, V) "lambda" topic-word variational
    parameter, so the topic-word distribution, Σλ_k, E[β_k] and peak are
    derived identically — but STM has NO per-topic Dirichlet α (its
    document-topic prior is logistic-normal, Γ/Σ), so the α column is dropped.

    When the model is gated, `topic_labels` (the length-K block names from the
    TopicBlockPartition) prefixes each line with its block, so background vs.
    foreground topics are distinguishable as they evolve. The framework calls
    this after each iteration as `(iter_num, global_params, elbo_trace)`;
    domain context rides in via closure capture (same pattern as LDA), and the
    runner wraps the call in try/except so a logging slip can't kill the fit.
    """
    from spark_vi.models.topic.diagnostics import topic_word_summary

    def _on_iter(iter_num: int, global_params: dict, _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        lam = global_params["lambda"]                          # (K, V)
        s = topic_word_summary(lam, top_n)
        # Heaviest topics first; printed k is the native (stable) index.
        order = np.argsort(s["row_sums"])[::-1]
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in order:
            ki = int(k)
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}({p:.3f})"
                for j, p in zip(s["top_indices"][ki], s["top_probs"][ki])
            )
            blk = (f" [{topic_labels[ki]:>10.10}]"
                   if topic_labels is not None else "")
            print(
                f"[driver]    topic {ki:>2}{blk}  "
                f"E[β]={s['mass_fraction'][ki]:.4f}  Σλ={s['row_sums'][ki]:.3g}  "
                f"peak={s['peak'][ki]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter


def _extract_categorical_levels(
    model_spec,
    categorical_cols: list[str],
) -> dict[str, dict]:
    """Extract {var: {"levels": [...], "reference": "..."}} from a formulaic ModelSpec.

    Uses encoder_state (formulaic >= 0.5) to read the full category list and
    derive the reference (dropped) level for treatment-coded contrasts.  With
    TreatmentContrasts the reference is the first sorted category when the base
    is UNSET, or the explicit base value otherwise.

    Returns an empty dict for any variable whose info cannot be recovered, so
    the caller degrades gracefully rather than raising.

    Mirrors analysis/local/fit_stm_local._extract_categorical_levels exactly.
    """
    result: dict[str, dict] = {}
    if model_spec is None:
        return result

    encoder_state = getattr(model_spec, "encoder_state", None) or {}
    for factor_key, state_tuple in encoder_state.items():
        if not (isinstance(state_tuple, tuple) and len(state_tuple) == 2):
            continue
        kind, state_dict = state_tuple
        if not (hasattr(kind, "value") and kind.value == "categorical"):
            continue
        categories = state_dict.get("categories") if isinstance(state_dict, dict) else None
        if not categories:
            continue
        categories = list(categories)

        import re as _re
        m = _re.match(r"^C\(\s*([^,)\s]+)", factor_key)
        var = m.group(1) if m else factor_key

        reference = ""
        contrasts_state = state_dict.get("contrasts") if isinstance(state_dict, dict) else None
        if contrasts_state is not None:
            try:
                c = contrasts_state.contrasts
                base = getattr(c, "base", None)
                if base is None or "UNSET" in str(base):
                    reference = categories[0]
                else:
                    reference = str(base)
            except Exception:
                reference = categories[0] if categories else ""

        result[var] = {"levels": categories, "reference": reference}

    return result


def build_topic_block_partition(*, group_var, background_k, foreground_arg, K):
    """Build a TopicBlockPartition from CLI args, or None when gating is off.

    foreground_arg is 'label:size,label:size'. Asserts background_k + sum(sizes)
    == K so a misconfigured partition fails before the (expensive) fit.
    """
    if background_k is None and foreground_arg is None:
        return None
    if background_k is None or foreground_arg is None:
        raise ValueError("--background-k and --foreground must be set together.")
    from spark_vi.models.topic.partition import TopicBlockPartition
    fg = []
    for piece in foreground_arg.split(","):
        label, _, size = piece.partition(":")
        fg.append((label.strip(), int(size)))
    part = TopicBlockPartition(group_var=group_var, background_k=int(background_k),
                               foreground=tuple(fg))
    if part.K != K:
        raise ValueError(
            f"gating partition K ({part.K}) != --K ({K}); "
            f"background_k + sum(foreground sizes) must equal --K.")
    return part


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
    p.add_argument("--print-topics-every", type=int, default=10,
                   help="Print top-N tokens per topic every N SVI iterations "
                        "(0 disables). Gated runs prefix each topic with its "
                        "background/foreground block label.")
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
    p.add_argument("--background-k", type=int, default=None,
                   help="Gating: size of the shared background topic block. "
                        "Set together with --foreground to enable background/"
                        "foreground gating; unset = canonical STM (no gating).")
    p.add_argument("--foreground", default=None,
                   help="Gating: per-group foreground block sizes as "
                        "'label:size,label:size' (e.g. 'cancer:10,dementia:10'). "
                        "The group labels are values of the gating column "
                        "(--group-var). background_k + sum(sizes) must equal --K.")
    p.add_argument("--group-var", default="source_cohort",
                   help="Gating: document column whose value selects a doc's "
                        "foreground block (default: source_cohort). Must NOT also "
                        "appear in --covariate-formula.")
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

        # Build the gating partition first (None when --background-k/--foreground unset).
        partition = build_topic_block_partition(
            group_var=args.group_var, background_k=args.background_k,
            foreground_arg=args.foreground, K=args.K)
        if partition is not None and args.group_var != "source_cohort":
            raise SystemExit(
                f"--group-var {args.group_var!r} is not a materializable group "
                f"column; only 'source_cohort' (from the combined-cohort "
                f"patient_cohort doc-spec) is supported today.")

        # source_cohort's two independent roles, separated:
        source_cohort_is_covariate = "source_cohort" in cat_cols   # -> per-(person,cohort) cov keying
        need_source_cohort = source_cohort_is_covariate or (partition is not None)

        if need_source_cohort:
            if doc_spec.name != "patient_cohort":
                raise SystemExit(
                    "source_cohort requires the combined-cohort doc-spec "
                    f"(patient_cohort); got {doc_spec.name!r}. Use the "
                    "cancer_or_dementia cohort, or drop source_cohort from the "
                    "formula and gating.")
            # doc_id == "{source_cohort}:{person_id}"; recover the label.
            bow_df = bow_df.withColumn(
                "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))

        if source_cohort_is_covariate:
            labels = bow_df.select("person_id", "source_cohort").distinct()
            cov_key_cols = ["person_id", "source_cohort"]
            join_on = ["person_id", "source_cohort"]
        else:
            labels = None
            cov_key_cols = ["person_id"]
            join_on = "person_id"

        if partition is not None:
            # Zero-doc foreground block is a config error; a thin block is a warning.
            present = {r[args.group_var] for r in
                       bow_df.select(args.group_var).distinct().collect()}
            counts = {r[args.group_var]: r["count"] for r in
                      bow_df.groupBy(args.group_var).count().collect()}
            for g in partition.groups:
                if g not in present:
                    raise SystemExit(
                        f"gating group {g!r} has zero documents ({args.group_var} "
                        f"values present: {sorted(present)}). Fix --foreground "
                        f"labels to match the corpus.")
                if counts.get(g, 0) < 100:
                    print(f"[driver]   WARNING: foreground group {g!r} has only "
                          f"{counts.get(g, 0)} docs; its block may be unstable.",
                          flush=True)

        # --- Diagnostics: corpus doc counts + per-cohort breakdown ---
        # Answers "is the doc count right, or is the join dropping docs?" The
        # per-source_cohort split is the combined-cohort validation readout.
        with _phase("corpus doc-count diagnostics"):
            n_bow = bow_df.count()
            print(f"[driver]   corpus docs (pre-join) = {n_bow}", flush=True)
            if need_source_cohort:
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
            if source_cohort_is_covariate:
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
                topic_blocks=partition,
                doc_group_col=(args.group_var if partition is not None else None),
            )
            # Periodic top-terms logger (additive to the engine's per-iter
            # Γ/Σ diagnostics). Gated runs get per-topic block labels.
            on_iter = None
            if args.print_topics_every > 0:
                idx_to_cid = {idx: cid for cid, idx in vocab_map.items()}
                topic_labels = (partition.topic_labels()
                                if partition is not None else None)
                on_iter = _make_topic_evolution_logger(
                    top_n=8, every_n=args.print_topics_every,
                    idx_to_cid=idx_to_cid, name_by_id=name_by_id,
                    topic_labels=topic_labels,
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
                on_iteration=on_iter,
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
                "topic_block_spec": (partition.to_dict() if partition is not None else None),
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
                # categorical_levels enables build_dashboard_cloud to write
                # accurate reference levels in covariate_schema.json without
                # relying on formulaic model_spec introspection (which is
                # unreliable after deserialization). Mirrors the local fitter.
                "categorical_levels": _extract_categorical_levels(
                    model_spec, cat_cols,
                ),
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
