"""PG-STM (Pólya-Gamma / full-Bayes gated STM) fit driver — sub-project 2.

Mirrors ``stm_bigquery_cloud.py``'s corpus + covariate loading (SAME cache path, SAME
gating partition, SAME per-doc STMDocument construction via ``_vector_to_stm_document``),
then runs the DISTRIBUTED PG engine instead of the softmax point-EM STM:

  Phase 1 — ``StreamingPGSTM`` mini-batch PG-SVI (the runaway-cure test). ``--sigma-mode``
    selects the IW block posterior (the cure) or the un-regularized ``scatter/n`` point
    estimate (the contrast arm). Records ``sigma_max_trace`` / final Sigma eigmin+max|Sigma|.
  Phase 2 (optional, ``--gibbs-sweeps > 0``) — ``pg_stm_sigma_readout``: the comorbidity
    Sigma-correlation read-out via the validated exact Gibbs on a driver-collected
    subsample (mean-field VI cannot produce it — insight 0044).

Saves beta / Gamma / Sigma / sigma_max_trace (+ the Phase-2 Sigma) as an .npz plus a
manifest.json. Dashboard export is sub-project 3, not here.

NOTE: this driver's end-to-end BigQuery path is exercised on the CLUSTER (make exp
ID=50/51); the engine it calls (StreamingPGSTM, pg_stm_sigma_readout) is unit-validated
locally. The Pólya-Gamma sampler is pure-numpy (spark_vi.models.topic._pg) — NO native
polyagamma dependency on any node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import functions as F

from _driver_common import _phase, configure_logging, make_spark_session
from _corpus_load import load_or_build_corpus
from _covariates_load import (
    load_or_build_covariates,
    covariate_key_cols,
    validate_label_not_covariate,
)


def build_topic_block_partition(*, group_var, background_k, foreground_arg, K):
    """Build a TopicBlockPartition from CLI args, or None when gating is off (identical
    to the STM driver's helper)."""
    if background_k is None or not foreground_arg:
        return None
    from spark_vi.models.topic.partition import TopicBlockPartition
    foreground = []
    for tok in str(foreground_arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        label, size = tok.split(":")
        foreground.append((label.strip(), int(size)))
    part = TopicBlockPartition(group_var=group_var, background_k=int(background_k),
                              foreground=tuple(foreground))
    if part.K != int(K):
        raise SystemExit(
            f"gating blocks sum to K={part.K} but --K={K} "
            f"(background_k {background_k} + foreground {foreground}).")
    return part


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Distributed PG-STM fit (SVI + Sigma read-out).")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--cohort", default=None)
    p.add_argument("--prior-obs-days", type=int, default=365)
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--doc-spec", default="patient_year")
    p.add_argument("--doc-min-length", type=int, default=20)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--subsampling-rate", type=float, default=0.2)
    p.add_argument("--tau0", type=float, default=64.0)
    p.add_argument("--kappa", type=float, default=0.7)
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--covariate-formula", required=True)
    p.add_argument("--categorical-cols", required=True)
    p.add_argument("--continuous-cols", required=True)
    # PG-specific.
    p.add_argument("--sigma-mode", choices=["iw", "mle"], default="iw",
                   help="Sigma M-step: 'iw' block inverse-Wishart posterior (the runaway "
                        "cure) or 'mle' un-regularized scatter/n (the contrast arm).")
    p.add_argument("--gibbs-sweeps", type=int, default=0,
                   help="Phase-2 exact-Gibbs Sigma read-out sweeps (0 = skip Phase 2).")
    p.add_argument("--gibbs-burn", type=int, default=None,
                   help="Phase-2 burn-in (default: gibbs-sweeps // 2).")
    p.add_argument("--sigma-readout-subsample", type=int, default=20000,
                   help="Phase-2 subsample size for the Sigma read-out (0 = whole corpus).")
    # Gating.
    p.add_argument("--background-k", type=int, default=None)
    p.add_argument("--foreground", default=None)
    p.add_argument("--group-var", default="source_cohort")
    p.add_argument("--known-sex-only", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused for PG-STM (accepted for run_experiment parity).")
    return p.parse_args(argv)


def _sigma_diagnostics(Sigma):
    S = 0.5 * (np.asarray(Sigma) + np.asarray(Sigma).T)
    eig = float(np.linalg.eigvalsh(S).min())
    pd = True
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        pd = False
    return {"eigmin": eig, "max_abs": float(np.max(np.abs(S))), "pd": pd}


def main() -> int:
    args = parse_args()
    configure_logging()
    cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.continuous_cols.split(",") if c.strip()]

    with make_spark_session(app_name="pg-stm-fit") as spark:
        from charmpheno.omop import doc_spec_from_cli, load_person_table
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        from spark_vi.mllib.topic.pg_stm import StreamingPGSTM, pg_stm_sigma_readout

        doc_spec = doc_spec_from_cli(args.doc_spec, min_doc_length=args.doc_min_length)

        with _phase("corpus load"):
            bow_df, vocab_map, name_by_id = load_or_build_corpus(
                spark, doc_spec=doc_spec, cdr=args.cdr, billing=args.billing,
                source_table=args.source_table, person_mod=args.person_mod,
                vocab_size=args.vocab_size, min_df=args.min_df,
                min_patient_count=args.min_patient_count, cache_uri=args.cache_uri,
                cohort=args.cohort, prior_obs_days=args.prior_obs_days,
                length_report_group_col=args.group_var)

        partition = build_topic_block_partition(
            group_var=args.group_var, background_k=args.background_k,
            foreground_arg=args.foreground, K=args.K)
        if partition is None:
            raise SystemExit("PG-STM requires gating (--background-k + --foreground).")
        if args.group_var != "source_cohort":
            raise SystemExit(
                f"--group-var {args.group_var!r} unsupported; only 'source_cohort'.")
        validate_label_not_covariate(cat_cols, cont_cols, label=args.group_var)

        if doc_spec.name != "patient_cohort":
            raise SystemExit(
                f"gating on {args.group_var!r} requires the patient_cohort doc-spec; "
                f"got {doc_spec.name!r} (use the cancer_or_dementia cohort).")
        bow_df = bow_df.withColumn(
            args.group_var, F.split(F.col("doc_id"), ":").getItem(0))
        labels = bow_df.select("person_id", args.group_var).distinct()
        cov_key_cols = covariate_key_cols(gated=True, label=args.group_var)
        join_on = ["person_id", args.group_var]

        present = {r[args.group_var] for r in
                   bow_df.select(args.group_var).distinct().collect()}
        for g in partition.groups:
            if g not in present:
                raise SystemExit(
                    f"gating group {g!r} has zero documents ({args.group_var} present: "
                    f"{sorted(present)}).")

        with _phase("person table load"):
            person_df = load_person_table(
                spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                person_sample_mod=args.person_mod, cohort=args.cohort,
                known_sex_only=args.known_sex_only)
            person_df = person_df.join(labels, on="person_id", how="inner")

        with _phase("covariates load"):
            cov_df, model_spec, covariate_names = load_or_build_covariates(
                spark, person_df=person_df, covariate_formula=args.covariate_formula,
                categorical_cols=cat_cols, continuous_cols=cont_cols, cdr=args.cdr,
                source_table=args.source_table, cohort=args.cohort,
                person_mod=args.person_mod, cache_uri=args.cache_uri,
                key_cols=cov_key_cols, prior_obs_days=args.prior_obs_days)

        with _phase("corpus + covariates join"):
            joined = bow_df.join(F.broadcast(cov_df), on=join_on, how="inner")
            n_joined = joined.count()
            print(f"[driver]   joined docs = {n_joined}", flush=True)

        doc_rdd = joined.rdd.map(lambda row: _vector_to_stm_document(
            row, features_col="features", covariates_col="covariates",
            group_col=args.group_var)).persist()

        K = int(args.K)
        V = len(vocab_map)
        P = len(covariate_names)
        seed = args.random_seed if args.random_seed is not None else 0

        with _phase(f"PG-SVI fit (sigma_mode={args.sigma_mode})"):
            est = StreamingPGSTM(K=K, V=V, partition=partition, P=P,
                                 sigma_mode=args.sigma_mode, seed=seed)
            svi = est.fit(doc_rdd, max_iter=args.max_iter,
                          batch=args.subsampling_rate, tau0=args.tau0, kappa=args.kappa)
            sdiag = _sigma_diagnostics(svi["Sigma"])
            print(f"[driver]   Phase-1 Sigma: eigmin={sdiag['eigmin']:.4g} "
                  f"max|Sigma|={sdiag['max_abs']:.4g} PD={sdiag['pd']} "
                  f"max(sigma_max_trace)={max(svi['sigma_max_trace']):.4g}", flush=True)

        readout = None
        if args.gibbs_sweeps and args.gibbs_sweeps > 0:
            burn = args.gibbs_burn if args.gibbs_burn is not None else args.gibbs_sweeps // 2
            with _phase(f"Phase-2 Sigma read-out ({args.gibbs_sweeps} sweeps)"):
                readout = pg_stm_sigma_readout(
                    doc_rdd, K=K, V=V, partition=partition, P=P,
                    subsample_n=args.sigma_readout_subsample,
                    n_iter=args.gibbs_sweeps, burn=burn, seed=seed)

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            arrays = {"beta": svi["beta"], "Gamma": svi["Gamma"], "Sigma": svi["Sigma"],
                      "sigma_max_trace": np.asarray(svi["sigma_max_trace"])}
            if readout is not None:
                arrays["Sigma_gibbs"] = readout["Sigma"]
                arrays["beta_gibbs"] = readout["beta"]
            np.savez(out / "pg_stm_result.npz", **arrays)
            vocab_list = [None] * len(vocab_map)
            for cid, idx in vocab_map.items():
                vocab_list[idx] = cid
            manifest = {
                "model_class": "pg_stm",
                "sigma_mode": args.sigma_mode,
                "K": K, "background_k": args.background_k, "foreground": args.foreground,
                "group_var": args.group_var, "P": P, "vocab_size": V,
                "n_docs": int(n_joined), "max_iter": args.max_iter,
                "covariate_names": list(covariate_names),
                "covariate_formula": args.covariate_formula,
                "topic_block_spec": partition.to_dict(),
                "phase1_sigma": sdiag,
                "gibbs_sweeps": args.gibbs_sweeps,
                "corpus_manifest": {
                    "cdr": args.cdr, "source_table": args.source_table,
                    "cohort": args.cohort, "prior_obs_days": args.prior_obs_days,
                    "person_mod": args.person_mod, "doc_spec": doc_spec.manifest(),
                    "vocab": vocab_list, "name_by_id": name_by_id},
            }
            if readout is not None:
                manifest["phase2_sigma"] = _sigma_diagnostics(readout["Sigma"])
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved PG-STM result to {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
