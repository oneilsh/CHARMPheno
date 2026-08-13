"""Cloud fit+eval driver for Gated-PC rare-disease case-finding (smooshed vocab).

The forward test named in insight 0066: PC topic-shaping should help in the
HIDDEN-LOW-MASS regime — a rare phenotype an unsupervised fit spends its K
topics missing. This driver runs that test on real All-of-Us OMOP data with a
single smooshed (fused-vocab) BOW, comparing, on the SAME gated hierarchy and
per-node label:

  arm "unsup_gated"  — gated topics, NO supervision (OnlinePCLDA weightY=0). The
                       controlled incumbent: the same gated E-step + K + init as
                       the PC arm, only the head is off. (The dag_placement_cloud
                       node_affinity model is the production incumbent; this is
                       the pc_topics_lr-comparable twin.)
  arm "gated_pc"     — gate + FLAT PC head (weightY>0). Inject the hierarchy ONCE
                       (ADR 0042: gate+flat wins; gate+closure-head collapses).
  arm "dag_head"     — (optional, --with-dag-head) UNGATED + DAG-closure head, the
                       label-side-only alternative to the gate.

Headline metric: `pc_topics_lr` — a fresh post-hoc LogisticRegression on each
arm's FINAL per-doc theta against the per-node label, macro AUC over nodes
(insight 0066: this isolates representation quality from the co-fit head's own
convergence, and is directly comparable across arms). The gated_pc arm also
reports its co-fit head P(node) AUC (secondary; the head's own readout).

The per-node (label, labelMask) is emitted by the Step-A adapter
(case_finding_assembly.emit_labels): label[c]=1 iff node c is in the is-a
closure of the doc's frontier; C = len(int2cid). Mirrors dag_placement_cloud's
assemble->fit->score->save shape (cached CaseFindingBundle, npz + manifest.json).

CANNOT RUN HERE: every fact table is read from the workspace CDR via the
spark-bigquery connector, so main() only executes inside the All-of-Us Dataproc
workspace. Unit tests cover parse_args + the pure pc_topics_lr scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session


# --------------------------------------------------------------------------- #
# Pure scoring (no SparkSession; unit-tested numpy-level).                     #
# --------------------------------------------------------------------------- #
def pc_topics_lr_bundle(Pi_tr, y_tr, mask_tr, Pi_te, y_te, mask_te, C,
                        min_count=0):
    """`pc_topics_lr`: fit a fresh per-node LogisticRegression on TRAIN theta and
    score it on TEST theta, per-node AUC/AP macro'd over the observed cells.

    The convergence-robust representation-quality metric (insight 0066): it reads
    ONLY the final per-doc theta of an arm, so it measures whether that arm's
    topics carry per-node predictive signal — independent of any co-fit head's
    own optimization. Reuses the analysis.pc.evaluate masked-LR + masked-scorer
    so the number is identical machinery to the antidepressant driver's
    two-stage/pc_topics_lr readout. Pure (numpy in, dict out)."""
    from analysis.pc.evaluate import _bundle_masked, _lr_proba_per_label_masked

    proba = _lr_proba_per_label_masked(Pi_tr, y_tr, mask_tr, Pi_te, C)
    return _bundle_masked(proba, y_te, mask_te, C, min_count)


def dag_closure_parents(parent_int, C):
    """Length-C list of parent-LABEL-index lists for the ungated DAG-closure head.

    closure_parents[c] lists node c's direct parents in the [0, C) engine-id label
    space (root and any parentless node -> []); this is exactly `parent_int` (root
    omitted) densified over range(C). Feeds OnlinePCLDAEstimator.setClosureParents."""
    return [[int(p) for p in parent_int.get(c, [])] for c in range(C)]


# --------------------------------------------------------------------------- #
# Spark collectors (cluster-covered; not unit-tested).                        #
# --------------------------------------------------------------------------- #
def _collect_theta_labels(df, C, *, label_col="label", mask_col="labelMask",
                          topic_col="topicDistribution"):
    """Collect ONLY the K-dim per-doc theta + the (C,) label/mask arrays to numpy
    (never the dense BOW), so it stays on the driver's memory budget at cohort
    scale. Returns (Pi (D,K), y_DC (D,C), mask_DC (D,C), person_order). Empty df
    -> correctly-shaped zero arrays. Mirrors pc_antidepressant_cloud's
    _collect_topics_labels but with configurable label/mask column names (this
    corpus uses 'label'/'labelMask' from the Step-A adapter)."""
    rows = df.select("person_id", topic_col, label_col, mask_col).collect()
    person_order = [r["person_id"] for r in rows]
    if rows:
        Pi = np.asarray([r[topic_col].toArray() for r in rows], dtype=np.float64)
        y_DC = np.asarray([[float(v) for v in r[label_col]] for r in rows],
                          dtype=np.float64)
        mask_DC = np.asarray([[float(v) for v in r[mask_col]] for r in rows],
                             dtype=np.float64)
    else:
        Pi = np.zeros((0, 0), dtype=np.float64)
        y_DC = np.zeros((0, C), dtype=np.float64)
        mask_DC = np.zeros((0, C), dtype=np.float64)
    return Pi, y_DC, mask_DC, person_order


def _collect_head_proba(df, C, *, prob_col="probability", label_col="label",
                        mask_col="labelMask"):
    """Collect the co-fit head's per-node P(node)=sigmoid(w_CK.theta) + label/mask
    to (proba (D,C), y (D,C), mask (D,C)). Only meaningful for a supervised
    (weightY>0) transform, which appends `prob_col`. Cluster-covered."""
    rows = df.select(prob_col, label_col, mask_col).collect()
    if not rows:
        z = np.zeros((0, C), dtype=np.float64)
        return z, z, z.copy()
    proba = np.asarray([r[prob_col].toArray() for r in rows], dtype=np.float64)
    y_DC = np.asarray([[float(v) for v in r[label_col]] for r in rows],
                      dtype=np.float64)
    mask_DC = np.asarray([[float(v) for v in r[mask_col]] for r in rows],
                         dtype=np.float64)
    return proba, y_DC, mask_DC


def _macro_line(name, bundle):
    """One-line macro summary of a `_bundle_masked` result for the driver log."""
    m = bundle["macro"]
    auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f}"
    ap = "n/a" if m["ap"] is None else f"{m['ap']:.4f}"
    return (f"{name}: macro AUC={auc} AP={ap} "
            f"(scored {m['n_labels_scored']}/{m['n_labels_scored'] + m['n_labels_skipped']} nodes)")


def _build_pc_estimator(args, *, weight_y, gated, closure_parents=None):
    """Construct an OnlinePCLDAEstimator for one arm. `gated` injects the gated
    topic engine (gateParent set post-construction, since the Param is a JSON
    string); `closure_parents` selects the DAG-closure head. K comes from the gate
    layout when gated, else --k."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    est = OnlinePCLDAEstimator(
        featuresCol="features", labelCol="label", labelMaskCol="labelMask",
        numLabels=args._C, weightY=float(weight_y), k=args.k,
        maxIter=args.max_iter, seed=args.seed,
        subsamplingRate=args.subsampling_rate,
        learningOffset=args.tau0, learningDecay=args.kappa,
        gammaShape=args.gamma_shape, caviMaxIter=args.cavi_max_iter,
        caviTol=args.cavi_tol, gradCaviIters=args.grad_cavi_iters,
        topicTrust=args.topic_trust, weightYWarmupIters=args.weight_y_warmup_iters,
        headOptimizer=args.head_optimizer, headLr=args.head_lr,
        headNewtonRidge=args.head_newton_ridge, headL2=args.head_l2,
        optimizeDocConcentration=args.optimize_doc_concentration,
        frontierCol="frontier", gateNBg=args.n_bg, gateTpn=args.tpn,
    )
    if gated:
        est.setGateParent(args._parent_int)      # JSON-encodes the DAG map
    if closure_parents is not None:
        est.setClosureParents(closure_parents)
    return est


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gated-PC rare-disease case-finding fit + pc_topics_lr eval.")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--doc-min-length", type=int, default=0)
    p.add_argument("--prior-obs-days", type=int, default=365)
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--window-mode", choices=["forward", "lookback"], default="forward")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--label-window-days", type=int, default=365)
    # assembly / DAG. `disease` selects the foreground cohort AND the label-DAG
    # anchors (cohorts.disease_anchors); rare6 = the six-anchor rare-disease forest.
    p.add_argument("--disease", default="rare6")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")
    p.add_argument("--label-mask-mode", choices=["full", "closure"], default="full",
                   help="per-node observation policy for the PC label (Step-A "
                        "adapter): 'full' (ones(C); background is a true negative "
                        "everywhere) or 'closure' (observe the active closure + DAG "
                        "siblings only).")
    # gate topic-block layout (topics per node); K is emergent = n_bg + nodes*tpn.
    p.add_argument("--n-bg", type=int, default=2)
    p.add_argument("--tpn", type=int, default=1)
    p.add_argument("--k", type=int, default=50,
                   help="K for the UNGATED --with-dag-head arm only; the gated arms "
                        "derive K from the layout (n_bg + nodes*tpn).")
    p.add_argument("--optimize-doc-concentration", action="store_true")
    # PC head + SVI
    p.add_argument("--weight-y", type=float, default=50.0,
                   help="PC prediction weight (>0). Hughes ~ tokens/doc; tune on "
                        "validation. The unsup_gated arm always refits at 0.")
    p.add_argument("--head-optimizer", choices=["sgd", "newton"], default="newton",
                   help="head optimizer; 'newton' is the settled convergent head "
                        "(ADR 0039) — the default here.")
    p.add_argument("--head-lr", type=float, default=0.5)
    p.add_argument("--head-newton-ridge", type=float, default=0.01)
    p.add_argument("--head-l2", type=float, default=1e-3,
                   help="ABSOLUTE ridge on w_CK (= Hughes lambda_w; ADR 0041). "
                        "0.0 BLOWS UP on the separable topics PC creates.")
    p.add_argument("--grad-cavi-iters", type=int, default=20)
    p.add_argument("--topic-trust", type=float, default=0.1)
    p.add_argument("--weight-y-warmup-iters", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--subsampling-rate", type=float, default=0.05)
    p.add_argument("--tau0", type=float, default=64.0,
                   help="Robbins-Monro offset. On smaller cohorts try ~10-64 so the "
                        "head actually moves.")
    p.add_argument("--kappa", type=float, default=0.51)
    p.add_argument("--gamma-shape", type=float, default=100.0)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None)
    # arms + eval
    p.add_argument("--skip-unsup-gated", action="store_true",
                   help="skip the weightY=0 gated baseline (fit only the PC arm).")
    p.add_argument("--with-dag-head", action="store_true",
                   help="also fit the UNGATED + DAG-closure-head arm (label-side "
                        "hierarchy alternative to the gate).")
    p.add_argument("--baseline-max-iter", type=int, default=-1,
                   help="cap the unsup_gated fit at N iters; <=0 reuses --max-iter "
                        "(unsupervised topics converge faster than the head).")
    p.add_argument("--min-label-count", type=int, default=20,
                   help="mask any node whose heldout column has < this many cells "
                        "of either class from the macro (AoU small-cell floor).")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused; accepted for run_experiment parity.")
    return p.parse_args(argv)


def main() -> int:
    from _case_finding_cache import load_or_build_case_finding_bundle
    from spark_vi.models.topic.dag_placement import DagLayout

    args = parse_args()
    configure_logging()
    with make_spark_session(app_name="gated-pc-fit") as spark:
        with _phase("assemble corpus (cached, emit_labels)"):
            bundle = load_or_build_case_finding_bundle(
                spark, cache_uri=args.cache_uri,
                cdr=args.cdr, billing=args.billing, source_table=args.source_table,
                person_mod=args.person_mod, vocab_size=args.vocab_size,
                min_df=args.min_df, min_patient_count=args.min_patient_count,
                doc_min_length=args.doc_min_length, prior_obs_days=args.prior_obs_days,
                window_days=args.window_days, disease=args.disease, min_n=args.min_n,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                strip_mode=args.strip_mode, window_mode=args.window_mode,
                lookback_days=args.lookback_days,
                label_window_days=args.label_window_days,
                emit_labels=True, label_mask_mode=args.label_mask_mode)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        C = len(bundle.int2cid)               # label heads = engine nodes incl. root
        # Stash on args so the estimator builder can reach them without re-threading.
        args._C = C
        args._parent_int = bundle.parent_int
        print(f"[driver]   corpus: V={len(bundle.vocab_map)} vocab, "
              f"K={lay.K} gated topics ({args.n_bg} bg + {len(lay.nodes)} nodes x "
              f"{args.tpn} tpn), C={C} label heads", flush=True)

        results = {}

        with _phase(f"gated_pc fit (weightY={args.weight_y}, K={lay.K})"):
            pc_est = _build_pc_estimator(
                args, weight_y=args.weight_y, gated=True)
            pc_model = pc_est.fit(bundle.train_df)
            # Transform each split ONCE (each transform re-runs CAVI over the split);
            # the supervised transform appends BOTH topicDistribution and probability.
            train_scored = pc_model.transform(bundle.train_df).cache()
            test_scored = pc_model.transform(bundle.test_df).cache()
            Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
            Pi_te, y_te, m_te, _ = _collect_theta_labels(test_scored, C)
            pc_lr = pc_topics_lr_bundle(
                Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, args.min_label_count)
            results["gated_pc__pc_topics_lr"] = pc_lr["macro"]
            print("[driver]   " + _macro_line("gated_pc pc_topics_lr", pc_lr),
                  flush=True)
            # co-fit head's own per-node P(node) readout (secondary), from the SAME
            # scored test frame (no second CAVI pass).
            hp, hy, hm = _collect_head_proba(test_scored, C)
            from analysis.pc.evaluate import _bundle_masked
            head_bundle = _bundle_masked(hp, hy, hm, C, args.min_label_count)
            train_scored.unpersist(); test_scored.unpersist()
            results["gated_pc__head"] = head_bundle["macro"]
            print("[driver]   " + _macro_line("gated_pc co-fit head", head_bundle),
                  flush=True)

        if not args.skip_unsup_gated:
            n_it = (args.baseline_max_iter if args.baseline_max_iter and
                    args.baseline_max_iter > 0 else args.max_iter)
            with _phase(f"unsup_gated fit (weightY=0, {n_it} iters)"):
                us_args = argparse.Namespace(**vars(args))
                us_args.max_iter = int(n_it)
                us_est = _build_pc_estimator(us_args, weight_y=0.0, gated=True)
                us_model = us_est.fit(bundle.train_df)
                Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                    us_model.transform(bundle.train_df), C)
                Pi_te, y_te, m_te, _ = _collect_theta_labels(
                    us_model.transform(bundle.test_df), C)
                us_lr = pc_topics_lr_bundle(
                    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, args.min_label_count)
                results["unsup_gated__pc_topics_lr"] = us_lr["macro"]
                print("[driver]   " + _macro_line("unsup_gated pc_topics_lr", us_lr),
                      flush=True)

        if args.with_dag_head:
            with _phase(f"dag_head fit (ungated + DAG-closure head, K={args.k})"):
                cp = dag_closure_parents(bundle.parent_int, C)
                dh_est = _build_pc_estimator(
                    args, weight_y=args.weight_y, gated=False, closure_parents=cp)
                dh_model = dh_est.fit(bundle.train_df)
                Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                    dh_model.transform(bundle.train_df), C)
                Pi_te, y_te, m_te, _ = _collect_theta_labels(
                    dh_model.transform(bundle.test_df), C)
                dh_lr = pc_topics_lr_bundle(
                    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, args.min_label_count)
                results["dag_head__pc_topics_lr"] = dh_lr["macro"]
                print("[driver]   " + _macro_line("dag_head pc_topics_lr", dh_lr),
                      flush=True)

        # Headline: does supervision improve the representation (pc_topics_lr)?
        pc_auc = results.get("gated_pc__pc_topics_lr", {}).get("auc")
        us_auc = results.get("unsup_gated__pc_topics_lr", {}).get("auc")
        if pc_auc is not None and us_auc is not None:
            print(f"[driver]   HEADLINE: gated_pc pc_topics_lr {pc_auc:.4f} vs "
                  f"unsup_gated {us_auc:.4f} (delta {pc_auc - us_auc:+.4f}); PC "
                  f"should help in the hidden-low-mass regime (insight 0066).",
                  flush=True)

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            gp = pc_model.result.global_params
            np.savez(out / "gated_pc_result.npz",
                     **{"lambda": gp["lambda"], "alpha": gp["alpha"],
                        "w_CK": gp["w_CK"]})
            manifest = {
                "model_class": "gated_pc",
                "disease": args.disease, "min_n": args.min_n,
                "strip_mode": args.strip_mode, "label_mask_mode": args.label_mask_mode,
                "window_mode": args.window_mode, "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "n_bg": args.n_bg, "tpn": args.tpn, "K": lay.K, "C": C,
                "weight_y": args.weight_y, "head_optimizer": args.head_optimizer,
                "head_l2": args.head_l2, "head_lr": args.head_lr,
                "weight_y_warmup_iters": args.weight_y_warmup_iters,
                "grad_cavi_iters": args.grad_cavi_iters, "topic_trust": args.topic_trust,
                "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
                "kappa": args.kappa, "max_iter": args.max_iter,
                "min_label_count": args.min_label_count,
                "with_dag_head": args.with_dag_head,
                "skip_unsup_gated": args.skip_unsup_gated,
                "results": results, "ledger": bundle.ledger,
                "corpus_manifest": {
                    "cdr": args.cdr, "source_table": args.source_table,
                    "person_mod": args.person_mod, "vocab_size": args.vocab_size,
                    "min_df": args.min_df, "min_patient_count": args.min_patient_count,
                    "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
                    "holdout_frac": args.holdout_frac,
                    "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                    "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()}},
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved gated_pc result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
