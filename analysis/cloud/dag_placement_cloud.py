"""Cloud fit+eval driver for the gated-SVI hierarchical case-finding engine.

Assembles the diabetes case-finding corpus (piece-2 assemble_case_finding_corpus,
cached), fits the gated MLlib shim (GatedLDAEstimator), scores held-out placement
inline (dag_placement.evaluate), and saves an npz + manifest.json artifact (the
pg_stm methods-experiment pattern; the NPMI coherence eval cannot score a
placement model). K is EMERGENT (n_bg + surviving-DAG-nodes * tpn), so there is
no --K. Resume is unsupported (GatedLDAModel is not persistable in v1).

The init flag (random | spectral) is the pre-registered A/B: spectral uses the
dense block-aligned anchor-word seed (Arora et al. 2013) collected to the driver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session


def profiles_from_scored_rows(rows, lay):
    """Adapt transform() output rows to dag_placement.evaluate inputs.

    Each row's `nodeAffinity` is a DenseVector ordered by lay.nodes; the profile
    is dict(zip(lay.nodes, affinity)). `frontier` (engine-ids) becomes the truth
    set. Pure; the driver collects the test set (held-out scale) before calling."""
    profiles, test_labels = [], []
    for r in rows:
        aff = r["nodeAffinity"].toArray()
        profiles.append({u: float(aff[i]) for i, u in enumerate(lay.nodes)})
        test_labels.append({int(x) for x in r["frontier"]})
    return profiles, test_labels


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gated-SVI hierarchical case-finding fit + inline placement eval.")
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
    # assembly / DAG
    p.add_argument("--anchor", type=int, default=201820)
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    # gating
    p.add_argument("--n-bg", type=int, default=2)
    p.add_argument("--tpn", type=int, default=1)
    # SVI
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    p.add_argument("--spectral-max-vocab", type=int, default=8000)
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused (GatedLDAModel is not persistable in v1); "
                        "accepted for run_experiment parity.")
    return p.parse_args(argv)


def main() -> int:
    from _case_finding_cache import load_or_build_case_finding_bundle
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout, evaluate, render_profile

    args = parse_args()
    configure_logging()
    with make_spark_session(app_name="dag-placement-fit") as spark:
        with _phase("assemble corpus (cached)"):
            bundle = load_or_build_case_finding_bundle(
                spark, cache_uri=args.cache_uri,
                cdr=args.cdr, billing=args.billing, source_table=args.source_table,
                person_mod=args.person_mod, vocab_size=args.vocab_size,
                min_df=args.min_df, min_patient_count=args.min_patient_count,
                doc_min_length=args.doc_min_length, prior_obs_days=args.prior_obs_days,
                window_days=args.window_days, anchor=args.anchor, min_n=args.min_n,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)

        with _phase(f"gated-svi fit (init={args.init}, K={lay.K})"):
            est = GatedLDAEstimator(
                featuresCol="features", labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab)
            model = est.fit(bundle.train_df)

        with _phase("transform + inline placement eval"):
            scored = model.transform(bundle.test_df).select("nodeAffinity", "frontier")
            rows = scored.collect()
            profiles, test_labels = profiles_from_scored_rows(rows, lay)
            metrics = evaluate(profiles, test_labels, lay)
            print(f"[driver]   placement metrics: "
                  f"auc_by_depth={metrics['auc_by_depth']} mrr={metrics['mrr']:.3f} "
                  f"top2={metrics['top2']:.3f} mean_hops={metrics['mean_hops']:.2f} "
                  f"frontier_size_mean={metrics['frontier_size_mean']:.2f} "
                  f"multi_frontier_rate={metrics['multi_frontier_rate']:.3f}",
                  flush=True)
            # Spot-check render for a few foreground held-out docs. names must be
            # ENGINE-id-keyed (remap concept-id name_by_id via int2cid).
            names = {i: bundle.name_by_id[c] for i, c in bundle.int2cid.items()
                     if c in bundle.name_by_id}
            for pr, lab in list(zip(profiles, test_labels))[:5]:
                if lab:
                    print(render_profile(pr, lay, names=names, true_node=lab),
                          flush=True)

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            gp = model.result.global_params
            np.savez(out / "dag_placement_result.npz",
                     **{"lambda": gp["lambda"], "alpha": gp["alpha"]})
            manifest = {
                "model_class": "dag_placement",
                "init": args.init, "K": lay.K, "n_bg": args.n_bg, "tpn": args.tpn,
                "anchor": args.anchor, "min_n": args.min_n,
                "max_iter": args.max_iter, "metrics": metrics, "ledger": bundle.ledger,
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
            print(f"[driver]   saved dag_placement result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
