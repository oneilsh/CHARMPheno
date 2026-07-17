"""Cloud fit+eval driver for the gated-SVI hierarchical case-finding engine.

Assembles the case-finding corpus for a chosen `disease` (piece-2
assemble_case_finding_corpus, cached) — a single-disease type taxonomy
(diabetes, eds) or the multi-disease rare-disease forest (rare6) — fits the
gated MLlib shim (GatedLDAEstimator), scores held-out placement
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


def _log_corpus_stats(bundle, lay):
    """Log + return train/test corpus stats: doc counts, per-source_cohort
    breakdown, how many docs carry a frontier (rankable foreground), and the vocab
    / topic-structure dimensions. One lightweight aggregation pass per split."""
    from pyspark.sql import functions as F

    def _stats(df, name):
        agg = (df.groupBy("source_cohort")
               .agg(F.count(F.lit(1)).alias("n"),
                    F.sum((F.size("frontier") > 0).cast("long")).alias("fg"))
               .collect())
        by = {r["source_cohort"]: (int(r["n"]), int(r["fg"])) for r in agg}
        total = sum(n for n, _ in by.values())
        fg = sum(f for _, f in by.values())
        print(f"[driver]   corpus[{name}]: {total} docs, {fg} with a frontier; "
              + ", ".join(f"{k}={n}" for k, (n, _) in sorted(by.items())), flush=True)
        return {"n_docs": total, "n_frontier": fg,
                "by_source_cohort": {k: n for k, (n, _) in by.items()}}

    stats = {"train": _stats(bundle.train_df, "train"),
             "test": _stats(bundle.test_df, "test"),
             "vocab_size": len(bundle.vocab_map), "K": lay.K,
             "n_nodes": len(lay.nodes), "n_bg": lay.n_bg, "tpn": lay.tpn}
    print(f"[driver]   corpus: V={stats['vocab_size']} vocab, K={lay.K} topics "
          f"({lay.n_bg} bg + {len(lay.nodes)} nodes x {lay.tpn} tpn)", flush=True)
    return stats


def _topic_node_labels(lay, int2cid, name_by_id, n_bg):
    """Length-K block labels: background topics -> 'bg'; each node's topic block ->
    the node's condition name (engine-id -> concept-id -> name). Lets the per-iter
    topic log show which DAG node each topic belongs to as it evolves."""
    labels = ["bg"] * lay.K
    for u in lay.nodes:
        cid = int2cid.get(u)
        nm = name_by_id.get(cid, str(cid))
        for k in lay.block[u]:
            labels[k] = nm
    return labels


def _vocab_concept_names(spark, cdr, billing, vocab_map):
    """{concept_id: concept_name} for the vocabulary (for the per-iter top-terms
    log). A small filtered read of `concept`, mirroring _corpus_load's lookup;
    done only when topic logging is on, so a cache-hit fit without logging skips
    the BigQuery round-trip."""
    from pyspark.sql import functions as F
    cids = [int(c) for c in vocab_map.keys()]
    rows = (spark.read.format("bigquery")
            .option("table", f"{cdr}.concept")
            .option("parentProject", billing).load()
            .select("concept_id", "concept_name")
            .where(F.col("concept_id").isin(cids))
            .collect())
    return {int(r["concept_id"]): r["concept_name"] for r in rows}


def _make_topic_evolution_logger(top_n, every_n, idx_to_cid, vocab_names, topic_labels):
    """Build a per-iteration callback that prints top-N terms per topic (STM parity,
    mirrors stm_bigquery_cloud._make_topic_evolution_logger). Each topic line shows
    its DAG-node block, E[beta], sum(lambda), peak, and its heaviest vocab terms
    (concept name, falling back to concept-id). Heaviest topics first. The runner
    wraps the call so a logging slip can't kill the fit."""
    from spark_vi.models.topic.diagnostics import topic_word_summary

    def _on_iter(iter_num, global_params, _elbo):
        if every_n <= 0 or iter_num % every_n != 0:
            return
        lam = global_params["lambda"]
        s = topic_word_summary(lam, top_n)
        order = np.argsort(s["row_sums"])[::-1]
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in order:
            ki = int(k)
            terms = ", ".join(
                f"{str(vocab_names.get(idx_to_cid.get(int(j)), idx_to_cid.get(int(j), '?')))[:24]}"
                f"({p:.3f})"
                for j, p in zip(s["top_indices"][ki], s["top_probs"][ki]))
            blk = f" [{topic_labels[ki]:>16.16}]" if topic_labels is not None else ""
            print(f"[driver]    topic {ki:>2}{blk}  E[β]={s['mass_fraction'][ki]:.4f}  "
                  f"Σλ={s['row_sums'][ki]:.3g}  peak={s['peak'][ki]:.3f}  | {terms}",
                  flush=True)
    return _on_iter


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
    # assembly / DAG. `disease` selects both the foreground cohort and the label-
    # DAG anchors (cohorts.disease_anchors): single-disease (diabetes, eds) roots
    # at one anchor; a multi-disease name (rare6) builds a forest of subtrees.
    p.add_argument("--disease", default="diabetes")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")
    # gating
    p.add_argument("--n-bg", type=int, default=20)
    p.add_argument("--tpn", type=int, default=5)
    p.add_argument("--node-alpha-scale", type=float, default=1.0,
                   help="Multiplier on the per-node-topic Dirichlet alpha vs the "
                        "background alpha (1/K). 1.0 = symmetric (default); <1 "
                        "down-weights disease-node topics (asymmetric prior).")
    # SVI
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--mini-batch-fraction", type=float, default=0.0,
                   help="SVI mini-batch fraction in (0,1]; 0 = full-batch. "
                        "Mini-batching makes the decaying step size legitimate.")
    p.add_argument("--learning-rate-tau0", type=float, default=1.0,
                   help="SVI step-size delay tau0 in rho=(tau0+t+1)^-kappa; larger "
                        "= gentler slow start (less aggressive early steps).")
    p.add_argument("--learning-rate-kappa", type=float, default=0.7,
                   help="SVI step-size decay exponent kappa in (0.5,1]; larger "
                        "decays faster.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    p.add_argument("--spectral-max-vocab", type=int, default=8000)
    p.add_argument("--spectral-method", choices=["auto", "dense", "scalable"],
                   default="auto",
                   help="Spectral anchor-word init strategy: 'auto' routes by "
                        "vocab size vs --spectral-max-vocab, 'dense' forces the "
                        "exact single-driver path, 'scalable' forces the "
                        "distributed random-projection path (ADR 0037).")
    p.add_argument("--anchor-scope", choices=["closure", "frontier"],
                   default="closure",
                   help="Which docs feed each spectral anchor set: 'closure' "
                        "(node u from every doc with u in its closure, background "
                        "from all docs) or 'frontier' (node u only from docs where "
                        "u is the most-specific attested node, background only from "
                        "empty-frontier docs) — 'frontier' stops background/ancestors "
                        "stealing a descendant's defining anchor.")
    # per-iter topic logging (STM parity)
    p.add_argument("--print-topics-every", type=int, default=0,
                   help="Print top-N terms per topic every N iters (0 = off). "
                        "Resolves vocab names via a small concept read.")
    p.add_argument("--top-n-tokens", type=int, default=8)
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
                window_days=args.window_days, disease=args.disease, min_n=args.min_n,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                strip_mode=args.strip_mode)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        corpus_stats = _log_corpus_stats(bundle, lay)

        from spark_vi.mllib.topic.stm import resolve_spectral_method
        resolved_spectral = resolve_spectral_method(
            args.spectral_method, len(bundle.vocab_map), threshold=args.spectral_max_vocab)
        print(f"[driver]   spectral init: requested={args.spectral_method} "
              f"resolved={resolved_spectral}", flush=True)

        with _phase(f"gated-svi fit (init={args.init}, K={lay.K})"):
            est = GatedLDAEstimator(
                featuresCol="features", labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab,
                spectralMethod=args.spectral_method,
                anchorScope=args.anchor_scope,
                nodeAlphaScale=args.node_alpha_scale,
                miniBatchFraction=args.mini_batch_fraction,
                learningRateTau0=args.learning_rate_tau0,
                learningRateKappa=args.learning_rate_kappa)
            if args.print_topics_every > 0:
                vocab_names = _vocab_concept_names(
                    spark, args.cdr, args.billing, bundle.vocab_map)
                idx_to_cid = {idx: cid for cid, idx in bundle.vocab_map.items()}
                topic_labels = _topic_node_labels(
                    lay, bundle.int2cid, bundle.name_by_id, args.n_bg)
                est.setOnIteration(_make_topic_evolution_logger(
                    args.top_n_tokens, args.print_topics_every,
                    idx_to_cid, vocab_names, topic_labels))
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
                  f"multi_frontier_rate={metrics['multi_frontier_rate']:.3f} "
                  f"ap_macro={metrics['ap_macro']:.3f} "
                  f"ap_prevalence_weighted={metrics['ap_prevalence_weighted']:.3f} "
                  f"recall_at_k={metrics['recall_at_k']} "
                  f"test_coarsening_rate={bundle.ledger.get('test_coarsening_rate')}",
                  flush=True)
            det = metrics["detection"]
            op90 = det["operating_points"].get("0.90", {})
            print(f"[driver]   detection (case vs background): "
                  f"auc={det['auc']:.3f} ap={det['ap']:.3f} "
                  f"(prevalence={det['prevalence']:.3f}, "
                  f"n_fg={det['n_foreground']}/n_bg={det['n_background']}); "
                  f"disease_mass auc={det['auc_disease_mass']:.3f}; "
                  f"@90%-sens: bg_fpr={op90.get('bg_fpr', float('nan')):.3f} "
                  f"precision={op90.get('precision', float('nan')):.3f}; "
                  f"bg_mass mean bg={det['bg_mass_background_mean']:.3f} "
                  f"fg={det['bg_mass_foreground_mean']:.3f}", flush=True)
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
                "spectral_method_requested": args.spectral_method,
                "spectral_method_resolved": resolved_spectral,
                "anchor_scope": args.anchor_scope,
                "disease": args.disease, "min_n": args.min_n, "strip_mode": args.strip_mode,
                "node_alpha_scale": args.node_alpha_scale,
                "mini_batch_fraction": args.mini_batch_fraction,
                "learning_rate_tau0": args.learning_rate_tau0,
                "learning_rate_kappa": args.learning_rate_kappa,
                "max_iter": args.max_iter, "metrics": metrics, "ledger": bundle.ledger,
                "corpus_stats": corpus_stats,
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
