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
    set. `features` (the BOW vector) is summed to a per-doc token count, fed to
    evaluate's length-conditioned FDR block. Pure; the driver collects the test
    set (held-out scale) before calling."""
    profiles, test_labels, doc_lengths = [], [], []
    for r in rows:
        aff = r["nodeAffinity"].toArray()
        profiles.append({u: float(aff[i]) for i, u in enumerate(lay.nodes)})
        test_labels.append({int(x) for x in r["frontier"]})
        doc_lengths.append(float(r["features"].toArray().sum()))
    return profiles, test_labels, doc_lengths


def _csv_list(s):
    """argparse type: 'a,b,c' -> ['a','b','c'] (empty/whitespace -> [])."""
    return [t.strip() for t in s.split(",") if t.strip()]


def _row_has(row, col):
    """True if a collected row (dict OR pyspark Row) carries field `col`."""
    fields = row.__fields__ if hasattr(row, "__fields__") else row.keys()
    return col in fields


def covariates_from_scored_rows(rows, col="covariates"):
    """Aligned (D, P) patient-covariate matrix from the SAME collected rows that
    `profiles_from_scored_rows` consumes, so the matrix aligns to evaluate()'s
    profiles by position. Returns None when `col` is absent (the baseline path).

    A null covariate cell (a doc with no sidecar match after the left join) is
    zero-filled rather than dropped: the 2x2 must compare an identical doc set
    across cells, so covariate presence never changes which docs are scored."""
    if not rows or not _row_has(rows[0], col):
        return None
    P = None
    for r in rows:
        v = r[col]
        if v is not None:
            P = len(v)
            break
    if P is None:
        return None                                    # column present but all null
    out = np.zeros((len(rows), P), dtype=float)
    for i, r in enumerate(rows):
        v = r[col]
        if v is not None:
            out[i] = v.toArray()
    return out


def join_covariates(scored_df, cov_df, *, key="person_id"):
    """Left-join the per-person covariate sidecar (`key`, covariates: Vector) onto
    the scored test docs, preserving EVERY scored doc (unmatched -> null covariate,
    zero-filled downstream). Left (not inner, as STM uses) so the doc set is stable
    across the 2x2 cells. Broadcasts the small sidecar."""
    from pyspark.sql import functions as F
    return scored_df.join(F.broadcast(cov_df), on=key, how="left")


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


def _node_coverage(train_df, lay):
    """Per-node training COVERAGE: the number of train docs where the node is in the
    allowed set (its closure was attested) — exactly the docs the node's learned
    alpha is estimated from. Returns {node_id: doc_count}. Collected from the frontier
    column at foreground scale (one groupBy), so cheap; only called when the learned
    alpha is logged."""
    from pyspark.sql import functions as F
    counts = (train_df.select("frontier").rdd
              .map(lambda r: frozenset(int(x) for x in (r[0] or [])))
              .countByValue())
    cov = {u: 0 for u in lay.nodes}
    for frontier, cnt in counts.items():
        covered = set()
        for f in frontier:
            covered.update(lay.closure(f))           # closure = node + all ancestors (node ids)
        for u in covered:
            if u in cov:
                cov[u] += int(cnt)
    return cov


def _spearman(x, y):
    """Spearman rank correlation via numpy (no scipy dependency in the driver)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _log_learned_alpha(model, lay, int2cid, name_by_id, n_bg, train_df):
    """Print the learned per-node Dirichlet alpha (optimizeDocConcentration) sorted
    high->low, mapped to condition names, alongside the background alpha AND each
    node's training coverage (docs the alpha is estimated from). Reports the Spearman
    correlation between learned alpha and coverage: does the learned alpha track node
    prevalence/coverage, or something else (e.g. footprint diffuseness)? Single-seed
    fits are multimodal (insight 0059), so read the range + correlation, not per-node
    point values."""
    alpha = model.result.global_params["alpha"]
    bg_alpha = float(alpha[0])                       # background block (all n_bg share it)
    cov = _node_coverage(train_df, lay)
    rows = []
    for u in lay.nodes:
        a_u = float(alpha[lay.block[u][0]])          # tying: all tpn topics share it
        nm = name_by_id.get(int2cid.get(u), str(u))
        rows.append((a_u, u, nm, cov.get(u, 0)))
    rows.sort(reverse=True)
    lo, hi = rows[-1][0], rows[0][0]
    # correlation over COVERED nodes only (uncovered alpha is just the init, uninformative)
    covered = [(a, c) for a, _, _, c in rows if c > 0]
    rho = _spearman([a for a, _ in covered], [c for _, c in covered]) if covered else float("nan")
    print(f"[driver]   learned alpha (optimizeDocConcentration): background={bg_alpha:.4g}; "
          f"node blocks min={lo:.4g} max={hi:.4g} (init was 1/K={1.0/lay.K:.4g}); "
          f"Spearman(alpha, coverage) over {len(covered)} covered nodes = {rho:.3f}", flush=True)
    for a_u, u, nm, c in rows:
        rel = ">bg" if a_u > bg_alpha else "<bg"
        print(f"[driver]     alpha[node {u:>2} {nm[:34]:<34}] = {a_u:.4g}  {rel}  "
              f"coverage={c}", flush=True)


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
    p.add_argument("--window-mode", choices=["forward", "lookback"], default="forward",
                   help="'forward': existing single-window behavior (label window "
                        "runs forward from the observation window). 'lookback': "
                        "pre-index feature frame (--lookback-days back from an "
                        "index date) + forward label frame (--label-window-days), "
                        "leakage-free by construction.")
    p.add_argument("--lookback-days", type=int, default=365,
                   help="window_mode=lookback only: how far back from the index "
                        "date the feature window extends.")
    p.add_argument("--label-window-days", type=int, default=365,
                   help="window_mode=lookback only: how far forward from the "
                        "index date the label window extends.")
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
                        "down-weights disease-node topics (asymmetric prior). When "
                        "--optimize-doc-concentration is set this is the INITIAL alpha.")
    p.add_argument("--optimize-doc-concentration", action="store_true",
                   help="Learn an asymmetric per-node Dirichlet alpha from data "
                        "(Wallach et al. 2009); node-alpha-scale sets the initial "
                        "alpha, the gated Newton step refines it. Off by default.")
    p.add_argument("--transform-alpha-mode", default="fitted",
                   choices=["fitted", "symmetric", "block_balanced"],
                   help="Deployment (fold-in) Dirichlet alpha: 'fitted' (default; "
                        "the fitted alpha) | 'symmetric' (flat 1/K or --transform-alpha; "
                        "neutral between nodes) | 'block_balanced' (nodes equal, "
                        "background collective mass = --transform-bg-weight). Decouples "
                        "the fitting-aid alpha from the deployment prior.")
    p.add_argument("--transform-alpha", type=float, default=0.0,
                   help="Per-topic alpha for --transform-alpha-mode symmetric (<=0 -> 1/K).")
    p.add_argument("--transform-bg-weight", type=float, default=0.5,
                   help="Collective background prior mass in (0,1) for "
                        "--transform-alpha-mode block_balanced.")
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
    p.add_argument("--spectral-topo-order", choices=["forward", "reverse"],
                   default="forward",
                   help="spectral init deflation order (forward=ancestors-first, "
                        "reverse=leaves-first); default forward")
    # per-iter topic logging (STM parity)
    p.add_argument("--print-topics-every", type=int, default=0,
                   help="Print top-N terms per topic every N iters (0 = off). "
                        "Resolves vocab names via a small concept read.")
    p.add_argument("--top-n-tokens", type=int, default=8)
    # Patient covariates (cheap prediction axis; see the 2x2 plan). Absent
    # formula -> baseline path, byte-identical to today. `x_d` is demographic /
    # nuisance adjusters (age, sex, site, birth-era) -- NOT the gating label.
    p.add_argument("--covariate-formula", default=None,
                   help="Formulaic patient-covariate formula (e.g. 'age + C(sex)'); "
                        "absent -> no covariates (baseline). Loaded via the shared "
                        "sidecar and joined to the corpus by person_id.")
    p.add_argument("--covariate-categorical", default=[], type=_csv_list,
                   help="Comma-separated categorical covariate source columns.")
    p.add_argument("--covariate-continuous", default=[], type=_csv_list,
                   help="Comma-separated continuous covariate source columns.")
    p.add_argument("--known-sex-only", action="store_true",
                   help="Restrict the person table to rows with a known sex "
                        "(mirrors the STM covariate driver).")
    p.add_argument("--pred-cov", choices=["on", "off"], default="off",
                   help="Prediction axis: 'on' feeds joined covariates to "
                        "evaluate() for the covariate-adjusted per-node readout. "
                        "Requires --covariate-formula.")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused (GatedLDAModel is not persistable in v1); "
                        "accepted for run_experiment parity.")
    return p.parse_args(argv)


def _load_case_finding_covariates(spark, args):
    """Load-or-build the patient-covariate sidecar for the case-finding corpus,
    keyed by person_id (ungated: one covariate vector per person). Returns
    (cov_df, covariate_names).

    Reuses the SAME shared sidecar loader/cache as the STM pipeline
    (`_covariates_load`) rather than a case-finding-specific copy, so the two
    pipelines converge on one covariate path and the loader survives an eventual
    STM removal. `x_d` is demographic/nuisance only; `validate_label_not_covariate`
    rejects a formula that smuggles in the gating label."""
    from _covariates_load import (
        load_or_build_covariates, validate_label_not_covariate,
    )
    from charmpheno.omop import load_person_table
    validate_label_not_covariate(
        args.covariate_categorical, args.covariate_continuous)
    with _phase("person table load"):
        person_df = load_person_table(
            spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
            person_sample_mod=args.person_mod, cohort=None,
            known_sex_only=args.known_sex_only)
    with _phase("covariates load"):
        cov_df, _spec, names = load_or_build_covariates(
            spark, person_df=person_df,
            covariate_formula=args.covariate_formula,
            categorical_cols=args.covariate_categorical,
            continuous_cols=args.covariate_continuous,
            cdr=args.cdr, source_table=args.source_table, cohort=None,
            person_mod=args.person_mod, cache_uri=args.cache_uri,
            key_cols=("person_id",), prior_obs_days=args.prior_obs_days)
    return cov_df, names


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
                strip_mode=args.strip_mode, window_mode=args.window_mode,
                lookback_days=args.lookback_days,
                label_window_days=args.label_window_days)
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
                spectralTopoOrder=args.spectral_topo_order,
                nodeAlphaScale=args.node_alpha_scale,
                optimizeDocConcentration=args.optimize_doc_concentration,
                transformAlphaMode=args.transform_alpha_mode,
                transformAlpha=args.transform_alpha,
                transformBgWeight=args.transform_bg_weight,
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
            if args.optimize_doc_concentration:
                _log_learned_alpha(model, lay, bundle.int2cid,
                                   bundle.name_by_id, args.n_bg, bundle.train_df)

        # Prediction axis (cheap): load + join per-person covariates when opted in.
        # Absent --pred-cov -> cov_df stays None and the block below is the exact
        # baseline path (same select, no covariates arg to evaluate).
        cov_df, covariate_names = (None, None)
        if args.pred_cov == "on":
            cov_df, covariate_names = _load_case_finding_covariates(spark, args)

        with _phase("transform + inline placement eval"):
            scored = model.transform(bundle.test_df)
            if cov_df is not None:
                scored = join_covariates(
                    scored.select("person_id", "nodeAffinity", "frontier", "features"),
                    cov_df, key="person_id")
            else:
                scored = scored.select("nodeAffinity", "frontier", "features")
            rows = scored.collect()
            profiles, test_labels, doc_lengths = profiles_from_scored_rows(rows, lay)
            covariates = covariates_from_scored_rows(rows) if cov_df is not None else None
            if covariates is not None:
                n_missing = int((~covariates.any(axis=1)).sum())
                print(f"[driver]   covariate-adjusted prediction ON: "
                      f"P={covariates.shape[1]} covariates "
                      f"({','.join(covariate_names or [])}); {n_missing}/{len(rows)} "
                      f"docs zero-filled (no sidecar match)", flush=True)
            metrics = evaluate(profiles, test_labels, lay,
                               doc_lengths=doc_lengths, covariates=covariates)
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
            ca = metrics.get("covariate_adjusted")
            if ca is not None:
                print(f"[driver]   covariate-adjusted (OOF-CV logistic, "
                      f"P={ca['n_covariates']}): detection auc "
                      f"score_cv={ca['detection_auc_score_cv']:.3f} -> "
                      f"adj={ca['detection_auc_adj']:.3f}; per-node macro auc "
                      f"score_cv={ca['auc_score_cv_macro']:.3f} -> "
                      f"adj={ca['auc_adj_macro']:.3f}; ap macro "
                      f"score_cv={ca['ap_score_cv_macro']:.3f} -> "
                      f"adj={ca['ap_adj_macro']:.3f} "
                      f"(delta = covariate lift at the DECISION)", flush=True)
            fdr = metrics["fdr"]
            # NOTE: the by_q dict-comprehension is built as its own variable rather
            # than inlined via `{{...}}` in the f-string below — braces escaped with
            # `{{`/`}}` across ADJACENT (implicitly-concatenated) f-string literals do
            # not form one interpolated expression; each literal is parsed on its own,
            # so a naive `f"...{{expr}}" f"...more"` split prints the comprehension's
            # SOURCE TEXT verbatim instead of evaluating it (verified empirically).
            by_q_summary = {q: (v["n_discoveries"], round(v["precision"], 3),
                                 round(v["recall"], 3))
                            for q, v in fdr["by_q"].items()}
            print(f"[driver]   fdr: by_q={by_q_summary} "
                  f"multimorbidity={fdr['multimorbidity']} "
                  f"saturation={fdr['saturation_rate']:.3f} "
                  f"zib_gap_mean={fdr['zib_gap_mean']:.3f} "
                  f"bins={fdr['n_length_bins_effective']}", flush=True)
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
                "spectral_topo_order": args.spectral_topo_order,
                "disease": args.disease, "min_n": args.min_n, "strip_mode": args.strip_mode,
                "window_mode": args.window_mode, "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "node_alpha_scale": args.node_alpha_scale,
                "optimize_doc_concentration": args.optimize_doc_concentration,
                "transform_alpha_mode": args.transform_alpha_mode,
                "mini_batch_fraction": args.mini_batch_fraction,
                "learning_rate_tau0": args.learning_rate_tau0,
                "learning_rate_kappa": args.learning_rate_kappa,
                "max_iter": args.max_iter, "metrics": metrics, "fdr": metrics["fdr"],
                "covariates": {
                    "formula": args.covariate_formula,
                    "categorical": args.covariate_categorical,
                    "continuous": args.covariate_continuous,
                    "names": covariate_names,
                    "pred_cov": args.pred_cov},
                "ledger": bundle.ledger,
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
