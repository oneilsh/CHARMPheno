"""Cloud fit driver for the MULTI-DOMAIN (N-domain, MixEHR-style) gated topic
model: conditions (always domain 0) plus any number of extra domains
(drug_era, observation, ... selected via --domains) over independent
per-domain vocabularies, sharing the gated per-document theta with a
CONDITION-ONLY gate (gate ⟂ domain; SP3b arc design, generalized to N domains
+ a lookback window mode in SP3c). Mirrors dag_placement_cloud.py's structure
(argparse in parse_args, _driver_common, assemble -> GatedLDAEstimator.fit ->
save) but wires the N-domain path: it loads condition_era and every extra
domain's source table (DOMAIN_REGISTRY), windows them either to one shared
forward window (--window-mode forward, the original exp-0070 shape) or to a
leakage-free pre-index lookback feature window + forward condition-only label
window (--window-mode lookback), assembles a MultiDomainBundle
(charmpheno.omop.multi_domain.assemble_multidomain_from_events), fits
GatedLDAEstimator(featuresCols=["features_0", ..., "features_{N-1}"]) to a
per-domain dict lambda, logs a dead-node init-quality read, and writes the
VIResult via spark_vi.io.export.save_result (dict-lambda aware since SP3a).

K is EMERGENT (n_bg + surviving-DAG-nodes * tpn), so there is no --K. Resume is
unsupported (GatedLDAModel is not persistable in v1).

Only parse_args + the pure helpers (dead_node_report, _topic_block_labels,
_log_topics, _vocab_concept_names, _idx_to_name, DOMAIN_REGISTRY,
_domain_vocab_spec) are unit-tested (test_multidomain_cloud.py); the main()
body (live BQ load + Dataproc fit + artifact write) is CLUSTER-COVERED (make
multidomain-bq-smoke), not run here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session

# The clinical-semantics layer: which OMOP source tables can be domains, their
# per-domain event date column, and a short display/persistence name. Condition is
# always domain 0; the others are selected by --domains. The engine + assembler
# stay index-based and never see these names (SP3c design).
DOMAIN_REGISTRY = {
    "condition_era": {"date_col": "condition_era_start_date", "name": "condition",
                      "arg": "cond"},
    "drug_era":      {"date_col": "drug_era_start_date",      "name": "drug",
                      "arg": "drug"},
    "observation":   {"date_col": "observation_date",         "name": "observation",
                      "arg": "obs"},
}


def _domain_vocab_spec(args, source_table):
    """DomainVocabSpec for a source table, reading that domain's --<arg>-* controls
    (cond/drug/obs) off the parsed args."""
    from charmpheno.omop.multi_domain import DomainVocabSpec
    a = DOMAIN_REGISTRY[source_table]["arg"]
    return DomainVocabSpec(
        vocab_size=getattr(args, f"{a}_vocab_size"),
        min_df=getattr(args, f"{a}_min_df"),
        min_patient_count=getattr(args, f"{a}_min_patient_count"))


def _parse_float_list(s):
    """Parse a comma-separated float list ("1.0,0.5") to [float, ...], or None if
    unset. Returned to the estimator as-is: unset MUST stay None so omega /
    etaPerDomain reach the shim as the pre-multi-domain default (scalar 1/K),
    never omega=[1, 1, ...] which is only legal WITH domains (gated_lda.py)."""
    if s is None:
        return None
    return [float(x) for x in s.split(",") if x.strip() != ""]


def dead_node_report(lam_dict, lay, *, min_peak_ratio=5.0):
    """PURE (Spark-free) init-quality read: node ids whose fitted per-domain topic
    mass never rose off the ~uniform prior in ANY domain -- a "dead node".

    For each DAG node, each domain m, and each of the node's topics k
    (lay.block[u]), the peak-to-mean ratio max(lam_dict[m][k]) / mean(lam_dict[m]
    [k]) measures how far the fitted topic-word row departed from flat: a row
    still at the ~uniform Dirichlet prior has ratio ~1, a concentrated row has a
    large ratio. A node is DEAD iff EVERY (domain, topic) ratio is below
    min_peak_ratio -- i.e. it never concentrated anywhere. A node that
    concentrated in even one domain/topic is alive and spared.

    This is the concrete sanity read the cluster smoke asserts is empty: the
    scalable spectral init is seed-fragile (insight 0070), and a dead node is
    the observable signature of a projection draw the EM did not rescue.

    Args:
        lam_dict: fitted per-domain dict lambda {m: (K, V_m)} (the multi-domain
            gated model's storage; MixEHR-style, Li, Nair, Lu et al. 2020).
        lay: the DagLayout (gives node ids and each node's topic block).
        min_peak_ratio: peak/mean threshold below which a topic row counts as
            still-at-prior (default 5.0).

    Returns:
        Sorted list of dead node ids (engine ids). Empty = every node
        concentrated in at least one domain.
    """
    dead = []
    for u in lay.nodes:
        alive = False
        for m, lam in lam_dict.items():
            for k in lay.block[u]:
                row = np.asarray(lam[k], dtype=float)
                mean = float(row.mean())
                if mean <= 0.0:
                    continue
                if float(row.max()) / mean >= min_peak_ratio:
                    alive = True
                    break
            if alive:
                break
        if not alive:
            dead.append(int(u))
    return sorted(dead)


def _topic_block_labels(lay, node_names, n_bg):
    """PURE: length-K list of per-topic block labels. Background topics [0, n_bg)
    label 'bg'; each DAG node u's topic block (lay.block[u]) labels with the node's
    name (node_names[u], engine-id keyed), falling back to the id.

    node_names is the engine-id -> display-name map main() builds by remapping
    concept-id names through int2cid (the same {i: name_by_id[c]} bridge the
    dead-node labeling uses)."""
    labels = ["bg"] * lay.K
    for u in lay.nodes:
        nm = node_names.get(u, str(u))
        for k in lay.block[u]:
            labels[int(k)] = nm
    return labels


def _log_topics(lam_dict, idx2name_by_domain, labels, top_n, *, domain_tags=None):
    """PURE (Spark-free) final topic dump: for each topic (heaviest total Sigma-lambda
    first), print its block label and its top-N heaviest tokens in EACH domain,
    mapped to concept names.

    lam_dict: fitted per-domain dict lambda {m: (K, V_m)}.
    idx2name_by_domain: {m: {token_index: display_name}} — the per-domain vocab
        resolved to concept names (main() builds it from vocab_map_m + a BigQuery
        concept-name read).
    labels: length-K per-topic block labels (_topic_block_labels), or None.
    domain_tags: optional {m: short_tag} (e.g. {0: 'cond', 1: 'drug'}); defaults
        to 'm{m}'.

    Returns the topic ids in printed (heaviest-first) order — for tests. A topic
    still at the prior in a domain simply shows that domain's flattest tokens; the
    ordering is by summed mass across domains so the data-rich topics surface."""
    from spark_vi.models.topic.diagnostics import topic_word_summary
    summ = {m: topic_word_summary(np.asarray(lam, dtype=float), top_n)
            for m, lam in lam_dict.items()}
    K = len(labels) if labels is not None else next(iter(lam_dict.values())).shape[0]
    total = np.zeros(K, dtype=float)
    for s in summ.values():
        total += np.asarray(s["row_sums"], dtype=float)
    order = [int(k) for k in np.argsort(total)[::-1]]
    tags = domain_tags or {}
    print("[driver]   === final topics (top terms per domain, heaviest first) ===",
          flush=True)
    for ki in order:
        blk = f" [{labels[ki]:>18.18}]" if labels is not None else ""
        print(f"[driver]    topic {ki:>2}{blk}  Sigma-lam(total)={total[ki]:.3g}",
              flush=True)
        for m in sorted(lam_dict.keys()):
            s = summ[m]
            names = idx2name_by_domain.get(m, {})
            terms = ", ".join(
                f"{str(names.get(int(j), j))[:22]}({p:.3f})"
                for j, p in zip(s["top_indices"][ki], s["top_probs"][ki]))
            print(f"[driver]        {tags.get(m, f'm{m}'):>4}: {terms}", flush=True)
    return order


def _vocab_concept_names(spark, cdr, billing, vocab_map):
    """{concept_id: concept_name} for one domain's vocabulary (for the final topic
    dump). A small filtered read of `concept`, mirroring dag_placement_cloud's
    identical helper (duplicated rather than shared because sibling drivers are not
    on the spark-submit --py-files path, so multidomain_cloud cannot import it)."""
    from pyspark.sql import functions as F
    cids = [int(c) for c in vocab_map.keys()]
    if not cids:
        return {}
    rows = (spark.read.format("bigquery")
            .option("table", f"{cdr}.concept")
            .option("parentProject", billing).load()
            .select("concept_id", "concept_name")
            .where(F.col("concept_id").isin(cids))
            .collect())
    return {int(r["concept_id"]): r["concept_name"] for r in rows}


def _idx_to_name(vocab_map, names_by_cid):
    """PURE: {token_index: display_name} from a {concept_id: index} vocab map and a
    {concept_id: name} lookup (name falls back to the concept id as a string)."""
    return {int(idx): names_by_cid.get(int(cid), str(cid))
            for cid, idx in vocab_map.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-domain (N-domain, condition + --domains) gated "
                     "topic-model fit.")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--out-dir", required=True)
    # Insight 0070: the scalable spectral init is seed-FRAGILE; a real corpus may
    # expose a projection draw the EM does not fully rescue, so the seed must be a
    # DELIBERATE, recorded choice -- never the shim's silent seed=(seed or 0)
    # default. --seed is therefore REQUIRED (its absence is a SystemExit).
    p.add_argument("--seed", type=int, required=True,
                   help="Spectral-init / fit seed. REQUIRED (insight 0070: the "
                        "scalable init is seed-fragile; the seed must be a "
                        "deliberate, recorded choice, not a silent default).")
    p.add_argument("--source-table-cond", default="condition_era")
    # SP3c: superseded by --domains, which drives main()'s domain_tables list
    # directly by source-table name. Kept accepted (unused by main()) for CLI
    # back-compat with run_experiment.py's invocation.
    p.add_argument("--source-table-drug", default="drug_era")
    p.add_argument("--domains", default="drug_era",
                   help="Comma list of EXTRA domains beyond conditions (subset of "
                        "{drug_era, observation}); condition is always domain 0. "
                        "Default 'drug_era' = the two-domain exp-0070 shape.")
    p.add_argument("--window-mode", choices=["forward", "lookback"], default="forward",
                   help="forward = one shared window (exp 0070). lookback = pre-index "
                        "feature window (all domains) + forward condition label window "
                        "(leakage-free; parity with the single-domain rare6 exps).")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--label-window-days", type=int, default=365)
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--disease", default="diabetes")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")
    p.add_argument("--doc-min-length", type=int, default=0)
    p.add_argument("--prior-obs-days", type=int, default=365)
    p.add_argument("--window-days", type=int, default=365)
    # Per-domain vocabulary controls (independent because conditions and drugs
    # have very different natural vocabulary sizes; SP3b design).
    p.add_argument("--cond-vocab-size", type=int, default=5000)
    p.add_argument("--cond-min-df", type=int, default=20)
    p.add_argument("--cond-min-patient-count", type=int, default=20)
    p.add_argument("--drug-vocab-size", type=int, default=2000)
    p.add_argument("--drug-min-df", type=int, default=20)
    p.add_argument("--drug-min-patient-count", type=int, default=20)
    p.add_argument("--obs-vocab-size", type=int, default=1500)
    p.add_argument("--obs-min-df", type=int, default=20)
    p.add_argument("--obs-min-patient-count", type=int, default=20)
    # gating
    p.add_argument("--n-bg", type=int, default=20)
    p.add_argument("--tpn", type=int, default=5)
    # per-domain generative knobs (comma-separated; unset -> None -> shim default).
    # omega = per-modality tempering weight; eta-per-domain = per-domain topic-word
    # Dirichlet concentration. Both are only legal WITH domains (gated_lda.py).
    p.add_argument("--omega", default=None,
                   help="Per-domain modality weight, comma-separated (e.g. "
                        "'1.0,0.5'); unset = scalar default (all domains equal).")
    p.add_argument("--eta-per-domain", default=None,
                   help="Per-domain topic-word Dirichlet eta, comma-separated "
                        "(e.g. '0.1,0.2'); unset = the shim's scalar 1/K default.")
    # SVI
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    # SVI optimizer schedule (mirrors dag_placement_cloud). mini_batch_fraction
    # 0.0 = full-batch (every iter sees the whole corpus). A value in (0, 1]
    # switches to mini-batch stochastic VI (Hoffman et al. 2013), which is what
    # makes the decaying Robbins-Monro step legitimate; then the per-iter ELBO is
    # a noisy estimate so the relative-ELBO early stop is unreliable and max_iter
    # is the real budget (size it for enough epochs: max_iter * fraction).
    p.add_argument("--mini-batch-fraction", type=float, default=0.0,
                   help="SVI mini-batch fraction in (0,1]; 0 = full-batch.")
    p.add_argument("--learning-rate-tau0", type=float, default=1.0,
                   help="SVI slow-start (tau0); 10 tames noisy early mini-batches.")
    p.add_argument("--learning-rate-kappa", type=float, default=0.7,
                   help="SVI forgetting rate (kappa); 0.7 = standard text decay.")
    p.add_argument("--init", choices=["random", "spectral"], default="spectral")
    p.add_argument("--spectral-max-vocab", type=int, default=8000)
    p.add_argument("--spectral-method", choices=["auto", "dense", "scalable"],
                   default="auto")
    p.add_argument("--anchor-scope", choices=["closure", "frontier"],
                   default="closure")
    p.add_argument("--spectral-topo-order", choices=["forward", "reverse"],
                   default="forward")
    p.add_argument("--min-peak-ratio", type=float, default=5.0,
                   help="dead_node_report threshold: a node's per-domain topic "
                        "row is 'still at prior' if peak/mean < this.")
    p.add_argument("--top-n-tokens", type=int, default=8,
                   help="Final topic dump: top-N heaviest tokens to print per "
                        "topic per domain (0 disables the dump).")
    p.add_argument("--resume-from", default="",
                   help="Unused (GatedLDAModel is not persistable in v1); "
                        "accepted for run_experiment parity.")
    args = p.parse_args(argv)
    # Parse the two comma-lists to float lists (None when unset).
    args.omega = _parse_float_list(args.omega)
    args.eta_per_domain = _parse_float_list(args.eta_per_domain)
    # --domains is EXTRA domains beyond conditions (which is always domain 0 and
    # never itself a --domains entry); normalize the comma-list to a list of
    # source-table names and validate against DOMAIN_REGISTRY.
    args.domains = [d for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in args.domains if d not in DOMAIN_REGISTRY or d == "condition_era"]
    if unknown:
        p.error(f"--domains entries must be extra domains in "
                f"{sorted(k for k in DOMAIN_REGISTRY if k != 'condition_era')}; "
                f"got {unknown}")
    # --source-table-cond must be a registered domain (DOMAIN_REGISTRY[cond_table]
    # is looked up unguarded in main()); catch an unregistered condition source
    # table here with a clean error rather than a raw KeyError deep in main().
    if args.source_table_cond not in DOMAIN_REGISTRY:
        p.error(f"--source-table-cond must be one of {sorted(DOMAIN_REGISTRY)}; "
                f"got {args.source_table_cond!r}")
    return args


def _window_events_to_cohort(cond_windowed, dom_df, *,
                             cond_date_col, dom_date_col, window_days):
    """Window a secondary-domain event frame to the SAME per-patient cohort window
    as the (already-windowed) condition frame, carrying source_cohort across.
    Domain-neutral (was _window_drug_events_to_cohort; SP3c). Cluster-covered."""
    from pyspark.sql import functions as F
    bounds = (cond_windowed.groupBy("person_id", "source_cohort")
              .agg(F.min(cond_date_col).alias("_win_start")))
    return (
        dom_df.join(bounds, on="person_id", how="inner")
        .where(F.col(dom_date_col) >= F.col("_win_start"))
        .where(F.col(dom_date_col) < F.date_add(F.col("_win_start"), window_days))
        .drop("_win_start")
    )


def _log_corpus_stats(bundle, lay, domain_names):
    """Log + return train/test doc counts, per-source_cohort breakdown, how many
    docs carry a frontier, and the per-domain vocab / topic-structure dims.

    domain_names labels the printout / stats dict (bundle.vocab_maps is index-only;
    the driver owns the N clinical names, in domain order)."""
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
              + ", ".join(f"{k}={n}" for k, (n, _) in sorted(by.items())),
              flush=True)
        return {"n_docs": total, "n_frontier": fg,
                "by_source_cohort": {k: n for k, (n, _) in by.items()}}

    vocab_sizes = {n: len(vm) for n, vm in zip(domain_names, bundle.vocab_maps)}
    stats = {"train": _stats(bundle.train_df, "train"),
             "test": _stats(bundle.test_df, "test"),
             "vocab_sizes": vocab_sizes,
             "K": lay.K, "n_nodes": len(lay.nodes),
             "n_bg": lay.n_bg, "tpn": lay.tpn}
    v_str = ", ".join(f"V_{n}={v}" for n, v in vocab_sizes.items())
    print(f"[driver]   corpus: {v_str}, K={lay.K} topics "
          f"({lay.n_bg} bg + {len(lay.nodes)} nodes x {lay.tpn} tpn)", flush=True)
    return stats


def main(argv=None) -> int:
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.case_finding_assembly import (
        _FOREST_ROOT_CID, load_condition_dag)
    from charmpheno.omop.cohorts import (
        apply_population_disease_cohort, case_finding_index_table, disease_anchors)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    from charmpheno.omop.multi_domain import (
        assemble_multidomain_from_events, lookback_feature_frames)
    from spark_vi.io.export import save_result
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout

    args = parse_args(argv)
    configure_logging()
    with make_spark_session(app_name="multidomain-gated-fit") as spark:
        # Domain 0 is always conditions; --domains supplies the extra domains in
        # order. domain_tables/domain_names/date_cols/vocab_specs are all index-
        # aligned to this ordering and thread through the rest of main() (bundle
        # fields are index-only; the driver owns the clinical names).
        cond_table = args.source_table_cond
        domain_tables = [cond_table, *args.domains]     # source tables, domain order
        domain_names = [DOMAIN_REGISTRY[t]["name"] for t in domain_tables]
        date_cols = [DOMAIN_REGISTRY[t]["date_col"] for t in domain_tables]
        vocab_specs = [_domain_vocab_spec(args, t) for t in domain_tables]

        with _phase(f"load {len(domain_tables)} domains: {domain_names}"):
            # Load every domain WITHOUT `cohort=`: load_omop_bigquery's cohort
            # post-filter picks its date_col by CONDITION source-table and would
            # pick the wrong column for a non-condition domain (Task 1 footgun).
            # Windowing / frontier are handled below + inside the multi-domain
            # assembler, exactly as case_finding_assembly.assemble_case_finding_corpus
            # does (it also never passes cohort=).
            raws = [load_omop_bigquery(
                        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                        person_sample_mod=args.person_mod, source_table=t)
                    for t in domain_tables]

        with _phase(f"window ({args.window_mode}) + assemble"):
            cond_date_col = date_cols[0]
            if args.window_mode == "lookback":
                # Leakage-free: pre-index FEATURE window (all domains) + forward
                # condition-only LABEL window from ONE shared index (case_finding_
                # index_table), parity with the single-domain rare6 exps.
                index_df = case_finding_index_table(
                    raws[0], disease=args.disease, spark=spark,
                    cdr_dataset=args.cdr, billing_project=args.billing,
                    date_col=cond_date_col, prior_obs_days=args.prior_obs_days,
                    label_window_days=args.label_window_days)
                feats, cond_label = lookback_feature_frames(
                    raws, index_df, date_cols,
                    lookback_days=args.lookback_days,
                    label_window_days=args.label_window_days)
                cond_feature, extra_features, label_arg = feats[0], feats[1:], cond_label
            else:  # forward
                # condition cohort = whole-pop background + one disease foreground,
                # windowed + source_cohort-tagged; mirrors the single-domain forward
                # path. Extra domains are aligned to the SAME per-patient window
                # (condition-only gate: extra domains are features, never a
                # frontier/label).
                cond_feature = apply_population_disease_cohort(
                    raws[0], disease=args.disease, window_days=args.window_days,
                    spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                    date_col=cond_date_col, prior_obs_days=args.prior_obs_days)
                extra_features = [
                    _window_events_to_cohort(
                        cond_feature, raw, cond_date_col=cond_date_col,
                        dom_date_col=dc, window_days=args.window_days)
                    for raw, dc in zip(raws[1:], date_cols[1:])]
                label_arg = None

            anchors = disease_anchors(args.disease)
            root = _FOREST_ROOT_CID if len(anchors) > 1 else None
            before_dag = load_condition_dag(
                spark, anchors=anchors, root=root, cdr=args.cdr, billing=args.billing)

            doc_spec = PatientCohortDocSpec(min_doc_length=args.doc_min_length)
            bundle = assemble_multidomain_from_events(
                cond_feature, extra_features, before_dag, doc_spec=doc_spec,
                min_n=args.min_n, vocab_specs=vocab_specs,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                strip_mode=args.strip_mode, label_events=label_arg)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        corpus_stats = _log_corpus_stats(bundle, lay, domain_names)

        with _phase(f"multi-domain gated fit (init={args.init}, K={lay.K})"):
            feature_cols = [f"features_{i}" for i in range(len(domain_tables))]
            est = GatedLDAEstimator(
                featuresCols=feature_cols, labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                # explicit seed from --seed (insight 0070): a deliberate, recorded
                # choice, NOT the shim's silent seed=(seed or 0) default.
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab,
                spectralMethod=args.spectral_method,
                anchorScope=args.anchor_scope,
                spectralTopoOrder=args.spectral_topo_order,
                omega=args.omega, etaPerDomain=args.eta_per_domain,
                miniBatchFraction=args.mini_batch_fraction,
                learningRateTau0=args.learning_rate_tau0,
                learningRateKappa=args.learning_rate_kappa)
            model = est.fit(bundle.train_df)

        with _phase("dead-node init-quality read"):
            lam_dict = model.result.global_params["lambda"]
            dead = dead_node_report(lam_dict, lay, min_peak_ratio=args.min_peak_ratio)
            # names are concept-id keyed; remap engine-id -> concept-id -> name.
            names = {i: bundle.name_by_id.get(c, str(c))
                     for i, c in bundle.int2cid.items()}
            if dead:
                labeled = ", ".join(f"{u}({names.get(u, u)})" for u in dead)
                print(f"[driver]   DEAD NODES (topic stuck at prior in every "
                      f"domain, min_peak_ratio={args.min_peak_ratio}): "
                      f"{len(dead)} -> {labeled}", flush=True)
            else:
                print(f"[driver]   dead-node report: EMPTY (every node "
                      f"concentrated in >=1 domain; init OK)", flush=True)

        # Resolve each domain's vocabulary to concept names ONCE: used both for the
        # final topic dump and persisted into the manifest so the saved artifact is
        # self-describing (a later no-refit inspection can map token index ->
        # concept). N small filtered `concept` reads (one per domain's vocab ids).
        with _phase("resolve per-domain vocab names"):
            names_bycid = [_vocab_concept_names(spark, args.cdr, args.billing, vm)
                           for vm in bundle.vocab_maps]
            idx2name = {i: _idx_to_name(vm, names_bycid[i])
                        for i, vm in enumerate(bundle.vocab_maps)}

        if args.top_n_tokens > 0:
            with _phase("final topic dump (top terms per domain)"):
                labels = _topic_block_labels(lay, names, args.n_bg)
                _log_topics(lam_dict, idx2name, labels, args.top_n_tokens,
                            domain_tags={i: n for i, n in enumerate(domain_names)})

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            # dict-lambda-aware writer (SP3a): per-domain lambda {m: (K, V_m)} is
            # written as params/lambda_<m>.npy sidecars + dict_param_keys.
            save_result(model.result, out)
            manifest = {
                "model_class": "multidomain_gated",
                "init": args.init, "seed": args.seed,
                "K": lay.K, "n_bg": args.n_bg, "tpn": args.tpn,
                "disease": args.disease, "min_n": args.min_n,
                "strip_mode": args.strip_mode,
                "domains": domain_names, "window_mode": args.window_mode,
                "window_days": args.window_days,
                "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "omega": args.omega, "eta_per_domain": args.eta_per_domain,
                "mini_batch_fraction": args.mini_batch_fraction,
                "learning_rate_tau0": args.learning_rate_tau0,
                "learning_rate_kappa": args.learning_rate_kappa,
                "spectral_method": args.spectral_method,
                "anchor_scope": args.anchor_scope,
                "spectral_topo_order": args.spectral_topo_order,
                "min_peak_ratio": args.min_peak_ratio,
                "dead_nodes": dead,
                "corpus_stats": corpus_stats,
                "ledger": bundle.ledger,
                "corpus_manifest": {
                    "cdr": args.cdr,
                    "domain_tables": domain_tables,
                    "person_mod": args.person_mod,
                    # Per-domain vocab-fit knobs, keyed by NAME (domain 0 =
                    # condition, always first) -- generalizes the old hardcoded
                    # cond_vocab_size/drug_vocab_size pair to N domains.
                    "vocab_specs": {
                        n: {"vocab_size": vs.vocab_size, "min_df": vs.min_df,
                            "min_patient_count": vs.min_patient_count}
                        for n, vs in zip(domain_names, vocab_specs)},
                    "prior_obs_days": args.prior_obs_days,
                    "holdout_frac": args.holdout_frac,
                    "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                    "name_by_id": {str(c): n
                                   for c, n in bundle.name_by_id.items()},
                    # Per-domain vocabularies + concept names, keyed by NAME: makes
                    # the saved artifact self-describing so a later no-refit
                    # inspection can map a topic's token indices (domain i =
                    # domain_names[i]) to concepts. {concept_id: index} +
                    # {concept_id: name} per domain.
                    **{f"vocab_{domain_names[i]}": {str(c): j for c, j in vm.items()}
                       for i, vm in enumerate(bundle.vocab_maps)},
                    **{f"vocab_names_{domain_names[i]}":
                       {str(c): n for c, n in names_bycid[i].items()}
                       for i in range(len(bundle.vocab_maps))},
                },
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved multidomain_gated result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
