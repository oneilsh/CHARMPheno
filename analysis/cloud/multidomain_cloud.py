"""Cloud fit driver for the MULTI-DOMAIN (two-domain, MixEHR-style) gated topic
model: conditions (domain A) + drugs (domain B) over two independent
vocabularies, sharing the gated per-document theta with a CONDITION-ONLY gate
(gate ⟂ domain; SP3b arc design). Mirrors dag_placement_cloud.py's structure
(argparse in parse_args, _driver_common, assemble -> GatedLDAEstimator.fit ->
save) but wires the TWO-domain path: it loads condition_era AND drug_era
separately, windows both to the same per-patient cohort window, assembles a
TwoDomainBundle (charmpheno.omop.two_domain.assemble_two_domain_from_events),
fits GatedLDAEstimator(featuresCols=["features_a","features_b"]) to a per-domain
dict lambda, logs a dead-node init-quality read, and writes the VIResult via
spark_vi.io.export.save_result (dict-lambda aware since SP3a).

K is EMERGENT (n_bg + surviving-DAG-nodes * tpn), so there is no --K. Resume is
unsupported (GatedLDAModel is not persistable in v1).

Only parse_args + the pure dead_node_report helper are unit-tested
(test_multidomain_cloud.py); the main() body (live BQ load + Dataproc fit +
artifact write) is CLUSTER-COVERED (make multidomain-bq-smoke), not run here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session


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
        description="Multi-domain (condition+drug) gated topic-model fit.")
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
    p.add_argument("--source-table-drug", default="drug_era")
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
    return args


def _window_drug_events_to_cohort(cond_windowed, drug_df, *,
                                   cond_date_col, drug_date_col, window_days):
    """Window the drug frame to the SAME per-patient cohort window as the
    (already-windowed) condition frame, carrying source_cohort across.

    The condition cohort (apply_population_disease_cohort) has already windowed
    and tagged the condition events with source_cohort but dropped the index
    date. We reconstruct each (person, source_cohort) window START as the min
    in-window condition date and keep drug rows in [start, start+window_days).
    This keeps the two domains on the same window and gives every windowed
    condition-patient their aligned drug BOW (empty if none). Cluster-covered.
    """
    from pyspark.sql import functions as F
    bounds = (cond_windowed.groupBy("person_id", "source_cohort")
              .agg(F.min(cond_date_col).alias("_win_start")))
    return (
        drug_df.join(bounds, on="person_id", how="inner")
        .where(F.col(drug_date_col) >= F.col("_win_start"))
        .where(F.col(drug_date_col)
               < F.date_add(F.col("_win_start"), window_days))
        .drop("_win_start")
    )


def _log_corpus_stats(bundle, lay):
    """Log + return train/test doc counts, per-source_cohort breakdown, how many
    docs carry a frontier, and the per-domain vocab / topic-structure dims."""
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

    stats = {"train": _stats(bundle.train_df, "train"),
             "test": _stats(bundle.test_df, "test"),
             "vocab_size_a": len(bundle.vocab_map_a),
             "vocab_size_b": len(bundle.vocab_map_b),
             "K": lay.K, "n_nodes": len(lay.nodes),
             "n_bg": lay.n_bg, "tpn": lay.tpn}
    print(f"[driver]   corpus: V_a={stats['vocab_size_a']} (cond) "
          f"V_b={stats['vocab_size_b']} (drug), K={lay.K} topics "
          f"({lay.n_bg} bg + {len(lay.nodes)} nodes x {lay.tpn} tpn)", flush=True)
    return stats


def main(argv=None) -> int:
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.case_finding_assembly import (
        _FOREST_ROOT_CID, load_condition_dag)
    from charmpheno.omop.cohorts import (
        apply_population_disease_cohort, disease_anchors)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    from charmpheno.omop.two_domain import (
        DomainVocabSpec, assemble_two_domain_from_events)
    from spark_vi.io.export import save_result
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout

    args = parse_args(argv)
    configure_logging()
    with make_spark_session(app_name="multidomain-gated-fit") as spark:
        with _phase("load + window both domains (condition + drug)"):
            # Load BOTH domains WITHOUT `cohort=`: load_omop_bigquery's cohort
            # post-filter picks its date_col by CONDITION source-table and would
            # pick the wrong column for drug_era (Task 1 footgun). Windowing /
            # frontier are handled below + inside the two-domain assembler, exactly
            # as case_finding_assembly.assemble_case_finding_corpus does (it also
            # never passes cohort=).
            cond_raw = load_omop_bigquery(
                spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                person_sample_mod=args.person_mod,
                source_table=args.source_table_cond)
            drug_raw = load_omop_bigquery(
                spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                person_sample_mod=args.person_mod,
                source_table=args.source_table_drug)
            cond_date_col = "condition_era_start_date"
            drug_date_col = "drug_era_start_date"

            # condition cohort = whole-pop background + one disease foreground,
            # windowed + source_cohort-tagged (forward mode; mirrors the single-
            # domain forward path).
            cond_events = apply_population_disease_cohort(
                cond_raw, disease=args.disease, window_days=args.window_days,
                spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                date_col=cond_date_col, prior_obs_days=args.prior_obs_days)
            # align drugs to the same per-patient window (condition-only gate:
            # drugs are features, never a frontier/label).
            drug_events = _window_drug_events_to_cohort(
                cond_events, drug_raw, cond_date_col=cond_date_col,
                drug_date_col=drug_date_col, window_days=args.window_days)

        with _phase("build condition DAG"):
            anchors = disease_anchors(args.disease)
            root = _FOREST_ROOT_CID if len(anchors) > 1 else None
            before_dag = load_condition_dag(
                spark, anchors=anchors, root=root,
                cdr=args.cdr, billing=args.billing)

        with _phase("assemble two-domain bundle"):
            doc_spec = PatientCohortDocSpec(min_doc_length=args.doc_min_length)
            bundle = assemble_two_domain_from_events(
                cond_events, drug_events, before_dag,
                doc_spec=doc_spec, min_n=args.min_n,
                vocab_a=DomainVocabSpec(
                    vocab_size=args.cond_vocab_size, min_df=args.cond_min_df,
                    min_patient_count=args.cond_min_patient_count),
                vocab_b=DomainVocabSpec(
                    vocab_size=args.drug_vocab_size, min_df=args.drug_min_df,
                    min_patient_count=args.drug_min_patient_count),
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                strip_mode=args.strip_mode)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        corpus_stats = _log_corpus_stats(bundle, lay)

        with _phase(f"multi-domain gated fit (init={args.init}, K={lay.K})"):
            est = GatedLDAEstimator(
                featuresCols=["features_a", "features_b"], labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                # explicit seed from --seed (insight 0070): a deliberate, recorded
                # choice, NOT the shim's silent seed=(seed or 0) default.
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab,
                spectralMethod=args.spectral_method,
                anchorScope=args.anchor_scope,
                spectralTopoOrder=args.spectral_topo_order,
                omega=args.omega, etaPerDomain=args.eta_per_domain)
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
        # concept). Two small filtered `concept` reads (cond + drug vocab ids).
        with _phase("resolve per-domain vocab names"):
            names_a_bycid = _vocab_concept_names(
                spark, args.cdr, args.billing, bundle.vocab_map_a)
            names_b_bycid = _vocab_concept_names(
                spark, args.cdr, args.billing, bundle.vocab_map_b)
            idx2name = {0: _idx_to_name(bundle.vocab_map_a, names_a_bycid),
                        1: _idx_to_name(bundle.vocab_map_b, names_b_bycid)}

        if args.top_n_tokens > 0:
            with _phase("final topic dump (top terms per domain)"):
                labels = _topic_block_labels(lay, names, args.n_bg)
                _log_topics(lam_dict, idx2name, labels, args.top_n_tokens,
                            domain_tags={0: "cond", 1: "drug"})

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
                "strip_mode": args.strip_mode, "window_days": args.window_days,
                "omega": args.omega, "eta_per_domain": args.eta_per_domain,
                "spectral_method": args.spectral_method,
                "anchor_scope": args.anchor_scope,
                "spectral_topo_order": args.spectral_topo_order,
                "min_peak_ratio": args.min_peak_ratio,
                "dead_nodes": dead,
                "corpus_stats": corpus_stats,
                "ledger": bundle.ledger,
                "corpus_manifest": {
                    "cdr": args.cdr,
                    "source_table_cond": args.source_table_cond,
                    "source_table_drug": args.source_table_drug,
                    "person_mod": args.person_mod,
                    "cond_vocab_size": args.cond_vocab_size,
                    "cond_min_df": args.cond_min_df,
                    "cond_min_patient_count": args.cond_min_patient_count,
                    "drug_vocab_size": args.drug_vocab_size,
                    "drug_min_df": args.drug_min_df,
                    "drug_min_patient_count": args.drug_min_patient_count,
                    "prior_obs_days": args.prior_obs_days,
                    "holdout_frac": args.holdout_frac,
                    "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                    "name_by_id": {str(c): n
                                   for c, n in bundle.name_by_id.items()},
                    # Per-domain vocabularies + concept names: makes the saved
                    # artifact self-describing so a later no-refit inspection can
                    # map a topic's token indices (domain 0 = cond, 1 = drug) to
                    # concepts. {concept_id: index} + {concept_id: name} per domain.
                    "vocab_a": {str(c): i for c, i in bundle.vocab_map_a.items()},
                    "vocab_b": {str(c): i for c, i in bundle.vocab_map_b.items()},
                    "vocab_names_a": {str(c): n for c, n in names_a_bycid.items()},
                    "vocab_names_b": {str(c): n for c, n in names_b_bycid.items()}},
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved multidomain_gated result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
