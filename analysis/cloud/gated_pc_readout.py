"""Standalone post-hoc case-finding readout on a FINISHED gated_pc run (no re-fit).

Loads a completed run's saved globals (gated_pc_result.npz: lambda/alpha/w_CK) +
manifest.json, reconstructs the OnlinePCLDAModel, reloads the cached
CaseFindingBundle the run was fit against, re-transforms train+test, and prints
the FULL case-finding readout for the gated_pc arm:

  - pc_topics_lr:   ranking AUC/AP + per-node precision@recall / recall@FDR +
                    case-vs-background detection (AUC/AP + precision@recall).
  - co-fit head:    the same readout on the head's own P(node)=sigmoid(w_CK·θ).

The other arms' summary AUC/AP are echoed from manifest["results"] for context
(only the gated_pc arm's globals are saved, so only it can be re-scored here; a
run fit with the current driver already computes+stores every arm's full readout
inline — this script is for re-reading, richer operating points, or a run whose
inline readout predates this eval).

The bundle is located by RECOMPUTING its assembly cache key from the manifest. On
a MISS the tool no longer gives up: if the manifest carries the full assembly
parameters (every run of the current driver does — `corpus_manifest` is the corpus
spec verbatim) it REBUILDS the corpus through the same seam the fit uses, Mondo
DAG build and SNOMED-climb included, writes it through to the cache and proceeds.
That is the difference between "re-score a saved fit on a cluster whose cache is
empty" and "re-run the fit". `--no-rebuild` restores the old fail-fast behaviour,
and `--bundle-path` still short-circuits the key recompute entirely.

Rebuilding re-runs the assembly, which is deterministic in practice but not
guaranteed to be (the CDR moves; assembly source changes), so a rebuilt corpus is
CHECKED against the saved fit before anything is scored: per-domain vocabulary
sizes against the saved λ's own V dimensions, and the rebuilt engine-id -> concept
map against the one the manifest recorded. A mismatch aborts with the drift named,
rather than reporting numbers for a model scored on a corpus it was not fit on.

Both readout paths of ADR 0046 are available here, with the same
`--readout-mode {driver,distributed,auto}` / `--readout-ab-check` semantics as the
fit driver: this tool IS the recovery path when a finished fit's readout output was
lost (exp 0103), and at whole-Mondo C the driver-collect readout is exactly what
does not fit — so re-reading has to be able to run distributed without re-fitting.

SUBMIT NOTE: the distributed path's mapPartitions kernels pickle by MODULE
REFERENCE, so `analysis/cloud/distributed_readout.py` must be importable on the
EXECUTORS as top-level `distributed_readout` — i.e. it must ride in --py-files
(the `gated-pc-readout` Makefile target does this; a hand-rolled spark-submit must
add `--py-files ...,<repo>/analysis/cloud/distributed_readout.py` or the
distributed mode dies unpickling on the executors).

Cluster-covered (Spark + live cache); the pure helpers (build_parser,
bundle_key_from_manifest, reconstruct_model) are unit-tested and the scoring body
(run_readout) is covered against the driver path on a local-Spark fixture in
tests/scripts/test_readout_integration.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import os
from pathlib import Path

import numpy as np

from _driver_common import (_phase, configure_logging, hard_exit,
                            install_stdout_tee, make_spark_session)
from disk_telemetry import start_disk_telemetry
from gated_pc_cloud import (
    _MONDO_DAG_SOURCES, _dump_partial_results,
    _read_readout_heads, distributed_score_arm, format_arm_readout,
    format_stacked_readout, multidomain_cache_key, multidomain_load_or_build,
    stacked_readout,
)

# What the batched solve got before fits recorded their own cap. Runs older than
# `readout_max_iter` in the manifest were, in practice, record runs at the driver's
# own default of 200 — a DEV smoke from that era has to be recovered with an
# explicit --readout-max-iter 60, which is exactly the retyping the manifest field
# removes for every fit written since.
_LEGACY_READOUT_MAX_ITER = 200


def _cprint(msg, **kw) -> None:
    """print() through term_colors.highlight_line (ERROR/WARN/HIT/MISS/
    rebuild/macro-AUC/eta-boost -- see term_colors.py). Identity when color
    is disabled or nothing matches, so this is safe on every "[readout]"
    status line in this module.

    LOCAL import, not module-level: this module rides in a self-referencing
    `--py-files` chain from other diagnostics (`hpoa_stage2_probe.py` imports
    it at module level and itself ships its own mapPartitions kernel on
    `--py-files`) -- term_colors must never become a top-level dependency an
    executor could be asked to resolve. See term_colors.py's module
    docstring."""
    from term_colors import highlight_line
    print(highlight_line(msg), **kw)


def resolve_readout_max_iter(cli_value, manifest):
    """Pick the batched-L-BFGS iteration cap for a re-readout, and say who won.

    Precedence: explicit CLI > the fit's own recorded ``readout_max_iter`` >
    the legacy default. Same doctrine as ``readout_theta_topm``: a recovery should
    REPRODUCE the run it is rescuing (the fit's cap is what its lost readout would
    have used, CHARM_DEV capping already folded in), while the CLI stays available
    for deliberately re-solving a run harder or cheaper than it was.

    Returns ``(max_iter, source)``; `source` is a human label for the log line, so
    a recovery's output states on its face which budget it ran under — the number
    alone cannot be read back as "the fit's" or "mine".
    """
    if cli_value is not None:
        return int(cli_value), "CLI"
    recorded = manifest.get("readout_max_iter")
    if recorded:
        return int(recorded), "manifest"
    return _LEGACY_READOUT_MAX_ITER, "legacy default"





# The ridge every readout of record has used since exp 0122 (insight 0089).
_DEFAULT_READOUT_L2 = 100.0


def resolve_readout_l2(cli_value, manifest):
    """Ridge strength for a re-readout: explicit CLI > the fit's recorded
    ``readout_l2`` > 100. Fits stopped recording it in the 2026-10 cleanup (the fit
    no longer reads out), so the default is the record setting, not the pre-0122
    1.0; re-reading a pre-0122 run that never recorded it needs ``--readout-l2 1``.
    Returns ``(l2, source)``."""
    if cli_value is not None:
        return float(cli_value), "CLI"
    recorded = manifest.get("readout_l2")
    if recorded is not None:
        return float(recorded), "manifest"
    return _DEFAULT_READOUT_L2, "default"


def observe_root_everywhere(df, C, mask_col="labelMask"):
    """Return `df` with `mask_col[0]` (the ROOT's observation flag) set to 1.0 on
    every row; every other element untouched.

    The stacked arm's whole mechanism (spec 2026-10-06 Part A): under
    `label_mask_mode=closure` the root is observed only on foreground rows, all
    positive, so its readout head is the degenerate constant 1.0 and the closure
    product `Π_{a ∈ closure(c)} σ(z_a)` carries no population prior. Observing the
    root on every row makes its head a case-vs-background logistic — the one
    HSLDA's root conditional is — while the batched solve stays per-node
    independent (`solve_batched_lr`), so the heads of nodes ≥ 1 are byte-identical
    to the prevalent arm's. Applied to the TRAIN split only; the test mask is the
    corpus's own, so the ranking axis scores the same cells it always did.

    Pure column arithmetic on the `array<double>` mask (no UDF): `[1.0] ++
    mask[1:]`."""
    from pyspark.sql import functions as F
    C = int(C)
    tail = (F.slice(F.col(mask_col), 2, C - 1) if C > 1
            else F.array().cast("array<double>"))
    return df.withColumn(mask_col, F.concat(F.array(F.lit(1.0)), tail))


def most_specific_members(members, parent_int):
    """Members that are not an ancestor of another member. On a NESTED label space
    (exp 0152's whole-Mondo fit) a group's label descendants include intermediate
    terms whose labels are the closure of their children's, so "which member" needs the
    most specific ones; a patient coded only at an intermediate level has no member
    truth and is not scored. `parent_int` is the readout DAG (int or str keys)."""
    mem = [int(m) for m in members]
    anc = set()
    for m in mem:
        stack, seen = list(parent_int.get(m, parent_int.get(str(m), []))), set()
        while stack:
            a = int(stack.pop())
            if a in seen:
                continue
            seen.add(a)
            stack.extend(parent_int.get(a, parent_int.get(str(a), [])))
        anc |= seen
    return [m for m in mem if m not in anc]


def top_members(members, parent_int, group=None):
    """Members with no proper ancestor among the other members, the group node itself
    excluded: the group's TOP-LEVEL label terms. On a nested label space a term's label
    is the closure of its descendants', so a top-level member rolls up its subtypes (a
    patient coded at chronic hepatitis C is a hepatitis C patient). This is the
    natural differential; `most_specific_members` instead scores only the leaves and
    drops every patient coded at an intermediate term (exp 0152 read 4: viral
    hepatitis lost hepatitis C, its bulk). `parent_int` is the readout DAG."""
    mem = [int(m) for m in members if group is None or int(m) != int(group)]
    ms = set(mem)
    out = []
    for m in mem:
        stack, seen, hit = list(parent_int.get(m, parent_int.get(str(m), []))), set(), False
        while stack and not hit:
            a = int(stack.pop())
            if a in seen:
                continue
            seen.add(a)
            hit = a in ms
            stack.extend(parent_int.get(a, parent_int.get(str(a), [])))
        if not hit:
            out.append(m)
    return out


def readout_node_keep(parent_int, int2cid, keep_cids, C):
    """(C,) bool: readout nodes whose Mondo cid is in `keep_cids`, every transitive
    ancestor of one (the stacked product needs the whole closure), and the root.
    Exp 0152: solve the ribbon's heads on a whole-Mondo fit at ribbon cost."""
    keep_cids = {int(c) for c in keep_cids}
    keep = [False] * int(C)
    keep[0] = True
    stack = [int(n) for n, c in int2cid.items() if int(c) in keep_cids and int(n) < C]
    while stack:
        n = stack.pop()
        if keep[n]:
            continue
        keep[n] = True
        stack.extend(int(p) for p in parent_int.get(n, parent_int.get(str(n), [])))
    return keep


def mask_to_nodes(df, keep, *, mask_col="labelMask"):
    """Zero `mask_col` outside `keep`: an unkept node observes no cell, so the batched
    solve treats it as degenerate (constant fallback, converged at iteration 1) and
    every pass carries only the kept heads. Column arithmetic over a JSON literal
    (the `widen_labels_to_readout` pattern), no UDF."""
    import json as _json
    from pyspark.sql import functions as F
    lit = F.from_json(F.lit(_json.dumps([1.0 if k else 0.0 for k in keep])),
                      "array<double>")
    return df.withColumn(mask_col, F.transform(F.col(mask_col),
                                               lambda x, i: x * lit[i]))


def _raw_tokens_name(select):
    return ("raw_baseline_tokens.json" if select == "freq"
            else f"raw_baseline_tokens_{select}.json")


def _load_lambda(run_dir):
    """The saved fit's λ: one array, or the multi-domain {m: (K, V_m)} dict."""
    npz = np.load(Path(run_dir) / "gated_pc_result.npz")
    if "lambda" in npz.files:
        return npz["lambda"]
    return {int(f.split("_", 1)[1]): npz[f] for f in npz.files
            if f.startswith("lambda_")}


def raw_feature_cols(df):
    """The per-domain BOW columns of a bundle frame, in domain order."""
    cols = sorted((c for c in df.columns if c.startswith("features_")),
                  key=lambda c: int(c.split("_", 1)[1]))
    return cols or (["features"] if "features" in df.columns else [])


def select_raw_codes(train_df, n_top, feat_cols):
    """The `n_top` (domain, vocab idx) tokens carried by the most TRAIN documents
    (document frequency, all domains pooled), sorted by (domain, idx). The raw-code
    baseline's feature set (exp 0147). One RDD pass; only (domain, idx) counts come
    back to the driver."""
    cols = list(feat_cols)

    def _toks(row):
        for d, c in enumerate(cols):
            v = row[c]
            idx = getattr(v, "indices", None)
            if idx is None:
                arr = v.toArray() if hasattr(v, "toArray") else (v or [])
                idx = [i for i, x in enumerate(arr) if x]
            for i in idx:
                yield ((d, int(i)), 1)
    counts = (train_df.select(*cols).rdd.flatMap(_toks)
              .reduceByKey(lambda a, b: a + b).collect())
    counts.sort(key=lambda kv: (-kv[1], kv[0]))
    return sorted(k for k, _ in counts[:int(n_top)])


def select_model_codes(lam, n_bg, n_top):
    """The `n_top` (domain, vocab idx) tokens the FIT puts at the head of its node
    topics: round-robin over the block topics (indices >= `n_bg`; the background is
    the frequent, shared vocabulary), each topic contributing its next-ranked unseen
    token per round, until `n_top` are chosen. Sorted by (domain, idx).

    Why: the frequency-selected raw baseline (`select_raw_codes`) never consults the
    model, so "codes beat θ" leaves open what the model is for. This set is the
    model's own per-node vocabulary — a model-informed feature selection with the
    same budget — so raw(frequency) vs raw(model) asks whether the fit picks better
    predictors than counting does. Within a topic, tokens rank by their per-domain
    normalized λ (each domain's β sums to 1, as the engine normalizes them); `lam` is
    the multi-domain dict {m: (K, V_m)} or a single (K, V) array."""
    doms = lam if isinstance(lam, dict) else {0: lam}
    keys = sorted(doms)
    K = int(np.asarray(doms[keys[0]]).shape[0])
    ranked = []
    for k in range(int(n_bg), K):
        cand = []
        for m in keys:
            row = np.asarray(doms[m][k], dtype=np.float64)
            tot = row.sum()
            if tot > 0:
                p = row / tot
                cand.extend((float(p[i]), int(m), int(i)) for i in np.flatnonzero(p))
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        ranked.append([(m, i) for _, m, i in cand])
    chosen, seen, r = [], set(), 0
    while len(chosen) < int(n_top) and any(r < len(t) for t in ranked):
        for t in ranked:
            if r < len(t) and t[r] not in seen:
                seen.add(t[r]); chosen.append(t[r])
                if len(chosen) >= int(n_top):
                    break
        r += 1
    return sorted(chosen)


def select_relevance_codes(lam, n_bg, quota, *, rel_lambda=0.6, exclude=None):
    """Per-domain token sets chosen by the fit's node topics under LDAvis relevance
    (Sievert & Shirley 2014): r(w|k) = λ log p(w|k) + (1 - λ) log(p(w|k) / p(w)),
    with p(w) the fit's own word marginal in that domain. λ = 1 is plain top words;
    0.6 (the dashboard's setting) lifts codes distinctive of a topic over codes
    common everywhere.

    Why this shape: the plain top-words selector (`select_model_codes`) spent its
    budget on the nodes' own attesting codes, which by the new-diagnosis rule are
    absent before the index for every patient being scored, and drew almost no
    measurements (exp 0163). Here `exclude` ({topic: domain-0 vocab idx}, the node's
    and its descendants' own codes, `topic_npmi.own_code_exclusions`) drops those
    words from each topic's ranking, and `quota` ({domain: n}) fixes the per-domain
    count, matched to the frequency baseline's mix, so the comparison is about which
    codes within a domain, not how many of each domain. Within a domain the node
    topics take turns (round-robin by rank) until the quota is met. Sorted
    (domain, idx)."""
    doms = lam if isinstance(lam, dict) else {0: lam}
    exclude = exclude or {}
    chosen = []
    for m, q in sorted(quota.items()):
        L = np.asarray(doms[m], dtype=np.float64)
        pw = L.sum(axis=0) / max(L.sum(), 1e-300)
        ranked = []
        for k in range(int(n_bg), L.shape[0]):
            tot = L[k].sum()
            if tot <= 0:
                continue
            beta = L[k] / tot
            with np.errstate(divide="ignore", invalid="ignore"):
                lb = np.log(np.where(beta > 0, beta, np.nan))
                r = rel_lambda * lb + (1.0 - rel_lambda) * (lb - np.log(
                    np.where(pw > 0, pw, np.nan)))
            r = np.nan_to_num(r, nan=-np.inf)
            if m == 0 and k in exclude:
                r[list(exclude[k])] = -np.inf
            order = np.argsort(-r, kind="stable")
            ranked.append([int(i) for i in order if np.isfinite(r[i])])
        seen, r_i = set(), 0
        while len(seen) < int(q) and any(r_i < len(t) for t in ranked):
            for t in ranked:
                if r_i < len(t) and t[r_i] not in seen:
                    seen.add(t[r_i])
                    if len(seen) >= int(q):
                        break
            r_i += 1
        chosen.extend((int(m), i) for i in seen)
    return sorted(chosen)


def raw_presence_kernel(get_pos, n_cols):
    """`(*domain vectors) -> DenseVector(n)`: 1.0 where the document carries a selected
    token. `get_pos()` returns {(domain, idx): column}; a broadcast on the cluster
    (ADR 0047). Nested so cloudpickle ships it by value."""
    def kernel(*vecs):
        from pyspark.ml.linalg import Vectors
        pos = get_pos()
        out = [0.0] * int(n_cols)
        for d, v in enumerate(vecs):
            if v is None:
                continue
            idx = getattr(v, "indices", None)
            if idx is None:
                arr = v.toArray() if hasattr(v, "toArray") else v
                idx = [i for i, x in enumerate(arr) if x]
            for i in idx:
                j = pos.get((d, int(i)))
                if j is not None:
                    out[j] = 1.0
        return Vectors.dense(out)
    return kernel


def attach_raw_presence(df, selected, feat_cols, *, out_col="topicDistribution"):
    """Replace `out_col` with the binary presence of the `selected` tokens, so the
    unchanged readout machinery (solve, stacked product, prevalence check) runs on raw
    codes instead of θ (exp 0147)."""
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql import functions as F
    pos = {(int(d), int(i)): j for j, (d, i) in enumerate(selected)}
    bc = df.sparkSession.sparkContext.broadcast(pos)
    udf = F.udf(raw_presence_kernel(lambda: bc.value, len(selected)), VectorUDT())
    out = df.drop(out_col) if out_col in df.columns else df
    return out.withColumn(out_col, udf(*[F.col(c) for c in feat_cols]))





def collect_set_patients(test_df, members, n_domains, *, max_patients=5000, seed=0,
                         label_col="label", length="distinct", ever_df=None):
    """Test-split persons positive on any of `members` (fit-width label indices):
    `[(truth positions into members, {d: {vocab idx: count}}, {d: length})]`.

    Filter and label slice are column arithmetic over a JSON literal (the
    `widen_labels_to_readout` pattern: nothing array-shaped rides a closure); the
    collect carries only the members' label cells and the sparse feature vectors.
    Capped at `max_patients` by a seeded Spark sample. `length` is the record length
    the presence model gets per domain: `distinct` codes (default) or `tokens`.

    `ever_df` (exp 0149; `(person_id, __ever: array<int>)` positions into `members`
    from the E4 sidecar): a person ever coded with a member is collected too, and the
    truth is the union of label-year and ever-coded members."""
    import json as _json
    from pyspark.sql import functions as F
    lit = F.from_json(F.lit(_json.dumps([int(m) for m in members])), "array<int>")
    lab = F.col(label_col)
    fcols = ([f"features_{d}" for d in range(int(n_domains))]
             if f"features_0" in test_df.columns else ["features"])
    base = test_df
    if ever_df is not None:
        base = base.join(ever_df, "person_id", "left")
        keep = (F.exists(lit, lambda i: lab[i] > 0)
                | (F.coalesce(F.size("__ever"), F.lit(0)) > 0))
    else:
        base = base.withColumn("__ever", F.array().cast("array<int>"))
        keep = F.exists(lit, lambda i: lab[i] > 0)
    df = (base.where(keep)
              .select(F.transform(lit, lambda i: lab[i]).alias("__y"),
                      F.coalesce(F.col("__ever"), F.array().cast("array<int>"))
                       .alias("__ever"), *fcols))
    n = df.count()
    if n > int(max_patients):
        df = df.sample(False, min(1.0, 1.2 * int(max_patients) / n), seed=int(seed))
    out = []
    for row in df.collect()[:int(max_patients)]:
        truth = sorted({j for j, v in enumerate(row["__y"]) if v and v > 0}
                       | {int(j) for j in (row["__ever"] or [])})
        bags, lens = {}, {}
        for d, c in enumerate(fcols):
            vec = row[c]
            if vec is None:
                continue
            idx = getattr(vec, "indices", None)
            if idx is None:
                arr = np.asarray(vec.toArray() if hasattr(vec, "toArray") else vec)
                idx = np.flatnonzero(arr)
                vals = arr[idx]
            else:
                vals = vec.values
            bags[d] = {int(i): float(c) for i, c in zip(idx, vals)}
            lens[d] = (float(np.sum(vals)) if length == "tokens"
                       else float(len(idx)))
        out.append((truth, bags, lens))
    return out, n


def ever_member_frame(spark, run_dir, members, member_cids, parent_adj, *,
                      sidecar_uri=None):
    """`(person_id, __ever: array<int>)`: positions into `members` of every member the
    person has EVER been coded with (E4 sidecar, whole record — before or after the
    index; training only), a sidecar Mondo term counting for every member at or above
    it (`prevalence_check.sidecar_nodes_to_readout`'s is-a roll-up). Exp 0149."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    from conversion_sidecar import try_load_sidecar
    wit = json.loads((Path(run_dir) / "conversion_sidecar.json").read_text())
    uri = sidecar_uri or wit.get("sidecar_uri")
    first = try_load_sidecar(spark, uri, wit["key"])
    if first is None:
        raise SystemExit(f"[differential] sidecar MISS at {uri}/{wit['key']} — rebuild "
                         "it on this cluster (make build-conversion-sidecar ID=N)")
    nodes = [int(r[0]) for r in first.select("node_cid").distinct().collect()]
    # positions shifted by one: the roll-up never targets id 0 (the readout root)
    pos_of = {i + 1: int(c) for i, c in enumerate(member_cids)}
    n2p = pv.sidecar_nodes_to_readout(nodes, pos_of, parent_adj, root_cid=-10**9)
    rows = [(int(n), int(j) - 1) for n, js in n2p.items() for j in js]
    if not rows:
        return spark.createDataFrame([], "person_id long, __ever array<int>")
    mdf = spark.createDataFrame(rows, "node_cid long, __pos int")
    return (first.join(F.broadcast(mdf), "node_cid").groupBy("person_id")
            .agg(F.collect_set("__pos").alias("__ever")))


def _test_prior_pairs(test_df, run_dir, sidecar_uri=None):
    """`(person_id, node_cid)` for test persons: every sidecar Mondo node first attested
    BEFORE the person's index (index_horizon). Exp 0151."""
    from pyspark.sql import functions as F
    from conversion_sidecar import try_load_index_horizon, try_load_sidecar
    wit = json.loads((Path(run_dir) / "conversion_sidecar.json").read_text())
    uri = sidecar_uri or wit.get("sidecar_uri")
    spark = test_df.sparkSession
    first = try_load_sidecar(spark, uri, wit["key"])
    hz = try_load_index_horizon(spark, uri, wit["key"])
    if first is None or hz is None:
        raise SystemExit(f"[differential] sidecar MISS at {uri}/{wit['key']} — rebuild "
                         "it on this cluster (make build-conversion-sidecar ID=N)")
    persons = test_df.select("person_id").distinct()
    return (first.join(hz.select("person_id", "index_date").join(persons, "person_id"),
                       "person_id")
            .where(F.col("first_attested_date") < F.col("index_date"))
            .select("person_id", "node_cid").distinct())


def _incident_test_frame(test_df, prior_pairs, prior_nodes, target_cids, parent_adj):
    """`test_df` minus the persons with a pre-index attestation of any target Mondo
    term (a group's members and the group node) or anything under one — the INCIDENT
    rule of exp 0151 (insight 0103): their pre-index record is pre-diagnosis."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    tgt = {i + 1: int(c) for i, c in enumerate(target_cids)}
    hit = pv.sidecar_nodes_to_readout(prior_nodes, tgt, parent_adj, root_cid=-10**9)
    if not hit:
        return test_df
    ndf = test_df.sparkSession.createDataFrame([(int(n),) for n in hit], "node_cid long")
    # The excluded persons are collected and BROADCAST back, so the (large) test frame
    # is filtered in place, never shuffled (0151 launch 1: ~440 s per group from a
    # sort-merge anti-join that shuffled every feature vector twice).
    prev = [int(r[0]) for r in prior_pairs.join(F.broadcast(ndf), "node_cid")
            .select("person_id").distinct().collect()]
    if not prev:
        return test_df
    pdf = test_df.sparkSession.createDataFrame([(x,) for x in prev], "person_id long")
    return test_df.join(F.broadcast(pdf), "person_id", "left_anti")



def run_differential(test_df, train_df, run_dir, pattern, *, adhoc=False,
                     results_name="results_readout_stacked_mass99.json",
                     heads_tag="stacked_mass99", max_patients=600, max_train=2000,
                     thin_frac=0.1, own_top=5, rel_lambda=0.6, rho=0.3, seed=0,
                     tag=None, next_question=False, nq_pool=100, nq_hide=0.2,
                     train_ever=False, multi=False, crossfit=False, sidecar_uri=None,
                     fast=False, max_members=None, incident=False,
                     most_specific=None, groups_like=None, member_rule="auto"):
    """Framing 1 — DIFFERENTIAL WITH REASONS (spec 2026-10-13 clinician framings,
    R0/R1; `differential.py`). Four engines on the same held-out patients positive on
    exactly one member: E1 presence model with subtype profiles, E2 renormalized
    stacked heads, E3 a multiclass head fit on the train split, E4 the gated
    per-disease likelihood. Records lose attestation codes and `[measured]` tokens
    first. Tempers (E1, E4) are fit on one half of the patients and every engine is
    scored on the other. Writes `differential[_<tag>].md/.json` (pooled only)."""
    import time
    import differential as dfm
    import model_acquisition as ma
    import voi_tables as vt
    from gated_pc_cloud import closure_matrix
    run_dir = Path(run_dir)
    bm, cm = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
    names_csv = run_dir / "concept_names.csv"
    ctx = vt.load_voi_context(
        run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
        own_codes_path=str(cm) if cm.exists() else None, results_name=results_name,
        names_path=str(names_csv) if names_csv.exists() else None,
        thin_frac=thin_frac, own_top=own_top, rel_lambda=rel_lambda)
    npz = np.load(run_dir / "gated_pc_result.npz")
    lam_keys = sorted((k for k in npz.files if k.startswith("lambda_")),
                      key=lambda k: int(k.split("_", 1)[1]))
    lams_fit = [npz[k] for k in lam_keys] if lam_keys else [npz["lambda"]]
    alpha = npz["alpha"]
    hz = np.load(run_dir / f"readout_heads_gated_pc_{heads_tag}.npz")
    rd = json.loads((run_dir / "readout_dag.json").read_text())
    M = closure_matrix({int(c): [int(x) for x in ps] for c, ps in rd["parent_int"].items()},
                       int(rd["C"]))
    heads = (hz["V"], hz["b_raw"], hz["degenerate"], hz["const"], int(hz["theta_topm"]))
    measured = set()
    if "measurement" in ctx.dom_names and ctx.inv_maps:
        dm = ctx.dom_names.index("measurement")
        measured = {int(v) for v, tok in ctx.inv_maps[dm].items() if int(tok) % 100 == 0}
    ev = np.asarray(ctx.sh["evidence"], dtype=np.float64)
    member_cid, parent_adj = {}, None
    prior_pairs, prior_nodes = None, []
    if train_ever or incident:
        # Exp 0149: members' Mondo ids (bundle int2cid) and the Mondo is-a, so the
        # sidecar's finer terms roll up onto the members.
        from readout_dag import load_mondo_parent_adj
        bmj = json.loads(bm.read_text())
        member_cid = {int(e): int(c) for e, c in bmj["int2cid"].items()}
        cm_ = (json.loads((run_dir / "manifest.json").read_text())
               .get("corpus_manifest") or {})
        parent_adj, _nm = load_mondo_parent_adj(
            cm_.get("mondo_version"), cm_.get("mondo_cache_dir") or "data/mondo")
        member_cid.update({int(e): int(c) for e, c in
                           json.loads((run_dir / "readout_dag.json").read_text())
                           ["int2cid"].items()})
    if incident:
        # Exp 0151: (person, Mondo node) first attested BEFORE the person's index, for
        # the test persons — computed once; each group then drops the persons with
        # any such node at or under one of its members or the group node itself.
        prior_pairs = _test_prior_pairs(test_df, run_dir, sidecar_uri).cache()
        _np = prior_pairs.count()                      # materialize the cache ONCE
        prior_nodes = [int(r[0]) for r in
                       prior_pairs.select("node_cid").distinct().collect()]
        _cprint(f"[differential]   incident rule: {_np:,} (test person, node) pairs "
                f"attested before the index over {len(prior_nodes):,} Mondo nodes",
                flush=True)
    sets = vt.select_sets(ctx, pattern, adhoc=adhoc)
    # Member rule. A flat label set scores every label member ("all"). On a nested
    # label space (no label set: exp 0152) "top" scores the group's top-level label
    # terms, each rolling up its subtypes; "most-specific" scores the leaves only.
    if most_specific is not None:                     # legacy keyword
        member_rule = "most-specific" if most_specific else "all"
    if member_rule == "auto":
        _mf = json.loads((run_dir / "manifest.json").read_text())
        member_rule = ("all" if (_mf.get("corpus_manifest") or {}).get("label_set")
                       else "top")
    all_members = {}
    if member_rule != "all":
        _pi = ctx.rdag["parent_int"]
        _s2 = []
        for n_, g_, m_ in sets:
            ms = (most_specific_members(m_, _pi) if member_rule == "most-specific"
                  else top_members(m_, _pi, group=n_))
            all_members[g_] = list(m_)
            if len(ms) >= 2:
                _s2.append((n_, g_, ms))
        sets = _s2
    if groups_like:
        # Exp 0152 read 4: the groups a previous survey scored, matched by name.
        _names = set(json.loads(Path(groups_like).read_text()))
        sets = [x for x in sets if x[1] in _names]
    if max_members:
        # Exp 0150 survey: every Mondo group up to `max_members` label members (the
        # broad body-system groups are not a differential).
        sets = [x for x in sets if len(x[2]) <= int(max_members)]
    if not sets:
        raise SystemExit(f"[differential] no candidate set matches {pattern!r}")
    _cprint(f"[differential]   member rule: {member_rule}", flush=True)
    _cprint(f"[differential]   {len(sets)} candidate set(s)"
            f"{' (fast: E1/E4 skipped, framing 2 on E5)' if fast else ''}", flush=True)
    L_ = [f"# {run_dir.name} — differential with reasons (framing 1)",
          f"pattern {pattern!r}{' (ad hoc)' if adhoc else ''} · records minus attestation "
          f"codes and {len(measured):,} [measured] tokens · E1 presence subtypes · E2 "
          f"stacked heads ({heads_tag}) · E3 multiclass head (train split) · E4 gated "
          f"likelihood · tempers fit on one half, all scored on the other", ""]
    js = {}
    eng = None
    for n, gname, members in sets:
        t0 = time.time()
        sub = all_members.get(gname, members)        # every label node in the group
        excl = {0: vt.set_exclusions(ctx, n, sub)}
        # Name-matched attestation (0141): a condition concept whose name IS a member's
        # or the group's name ("Hypertrophic cardiomyopathy" 40565500, "Cardiomyopathy")
        # but which the code map missed is attestation too, not a finding.
        nm_targets = {str(ctx.rd_names.get(m, "")).strip().lower() for m in sub}
        nm_targets.add(str(gname).strip().lower())
        if ctx.names and ctx.inv_maps:
            for v, cid in ctx.inv_maps[0].items():
                if str(ctx.names.get(int(cid), "")).strip().lower() in nm_targets:
                    excl[0].add(int(v))
        if measured:
            excl[dm] = measured
        subtypes = {}
        for m in members:
            sig = ctx.sig.get(m) or [t for t, k, _, _ in ctx.kinds.get(m, [])
                                     if k != "starved"]
            subtypes[m] = [(t, ev[t]) for t in sig]
        if eng is None:
            eng = ma.ModelEngine(lams_fit, alpha, heads,
                                 {m: [int(a) for a in np.flatnonzero(M[m])]
                                  for m in members}, members)
        else:
            eng.__init__(lams_fit, alpha, heads,
                         {m: [int(a) for a in np.flatnonzero(M[m])] for m in members},
                         members)
        e1 = dfm.PresenceSubtypeEngine(lams_fit, subtypes, ctx.n_bg, ev, members, excl,
                                       rho=rho)
        bg = list(range(ctx.n_bg))
        e4 = dfm.GatedLikelihoodEngine(eng.eb, alpha,
                                       {m: bg + list(ctx.block[m]) for m in members},
                                       members)
        feat = bg + sorted({t for m in members for t in ctx.block[m]})

        def prep(rows, keep_multi):
            out = []
            for truth, bags, _ in rows:
                if not truth or (len(truth) != 1 and not keep_multi):
                    continue
                fb, lens = dfm.filter_record(bags, excl)
                gi = [eng.gidx(d, v) for d, b in fb.items() for v in b]
                gc = [c for d, b in fb.items() for c in b.values()]
                out.append((truth, fb, lens, gi, gc))
            return out
        ever = None
        if train_ever:
            # Exp 0149 item 1: training members = ever coded (E4 sidecar), not only
            # coded in the label year. The test truth stays the label year.
            ever = ever_member_frame(train_df.sparkSession, run_dir, members,
                                     [member_cid[int(m)] for m in members], parent_adj,
                                     sidecar_uri=sidecar_uri)
        te_src = test_df
        if incident:
            te_src = _incident_test_frame(test_df, prior_pairs, prior_nodes,
                                          [member_cid[int(m)] for m in sub]
                                          + ([member_cid[int(n)]] if int(n) > 0
                                             and int(n) in member_cid else []),
                                          parent_adj)
        te_raw, n_te = collect_set_patients(te_src, members, len(ctx.lams),
                                            max_patients=max_patients, seed=seed)
        tr_raw, n_tr = collect_set_patients(train_df, members, len(ctx.lams),
                                            max_patients=max_train, seed=seed,
                                            ever_df=ever)
        n_multi = sum(len(t) > 1 for t, _, _ in te_raw)
        te_all = prep(te_raw, True)
        te = [r for r in te_all if len(r[0]) == 1]
        te_multi = [r for r in te_all if len(r[0]) > 1] if multi else []
        tr = prep(tr_raw, multi)
        n_tr_multi = sum(len(r[0]) > 1 for r in tr)
        _cprint(f"[differential]   {gname}: {len(te)} single-positive test patients "
                f"({n_multi} multi-positive{' scored apart' if multi else ' set aside'}), "
                f"{len(tr)} train ("
                f"{f'{n_tr_multi} multi-positive as soft targets' if multi else 'single-member only'}"
                f"{'; members ever coded' if train_ever else ''})", flush=True)
        if len(te) < 2 * dfm.EGRESS_FLOOR:
            L_.append(f"## {gname}\n(fewer than {2 * dfm.EGRESS_FLOOR} single-positive "
                      f"held-out patients; suppressed)")
            continue
        def theta(row):
            g = eng.fold_in(row[3], row[4])
            return g / g.sum()
        th_te = np.stack([theta(r) for r in te])
        gsum_te = [float(eng.alpha.sum() + sum(r[4])) for r in te]
        truths = [r[0] for r in te]
        e3 = e5 = e6 = None
        ytr = [list(r[0]) if multi else r[0][0] for r in tr]
        Ytr = dfm.label_matrix(ytr, len(members))
        if int((Ytr.sum(axis=0) > 0).sum()) >= 2:
            th_tr = np.stack([theta(r) for r in tr])
            e3 = dfm.MulticlassHead(feat, len(members), seed=seed).fit(th_tr, ytr)
            # Exp 0148 (insight 0101): the same softmax on the record's CODES (E5) and
            # on codes + log θ (E6) — same filtered records, same patients.
            e5 = dfm.CodeHead(len(members), seed=seed).fit([r[1] for r in tr], ytr)
            e6 = dfm.CodeHead(len(members), theta_topics=feat, seed=seed).fit(
                [r[1] for r in tr], ytr, th_tr)
        if fast:
            s1 = s4 = lphi4 = None
        else:
            s1 = [e1.scores(e1.subtype_ll(r[1], r[2])) for r in te]
            s4, lphi4 = zip(*[e4.elbos(r[3], r[4]) for r in te])
        # Prior over members from the TRAIN split (soft counts), add-one smoothed: a
        # member absent from the scored-results file (0140 first run) got prior 1e-12
        # and could never be called, and a test-derived prior leaks.
        pri = Ytr.sum(axis=0) + 1.0
        pri = pri / pri.sum()
        lp = np.log(pri)

        def engine_rows(idx, t1, t4):
            out = {"prior": [pri for _ in idx]}
            if not fast:
                out["E1 presence subtypes"] = [dfm.posterior_from_scores(lp, s1[i], t1)
                                               for i in idx]
            out["E2 stacked heads"] = [eng.posterior(th_te[i]) for i in idx]
            if not fast:
                out["E4 gated likelihood"] = [dfm.posterior_from_scores(lp, s4[i], t4)
                                              for i in idx]
            if e3 is not None:
                out["E3 multiclass head"] = list(e3.predict(th_te[idx]))
                out["E5 code head"] = list(e5.predict([te[i][1] for i in idx]))
                out["E6 θ + codes"] = list(e6.predict([te[i][1] for i in idx],
                                                       th_te[idx]))
            return out
        # Exp 0149 item 3: CROSS-FIT — tempers fit on one half and every engine scored
        # on the other, then the halves swap, so every patient is scored once.
        halves = [(list(range(0, len(te), 2)), list(range(1, len(te), 2)))]
        if crossfit:
            halves.append((halves[0][1], halves[0][0]))
        rows, tv, ev_all, taus = {}, [], [], []
        for fi, ei in halves:
            if fast:
                t1 = t4 = float("nan")
            else:
                t1, _ = dfm.fit_tau_scores(lp, [s1[i] for i in fi],
                                           [truths[i] for i in fi])
                t4, _ = dfm.fit_tau_scores(lp, [s4[i] for i in fi],
                                           [truths[i] for i in fi])
            taus.append((t1, t4))
            for k, v in engine_rows(ei, t1, t4).items():
                rows.setdefault(k, []).extend(v)
            tv += [truths[i] for i in ei]
            ev_all += ei
        fit_i, ev_i = halves[0]
        tau1, tau4 = taus[0]
        res = {"members": [int(m) for m in members], "member_rule": member_rule,
               "n_eval": len(ev_all), "n_fit": len(fit_i), "n_train": len(tr),
               "n_train_multi": n_tr_multi, "crossfit": bool(crossfit),
               "train_ever": bool(train_ever),
               "tau_e1": [t[0] for t in taus], "tau_e4": [t[1] for t in taus],
               "e3_lambda": (e3.lam if e3 is not None else None),
               "e5_lambda": (e5.lam if e5 is not None else None),
               "e6_lambda": (e6.lam if e6 is not None else None),
               "e5_n_codes": (e5.p_codes if e5 is not None else None)}
        res["metrics"] = {k: dfm.metrics(v, tv) for k, v in rows.items()}
        if te_multi and e3 is not None:
            # Exp 0149 item 2: test patients coded with 2+ members, scored apart:
            # top-1 / top-3 hit ANY true member; all_in_top3 = every true member there.
            thm = np.stack([theta(r) for r in te_multi])
            tm = [r[0] for r in te_multi]
            pm = {"prior": [pri for _ in te_multi],
                  "E3 multiclass head": list(e3.predict(thm)),
                  "E5 code head": list(e5.predict([r[1] for r in te_multi])),
                  "E6 θ + codes": list(e6.predict([r[1] for r in te_multi], thm))}
            res["metrics_multi"] = {}
            for k, v in pm.items():
                mm = dfm.metrics(v, tm)
                mm["all_in_top3"] = float(np.mean([
                    set(t) <= set(np.argsort(q)[::-1][:3].tolist())
                    for q, t in zip(v, tm)]))
                res["metrics_multi"][k] = mm
        # reasons on correctly called evaluation patients
        r1, r3, r4, r5 = [], [], [], []
        for j, i in enumerate(ev_all):
            t = truths[i][0]
            m = members[t]
            if e3 is not None and int(np.argmax(rows["E3 multiclass head"][j])) == t \
                    and te[i][3]:
                c3 = dfm.e3_reasons(e3, th_te[i], gsum_te[i], eng.eb[:, te[i][3]],
                                    te[i][4], t)
                r3.append((m, {_split_gidx(eng, g): float(x)
                               for g, x in zip(te[i][3], c3)}))
            if e5 is not None and int(np.argmax(rows["E5 code head"][j])) == t:
                r5.append((m, e5.reasons(te[i][1], t)))
            if fast:
                continue
            if int(np.argmax(rows["E1 presence subtypes"][j])) == t:
                r1.append((m, e1.reasons(e1.subtype_ll(te[i][1], te[i][2]), m)))
            if int(np.argmax(rows["E4 gated likelihood"][j])) == t:
                rs = e4.reasons(te[i][3], te[i][4], lphi4[i], m)
                r4.append((m, {_split_gidx(eng, g): x for g, x in rs.items()}))
        res["reasons_e3"] = dfm.pool_reasons(r3)
        res["reasons_e1"] = dfm.pool_reasons(r1)
        res["reasons_e4"] = dfm.pool_reasons(r4)
        res["reasons_e5"] = dfm.pool_reasons(r5)
        named = {m: ctx.rd_names.get(m, str(m)) for m in members}
        L_.append(f"## {gname} — {len(members)} members · {len(ev_all)} scored patients "
                  f"({'cross-fit, both halves' if crossfit else f'τ fit on {len(fit_i)}'})"
                  f" · {len(tr)} train patients ({n_tr_multi} multi-positive"
                  f"{', members ever coded' if train_ever else ''}) · τ E1 "
                  f"{tau1:.3g}, E4 {tau4:.3g}")
        L_.append("")
        L_.append("| engine | top-1 | top-3 | bal top-1 | log loss | ECE |")
        L_.append("|---|--:|--:|--:|--:|--:|")
        for k, mtr in res["metrics"].items():
            L_.append(f"| {k} | {mtr['top1']:.3f} | {mtr['top3']:.3f} | "
                      f"{mtr['bal_top1']:.3f} | {mtr['logloss']:.3f} | {mtr['ece']:.3f} |")
        if res.get("metrics_multi") and len(te_multi) >= dfm.EGRESS_FLOOR:
            L_ += ["", f"multi-positive test patients ({len(te_multi)}; coded with 2+ "
                   "members): top-1 / top-3 = any true member, all-in-top-3 = every "
                   "one", "", "| engine | top-1 | top-3 | all in top-3 | log loss |",
                   "|---|--:|--:|--:|--:|"]
            for k, mtr in res["metrics_multi"].items():
                L_.append(f"| {k} | {mtr['top1']:.3f} | {mtr['top3']:.3f} | "
                          f"{mtr['all_in_top3']:.3f} | {mtr['logloss']:.3f} |")
        for key, lab in (("reasons_e3", "E3"), ("reasons_e1", "E1"),
                         ("reasons_e4", "E4"), ("reasons_e5", "E5")):
            if res[key]:
                L_.append("")
                L_.append(f"{lab} reasons (top supporting codes, correctly called "
                          f"patients, members with >= {dfm.EGRESS_FLOOR}):")
                for m, codes in res[key].items():
                    L_.append(f"- {named[m][:40]}: " + " · ".join(
                        ctx.code_name(d, v) for d, v in codes))
        L_.append("")
        if next_question and e3 is not None and fast:
            nq = _next_question_block(ctx, eng, None, members, te, ev_i, truths, pri,
                                      excl, named, nq_pool, seed, frac=nq_hide,
                                      code_head=e5, code_name="E5 codes",
                                      codes_only=True)
            L_.extend(nq["lines"])
            res["next_question"] = nq["res"]
        elif next_question and e3 is not None:
            # The code engine for framing 2: E5 or E6, whichever has the lower log loss
            # on the tau-fit half (never the evaluation half).
            def _ll(P, ts):
                return float(np.mean([-np.log(max(P[j][t].sum(), 1e-300))
                                      for j, t in enumerate(ts)]))
            ts_fit = [truths[i] for i in fit_i]
            ll5 = _ll(e5.predict([te[i][1] for i in fit_i]), ts_fit)
            ll6 = _ll(e6.predict([te[i][1] for i in fit_i], th_te[fit_i]), ts_fit)
            ce, ce_name = (e5, "E5 codes") if ll5 <= ll6 else (e6, "E6 θ + codes")
            res["nq_code_engine"] = {"chosen": ce_name, "fit_logloss_e5": ll5,
                                     "fit_logloss_e6": ll6}
            nq = _next_question_block(ctx, eng, e3, members, te, ev_i, truths, pri,
                                      excl, named, nq_pool, seed, frac=nq_hide,
                                      e1=e1, e1_lp=lp, e1_tau=tau1, code_head=ce,
                                      code_name=ce_name)
            L_.extend(nq["lines"])
            res["next_question"] = nq["res"]
        _cprint(f"[differential]   {gname}: done in {time.time() - t0:.0f}s", flush=True)
        js[gname] = {k: v for k, v in res.items() if not k.startswith("reasons")}
    text = "\n".join(L_).rstrip() + "\n"
    stem = "differential" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(js, indent=1, default=str))
    if max_members:
        sv = differential_survey(js, sets, run_dir)
        (run_dir / f"{stem}_survey.md").write_text(sv)
        print(sv, flush=True)
    print(text, flush=True)
    _cprint(f"[differential]   wrote {run_dir / (stem + '.md')}", flush=True)
    return js


def differential_survey(js, sets, run_dir, *, top=40):
    """Exp 0150: one row per surveyed Mondo group — framing 1 (balanced top-1 of the
    prior, E3 θ, E5 codes, E6 θ + codes; the best engine's lift over chance = bal × M;
    E6 top-3 vs the prior's), framing 2 (E5: P(true) visible -> VOI@5 -> full, and the
    share of the gap five questions close; VOI vs the population table), framing 3
    (members' mean new-diagnosis AUC from the whole-history prevalence checks, θ and
    raw codes, when those files are in the run dir). Pooled per group; groups under
    the egress floor never reach `js`."""
    from prevalence_check import framing3_sources
    th, rw = framing3_sources(run_dir)

    def f3(members, src):
        if not src or "history_stacked" not in src:
            return float("nan")
        pn = src["history_stacked"]["per_node"]
        v = [pn[str(m)]["auc_incident"] for m in members if str(m) in pn]
        return float(np.mean(v)) if v else float("nan")
    mem = {g: m for _n, g, m in sets}
    rows = []
    for g, r in js.items():
        mt = r.get("metrics") or {}
        if "E6 θ + codes" not in mt:
            continue
        M = len(mem.get(g, []))
        bal = {k: mt[k]["bal_top1"] for k in ("prior", "E3 multiclass head",
                                              "E5 code head", "E6 θ + codes")}
        best = max(("E3 multiclass head", "E5 code head", "E6 θ + codes"),
                   key=lambda k: bal[k])
        nq = r.get("next_question") or {}
        if nq:
            v0, vf = nq["visible"]["p_true"], nq["full"]["p_true"]
            v5 = nq["by"]["VOI (codes)@5"]["p_true"]
            t5 = nq["by"]["population table@5"]["p_true"]
            gap = (v5 - v0) / (vf - v0) if vf > v0 else float("nan")
        else:
            v0 = vf = v5 = t5 = gap = float("nan")
        rows.append({"group": g, "M": M, "n": r["n_eval"], "prior": bal["prior"],
                     "e3": bal["E3 multiclass head"], "e5": bal["E5 code head"],
                     "e6": bal["E6 θ + codes"], "best": best.split()[0],
                     "lift": bal[best] * M, "top3_e6": mt["E6 θ + codes"]["top3"],
                     "top3_prior": mt["prior"]["top3"], "v0": v0, "v5": v5, "t5": t5,
                     "vf": vf, "gap": gap, "f3_theta": f3(mem.get(g, []), th),
                     "f3_raw": f3(mem.get(g, []), rw)})
    if not rows:
        return "# differential survey\n(no group scored)\n"
    rows.sort(key=lambda x: -x["lift"])
    med = lambda k: float(np.nanmedian([x[k] for x in rows]))
    win = {e: sum(x["best"] == e for x in rows) for e in ("E3", "E5", "E6")}
    L = ["# differential survey (exp 0150) — framings 1-3 across Mondo groups", "",
         f"{len(rows)} groups scored · best engine by balanced top-1: E3 {win['E3']}, "
         f"E5 {win['E5']}, E6 {win['E6']} · median lift over chance {med('lift'):.2f}× · "
         f"median E6 − E3 balanced top-1 {float(np.median([x['e6'] - x['e3'] for x in rows])):+.3f}"
         f" · median share of the visible→full gap 5 questions close "
         f"{med('gap'):.2f} · median VOI − table P(true)@5 "
         f"{float(np.nanmedian([x['v5'] - x['t5'] for x in rows])):+.3f}", "",
         f"Top {min(top, len(rows))} by lift (balanced top-1 of the best engine × members):",
         "", "| group | M | n | prior | E3 | E5 | E6 | lift | top-3 E6 (prior) | "
         "F2 P(true) vis → @5 → full | gap closed | VOI − table | F3 AUC θ / raw |",
         "|---|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|---|"]
    for x in rows[:top]:
        L.append(f"| {str(x['group'])[:40]} | {x['M']} | {x['n']} | {x['prior']:.2f} | "
                 f"{x['e3']:.2f} | {x['e5']:.2f} | {x['e6']:.2f} | {x['lift']:.2f} | "
                 f"{x['top3_e6']:.2f} ({x['top3_prior']:.2f}) | {x['v0']:.2f} → "
                 f"{x['v5']:.2f} → {x['vf']:.2f} | {x['gap']:.2f} | "
                 f"{x['v5'] - x['t5']:+.3f} | {x['f3_theta']:.2f} / {x['f3_raw']:.2f} |")
    L.append("")
    return "\n".join(L) + "\n"


def _next_question_block(ctx, eng, head, members, te, ev_i, truths, pri, excl, named,
                         pool_per_dom, seed, *, frac=0.2, e1=None, e1_lp=None,
                         e1_tau=1.0, code_head=None, code_name="codes",
                         codes_only=False):
    """Framing 2 (spec R2) on the framing-1 evaluation half: the candidate pool is the
    population-EIG table over condition, drug and measurement-result codes (attestation
    and [measured] excluded), the differential is E3's. Candidates whose presence
    alone would give one member >= 90% (presence model, prior `pri`) are flagged as
    likely treatments / diagnostic consequences (R2.2)."""
    import differential as dfm
    import voi_acquisition as va
    import voi_tables as vt
    doms = [i for i, nm in enumerate(ctx.dom_names)
            if nm in ("condition", "drug", "measurement")]
    profs, bgs, used = {}, {}, None
    for d in doms:
        pr, bg = vt.set_profiles(ctx, members, d)
        u = [j for j, m in enumerate(members) if pr.get(m) is not None]
        used = u if used is None else [j for j in used if j in u]
        profs[d], bgs[d] = pr, bg
    lines = [f"### Next question (framing 2; {100 * frac:.0f}% of each record hidden; "
             f"differentials E3 and E1)"]
    if not used or len(used) < 2:
        return {"lines": lines + ["(fewer than 2 members with a profile)"], "res": None}
    pu = np.asarray(pri)[used]
    pu = pu / pu.sum()
    P = {d: np.stack([profs[d][members[j]] for j in used]) for d in doms}
    pool_items, _ = va.build_pool(pu, P, bgs, excl, pool=pool_per_dom)
    flagged = {}
    for j, (d, v) in enumerate(pool_items):
        pm = vt.presence_prob(P[d][:, v], bgs[d][v])
        q = pu * pm
        flagged[j] = bool(q.max() / q.sum() >= 0.9)
    pats = [(truths[i], te[i][1]) for i in ev_i]
    rng = np.random.default_rng(int(seed))
    if codes_only:
        r = dfm.evaluate_next_question_codes(code_head, pats, pool_items, rng, frac=frac)
        if r["n"] < dfm.EGRESS_FLOOR:
            return {"lines": lines + ["(fewer than 20 evaluable patients; suppressed)"],
                    "res": None}
        v, f = r["visible"], r["full"]
        lines += [f"{r['n']} patients · pool {len(pool_items)} codes · differential "
                  f"{code_name}", "", "| strategy | k | yield | P(true) | top-1 |",
                  "|---|--:|--:|--:|--:|",
                  f"| visible record ({100 - 100 * frac:.0f}%) | — | — | "
                  f"{v['p_true']:.3f} | {v['top1']:.3f} |"]
        for k in r["ks"]:
            for s_ in tuple(r["strategies"]) + ("random hidden",):
                x = r["by"][f"{s_}@{k}"]
                lines.append(f"| {s_} | {k} | {x['yield']:.3f} | {x['p_true']:.3f} | "
                             f"{x['top1']:.3f} |")
        lines += [f"| full record | — | — | {f['p_true']:.3f} | {f['top1']:.3f} |", ""]
        return {"lines": lines, "res": r}
    r = dfm.evaluate_next_question(eng, head, pats, pool_items, flagged, rng, frac=frac,
                                   e1=e1, e1_lp=e1_lp, e1_tau=e1_tau,
                                   code_head=code_head)
    if r["n"] < dfm.EGRESS_FLOOR:
        return {"lines": lines + ["(fewer than 20 evaluable patients; suppressed)"],
                "res": None}
    v, f = r["visible"], r["full"]
    lines += [f"{r['n']} patients · pool {len(pool_items)} codes (condition, drug, "
              f"measurement results) · likely-treatment flags in VOI top 3: "
              f"{100 * r['flag_share_top3']:.0f}%", "",
              f"code differential: {code_name}", "",
              "| strategy | k | yield | P(true) E3 | top-1 E3 | P(true) E1 | top-1 E1 | "
              "P(true) codes | top-1 codes |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
              f"| visible record ({100 - 100 * frac:.0f}%) | — | — | {v['p_true']:.3f} | "
              f"{v['top1']:.3f} | {v['p_true_e1']:.3f} | {v['top1_e1']:.3f} | "
              f"{v['p_true_c']:.3f} | {v['top1_c']:.3f} |"]
    for k in r["ks"]:
        for s_ in tuple(r["strategies"]) + ("random hidden",):
            x = r["by"][f"{s_}@{k}"]
            lines.append(f"| {s_} | {k} | {x['yield']:.3f} | {x['p_true']:.3f} | "
                         f"{x['top1']:.3f} | {x['p_true_e1']:.3f} | {x['top1_e1']:.3f} | "
                         f"{x['p_true_c']:.3f} | {x['top1_c']:.3f} |")
    lines.append(f"| full record | — | — | {f['p_true']:.3f} | {f['top1']:.3f} | "
                 f"{f['p_true_e1']:.3f} | {f['top1_e1']:.3f} | {f['p_true_c']:.3f} | "
                 f"{f['top1_c']:.3f} |")
    top = sorted(((j, c) for j, c in r["recommended"].items()
                  if c >= dfm.EGRESS_FLOOR), key=lambda x: -x[1])[:10]
    if top:
        lines += ["", "Most often in the VOI (E3) top 3 (>= 20 patients; † = likely "
                  "treatment / consequence):"]
        lines.append(" · ".join(ctx.code_name(*pool_items[j]) + (" †" if flagged[j] else "")
                                for j, _ in top))
    topc = sorted(((j, c) for j, c in r.get("recommended_codes", {}).items()
                   if c >= dfm.EGRESS_FLOOR), key=lambda x: -x[1])[:10]
    if topc:
        lines += ["", f"Most often in the VOI (codes) top 3 (>= 20 patients; † = likely "
                  "treatment / consequence):"]
        lines.append(" · ".join(ctx.code_name(*pool_items[j]) + (" †" if flagged[j] else "")
                                for j, _ in topc))
    lines.append("")
    res = {k: v for k, v in r.items() if k not in ("recommended", "recommended_codes")}
    return {"lines": lines, "res": res}


def _split_gidx(eng, g):
    """Global concatenated vocab id -> (domain, vocab idx)."""
    d = int(np.searchsorted(eng.offsets, g, side="right") - 1)
    return d, int(g - eng.offsets[d])



def _whole_history_prior(test_scored, keys, run_dir, *, C, readout_int2cid,
                         parent_adj, sidecar_uri=None):
    """Exp 0145 (spec 2026-10-13 label redesign R1): prior[d, X] from the E4 sidecar —
    a first attestation of X's Mondo term or a descendant dated BEFORE the person's
    index (the sidecar's own index_horizon, re-derived by the assembler's index
    builder at sidecar build). Whole history, not the 5-year lookback. Returns
    (prior, receipt)."""
    import prevalence_check as pv
    from pyspark.sql import functions as F
    from conversion_sidecar import try_load_index_horizon, try_load_sidecar
    import distributed_readout as _dr
    wp = Path(run_dir) / "conversion_sidecar.json"
    if not wp.exists():
        raise SystemExit("[prevalence] --prev-history needs <run>/conversion_sidecar.json "
                         "(make build-conversion-sidecar ID=N)")
    wit = json.loads(wp.read_text())
    uri = sidecar_uri or wit.get("sidecar_uri")
    spark = test_scored.sparkSession
    first = try_load_sidecar(spark, uri, wit["key"])
    hz = try_load_index_horizon(spark, uri, wit["key"])
    if first is None or hz is None:
        raise SystemExit(f"[prevalence] sidecar MISS at {uri}/{wit['key']} — rebuild it "
                         "on this cluster (make build-conversion-sidecar ID=N)")
    persons = test_scored.select("person_id").distinct()
    hz_t = hz.join(persons, "person_id").select("person_id", "index_date").cache()
    n_docs, n_idx = len(keys), hz_t.count()
    pairs = (first.join(hz_t, "person_id")
             .where(F.col("first_attested_date") < F.col("index_date"))
             .select("person_id", "node_cid").distinct().cache())
    nodes = [int(r[0]) for r in pairs.select("node_cid").distinct().collect()]
    n2r = pv.sidecar_nodes_to_readout(nodes, readout_int2cid, parent_adj)
    mrows = [(int(n), int(r)) for n, rs in n2r.items() for r in rs]
    mdf = spark.createDataFrame(mrows, "node_cid long, rnode int")
    got = (pairs.join(F.broadcast(mdf), "node_cid").groupBy("person_id")
           .agg(F.collect_set("rnode").alias("ns")).collect())
    by_person = {int(r["person_id"]): r["ns"] for r in got}
    prior = pv.prior_from_sets(keys, by_person, int(C), _dr.person_of)
    pairs.unpersist(); hz_t.unpersist()
    return prior, {"n_test_docs": n_docs, "n_with_index": n_idx,
                   "n_sidecar_nodes_prior": len(nodes), "n_mapped_to_readout": len(n2r)}


def run_prevalence_check(test_scored, run_dir, *, C, parent_int,
                         heads_tag="stacked_mass99", tag=None, top=15,
                         history=None, node_subset=None):
    """How much of the case-finding AUC is prevalent recall? (insight 0100;
    `prevalence_check.py`). Scores the test split with the saved heads (no solve), flags
    each (patient, node) with a mapped attestation code of the node or a label
    descendant in the PRE-INDEX condition features, and reports per-node AUC over all
    patients vs with those patients set aside — flat heads and the stacked product.
    Writes `prevalence_check[_<tag>].md/.json`.

    `history` (exp 0145): a dict of `_whole_history_prior` kwargs; adds the
    WHOLE-HISTORY prior rule beside the lookback one, with a containment check."""
    import prevalence_check as pv
    import voi_tables as vt
    from gated_pc_cloud import (_collect_lean_proba, _doc_key_column, closure_matrix,
                                stacked_proba)
    run_dir = Path(run_dir)
    bm, cm = run_dir / "bundle_meta.json", run_dir / "code_map.tsv"
    if not cm.exists():
        raise SystemExit("[prevalence] needs <run>/code_map.tsv (the attestation map)")
    ctx = vt.load_voi_context(run_dir, bundle_meta_path=str(bm) if bm.exists() else None,
                              own_codes_path=str(cm))
    hp = run_dir / f"readout_heads_gated_pc_{heads_tag}.npz"
    z = np.load(hp)
    if int(z["C"]) != int(C):
        raise SystemExit(f"[prevalence] heads {hp.name} are C={int(z['C'])}, frames C={C}")
    proba, y, _m, keys, _e = _collect_lean_proba(
        test_scored, int(C), z["V"], z["b_raw"], degenerate=z["degenerate"],
        const=z["const"], theta_topm=int(z["theta_topm"]))
    P = stacked_proba(proba, closure_matrix(parent_int, int(C)))
    col = "features_0" if "features_0" in test_scored.columns else "features"

    def _codes(row):
        v = row[col]
        idx = getattr(v, "indices", None)
        if idx is None:
            arr = np.asarray(v.toArray() if hasattr(v, "toArray") else v)
            idx = np.flatnonzero(arr)
        return int(row["doc_key"]), [int(i) for i in idx]
    rows = (test_scored.withColumn("doc_key", _doc_key_column(test_scored, "person_id"))
            .select("doc_key", col).rdd.map(_codes).collect())
    codes = dict(rows)
    v2n = pv.vocab_to_nodes(ctx.own_idx, ctx.rdag["pos_sources"])
    prior = pv.prior_matrix([int(k) for k in keys], codes, v2n, int(C))
    nodes = list(range(1, int(C)))
    if node_subset is not None:
        # Exp 0152: only the solved heads — masked heads are constants (AUC 0.5).
        nodes = [n for n in nodes if node_subset[n]]
    res = {}
    L_ = [f"# {run_dir.name} — prevalence check: AUC with prior-diagnosis patients set "
          f"aside", f"heads {hp.name} · {len(keys):,} test patients · LOOKBACK prior = a "
          f"mapped attestation code of the node (or a label descendant) in the 5-year "
          f"pre-index condition features (a lower bound: older diagnoses count as "
          f"incident)", ""]
    rules = [("", prior)]
    if history:
        prior_h, rec = _whole_history_prior(test_scored, keys, run_dir, C=int(C),
                                            **history)
        res["history_receipt"] = rec
        res["containment_lookback_in_history"] = pv.containment(prior, prior_h)
        L_ += [f"WHOLE-HISTORY prior (exp 0145) = a sidecar first attestation of the "
               f"node's Mondo term or a descendant before the index · test docs "
               f"{rec['n_test_docs']:,}, with a sidecar index {rec['n_with_index']:,} · "
               f"prior cells: lookback {int(prior.sum()):,}, whole history "
               f"{int(prior_h.sum()):,} · lookback cells inside whole-history "
               f"{res['containment_lookback_in_history']:.3f} (expect ~1; low = index "
               f"or node-id mismatch)", ""]
        rules.append(("history_", prior_h))
    for pre, pri in rules:
        for arm0, score in (("stacked", P), ("flat", proba)):
            arm = pre + arm0
            per = pv.node_aucs(score, y, pri, nodes)
            sm = pv.summarize(per)
            res[arm] = {"summary": sm, "per_node": per}
            if not sm.get("n_nodes"):
                L_.append(f"## {arm}: no node with >= {pv.EGRESS_FLOOR} positives both ways")
                continue
            L_ += [f"## {arm} — {sm['n_nodes']} nodes · macro AUC all {sm['macro_auc_all']:.4f}"
                   f" → incident {sm['macro_auc_incident']:.4f} · gap median "
                   f"{sm['gap_median']:+.4f}, p90 {sm['gap_p90']:+.4f}, > 0.05 on "
                   f"{sm['n_gap_over_0p05']} nodes · median prior share of positives "
                   f"{sm['prior_share_pos_median']:.2f}", "",
                   "| prior share of positives | nodes | AUC all | AUC incident | median gap |",
                   "|---|--:|--:|--:|--:|"]
            for band, b in sm["by_prior_share"].items():
                L_.append(f"| {band} | {b['n_nodes']} | {b['auc_all']:.3f} | "
                          f"{b['auc_incident']:.3f} | {b['median_gap']:+.3f} |")
            worst = sorted(per.items(), key=lambda kv: -(kv[1]["auc_all"]
                                                         - kv[1]["auc_incident"]))[:int(top)]
            L_ += ["", f"largest gaps ({arm}):"]
            for c, v in worst:
                L_.append(f"- {str(ctx.rd_names.get(c, c))[:40]}: {v['auc_all']:.3f} → "
                          f"{v['auc_incident']:.3f} · prior share of positives "
                          f"{v['prior_share_pos']:.2f} · positives {v['n_pos_all']} → "
                          f"{v['n_pos_incident']}")
            L_.append("")
    text = "\n".join(L_).rstrip() + "\n"
    stem = "prevalence_check" + (f"_{tag}" if tag else "")
    (run_dir / f"{stem}.md").write_text(text)
    (run_dir / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str))
    print(text, flush=True)
    _cprint(f"[prevalence]   wrote {run_dir / (stem + '.md')}", flush=True)
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Post-hoc case-finding readout on a finished gated_pc run "
                    "(no re-fit): pc_topics_lr + precision@recall / recall@FDR / "
                    "detection for the gated_pc arm.",
        epilog="Submit via `make gated-pc-readout ID=N` (analysis/cloud/Makefile). A "
               "hand-rolled spark-submit MUST pass --py-files "
               "<repo>/analysis/cloud/distributed_readout.py alongside the source "
               "zips: --readout-mode distributed ships its partition kernels by "
               "module reference and they must import on the executors.")
    p.add_argument("--run-dir", required=True,
                   help="Run dir with gated_pc_result.npz + manifest.json.")
    p.add_argument("--cache-uri", default=None,
                   help="Bundle cache root the fit used (its --cache-uri). Defaults "
                        "to the one the manifest recorded; required only when "
                        "neither that nor --bundle-path is available.")
    p.add_argument("--bundle-path", default=None,
                   help="Exact cached bundle dir ({cache_uri}/{key}); bypasses the "
                        "key recompute if the manifest is incomplete.")
    p.add_argument("--doc-min-length", type=int, default=None,
                   help="Override doc_min_length for the cache key (older manifests "
                        "did not record it; the current driver does).")
    p.add_argument("--differential", default=None, metavar="REGEX",
                   help="Framing 1 (spec 2026-10-13): DIFFERENTIAL WITH REASONS for the "
                        "candidate sets matching REGEX — four engines (presence "
                        "subtypes, stacked heads, multiclass head, gated likelihood) "
                        "scored on held-out single-positive patients. --acq-adhoc, "
                        "--acq-heads and --acq-tag apply. Writes "
                        "differential[_<tag>].md/.json.")
    p.add_argument("--prevalence-check", action="store_true",
                   help="Insight 0100: per-node AUC of the saved heads (--acq-heads) "
                        "over all test patients vs with prior-diagnosis patients set "
                        "aside (mapped own code in the pre-index features). Transforms "
                        "the test split only.")
    p.add_argument("--raw-select", choices=["freq", "model", "relevance"],
                   default="freq",
                   help="How --raw-baseline picks its N tokens: 'freq' = train document "
                        "frequency (exp 0147); 'model' = round-robin over the fit's node "
                        "topics' top words (select_model_codes; tag 'rawm<N>'); "
                        "'relevance' = the node topics' LDAvis relevance at "
                        "--raw-rel-lambda with each topic's own and descendant codes "
                        "excluded and the frequency set's per-domain mix "
                        "(select_relevance_codes; tag 'rawr<N>').")
    p.add_argument("--raw-rel-lambda", type=float, default=0.6,
                   help="relevance weight λ for --raw-select relevance (1 = plain "
                        "top words; 0.6 = the dashboard's setting)")
    p.add_argument("--raw-baseline", type=int, default=0, metavar="N",
                   help="Exp 0147: RAW-CODE BASELINE. Skip the θ transform; the heads "
                        "are fit on the binary presence of the N most frequent train "
                        "tokens (all domains pooled) with the same solver, l2, readout "
                        "DAG and stacked arm; outputs tagged 'raw<N>'. With "
                        "--prev-history the whole-history prevalence check runs on "
                        "those heads. Do not pass --readout-theta-mass.")
    p.add_argument("--prev-history", action="store_true",
                   help="(--prevalence-check) exp 0145: add the WHOLE-HISTORY prior rule "
                        "from the E4 sidecar (first attestation before the index) beside "
                        "the 5-year lookback rule. Needs <run>/conversion_sidecar.json.")
    p.add_argument("--prev-sidecar-uri", default=None,
                   help="(--prev-history) sidecar root; default the witness's")
    p.add_argument("--readout-nodes", default="", metavar="TSV",
                   help="Exp 0152: solve only the heads of the Mondo terms in this "
                        "label-set TSV (e.g. the DisMech ribbon) and their readout "
                        "ancestors; every other head is masked to a constant. Outputs "
                        "tagged 'nodes_…'; with --readout-stacked --prev-history the "
                        "whole-history prevalence check follows on those heads.")
    p.add_argument("--diff-max-patients", type=int, default=600)
    p.add_argument("--diff-survey", type=int, default=0, metavar="MAX_MEMBERS",
                   help="(--differential) exp 0150 SURVEY: every Mondo group matching "
                        "the pattern with 2..MAX_MEMBERS label members, fast engines "
                        "(E1/E4 skipped; framing 2 on E5), plus <tag>_survey.md")
    p.add_argument("--diff-groups-like", default=None, metavar="JSON",
                   help="(--differential) exp 0152: score only the groups whose names are "
                        "keys of this JSON (e.g. a previous run's differential_<tag>.json)")
    p.add_argument("--npmi", action="store_true",
                   help="Exp 0159: GATED NPMI of every topic from the saved fit — each "
                        "node block scored on the documents its gate admits, and on all "
                        "documents for comparison. No transform, no heads. Writes "
                        "npmi[_<acq-tag>].md (pooled) and .json (workspace-internal).")
    p.add_argument("--npmi-top", type=int, default=10, help="(--npmi) top-N words")
    p.add_argument("--npmi-split", choices=("train", "test"), default="train",
                   help="(--npmi) reference documents: the split the fit trained on "
                        "(default) or the held-out test split")
    p.add_argument("--npmi-domains", default="0",
                   help="(--npmi) comma-separated domain indices (0 = conditions)")
    p.add_argument("--diff-members", default="auto",
                   choices=("auto", "all", "top", "most-specific"),
                   help="(--differential) which label terms are a group's members: all "
                        "(flat label set), top (nested: top-level terms, each rolling up "
                        "its subtypes), most-specific (nested: leaves only — drops "
                        "patients coded at intermediate terms). auto = all with a label "
                        "set, else top")
    p.add_argument("--diff-incident", action="store_true",
                   help="(--differential) exp 0151: score only test patients with no "
                        "member or group diagnosis anywhere BEFORE the index (E4 sidecar, "
                        "Mondo is-a) — a pre-diagnosis record; training unchanged. Needs "
                        "<run>/conversion_sidecar.json")
    p.add_argument("--diff-crossfit", action="store_true",
                   help="(--differential) exp 0149: tempers fit on each half in turn, "
                        "every patient scored once (doubles the scored patients)")
    p.add_argument("--diff-max-train", type=int, default=2000)
    p.add_argument("--diff-next-question", action="store_true",
                   help="(--differential) also run framing 2 — the next-question "
                        "evaluation (hide --diff-nq-hide of each record; yield and gain "
                        "of the top-k recommendations under the E3 and E1 "
                        "differentials)")
    p.add_argument("--diff-nq-hide", type=float, default=0.2,
                   help="(--diff-next-question) fraction of each record hidden; 0.8 = "
                        "early in a workup (20%% visible)")
    p.add_argument("--acq-adhoc", action="store_true",
                   help="(--acquisition) REGEX picks label nodes as ONE set")
    p.add_argument("--acq-tag", default=None)
    p.add_argument("--acq-heads", default="stacked_mass99",
                   help="(--acq-engine model) readout_heads_gated_pc_<TAG>.npz")
    p.add_argument("--bundle-only", action="store_true",
                   help="Load (or rebuild) the bundle, write <run>/bundle_meta.json "
                        "and <run>/code_map.tsv, and stop: no transform, no solve. "
                        "What a fresh cluster runs before `inspect-topics` on an old "
                        "run (~20 min on a MISS).")
    p.add_argument("--no-rebuild", action="store_true",
                   help="FAIL FAST on a cache miss instead of re-assembling the "
                        "corpus from the manifest's parameters. The old behaviour: "
                        "use it when a miss should be investigated (a wrong "
                        "--cache-uri, a drifted assembly source) rather than paid "
                        "for — a whole-Mondo rebuild is ~5 min of BigQuery plus the "
                        "assembly itself.")
    # Rebuild inputs a manifest may predate. Each defaults to the manifest value;
    # pass them when re-reading a run whose manifest was written before the corpus
    # spec was recorded in full (they are cache-KEY inputs, so a wrong value misses
    # the cache rather than mis-scoring).
    p.add_argument("--billing", default=None,
                   help="Billing project for a rebuild (default: the manifest's).")
    p.add_argument("--dag-source", choices=["snomed", "mondo", "mondo_native"],
                   default=None,
                   help="The fit's --dag-source. Default: the manifest's, which for "
                        "runs predating this field is absent — so a MONDO run from "
                        "before it was recorded must pass --dag-source mondo, or it "
                        "keys (and would rebuild) as the disease-anchored SNOMED "
                        "corpus instead. Every exp-0110 (mondo_native) run records "
                        "it, so an override there can only contradict the manifest "
                        "and is rejected (exit 2).")
    p.add_argument("--cache-write", choices=["on", "off"], default="on",
                   help="write a rebuilt bundle through to --cache-uri (default on, "
                        "so the next readout of this run is a HIT). 'off' keeps the "
                        "rebuild in memory only.")
    p.add_argument("--mondo-version", default=None,
                   help="mondo runs: the fit's --mondo-version (default: manifest).")
    p.add_argument("--mondo-branch", default=None,
                   help="mondo runs: the fit's --mondo-branch, '' = whole Mondo "
                        "(default: manifest).")
    p.add_argument("--min-positives", type=int, default=None,
                   help="mondo runs: the fit's --min-positives (default: manifest).")
    p.add_argument("--mondo-cache-dir", default=None,
                   help="mondo runs: local Mondo OBO cache dir for a rebuild "
                        "(default: the manifest's, else data/mondo).")
    p.add_argument("--recall-targets", default="0.5,0.8,0.9")
    p.add_argument("--fdr-targets", default="0.1,0.25,0.5")
    p.add_argument("--min-label-count", type=int, default=None,
                   help="Small-cell mask floor; default = the run's own value.")
    p.add_argument("--readout-mode", choices=["distributed"], default="distributed",
                   help="accepted for command compatibility; the distributed "
                        "batched solve is the only readout since the 2026-10 cleanup")
    p.add_argument("--readout-theta-mass", type=float, default=None,
                   help="OVERRIDE: resolve the top-m width from a theta-MASS target "
                        "on the train frame (spec 2026-10-07 §D4; smallest m in "
                        "{16,32,...,K} with p10 coverage >= target). Tags outputs "
                        "mass<pct>. Default: the manifest's resolved "
                        "readout_theta_topm; a fit-only manifest that recorded a "
                        "readout_theta_mass but no resolved width re-resolves it.")
    p.add_argument("--readout-theta-topm", type=int, default=None,
                   help="OVERRIDE the manifest's readout_theta_topm for this re-readout "
                        "(0 forces full-K). Default: the manifest's value, which "
                        "reproduces the fit's own design matrix. The override exists "
                        "for exactly one job: PRICING truncation — re-read a full-K "
                        "run (e.g. the 0103 cardiovascular record) with top-m forced "
                        "on and compare per-node AUC against its recorded numbers, so "
                        "the whole-Mondo enablement decision rests on a measured "
                        "delta, not the mass-coverage heuristic alone.")
    p.add_argument("--readout-max-iter", type=int, default=None,
                   help="OVERRIDE the batched L-BFGS iteration cap for the "
                        "distributed re-readout. Default: the manifest's recorded "
                        "readout_max_iter — the cap the fit itself used, CHARM_DEV "
                        "capping included, so a recovery reproduces the run it is "
                        "rescuing without being told. Manifests written before that "
                        f"field fall back to {_LEGACY_READOUT_MAX_ITER} (the "
                        "record-run budget); recovering a DEV smoke from one of "
                        "those still needs an explicit 60.")
    p.add_argument("--readout-warm-from", default=None, metavar="TAG",
                   help="Start the pc_topics_lr solve from this run's saved heads "
                        "readout_heads_gated_pc_<TAG>.npz (e.g. stacked_nodes_mass99) "
                        "to continue an iteration-capped solve. Outputs get a "
                        "'_warm' tag suffix, so the record is untouched.")
    p.add_argument("--readout-l2", type=float, default=None,
                   help="OVERRIDE the ridge strength of the per-node readout "
                        "heads (SUMMED log-loss; 1.0 == sklearn C=1.0). Default: "
                        "the manifest's recorded readout_l2, else 100 — so a "
                        "recovery reproduces the fit's own readout. At record "
                        "cell counts 1.0 is near-unregularized over K "
                        "standardized features: 0120 went 0.7555 -> 0.7927 at "
                        "100 (insight 0089). A sweep is a named re-readout.")
    p.add_argument("--readout-stacked", action="store_true",
                   help="Add the STACKED arm (spec 2026-10-06 Part A): the root "
                        "head is fit on every train row (case vs background) in "
                        "the same batched solve, and each node is scored by the "
                        "closure PRODUCT of its ancestors' heads, "
                        "P_stack(c)=prod_{a in closure(c)} sigma(z_a) — HSLDA's "
                        "label-side gating on post-hoc heads. Outputs go to "
                        "results_readout_stacked.json / "
                        "readout_heads_gated_pc_stacked.npz (the record files are "
                        "untouched); the block carries the flat, root-only and "
                        "stacked detection side by side. Distributed readout + "
                        "driver eval path only.")
    p.add_argument("--readout-hierarchy", choices=["auto", "mondo", "flat"],
                   default="auto",
                   help="WP-B (spec 2026-10-07 §D1/§D4, insight 0096): read a FLAT "
                        "label-set fit through the Mondo hierarchy. 'mondo' builds "
                        "the readout DAG over the fit's label nodes + every Mondo "
                        "ancestor grouping >= 2 of them (root aliases folded, rungs "
                        "dropped), widens label/labelMask on the scored frames so a "
                        "node is observed only where it or a sibling is positive "
                        "(negatives = siblings under the nearest grouping ancestor), "
                        "appends the ancestor heads, and feeds that DAG to the "
                        "stacked product. 'auto' (default) = 'mondo' for label-set "
                        "fits, 'flat' otherwise (every nested run is byte-identical). "
                        "'flat' on a label-set fit tags its outputs _flat. Writes "
                        "<run>/readout_dag.json. Incompatible with "
                        "--readout-feature-mask; the incident arm is skipped.")
    return p


def resolve_run_dir(pattern):
    """Resolve --run-dir to a single gated_pc run directory.

    An exact dir that holds gated_pc_result.npz is used as-is. Otherwise the value
    is treated as a glob (the Makefile passes `.../<id>-*` QUOTED so the shell does
    not pre-expand it) and filtered to matched dirs that contain gated_pc_result.npz
    — so a runs/ numbering COLLISION with a non-gated_pc experiment of the same id
    (e.g. a stale `0076-multidomain-...` from another branch) is discarded
    automatically. Requires exactly one gated_pc match; a clear error otherwise."""
    p = Path(pattern)
    if p.is_dir() and (p / "gated_pc_result.npz").exists():
        return p
    matches = [Path(m) for m in glob.glob(str(pattern))]
    hits = [m for m in matches if m.is_dir() and (m / "gated_pc_result.npz").exists()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[readout] no run dir with gated_pc_result.npz matches {pattern!r} "
            f"(matched: {[m.name for m in matches]}). Has the fit finished + saved? "
            f"Pass an exact --run-dir / GPR_RUN_DIR.")
    raise SystemExit(
        f"[readout] {len(hits)} gated_pc run dirs match {pattern!r}: "
        f"{[m.name for m in hits]}. Pass an exact --run-dir / GPR_RUN_DIR.")


# The single-domain assembler's kwargs that are also cache-key inputs (billing is
# an assembly input only). Kept next to the spec builder so "what the key needs"
# and "what a rebuild needs" cannot drift apart.
#
# `doc_spec` is NOT here for the same reason it is not in
# `gated_pc_cloud._SPEC_ASSEMBLY_KEYS`: these names are forwarded to the
# assembler (`rebuild_bundle` below), which takes no doc-spec kwarg. It is
# key-only, and is passed explicitly at each of the two sites that need it.
_SNOMED_KEY_KEYS = (
    "source_table", "person_mod", "vocab_size", "min_df", "min_patient_count",
    "doc_min_length", "prior_obs_days", "window_days", "disease", "min_n",
    "holdout_frac", "n_bg", "tpn", "cdr", "strip_mode", "window_mode",
    "lookback_days", "label_window_days", "emit_labels", "label_mask_mode",
)


def corpus_spec_from_manifest(manifest: dict, *, doc_min_length=None, billing=None,
                              dag_source=None, mondo_version=None, mondo_branch=None,
                              min_positives=None, mondo_cache_dir=None,
                              dag_collapse=None, preindex_closure=None,
                              label_set_path=None) -> dict:
    """The corpus SPEC a gated_pc manifest describes — key inputs + rebuild inputs.

    The current driver writes this dict into `corpus_manifest` verbatim, so for any
    recent run this is just a read-back. Older manifests kept some fields only at
    the top level (and the Mondo build inputs not at all), so every field falls back
    `corpus_manifest` -> top level -> a documented default, and the CLI overrides
    supply what neither has. Raises KeyError naming the field when a required one is
    missing everywhere."""
    from _case_finding_cache import DEFAULT_DOC_SPEC

    cm = manifest.get("corpus_manifest", {})

    _MISSING = object()

    def _pick(name, default=_MISSING):
        if name in cm and cm[name] is not None:
            return cm[name]
        if name in manifest and manifest[name] is not None:
            return manifest[name]
        if default is _MISSING:
            raise KeyError(
                f"{name} is not in the manifest (neither corpus_manifest nor the "
                f"top level); it is a cache-key input, so the bundle cannot be "
                f"located. Pass --bundle-path at the exact cached dir.")
        return default

    dml = doc_min_length if doc_min_length is not None else cm.get("doc_min_length")
    if dml is None:
        raise KeyError(
            "doc_min_length is not in the manifest and no --doc-min-length was "
            "given; it is a cache-key input, so the bundle cannot be located. "
            "Pass --doc-min-length (the fit's value) or --bundle-path.")
    dag_source = str(dag_source if dag_source is not None
                     else _pick("dag_source", "snomed"))
    # Both Mondo flavours (exp 0088/0104's anchor hierarchy and exp 0110's native
    # label space) share the population index, min_n=0 and the Mondo build inputs.
    mondo = dag_source in _MONDO_DAG_SOURCES
    extra_domains = list(_pick("extra_domains", []) or [])
    spec = {
        "dag_source": dag_source,
        "extra_domains": extra_domains,
        # The Mondo fit indexes the whole POPULATION and passes min_n=0 (its DAG is
        # already powered); both are defaults here so a manifest written before
        # corpus_manifest carried them still recomputes the fit's own key.
        "index_mode": _pick("index_mode", "population" if mondo else "disease"),
        "min_n": (cm["min_n"] if "min_n" in cm
                  else (0 if mondo else manifest["min_n"])),
        "disease": _pick("disease"),
        "cdr": _pick("cdr", None),
        "billing": billing if billing is not None else _pick("billing", None),
        "source_table": _pick("source_table", "condition_era"),
        "person_mod": _pick("person_mod"), "vocab_size": _pick("vocab_size"),
        "min_df": _pick("min_df"), "min_patient_count": _pick("min_patient_count"),
        "doc_min_length": int(dml),
        "prior_obs_days": _pick("prior_obs_days", 0),
        "window_days": _pick("window_days", 0),
        "holdout_frac": _pick("holdout_frac"),
        "n_bg": _pick("n_bg"), "tpn": _pick("tpn"),
        "strip_mode": _pick("strip_mode"), "window_mode": _pick("window_mode"),
        "lookback_days": _pick("lookback_days"),
        "label_window_days": _pick("label_window_days"),
        "label_mask_mode": _pick("label_mask_mode", "full"),
        "emit_labels": True,               # a gated_pc corpus always carries labels
        "mondo_version": (mondo_version if mondo_version is not None
                          else _pick("mondo_version", "")),
        "mondo_branch": (mondo_branch if mondo_branch is not None
                         else _pick("mondo_branch", "")),
        "min_positives": (min_positives if min_positives is not None
                          else _pick("min_positives", 0)),
        "mondo_cache_dir": (mondo_cache_dir if mondo_cache_dir is not None
                            else _pick("mondo_cache_dir", "data/mondo")),
        # exp 0109. Defaults to False, which is what EVERY manifest written before
        # the flag existed means — so an old run still recomputes its own key.
        # On the exp-0110 native path the splice is intrinsic to the build, so the
        # flag is pinned False there and cannot double-apply (this mirrors
        # `gated_pc_cloud.multidomain_corpus_spec`, which is what wrote the field).
        "dag_collapse": (False if dag_source == "mondo_native" else
                         bool(dag_collapse if dag_collapse is not None
                              else _pick("dag_collapse", False))),
        # Spec 2026-10-07 §D1: the ribbon. `label_set` is the KEY identity and is
        # always the manifest's (a readout cannot be told to key under a different
        # ribbon than the fit used); `label_set_path` is the rebuild input and may
        # be overridden when the TSV moved. Both default to '' — what every
        # manifest written before the field existed means.
        "label_set": str(_pick("label_set", "") or ""),
        "label_set_path": str(label_set_path if label_set_path is not None
                              else _pick("label_set_path", "") or ""),
        # The DOC UNIT (R5.3 / audit seam 4). A cache-key input as of this
        # change, and the default is what EVERY manifest written before the field
        # existed means — every corpus in the repo was assembled under it — so an
        # old run still recomputes its own byte-identical key. There is no CLI
        # override: unlike the Mondo build inputs, the doc unit is not something a
        # rebuild can be told to differ on, because the assembler hard-codes it.
        "doc_spec": str(_pick("doc_spec", DEFAULT_DOC_SPEC)),
        # exp 0111 WP-D2: the external episode/random index identity. Defaults to
        # None (absent) — what every manifest written before the arms existed means,
        # so an old run recomputes its byte-identical key — and is folded into the
        # bundle key ONLY on the `index_mode="external"` path (multidomain_cache_key),
        # so a re-readout of an episode bundle recomputes the fit's OWN key. The
        # sidecar URI rides too but is NOT a key input; it only tells a re-readout
        # MISS where to reload/rebuild the index's sidecar.
        "episode_sampling": _pick("episode_sampling", None),
        "episode_sidecar_uri": _pick("episode_sidecar_uri", ""),
        # exp 0111 WP-D3 (D13): the gate/label-separation identity. Defaults to None
        # (absent) — what every manifest written before D13 means, so an old run
        # recomputes its byte-identical key — and is folded into the bundle key ONLY
        # on the external path (multidomain_cache_key), so a re-readout of a gated
        # episode bundle recomputes the fit's OWN key. The readout reads the 365-day
        # `label`/`labelMask` regardless; this field only routes the cache key.
        "gate_frontier_mode": _pick("gate_frontier_mode", None),
        # Label redesign R2: the all-ever train gate (absent = label year), so a
        # re-readout of an `ever` fit recomputes the fit's own bundle key.
        "gate_attestation": _pick("gate_attestation", None),
        # E1 (exp E-census onward). Mondo-only, defaults to False — what every
        # manifest written before the flag existed means — and the CLI wins in
        # both directions for a fit whose manifest omits or contradicts it, the
        # same tri-state shape `dag_collapse` uses. It is a cache-key input, so a
        # wrong value MISSES the cache rather than mis-scoring.
        "preindex_closure": (bool(preindex_closure
                                  if preindex_closure is not None
                                  else _pick("preindex_closure", False))
                             if mondo else False),
    }
    return spec


def spec_is_multidomain(spec) -> bool:
    """True when the corpus came from the multi-domain assembler (either Mondo
    flavour, or plain extra domains) — the shapes whose cache key carries the
    extra fold. Missing `mondo_native` here would recompute a SNOMED key for an
    exp-0110 run, i.e. a guaranteed MISS followed by a ~20-minute rebuild of the
    wrong corpus, which is exactly the failure `mondo_spec_mismatch` was written
    for after exp 0104's fresh-cluster recovery."""
    return (str(spec.get("dag_source", "snomed")) in _MONDO_DAG_SOURCES
            or bool(spec.get("extra_domains")))


def native_spec_mismatch(spec, manifest) -> bool:
    """True when the SAVED FIT is an exp-0110 native run but the rebuild spec
    resolved to something else.

    The sibling of `mondo_spec_mismatch` for the native label space, and it needs
    its own witness: a native fit's `int2cid` values are plain ints (the Mondo
    curie's numeric part — see `mondo_native_dag`'s docstring on why they cannot
    be curie strings), so the `MONDO:`-prefix test cannot see it. The manifest's
    own `corpus_manifest.dag_source` is the witness instead, which every native
    run records; the only way to reach here is a CLI override that contradicts
    it, and silently rebuilding a DIFFERENT label space under a wrong key is worth
    an exit 2 rather than a drift-gate failure 20 minutes later."""
    cm = manifest.get("corpus_manifest") or {}
    saved = str(cm.get("dag_source") or manifest.get("dag_source") or "")
    return saved == "mondo_native" and str(spec.get("dag_source")) != "mondo_native"


def mondo_spec_mismatch(spec, manifest) -> bool:
    """True when the rebuild spec says snomed but the SAVED FIT is a Mondo run.

    A manifest written before ``corpus_manifest`` recorded ``dag_source`` resolves
    to the snomed default, and a Mondo fit then keys — and, on the MISS that wrong
    key guarantees — REBUILDS the wrong corpus. That cost a fresh-cluster recovery
    of exp 0104 a rebuild attempt (it also surfaced as a billing NPE, but the
    corpus was already wrong before BigQuery was ever reached). The fit itself is
    the witness the manifest lacks: a Mondo run's label space is Mondo ids, so
    ``int2cid`` carries the ``MONDO:`` prefix. Detecting the mismatch here — BEFORE
    the key is computed — turns ~20 minutes of rebuilding the wrong thing (caught
    only by the drift gate, exit 3) into an immediate exit 2 naming the flags."""
    int2cid = manifest.get("int2cid") or []
    cids = int2cid.values() if isinstance(int2cid, dict) else int2cid
    looks_mondo = any(str(c).startswith("MONDO:") for c in cids)
    return looks_mondo and str(spec.get("dag_source")) != "mondo"


def bundle_key_from_manifest(manifest: dict, *, doc_min_length=None, **spec_over):
    """Recompute the bundle cache key from a gated_pc manifest.

    Routes on the corpus the manifest describes: a mondo / extra-domains run gets
    the MULTI-DOMAIN key (the same `multidomain_cache_key` the fit computes, folding
    extra_domains / index_mode / the Mondo build inputs), anything else the
    single-domain key, byte-identical to what it has always been. `doc_min_length`
    and the `spec_over` fields override the manifest (older manifests omit them)."""
    from _case_finding_cache import compute_bundle_cache_key

    spec = corpus_spec_from_manifest(manifest, doc_min_length=doc_min_length,
                                     **spec_over)
    if spec_is_multidomain(spec):
        return multidomain_cache_key(spec)
    return compute_bundle_cache_key(doc_spec=spec["doc_spec"],
                                    **{k: spec[k] for k in _SNOMED_KEY_KEYS})


def rebuild_bundle(spark, spec, *, cache_uri=None):
    """Re-assemble the corpus a finished fit was built against, and write it through.

    The SAME seam the fit uses — `multidomain_load_or_build` (Mondo DAG build +
    SNOMED-climb provider included, on the miss it is about to take) for a
    multi-domain corpus, `load_or_build_case_finding_bundle` for a single-domain
    one — so the bundle this produces lands under the key the fit's own bundle would
    have, and the next readout (or fit) of this corpus is a HIT."""
    if spec_is_multidomain(spec):
        return multidomain_load_or_build(spark, spec, cache_uri=cache_uri)
    from _case_finding_cache import load_or_build_case_finding_bundle
    params = {k: spec[k] for k in _SNOMED_KEY_KEYS}
    return load_or_build_case_finding_bundle(
        spark, cache_uri=cache_uri, billing=spec["billing"],
        # key-only, exactly as the fit passes it (gated_pc_cloud's single-domain
        # call site), so a rebuild lands under the key the fit stored under.
        _key_extra={"doc_spec": spec["doc_spec"]}, **params)


def lambda_vocab_sizes(global_params) -> list:
    """The V dimension of each domain's λ, in domain order — the saved fit's own
    record of how wide each vocabulary was. Multi-domain λ is a {domain: (K, V_m)}
    dict; single-domain is one (K, V) array."""
    lam = global_params["lambda"]
    if isinstance(lam, dict):
        return [int(lam[m].shape[1]) for m in sorted(lam)]
    return [int(lam.shape[1])]


def bundle_drift_report(bundle, manifest, fit_vocab_sizes) -> list:
    """Reasons the corpus in hand cannot be scored against the saved fit (empty = ok).

    Re-assembly is deterministic in practice but nothing guarantees it: the CDR
    advances, an assembly-source edit changes the split or the vocabulary fit, a
    Mondo release re-parents a branch. The λ that comes back from the npz is
    (K, V_m) per domain and the per-node head is (C, K) over engine ids, so the two
    things that MUST still line up are each domain's vocabulary width and the
    engine-id -> concept-id map. Both are recorded — the widths implicitly in λ, the
    map explicitly in the manifest — so drift is detectable before a single number
    is computed, and reporting AUCs for a model scored on a corpus it was not fit on
    is not a failure mode this tool has to have."""
    problems = []
    vocab_maps = getattr(bundle, "vocab_maps", None)
    if vocab_maps is None:
        vocab_maps = [bundle.vocab_map]
    sizes = [len(vm) for vm in vocab_maps]
    if len(sizes) != len(fit_vocab_sizes):
        problems.append(
            f"the saved fit has {len(fit_vocab_sizes)} domain(s) "
            f"(lambda {fit_vocab_sizes}) but the corpus has {len(sizes)} "
            f"(vocab sizes {sizes})")
    else:
        for i, (v_fit, v_corpus) in enumerate(zip(fit_vocab_sizes, sizes)):
            if int(v_fit) != int(v_corpus):
                problems.append(
                    f"domain {i}: the fit's lambda is V={v_fit} wide, the corpus "
                    f"vocabulary has {v_corpus} concepts")
    got = {int(i): int(c) for i, c in bundle.int2cid.items()}
    C = manifest.get("C")
    if C is not None and len(got) != int(C):
        problems.append(
            f"the fit has C={int(C)} label heads, the corpus DAG has {len(got)} nodes")
    saved = (manifest.get("corpus_manifest") or {}).get("int2cid")
    if saved:
        want = {int(i): int(c) for i, c in saved.items()}
        if got != want:
            diff = sorted(i for i in set(got) | set(want)
                          if got.get(i) != want.get(i))
            problems.append(
                f"the engine-id -> concept-id map differs at {len(diff)} node(s) "
                f"(first: {diff[:5]}); the label DAG is not the one the head was "
                f"fit against")
    return problems


def reconstruct_model(run_dir: Path, manifest: dict):
    """Rebuild a scoreable OnlinePCLDAModel from the saved gated_pc_result.npz.

    The driver saves raw globals (lambda/alpha/w_CK), not a persisted model, so we
    wrap them in a VIResult and an OnlinePCLDAModel and set weightY>0 + numLabels so
    transform appends both topicDistribution (θ) and the head probability. The CAVI
    read-out knobs (gammaShape/caviMaxIter/caviTol) come from the model defaults,
    which match the fit's defaults for this experiment family.

    MULTI-DOMAIN (the Mondo path): `np.savez` cannot store a dict, so `_save_fit`
    writes a per-domain λ as `lambda_0, lambda_1, ...`; it is reassembled here into
    the `{domain: (K, V_m)}` dict `_transform` fuses, and `featuresCols` is set to
    the matching `features_0..` so the transform reads the per-domain BOW columns
    instead of a `features` column the multi-domain corpus does not have."""
    from spark_vi.core.result import VIResult
    from spark_vi.mllib.topic.pc import OnlinePCLDAModel

    npz = np.load(run_dir / "gated_pc_result.npz")
    files = set(npz.files)
    if "lambda" in files:
        lam = npz["lambda"]
    else:
        dom = sorted(int(f.split("_", 1)[1]) for f in files
                     if f.startswith("lambda_"))
        if not dom:
            raise KeyError(
                f"{run_dir}/gated_pc_result.npz has no lambda: found {sorted(files)}")
        lam = {m: npz[f"lambda_{m}"] for m in dom}
    gp = {"lambda": lam, "alpha": npz["alpha"], "w_CK": npz["w_CK"]}
    if "b_CK" in files:
        # The per-node intercept (--head-intercept) is part of the head; without it
        # the head arm's P(node) would be the intercept-free sigmoid of a model that
        # was fit with one. Absent on runs that predate the save.
        gp["b_CK"] = npz["b_CK"]
    result = VIResult(global_params=gp, elbo_trace=[],
                      n_iterations=int(manifest.get("max_iter", 0)), converged=True)
    model = OnlinePCLDAModel(result)
    model._set(weightY=float(manifest.get("weight_y", 1.0)),
               numLabels=int(manifest["C"]), closureParents="")
    if isinstance(lam, dict):
        model._set(featuresCols=[f"features_{i}" for i in range(len(lam))])
    return model


def run_readout(train_scored, test_scored, manifest, *, recall_targets, fdr_targets,
                min_count, out_dir=None, theta_topm=None, readout_max_iter=None,
                readout_l2=None, mask_tag=None, stacked=False, parent_int=None,
                theta_mass=None, C=None, warm_start=None):
    """Fit and score the pc_topics_lr heads off two already-TRANSFORMED splits.

    One path (cleanup 2026-10): the per-node L2 logistic heads are fit by the
    distributed batched L-BFGS solve (`distributed_score_arm`), the test split's
    lean (D, C) proba is collected for the metrics, and `stacked=True` adds the
    closure-product block on that same proba. Taking DataFrames rather than a
    SparkSession + run dir keeps it callable on a local Spark fixture.

    `theta_topm` / `theta_mass` / `readout_max_iter` / `readout_l2` are None-means-
    ask-the-manifest, so a re-readout of an old fit reproduces its own design
    (manifests written since the cleanup carry none of them; the CLI sets them).
    `out_dir` gets `results_readout[_<tag>].json` after each block lands, the solver
    checkpoint while it runs, and the heads sidecar after.

    `stacked=True` (spec 2026-10-06 Part A): the TRAIN mask gets the root observed
    on every row (`observe_root_everywhere`), so the root head is case vs background,
    and `stacked_readout` scores the closure product. Needs `parent_int`.

    `C` overrides the manifest's label width when the frames were widened to the
    readout node space (`readout_dag.widen_labels_to_readout`); `parent_int` is then
    the readout DAG. `warm_start=(V, b_raw)` (`--readout-warm-from`) starts the solve
    from saved heads of the same problem."""
    C = int(manifest["C"] if C is None else C)
    K = int(manifest.get("K") or 0)
    if not K:
        K = int(train_scored.select("topicDistribution").head()[0].size)
    _cprint(f"[readout]   distributed readout (C={C}, K={K})", flush=True)

    arm = "gated_pc" if not mask_tag else f"gated_pc_{mask_tag}"
    results_name = ("results_readout.json" if not mask_tag
                    else f"results_readout_{mask_tag}.json")
    if stacked:
        if parent_int is None:
            raise SystemExit("[readout] --readout-stacked needs the bundle's "
                             "parent_int (no DAG, no closure)")
        # The one change to the solve: the root is observed on every train row,
        # so its head is case-vs-background instead of the degenerate constant.
        train_scored = observe_root_everywhere(train_scored, C)
        _cprint("[readout]   STACKED arm: root head fit on EVERY train row (case vs "
              "background); heads of nodes >= 1 unchanged (per-node independent "
              "solve)", flush=True)

    def _dump(results):
        if out_dir is not None:
            _dump_partial_results(Path(out_dir), results, name=results_name)

    results = {}
    _mass = None
    if theta_mass is not None and float(theta_mass) > 0:
        _mass = float(theta_mass)
        _cprint(f"[readout]   theta top-m by MASS target {_mass:g} (CLI)", flush=True)
    elif theta_topm is None and not int(manifest.get("readout_theta_topm", 0) or 0) \
            and float(manifest.get("readout_theta_mass", 0) or 0) > 0:
        # An older fit-only manifest recorded the mass target, not the width.
        _mass = float(manifest["readout_theta_mass"])
        _cprint(f"[readout]   theta top-m by MASS target {_mass:g} (from the "
              "manifest; width not yet resolved there)", flush=True)
    if _mass is not None:
        import distributed_readout as _drm
        theta_topm, _cov = _drm.resolve_theta_topm_by_mass(
            train_scored, K or None, _mass)
        _cprint("[readout]   theta top-m by mass: "
              + " ".join(f"m={m}:{mean:.3f}/{p10:.3f}"
                         for m, (mean, p10) in sorted(_cov.items()))
              + " (mean/p10) -> " + (f"m={theta_topm}" if theta_topm else
                                     f"FULL K={K}"), flush=True)
    elif theta_topm is None:
        theta_topm = int(manifest.get("readout_theta_topm", 0) or 0)
        if theta_topm:
            _cprint(f"[readout]   theta top-m={theta_topm} (from manifest)", flush=True)
    else:
        theta_topm = int(theta_topm)
        _cprint(f"[readout]   theta top-m={theta_topm} (CLI)", flush=True)
    readout_max_iter, _mi_src = resolve_readout_max_iter(readout_max_iter, manifest)
    readout_l2, _l2_src = resolve_readout_l2(readout_l2, manifest)
    _cprint(f"[readout]   readout l2={readout_l2:g} (from {_l2_src})", flush=True)
    _cprint(f"[readout]   readout max_iter={readout_max_iter} (from {_mi_src})",
          flush=True)
    dist = distributed_score_arm(
        train_scored, test_scored, C, K, recall_targets=recall_targets,
        fdr_targets=fdr_targets, min_count=min_count, label=arm,
        theta_topm=theta_topm, max_iter=readout_max_iter, l2=readout_l2,
        warm_start=warm_start,
        # The solve is the multi-hour part: a death mid-solve costs at most
        # `checkpoint_every` iterations, because the next invocation finds the
        # checkpoint next to the manifest it already reads.
        checkpoint_dir=out_dir)
    results["gated_pc"] = dist[0]
    _dump(results)
    print(format_arm_readout(f"{arm} (pc_topics_lr)", results["gated_pc"]),
          flush=True)
    if stacked:
        # Same collected proba the arm was scored on — one closure product, no
        # second collect, no second solve.
        _stk = stacked_readout(
            dist[1], dist[2], dist[3], parent_int, C, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count,
            prevalent=results["gated_pc"], doc_keys=dist[4], arm_label=arm)
        results["gated_pc_stacked"] = _stk
        _dump(results)
        print(format_stacked_readout(_stk).replace("[driver]", "[readout]"),
              flush=True)
    return results


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    run_dir = resolve_run_dir(args.run_dir)
    # Every [readout]/[driver] line is teed to a durable readout_log.md IN THE
    # RUN DIR (the workspace disk survives cluster restarts; ~ and /tmp do not).
    # Appended, so a re-readout, an ablation sweep, and a ridge sweep all land
    # in one chronological file beside the results they produced. Separate from
    # the fit's driver_log.md, which `gated_pc_report --summary` digests.
    install_stdout_tee(run_dir / "readout_log.md")
    _cprint(f"[readout]   run dir: {run_dir}", flush=True)
    _argv = list(sys.argv[1:]) if argv is None else list(argv)
    _cprint(f"[readout]   argv: {' '.join(_argv)}", flush=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    rt = [float(x) for x in args.recall_targets.split(",") if x]
    ft = [float(x) for x in args.fdr_targets.split(",") if x]
    min_count = (args.min_label_count if args.min_label_count is not None
                 else int(manifest.get("min_label_count", 20)))

    spec_over = dict(billing=args.billing, dag_source=args.dag_source,
                     mondo_version=args.mondo_version,
                     mondo_branch=args.mondo_branch,
                     min_positives=args.min_positives,
                     mondo_cache_dir=args.mondo_cache_dir)
    # The fit records the cache root it used, so a recovery command need not
    # remember it; an explicit --cache-uri still wins.
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")

    with make_spark_session(app_name="gated-pc-readout") as spark:
        # Driver-disk telemetry, IN-BAND (see disk_telemetry's docstring): this
        # is the solve that keeps dying of ENOSPC from `sc.broadcast` with every
        # ADR 0047 destroy fix active, and the two local `diskwatch` loops that
        # were supposed to catch leak #2 both died with their cluster. Printing
        # from inside the driver puts the disk history in the persisted job log
        # and in driver_log.md, where it survives the machine.
        start_disk_telemetry(
            extra_dirs=[d for d in spark.sparkContext.getConf()
                        .get("spark.local.dir", "").split(",") if d],
            log=lambda msg: _cprint(f"[readout] {msg}", flush=True))

        from _case_finding_cache import try_load

        with _phase("reload cached bundle"):
            spec = None
            if args.bundle_path:
                base = args.bundle_path.rstrip("/")
                cache_uri, key = base.rsplit("/", 1)
            else:
                if not cache_uri:
                    _cprint("[readout] ERROR: pass --cache-uri or --bundle-path.",
                          flush=True)
                    return 1
                spec = corpus_spec_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
                if native_spec_mismatch(spec, manifest):
                    _cprint("[readout] ERROR: the saved fit is an exp-0110 NATIVE "
                          "Mondo run (corpus_manifest.dag_source=mondo_native), "
                          "but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — that keys, and on the MISS "
                          "rebuilds, a different label space. Drop the override, "
                          "or pass --dag-source mondo_native.", flush=True)
                    return 2
                if mondo_spec_mismatch(spec, manifest):
                    _cprint("[readout] ERROR: the saved fit's label space is MONDO "
                          "ids, but the rebuild spec resolved dag_source="
                          f"{spec['dag_source']!r} — this manifest predates "
                          "corpus_manifest recording the Mondo build inputs, so "
                          "the defaults would key (and on the MISS that wrong key "
                          "guarantees, rebuild) the WRONG corpus. Pass the fit's "
                          "own values from the experiment doc's front matter, "
                          "e.g. for exp 0104:", flush=True)
                    _cprint("[readout]   --dag-source mondo --mondo-version "
                          "2026-06-02 --min-positives 100", flush=True)
                    return 2
                key = bundle_key_from_manifest(
                    manifest, doc_min_length=args.doc_min_length, **spec_over)
            bundle = try_load(spark, cache_uri, key)
            if bundle is None and (args.no_rebuild or spec is None):
                # --no-rebuild, or a --bundle-path whose dir does not hold a bundle
                # (there is no spec to rebuild FROM in that case).
                _cprint(f"[readout] ERROR: bundle cache MISS at {cache_uri}/{key}. "
                      "The assembly source may have changed since the fit, or a "
                      "key field differs. Pass --bundle-path at the exact cached "
                      "dir, or --doc-min-length if it was omitted"
                      + (" (drop --no-rebuild to re-assemble it instead)."
                         if args.no_rebuild else "."), flush=True)
                return 2
            if bundle is None:
                # LOAD-OR-BUILD: a cold cache (a fresh cluster, a cleared bucket, or
                # a fit that predates the corpus being cached at all) is not a reason
                # to lose a finished fit. Rebuild through the fit's own seam and
                # write it through, so this is paid once.
                _cprint("[readout] cache MISS — rebuilding bundle from manifest "
                      "params (~20 min at whole-Mondo)", flush=True)
                # billing/cdr are ENVIRONMENT, not corpus identity: the fit takes
                # them from the sourced .workspace_env, and manifests from before
                # they were recorded carry neither. Fall back to the same env vars
                # here — the Makefile target sources .workspace_env before this
                # tool runs — rather than handing spark-bigquery a null
                # parentProject, which dies as an opaque py4j NPE (exp 0104's
                # fresh-cluster recovery, 08-28).
                for field, env_var, flag in (("billing", "GOOGLE_CLOUD_PROJECT",
                                              "--billing"),
                                             ("cdr", "WORKSPACE_CDR", None)):
                    if spec.get(field) is None:
                        env_val = os.environ.get(env_var)
                        if env_val:
                            spec[field] = env_val
                            _cprint(f"[readout]   {field} not in the manifest; "
                                  f"using {env_var} from the environment (the "
                                  "same source the fit used)", flush=True)
                        else:
                            _cprint(f"[readout] ERROR: a rebuild needs {field}, "
                                  "but the manifest predates recording it and "
                                  f"{env_var} is unset"
                                  + (f" — pass {flag}." if flag else
                                     " — source .workspace_env (make setup)."),
                                  flush=True)
                            return 2
                _cprint(f"[readout]   corpus: dag_source={spec['dag_source']} "
                      f"extra_domains={spec['extra_domains']} "
                      f"index_mode={spec['index_mode']} min_n={spec['min_n']} "
                      f"mondo_branch={spec['mondo_branch'] or 'ALL'} "
                      f"min_positives={spec['min_positives']} "
                      f"dag_collapse={spec['dag_collapse']} "
                      f"preindex_closure={spec['preindex_closure']}", flush=True)
                bundle = rebuild_bundle(
                    spark, spec,
                    cache_uri=(cache_uri if args.cache_write == "on" else None))
                _cprint(f"[readout]   bundle REBUILT"
                      + (f" and written to {cache_uri}/{key} (the next readout of "
                         "this run is a HIT)" if args.cache_write == "on"
                         else " (--cache-write off: not persisted)")
                      + f"; C={C}", flush=True)
            else:
                _cprint(f"[readout]   bundle loaded ({cache_uri}/{key}); C={C}",
                      flush=True)
            from gated_pc_cloud import write_bundle_meta
            _bmp = write_bundle_meta(run_dir, bundle)
            _cprint(f"[readout]   wrote bundle meta -> {_bmp}", flush=True)
            _ncm = getattr(bundle, "native_code_map", None)
            if _ncm and (args.bundle_only or not (run_dir / "code_map.tsv").exists()):
                # WP-C': a fit from before the code map rode the bundle gets it
                # on its first re-readout, so `--profile-census --own-codes`
                # works on 0135 without a refit.
                from gated_pc_cloud import write_code_map
                _cmp = write_code_map(run_dir, _ncm)
                _cprint(f"[readout]   wrote native code map ({len(_ncm)} pairs) -> "
                      f"{_cmp}", flush=True)

        if args.npmi:
            # Exp 0159: gated NPMI from the saved topics + the bundle's documents;
            # no transform, no heads.
            with _phase("gated NPMI"):
                from spark_vi.models.topic.dag_placement import DagLayout
                from topic_npmi import run_npmi
                _npz = np.load(run_dir / "gated_pc_result.npz")
                _lk = sorted((k for k in _npz.files if k.startswith("lambda_")),
                             key=lambda k: int(k.split("_", 1)[1]))
                _lams = [_npz[k] for k in _lk] if _lk else [_npz["lambda"]]
                _cm = manifest.get("corpus_manifest") or {}
                _lay = DagLayout({int(c): [int(p) for p in ps] for c, ps in
                                  bundle.parent_int.items()},
                                 n_bg=int(manifest["n_bg"]), tpn=int(manifest["tpn"]))
                run_npmi(bundle.train_df, bundle.test_df, run_dir, lay=_lay, lams=_lams,
                         n_bg=int(manifest["n_bg"]),
                         int2cid={int(k): int(v) for k, v in _cm.get("int2cid", {}).items()},
                         names={int(k): v for k, v in _cm.get("name_by_id", {}).items()},
                         split=args.npmi_split, top_n=args.npmi_top,
                         domains=[int(x) for x in str(args.npmi_domains).split(",")],
                         tag=args.acq_tag)
            return 0

        if args.differential:
            with _phase("differential with reasons (framing 1)"):
                run_differential(
                    bundle.test_df, bundle.train_df, run_dir, args.differential,
                    adhoc=args.acq_adhoc, heads_tag=args.acq_heads,
                    max_patients=args.diff_max_patients,
                    max_train=args.diff_max_train, tag=args.acq_tag,
                    next_question=args.diff_next_question,
                    nq_hide=args.diff_nq_hide, crossfit=args.diff_crossfit,
                    sidecar_uri=args.prev_sidecar_uri,
                    fast=bool(args.diff_survey), max_members=(args.diff_survey or None),
                    incident=args.diff_incident, groups_like=args.diff_groups_like,
                    member_rule=args.diff_members)
            return 0

        if args.bundle_only:
            _cprint("[readout]   --bundle-only: bundle meta + code map written; "
                  "stopping before the transform", flush=True)
            return 0

        with _phase("reconstruct model + transform"):
            # The theta collect (driver mode) now happens inside run_readout, so the
            # distributed mode can skip it entirely rather than paying for it here.
            model = reconstruct_model(run_dir, manifest)
            # DRIFT GATE, before a single number is computed: the corpus in hand
            # (cached or rebuilt) must be the one this λ/head was fit against.
            drift = bundle_drift_report(
                bundle, manifest, lambda_vocab_sizes(model.result.global_params))
            if drift:
                _cprint("[readout] ERROR: the corpus does not match the saved fit — "
                      "it has DRIFTED since the fit (a data refresh, or an "
                      "assembly-source change). Refusing to score:", flush=True)
                for line in drift:
                    _cprint(f"[readout]     - {line}", flush=True)
                _cprint("[readout]   Pass --bundle-path at the bundle this run was "
                      "actually fit against, or re-fit.", flush=True)
                return 3
            if args.raw_baseline:
                # Exp 0147: no θ needed — the baseline reads the bundle's own BOW.
                train_scored = bundle.train_df.cache()
                test_scored = bundle.test_df.cache()
            else:
                train_scored = model.transform(bundle.train_df).cache()
                test_scored = model.transform(bundle.test_df).cache()

        # WP-B (spec 2026-10-07 §D1/§D4): a label-set fit is FLAT, and the
        # bundle's closure mask on a flat forest is a full mask (insight 0096:
        # 70.7M cells). The read gets the hierarchy the fit was denied: the
        # Mondo readout DAG over the label nodes + grouping ancestors, applied to
        # the SCORED frames as column arithmetic. Nothing in the bundle, the
        # cache or a hashed module moves; the manifest's C stays the fit's.
        _rdag = None
        _C_ro = C
        _parent_ro = getattr(bundle, "parent_int", None)
        _cm = manifest.get("corpus_manifest") or {}
        from readout_dag import resolve_readout_hierarchy
        if resolve_readout_hierarchy(args.readout_hierarchy, manifest):
            with _phase("readout DAG (Mondo hierarchy over the flat fit)"):
                from readout_dag import (
                    build_readout_dag, format_readout_dag_report,
                    load_mondo_parent_adj, widen_labels_to_readout,
                    write_readout_dag)
                _mv = args.mondo_version or _cm.get("mondo_version")
                _mcd = args.mondo_cache_dir or _cm.get("mondo_cache_dir") or "data/mondo"
                if not _mv:
                    raise SystemExit("[readout] --readout-hierarchy mondo needs the "
                                     "Mondo version (manifest corpus_manifest."
                                     "mondo_version or --mondo-version)")
                _padj, _mnames = load_mondo_parent_adj(_mv, _mcd)
                _rdag = build_readout_dag(bundle.int2cid, _padj, names=_mnames)
                _cprint(format_readout_dag_report(_rdag), flush=True)
                _rp = write_readout_dag(run_dir, _rdag)
                _cprint(f"[readout]   wrote {_rp}", flush=True)
                _mm = str(_cm.get("label_mask_mode") or
                          manifest.get("label_mask_mode") or "full")
                _tr_w = widen_labels_to_readout(train_scored, _rdag, mask_mode=_mm).cache()
                _te_w = widen_labels_to_readout(test_scored, _rdag, mask_mode=_mm).cache()
                # Materialize the widened frames off the cached transforms, then
                # drop the fit-width copies: the solver's repeated passes read
                # the widened train frame, never these.
                # The prevalence check scores the test split only: never
                # materialize the (much larger) train transform for it.
                _n_tr = ("not materialized" if args.prevalence_check
                         else _tr_w.count())
                _n_te = _te_w.count()
                train_scored.unpersist(); test_scored.unpersist()
                train_scored, test_scored = _tr_w, _te_w
                _C_ro, _parent_ro = int(_rdag.C), _rdag.parent_int
                _cprint(f"[readout]   frames widened to the readout node space: "
                      f"C={_C_ro} (fit C={C}), mask mode={_mm}, train docs={_n_tr} "
                      f"test docs={_n_te}", flush=True)

        if args.prevalence_check:
            if _rdag is None:
                raise SystemExit("[readout] --prevalence-check needs the Mondo readout DAG")
            if args.raw_baseline:
                # Exp 0147 re-check from SAVED raw heads: the exact token list the
                # solve used (raw_baseline_tokens.json), test split only, no solve.
                _tj = json.loads((run_dir / _raw_tokens_name(args.raw_select))
                                 .read_text())
                if int(_tj["n"]) != int(args.raw_baseline):
                    raise SystemExit(f"[readout] raw_baseline_tokens.json has n="
                                     f"{_tj['n']}, not --raw-baseline {args.raw_baseline}")
                _sel = [(int(d), int(i)) for d, i in _tj["tokens"]]
                _te_r = attach_raw_presence(test_scored, _sel, _tj["feat_cols"]).cache()
                _cprint(f"[readout]   raw baseline (saved tokens): {len(_sel)} tokens, "
                        f"test docs={_te_r.count()}", flush=True)
                test_scored.unpersist()
                test_scored = _te_r
            with _phase("prevalence check (insight 0100)"):
                run_prevalence_check(
                    test_scored, run_dir, C=_C_ro, parent_int=_parent_ro,
                    heads_tag=args.acq_heads, tag=args.acq_tag,
                    history=(dict(readout_int2cid=_rdag.int2cid, parent_adj=_padj,
                                  sidecar_uri=args.prev_sidecar_uri)
                             if args.prev_history else None))
            return 0

        _manifest_ro = manifest
        if args.raw_baseline:
            if args.readout_theta_mass or args.readout_theta_topm:
                raise SystemExit("[readout] --raw-baseline: the features are binary "
                                 "codes; drop --readout-theta-mass/--readout-theta-topm")
            with _phase(f"raw-code baseline features (exp 0147, top {args.raw_baseline})"):
                _fcols = raw_feature_cols(train_scored)
                _sel = select_raw_codes(train_scored, args.raw_baseline, _fcols)
                if args.raw_select == "model":
                    _freq = set(_sel)
                    _sel = select_model_codes(_load_lambda(run_dir),
                                              int(manifest["n_bg"]), args.raw_baseline)
                    _cprint(f"[readout]   raw baseline, MODEL-selected: {len(_sel)} "
                            f"tokens from the node topics; {len(_freq & set(_sel))} also "
                            f"in the frequency top {args.raw_baseline}", flush=True)
                elif args.raw_select == "relevance":
                    import inspect_topics as _it
                    from spark_vi.models.topic.dag_placement import DagLayout
                    from topic_npmi import own_code_exclusions
                    _freq = set(_sel)
                    _quota = {}
                    for d, _i in _sel:
                        _quota[d] = _quota.get(d, 0) + 1
                    _lay = DagLayout({int(c): [int(q) for q in ps] for c, ps in
                                      bundle.parent_int.items()},
                                     n_bg=int(manifest["n_bg"]),
                                     tpn=int(manifest["tpn"]))
                    _vm0 = bundle.vocab_maps[0]
                    if _vm0 and isinstance(next(iter(_vm0)), str):
                        _vm0 = {int(k): v for k, v in _vm0.items()}
                    _own = _it.own_code_vocab_idx(
                        _it.load_code_map(run_dir / "code_map.tsv"),
                        bundle.int2cid, _vm0)
                    _excl = own_code_exclusions(_lay, _own)
                    if not _excl:
                        raise SystemExit("[readout] --raw-select relevance: no own codes "
                                         "mapped into the condition vocab (code_map.tsv "
                                         "vs bundle ids?) — refusing to run unexcluded")
                    _cprint(f"[readout]   relevance selector: own+descendant codes "
                            f"excluded on {len(_excl)} topics", flush=True)
                    _sel = select_relevance_codes(
                        _load_lambda(run_dir), int(manifest["n_bg"]), _quota,
                        rel_lambda=args.raw_rel_lambda, exclude=_excl)
                    _cprint(f"[readout]   raw baseline, RELEVANCE-selected (λ="
                            f"{args.raw_rel_lambda:g}, own+descendant codes excluded, "
                            f"per-domain quota {_quota}): {len(_sel)} tokens; "
                            f"{len(_freq & set(_sel))} also in the frequency top "
                            f"{args.raw_baseline}", flush=True)
                _tr_r = attach_raw_presence(train_scored, _sel, _fcols).cache()
                _te_r = attach_raw_presence(test_scored, _sel, _fcols).cache()
                _by_dom = {}
                for d, _i in _sel:
                    _by_dom[d] = _by_dom.get(d, 0) + 1
                _cprint(f"[readout]   raw baseline: {len(_sel)} tokens by train document "
                        f"frequency over {_fcols} (per domain {_by_dom}); train docs="
                        f"{_tr_r.count()} test docs={_te_r.count()}", flush=True)
                train_scored.unpersist(); test_scored.unpersist()
                train_scored, test_scored = _tr_r, _te_r
                _manifest_ro = dict(manifest, K=len(_sel), readout_theta_topm=0,
                                    readout_theta_mass=0)
                (run_dir / _raw_tokens_name(args.raw_select)).write_text(json.dumps(
                    {"n": len(_sel), "feat_cols": _fcols, "select": args.raw_select,
                     "tokens": [[int(d), int(i)] for d, i in _sel]}))
        _test_unmasked = test_scored
        if args.readout_nodes:
            if _rdag is None:
                raise SystemExit("[readout] --readout-nodes needs the Mondo readout DAG "
                                 "(--readout-hierarchy mondo)")
            from dismech_ribbon import load_label_set
            _ids, _ident, _rb = load_label_set(args.readout_nodes)
            _cids = [int(str(x).split(":", 1)[1]) for x in _ids]
            _keep = readout_node_keep(_rdag.parent_int, _rdag.int2cid, _cids, _C_ro)
            _cprint(f"[readout]   node subset {_ident}: {len(_ids)} terms -> "
                    f"{sum(_keep)} of {_C_ro} readout heads kept (with ancestors and "
                    f"root); the rest masked to constants", flush=True)
            train_scored = mask_to_nodes(train_scored, _keep)
            test_scored = mask_to_nodes(test_scored, _keep)
        with _phase("score"):
            _mtag = None
            if args.readout_theta_topm is not None:
                # A CLI top-m override is a pricing experiment, not the record:
                # tag its outputs so they never overwrite results_readout.json
                # (or a mask run at the manifest's top-m).
                _mtag = f"topm{int(args.readout_theta_topm)}"
            if args.readout_theta_mass is not None and float(args.readout_theta_mass) > 0:
                if args.readout_theta_topm is not None:
                    raise SystemExit("[readout] --readout-theta-mass and "
                                     "--readout-theta-topm are mutually exclusive")
                _mtag = f"mass{int(round(float(args.readout_theta_mass) * 100))}"
            if args.raw_baseline:
                _mtag = ({"model": "rawm", "relevance": "rawr"}.get(args.raw_select, "raw")
                         + f"{int(args.raw_baseline)}" + (f"_{_mtag}" if _mtag else ""))
            if args.readout_nodes:
                _mtag = "nodes" + (f"_{_mtag}" if _mtag else "")
            if args.readout_stacked:
                # Its own tag: the arm re-solves the SAME heads plus a changed
                # root head, and the record's results/heads files stay the
                # prevalent arm's.
                _mtag = "stacked" + (f"_{_mtag}" if _mtag else "")
                _cprint(f"[readout]   STACKED arm requested: results -> "
                      f"results_readout_{_mtag}.json, heads -> "
                      f"readout_heads_gated_pc_{_mtag}.npz (record untouched)",
                      flush=True)
            # A re-solve with a different budget or start is a convergence check,
            # not the record: tag it so the record's results/heads stay put. An
            # explicit cap EQUAL to the manifest's (a recovery) keeps the record's
            # name, as before.
            _warm = None
            if args.readout_warm_from:
                _wl = f"gated_pc_{args.readout_warm_from}"
                _wh = _read_readout_heads(run_dir, _wl, C=_C_ro,
                                          K=int(_manifest_ro["K"]))
                if _wh is None:
                    raise SystemExit(f"[readout] --readout-warm-from: no usable "
                                     f"readout_heads_{_wl}.npz in {run_dir}")
                _warm = (_wh[0], _wh[1])
                _mtag = (f"{_mtag}_" if _mtag else "") + "warm"
            if args.readout_max_iter is not None:
                _rec_mi = resolve_readout_max_iter(None, _manifest_ro)[0]
                if int(args.readout_max_iter) != _rec_mi or _warm is not None:
                    _mtag = (f"{_mtag}_" if _mtag else "") + f"it{int(args.readout_max_iter)}"
            if _warm is not None or args.readout_max_iter is not None:
                _cprint(f"[readout]   convergence re-solve: "
                      f"{'warm from ' + args.readout_warm_from if _warm else 'cold'}, "
                      f"max_iter={args.readout_max_iter or 'manifest'} -> tag {_mtag}",
                      flush=True)
            _rname = ("results_readout.json" if not _mtag
                      else f"results_readout_{_mtag}.json")
            run_readout(train_scored, test_scored, _manifest_ro, recall_targets=rt,
                        fdr_targets=ft, min_count=min_count,
                        out_dir=run_dir,
                        theta_topm=(0 if args.raw_baseline
                                    else args.readout_theta_topm),
                        readout_max_iter=args.readout_max_iter,
                        readout_l2=args.readout_l2,
                        mask_tag=_mtag, stacked=bool(args.readout_stacked),
                        parent_int=_parent_ro,
                        theta_mass=args.readout_theta_mass, C=_C_ro,
                        warm_start=_warm)
            _cprint(f"[readout]   arm results written to {run_dir / _rname}",
                  flush=True)
            if ((args.raw_baseline or args.readout_nodes or args.prev_history)
                    and args.readout_stacked):
                # --prev-history alone asks for the new-diagnosis check too (exp 0153:
                # a plain readout with it silently skipped the check).
                # The comparison that matters: these heads vs the record's on the
                # SAME unmasked test split under the same prior rule (0143/0145: the
                # record's stacked incident macro AUC 0.702 / 0.694, flat 0.616 /
                # 0.612).
                with _phase(f"prevalence check on the {_mtag} heads"):
                    run_prevalence_check(
                        _test_unmasked, run_dir, C=_C_ro, parent_int=_parent_ro,
                        heads_tag=_mtag, tag=_mtag,
                        node_subset=(_keep if args.readout_nodes else None),
                        history=(dict(readout_int2cid=_rdag.int2cid, parent_adj=_padj,
                                      sidecar_uri=args.prev_sidecar_uri)
                                 if args.prev_history and _rdag is not None else None))
                _test_unmasked.unpersist()
            train_scored.unpersist(); test_scored.unpersist()

            # Echo the other arms' stored summary (only gated_pc can be re-scored).
            # `results` is legitimately absent-or-null on a FIT-ONLY manifest:
            # `gated_pc_cloud` writes the npz + manifest as soon as the fit lands
            # (`partial="fit-only"`) so a readout death cannot cost the fit, which
            # is precisely the run this tool exists to rescue — so say what is
            # missing rather than echoing nothing, and never index into it.
            if manifest.get("partial"):
                _cprint(f"[readout]   manifest marked partial="
                      f"{manifest['partial']!r}: the fit landed but its own "
                      "readout did not, so there are no stored arm results to "
                      "echo (the numbers above are this re-readout's).",
                      flush=True)
            for name, res in (manifest.get("results") or {}).items():
                if name in ("gated_pc", "gated_pc_head"):
                    continue
                rk = res.get("ranking", res)   # tolerate old flat-macro manifests
                auc = rk.get("auc"); ap = rk.get("ap")
                _cprint(f"[readout]   (manifest) {name}: AUC="
                      f"{'n/a' if auc is None else f'{auc:.4f}'} "
                      f"AP={'n/a' if ap is None else f'{ap:.4f}'}", flush=True)
    return 0


if __name__ == "__main__":
    # A client-mode spark-submit driver does not end when main() returns: the
    # SparkSubmit JVM that owns this process waits on a non-daemon Dataproc
    # metrics-publisher thread stuck in a socket write (jstack, 2026-09-12),
    # and a CHAINED sweep waits on the JVM. `hard_exit` flushes the durable
    # log, takes the JVM down through the gateway, then os._exit's — see its
    # docstring in _driver_common for the full diagnosis.
    try:
        _rc = main()
    except SystemExit as _e:                      # argparse errors etc.
        _rc = _e.code if isinstance(_e.code, int) else 1
    except BaseException:                         # a crash must not wedge the sweep either
        import traceback
        traceback.print_exc()
        _rc = 1
    hard_exit(_rc)
