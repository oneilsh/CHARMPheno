"""Per-coordinate-class read-out for the DAG-offset engine (step 2). Assembles calibrated
increment posteriors for identified coordinates and machine-flagged non-answers (GAUGE /
UNRESOLVED) for design-wall directions. Fixed coordinate set: every non-root node always
appears, so a coordinate flips unresolved->number in place when data accrues.

Design-wall coordinates emit width/status/cause but NEVER a point estimate (Fable contract).

``dag_offset_readout`` is the step-2 ORCHESTRATOR: it wires a cheap VI warm-start, the
identifiability compiler, the co-sampled quotient Gibbs engine, and this module's assembly
into one call, so a caller never hand-threads the four phases together.
"""
import numpy as np

from spark_vi.models.topic.pg_stm import stick_layout
from spark_vi.models.topic.pg_stm_dag import PGSTMDag, _softgate_estep_doc, offset_penalty
from spark_vi.models.topic.pg_stm_dag_gibbs import PGSTMDagGibbs
from spark_vi.models.topic.dag_identify import (
    expected_closure_gram, foreground_grams, identifiability_spectrum,
    detect_confounds, build_quotient, classify_null_directions,
)


def node_prevalence(dag, doc_nodes, doc_candidates, memberships):
    """Per-node prevalence from partial-label resolution (v1: no latent-anchor recall
    correction; see below).

    Convention (labeled vs. inferred):
      - `labeled_mass[u]` = the COUNT of hard-attested documents (docs with exactly one
        candidate, i.e. `len(doc_candidates[d]) == 1`) whose closure contains node u.
        Soft/partial-label docs never contribute to labeled_mass.
      - `inferred_total[u]` = `labeled_mass[u]` + the summed partial-label membership
        mass (from `memberships`) landing on candidates whose closure contains u. This
        is the soft-resolved estimate of how many documents actually attest u.
      - `recall_ratio[u]` = `labeled_mass[u] / inferred_total[u]` (defined as 1.0 when
        there is no soft mass, i.e. `inferred_total[u] == 0`). Because labeled_mass is
        always a subset of inferred_total, recall_ratio <= 1.0: it measures how much
        the hard labels UNDERCOUNT node u relative to the soft-inferred total, not the
        other way around.

    `doc_nodes[d]` (a frozenset) is the hard/true node for document d; it is consulted
    ONLY on the hard-doc branch (`len(doc_candidates[d]) == 1`), where it stands in for
    the single candidate. `doc_candidates[d]` is `[(weight, nodes), ...]` and
    `memberships[d]` is an aligned np.ndarray of soft-gate E-step weights (a hard doc
    has `memberships[d] == [1.0]`).

    v1 scope: this counts only partial-label (soft-gate) resolution of ambiguous
    candidates. It does NOT correct for documents that never get any candidate at all
    (missing-not-at-random latent-anchor recall) -- that correction is DEFERRED.
    """
    nodes = range(1, dag.n_nodes)
    labeled = {u: 0.0 for u in nodes}
    inferred = {u: 0.0 for u in nodes}
    for dn, cands, wts in zip(doc_nodes, doc_candidates, memberships):
        hard = len(cands) == 1
        if hard:
            for u in nodes:
                if u in dag.closure(dn):
                    labeled[u] += 1.0
                    inferred[u] += 1.0
        else:
            for wgt, (_p, cnodes) in zip(wts, cands):
                for u in dag.closure(cnodes):
                    if u in inferred:
                        inferred[u] += float(wgt)
    return {u: {"labeled_mass": labeled[u], "inferred_total": inferred[u],
                "recall_ratio": (labeled[u] / inferred[u]) if inferred[u] > 0 else 1.0}
            for u in nodes}


def assemble_readout(dag, increment_draws, node_map, classification,
                     *, ci_level=0.90, fragility_margin=None, spectrum=None,
                     node_sticks=None,
                     doc_nodes=None, doc_candidates=None, memberships=None):
    """Assemble the per-coordinate-class ReadOut from the quotient increment draws.

    ``node_sticks`` (optional) maps an original node id -> the array of STICK indices on
    which that node's offset is IDENTIFIED (its own group's foreground sticks -- the sticks
    its documents actually activate; insight 0054). When supplied, an identified coordinate
    reports increment_mean / ci on ONLY those sticks (plus a `sticks` field naming them), so
    the object never carries a number for a stick the node's documents never touch (those
    sit pinned at the prior and are not the node's to claim -- insight 0057, the read-out
    half of insight 0053's block-granularity rule). When None, the full (K-1) vector is
    reported (backward-compatible)."""
    gauge = set(classification["gauge_nodes"])
    unresolved = dict(classification["unresolved"])
    lo_q, hi_q = (1 - ci_level) / 2, 1 - (1 - ci_level) / 2
    coords = {}
    for u in range(1, dag.n_nodes):
        parent = int(dag.parents[u][0]) if dag.parents[u] else 0
        if u in unresolved:
            rec = unresolved[u]
            q = int(node_map[u])
            width = float(np.sqrt(np.var(increment_draws[:, q - 1, :], axis=0).sum())) if q >= 1 else 0.0
            coords[u] = {"node": u, "parent": parent, "status": "unresolved",
                         "width": width, "reason": "design_null(no_own_documents)",
                         "recipe": {"attest_node": rec["attest_node"],
                                    "docs_needed": rec["docs_needed"]}}
        elif u in gauge:
            coords[u] = {"node": u, "parent": parent, "status": "gauge",
                         "reason": "design_null(partition_identity)",
                         "convention": "level fixed to the intercept gauge; increments only"}
        else:
            q = int(node_map[u])
            col = increment_draws[:, q - 1, :]            # (n_kept, Ksm1)
            status = "identified"
            entry = {"node": u, "parent": parent, "status": status}
            if node_sticks is not None and u in node_sticks:
                idx = np.asarray(node_sticks[u], dtype=int)
                sub = col[:, idx]
                entry["sticks"] = idx.tolist()
                entry["increment_mean"] = sub.mean(axis=0)
                entry["ci_low"] = np.quantile(sub, lo_q, axis=0)
                entry["ci_high"] = np.quantile(sub, hi_q, axis=0)
            else:
                entry["increment_mean"] = col.mean(axis=0)
                entry["ci_low"] = np.quantile(col, lo_q, axis=0)
                entry["ci_high"] = np.quantile(col, hi_q, axis=0)
            coords[u] = entry
    readout = {"calibration": "absolute", "coordinates": coords,
               "meta": {"n_draws": int(increment_draws.shape[0]), "ci_level": ci_level}}
    if doc_nodes is not None and doc_candidates is not None and memberships is not None:
        readout["prevalence"] = node_prevalence(dag, doc_nodes, doc_candidates, memberships)
    return readout


def _warm_start_nodes(dag, doc_candidates):
    """OBSERVABLE warm-start node placement per document, built only from the candidate
    set the model can actually see -- NEVER from doc_nodes' hidden truth (that would leak
    the answer into the very fit the compiler is meant to check for identifiability). A
    hard doc (single candidate) warm-starts at its own closure. A partial (multi-candidate)
    doc warm-starts one level up: the parent (in the original DAG) of its lowest-id
    candidate node, since the shared parent is a reasonable placement to seed beta/Sigma
    with (precision on the individual leaf comes from Phase B2's soft-gate resampling, not
    from this warm-start). A candidate with no parent (its node is the root) falls back to
    the root itself."""
    warm = []
    for cands in doc_candidates:
        if len(cands) == 1:
            warm.append(cands[0][1])
        else:
            u = min(min(nodes) for _p, nodes in cands)
            ps = dag.parents[u]
            warm.append(frozenset({ps[0] if ps else 0}))
    return warm


def dag_offset_readout(docs, doc_nodes, doc_candidates, doc_groups, partition, dag, *,
                       P=1, tol=1.0, lam_base=0.25, n_iter=200, burn=100, seed=0,
                       sigma_fixed=None):
    """End-to-end DAG-offset read-out: warm-start -> compile -> co-sampled Gibbs -> assemble.

    Phase A (VI warm-start): a cheap `PGSTMDag` VI fit on OBSERVABLE warm-start node
    placements (see `_warm_start_nodes` -- never doc_nodes' hidden truth) gives a
    starting beta/Sigma/coefficient block. For every partial (multi-candidate) document,
    `_softgate_estep_doc` scores its candidates under that warm start to get a soft
    membership posterior; a hard document's membership is the trivial [1.0].

    Phase B1 (compile on the expected moment): the membership-weighted candidates form
    the expected closure Gram (`expected_closure_gram`); the warm-start node placements
    form the per-group foreground Grams (`foreground_grams`). The identifiability
    compiler (`detect_confounds` -> `build_quotient` + `classify_null_directions`) reads
    the corpus's design wall off these two moments and produces the identified quotient
    DAG plus the GAUGE / UNRESOLVED classification of the null directions.

    Phase B2 (co-sampled quotient Gibbs): every document's candidate closures are
    remapped through the quotient's node_map. A quotient node created by merging >=2
    original nodes gets the SUMMED depth-scaled ridge penalty of the chain it replaces
    (insight-0055 fix -- a merged node must not be penalized as if it were a single
    shallow node). `PGSTMDagGibbs` then runs the exact co-sampled chain on the quotient
    DAG, warm-started from Phase A's beta.

    Phase C (assembly): `assemble_readout` turns the quotient increment draws into the
    fixed-keyset, per-original-node ReadOut (identified / gauge / unresolved), and
    per-node prevalence is attached from the (hard + soft) membership resolution.

    Domain-agnostic: everything here is integer node/token ids. Vocabulary size V is
    inferred from the corpus itself (max token id + 1) so callers never need to pass it
    separately.
    """
    V = int(max((int(np.max(doc.indices)) for doc in docs if len(doc.indices) > 0),
               default=-1)) + 1
    K = partition.K
    layout = stick_layout(partition)

    # ---- Phase A: VI warm-start (observable candidates only) + partial-doc membership ----
    warm_nodes = _warm_start_nodes(dag, doc_candidates)
    vi = PGSTMDag(K=K, V=V, partition=partition, dag=dag, P=P,
                 n_iter=min(n_iter, 60), lam_base=lam_base, seed=seed).fit(docs, warm_nodes)
    beta_warm = vi["beta"]
    Sigma_warm = vi["Sigma"]
    Cf_warm = np.vstack([vi["Gamma"], vi["B"][1:]])
    log_beta_warm = np.log(beta_warm)

    memberships = []
    for doc, cands in zip(docs, doc_candidates):
        if len(cands) == 1:
            memberships.append(np.array([1.0]))
        else:
            g = next(iter(doc.groups))
            triples = [(p, nodes, g) for p, nodes in cands]
            weights, _z_bar, _esteps = _softgate_estep_doc(
                doc, triples, layout["groups"], log_beta_warm, Cf_warm, Sigma_warm, dag,
                K=K, B=layout["B"], inner_rounds=8, inner_tol=1e-3)
            memberships.append(weights)

    # ---- Phase B1: compile on the expected moment ----
    doc_candidates_expected = []
    for cands, wts in zip(doc_candidates, memberships):
        if len(cands) == 1:
            doc_candidates_expected.append([(1.0, cands[0][1])])
        else:
            doc_candidates_expected.append(
                [(float(w), nodes) for w, (_p, nodes) in zip(wts, cands)])

    G = expected_closure_gram(dag, doc_candidates_expected)
    fg = foreground_grams(dag, warm_nodes, doc_groups, partition)
    spectrum = identifiability_spectrum(G)
    det = detect_confounds(dag, G, spectrum, tol=tol)
    q = build_quotient(dag, det)
    cls = classify_null_directions(dag, G, fg, det, tol=tol)

    # ---- Phase B2: co-sampled quotient Gibbs, with the merged-node penalty fix ----
    node_map = q["node_map"]
    qcand = [[(p, frozenset(int(node_map[u]) for u in nodes)) for p, nodes in cands]
             for cands in doc_candidates]

    penalty = offset_penalty(P, q["quotient_dag"], gamma_ridge=1e-6,
                             lam_base=lam_base, gamma_depth=1.0)
    orig_penalty = offset_penalty(P, dag, gamma_ridge=1e-6,
                                  lam_base=lam_base, gamma_depth=1.0)
    merged_members = {}
    for u in range(1, dag.n_nodes):
        merged_members.setdefault(int(node_map[u]), []).append(u)
    for qnode, members in merged_members.items():
        if len(members) >= 2:
            penalty[P + qnode - 1] = sum(orig_penalty[P + u - 1] for u in members)

    eng = PGSTMDagGibbs(K=K, V=V, partition=partition, dag=q["quotient_dag"], P=P,
                        n_iter=n_iter, burn=burn, lam_base=lam_base, seed=seed)
    out = eng.run(docs, qcand, beta_init=beta_warm, penalty_override=penalty,
                  sigma_fixed=sigma_fixed)

    # ---- identified sub-block per node: its group's foreground sticks (insight 0054/0057) ----
    # A node's offset is identified only on the sticks its own documents activate. Map each
    # node -> its anchor (the child-of-root on its path) -> that anchor's group (read off the
    # observable doc groups) -> that group's foreground sticks. The read-out then claims a
    # number only on those sticks, not on the other groups' prior-pinned sticks.
    def _anchor_of(u):
        for c in [u] + sorted(dag.ancestors(u)):
            if dag.parents[c] and 0 in dag.parents[c]:
                return c
        return None
    anchor_group = {}
    for g, wn in zip(doc_groups, warm_nodes):
        a = _anchor_of(min(wn))
        if a is not None:
            anchor_group[a] = g
    node_sticks = {}
    for u in range(1, dag.n_nodes):
        g = anchor_group.get(_anchor_of(u))
        if g is not None:
            node_sticks[u] = np.asarray(layout["groups"][g]["fg_sticks"], dtype=int)

    # ---- Phase C: assembly ----
    ro = assemble_readout(dag, out["increment_draws"], node_map, cls, ci_level=0.90,
                          node_sticks=node_sticks)
    ro["prevalence"] = node_prevalence(dag, doc_nodes, doc_candidates, memberships)
    # raw material for calibration diagnostics (joint/Mahalanobis coverage on the emitted
    # increment draws, marginal-vs-joint coverage-frame check; insight 0057). Not part of the
    # shipped read-out contract -- a private hook consumers ignore.
    ro["_debug"] = {"increment_draws": out["increment_draws"], "node_map": node_map,
                    "node_sticks": node_sticks}
    return ro
