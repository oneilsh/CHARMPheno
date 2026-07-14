"""Per-coordinate-class read-out for the DAG-offset engine (step 2). Assembles calibrated
increment posteriors for identified coordinates and machine-flagged non-answers (GAUGE /
UNRESOLVED) for design-wall directions. Fixed coordinate set: every non-root node always
appears, so a coordinate flips unresolved->number in place when data accrues.

Design-wall coordinates emit width/status/cause but NEVER a point estimate (Fable contract).
"""
import numpy as np


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
                     doc_nodes=None, doc_candidates=None, memberships=None):
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
            mean = col.mean(axis=0)
            status = "identified"
            entry = {"node": u, "parent": parent, "status": status,
                     "increment_mean": mean,
                     "ci_low": np.quantile(col, lo_q, axis=0),
                     "ci_high": np.quantile(col, hi_q, axis=0)}
            coords[u] = entry
    readout = {"calibration": "absolute", "coordinates": coords,
               "meta": {"n_draws": int(increment_draws.shape[0]), "ci_level": ci_level}}
    if doc_nodes is not None and doc_candidates is not None and memberships is not None:
        readout["prevalence"] = node_prevalence(dag, doc_nodes, doc_candidates, memberships)
    return readout
