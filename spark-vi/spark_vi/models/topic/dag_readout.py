"""Per-coordinate-class read-out for the DAG-offset engine (step 2). Assembles calibrated
increment posteriors for identified coordinates and machine-flagged non-answers (GAUGE /
UNRESOLVED) for design-wall directions. Fixed coordinate set: every non-root node always
appears, so a coordinate flips unresolved->number in place when data accrues.

Design-wall coordinates emit width/status/cause but NEVER a point estimate (Fable contract).
"""
import numpy as np


def assemble_readout(dag, increment_draws, node_map, classification,
                     *, ci_level=0.90, fragility_margin=None, spectrum=None):
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
    return {"calibration": "absolute", "coordinates": coords,
            "meta": {"n_draws": int(increment_draws.shape[0]), "ci_level": ci_level}}
