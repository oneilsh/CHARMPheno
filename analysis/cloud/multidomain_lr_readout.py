"""Post-hoc likelihood-ratio placement readout for a MULTI-DOMAIN gated run
(no re-fit). Loads a run dir's dict-lambda + manifest + the persisted held-out
test set (test_bow_<m>.npz + test_affinity.npy + test_meta.json, written
DRIVER-LOCAL by multidomain_cloud.py), and emits a per-rare-disease x
domain-subset LR-AUC table plus the theta-mass baseline. Self-contained and
Spark-FREE: no BigQuery, no bundle cache, no Spark session (choice C).

The multi-domain LR score is the per-domain SUM of the single-domain
lr_placement_scores; a domain subset is the per-domain decomposition. Per-disease
detection is max-over-subtree(anchor) vs frontier-hits-subtree.

Every function here is pure and unit-tested (build_parser, parse_alpha_grid,
children_map, subtree_nodes, per_disease_auc_row, load_lambda_dict,
load_test_set); main() wires them and is cluster-run (make multidomain-lr-readout
ID=N).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-domain per-domain LR placement-lift readout.")
    p.add_argument("--run-dir", required=True,
                   help="Run directory containing manifest.json + params/ + "
                        "test_bow_<m>.npz + test_affinity.npy + test_meta.json.")
    p.add_argument("--alpha-grid", default="0,1,10,100,inf",
                   help="Comma list of LR-shrinkage alphas (inf = the lift limit).")
    return p


def parse_alpha_grid(s):
    """['0','1','inf'] -> [0.0, 1.0, inf]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float("inf") if tok.lower() in ("inf", "infinity") else float(tok))
    return out


def children_map(parent_int):
    """{node: set(children)} inverted from {node: [parents]} (list-valued,
    multi-parent DAG per ConditionDag.to_engine)."""
    cmap = {}
    for child, parents in parent_int.items():
        for parent in parents:
            cmap.setdefault(int(parent), set()).add(int(child))
    return cmap


def subtree_nodes(parent_int, root):
    """`root` and all its DESCENDANTS (the descendant subtree, NOT the ancestral
    closure). Includes `root` itself."""
    cmap = children_map(parent_int)
    seen, stack = set(), [int(root)]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(cmap.get(u, ()))
    return seen


# --- driver-local test-set artifact: ONE source of truth for the sidecar names,
# imported by BOTH the writer (multidomain_cloud.py's persistence) via
# save_test_set and the reader (main) via load_test_set, so the two halves of the
# cross-process contract can never drift apart on a rename. ---
def _bow_path(run_dir, m):
    return Path(run_dir) / f"test_bow_{m}.npz"


def _aff_path(run_dir):
    return Path(run_dir) / "test_affinity.npy"


def _meta_path(run_dir):
    return Path(run_dir) / "test_meta.json"


def save_test_set(out_dir, bows, aff, frontiers, aff_frontiers):
    """Write the DRIVER-LOCAL held-out test set (the writer half of the contract
    load_test_set reads). `bows` = {m: scipy CSR [n x V_m]}; `aff` = dense
    [n x n_nodes] theta-mass affinity; `frontiers` pairs with `bows` (their shared
    collect), `aff_frontiers` pairs with `aff` (its own collect). Called by
    multidomain_cloud.py after the fit; kept here so writer and reader share these
    exact filenames/JSON keys."""
    for m, csr in bows.items():
        sp.save_npz(str(_bow_path(out_dir, m)), csr)
    np.save(str(_aff_path(out_dir)), aff)
    _meta_path(out_dir).write_text(json.dumps({
        "n_docs": len(frontiers),
        "frontiers": [[int(x) for x in fr] for fr in frontiers],
        "aff_frontiers": [[int(x) for x in fr] for fr in aff_frontiers],
    }))


def load_test_set(run_dir, n_dom):
    """Load the DRIVER-LOCAL held-out test set written by save_test_set:
    per-domain scipy CSR (test_bow_<m>.npz), the dense theta-mass affinity
    (test_affinity.npy), and the frontiers + count (test_meta.json). No Spark.

    Returns (bows {m: csr [n x V_m]}, frontiers list[list[int]], aff [n x n_nodes],
    aff_frontiers list[list[int]], n_docs). `frontiers` pairs with `bows` (same
    fit-time collect); `aff_frontiers` pairs with `aff` (its own collect) -- keep
    them separate so the theta baseline never scores against another collect's
    labels."""
    if not _meta_path(run_dir).exists():
        raise SystemExit(
            f"[lr] no test_meta.json under {run_dir} -- this run was fit before "
            "LR-readout persistence, or its test split was empty. Re-fit on the "
            "current code to produce a readable artifact.")
    meta = json.loads(_meta_path(run_dir).read_text())
    bows = {m: sp.load_npz(str(_bow_path(run_dir, m))) for m in range(n_dom)}
    aff = np.load(str(_aff_path(run_dir)))
    frontiers = [[int(x) for x in fr] for fr in meta["frontiers"]]
    aff_frontiers = [[int(x) for x in fr] for fr in meta["aff_frontiers"]]
    return bows, frontiers, aff, aff_frontiers, int(meta["n_docs"])


def per_disease_auc_row(scores, frontiers, anchor, lay, parent_int):
    """(auc, n_pos) for detecting disease `anchor` from a [n_docs x n_nodes] score
    matrix (columns in lay.nodes order). Positive = the doc's frontier intersects
    subtree(anchor) (anchor + descendants, scoreable); per-disease score = the max
    over that subtree's columns. One-class positive/negative -> auc nan."""
    from spark_vi.models.topic.dag_placement import _auc
    sub = subtree_nodes(parent_int, anchor) & set(lay.nodes)
    if not sub:
        return float("nan"), 0
    cols = [lay.nodes.index(u) for u in sub]
    node_score = scores[:, cols].max(axis=1)
    y = np.array([1 if (set(fr) & sub) else 0 for fr in frontiers], dtype=int)
    return _auc(node_score, y), int(y.sum())


def load_lambda_dict(run_dir):
    """Load the per-domain lambda dict from save_result's sidecars
    (params/lambda_<m>.npy), keyed by the integer domain suffix -- WITHOUT
    spark_vi.io.export.load_result.

    Why not load_result: the multidomain fit driver OVERWRITES save_result's
    manifest.json with its own (rich, domain-aware) manifest, which does not
    carry save_result's `param_names`/`dict_param_keys`, so load_result raises
    KeyError('param_names') on these run dirs. The lambda .npy sidecars are still
    present, so we read them directly. (The driver-clobbers-save_result-manifest
    is a separate hygiene bug; loading the sidecars is robust to it either way.)
    """
    import re
    params = Path(run_dir) / "params"
    lam = {}
    for p in sorted(params.glob("lambda_*.npy")):
        mo = re.match(r"lambda_(\d+)\.npy$", p.name)
        if mo:
            lam[int(mo.group(1))] = np.load(p)
    if not lam:
        raise SystemExit(
            f"[lr] no params/lambda_*.npy under {run_dir} -- not a multidomain "
            "fit artifact, or its lambda sidecars are missing.")
    return lam


def main(argv=None) -> int:
    from charmpheno.omop.cohorts import disease_anchors
    from spark_vi.models.topic.dag_placement import (
        DagLayout, lr_placement_scores_multidomain)

    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    lam_dict = load_lambda_dict(run_dir)                      # {m: [K x V_m]}
    n_dom = len(lam_dict)
    domain_names = manifest.get("domains", [f"m{i}" for i in range(n_dom)])

    cm = manifest["corpus_manifest"]
    parent_int = {int(k): [int(x) for x in v] for k, v in cm["parent_int"].items()}
    lay = DagLayout(parent_int, n_bg=manifest["n_bg"], tpn=manifest["tpn"])
    int2cid = {int(k): int(v) for k, v in cm["int2cid"].items()}
    cid2int = {c: i for i, c in int2cid.items()}
    name_by_id = {int(k): v for k, v in cm["name_by_id"].items()}
    # int2cid is engine-id -> concept-id; name_by_id is concept-id -> name. The
    # anchor loop below prints ENGINE ids, so it needs the composed engine-id ->
    # name map (mirrors multidomain_cloud.py's name_by_engine construction).
    name_by_engine = {i: name_by_id.get(c, str(c)) for i, c in int2cid.items()}
    alpha_grid = parse_alpha_grid(args.alpha_grid)

    # Driver-local test set (scipy CSR per domain + dense affinity + frontiers
    # json), written by multidomain_cloud.py's persistence. No Spark, no BQ.
    bows, frontiers, aff, aff_frontiers, n_docs = load_test_set(run_dir, n_dom)

    # rare6 anchor engine-ids (skip anchors pruned out of the DAG).
    anchors = []
    for cid in disease_anchors(manifest["disease"]):
        u = cid2int.get(int(cid))
        if u is not None and u in set(lay.nodes):
            anchors.append(u)

    # domain subsets: all, each-alone, leave-one-out (labeled by name).
    subsets = {"all": list(range(n_dom))}
    for i, nm in enumerate(domain_names):
        subsets[f"only:{nm}"] = [i]
    if n_dom > 1:
        for i, nm in enumerate(domain_names):
            subsets[f"drop:{nm}"] = [j for j in range(n_dom) if j != i]

    # Overall detection sweep (all domains, max-over-nodes vs is_fg) over the grid,
    # for continuity with the single-domain readout's output shape.
    from spark_vi.models.topic.dag_placement import lr_auc_sweep_multidomain
    is_fg = np.array([1 if (set(fr) & set(lay.nodes)) else 0 for fr in frontiers],
                     dtype=int)
    sweep = lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, alpha_grid=alpha_grid)
    print("[lr] === overall detection LR-AUC(alpha), all domains, max-over-nodes ===",
          flush=True)
    for a in alpha_grid:
        print(f"[lr]   alpha={a}: {sweep[a]:.3f}", flush=True)

    # Per-disease x domain-subset table at the alpha=inf lift limit (headline).
    # Score ONCE per subset (scores do not depend on the anchor), then loop anchors.
    a_head = alpha_grid[-1]
    subset_scores = {name: lr_placement_scores_multidomain(
                         bows, lam_dict, lay, alpha=a_head, domains=doms)
                     for name, doms in subsets.items()}
    print(f"[lr] === per-disease x domain-subset LR-AUC (alpha={a_head}) ===",
          flush=True)
    header = "disease".ljust(26) + "n+".rjust(5) + "  theta"
    for name in subsets:
        header += "  " + name[:12].rjust(12)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        theta_auc, n_pos = per_disease_auc_row(aff, aff_frontiers, u, lay, parent_int)
        line = dname.ljust(26) + str(n_pos).rjust(5) + f"  {theta_auc:5.3f}"
        for name in subsets:
            auc, _ = per_disease_auc_row(subset_scores[name], frontiers, u, lay,
                                         parent_int)
            line += "  " + f"{auc:12.3f}"
        print("[lr] " + line, flush=True)

    print(f"[lr] scored {n_docs} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
