"""Post-hoc likelihood-ratio placement readout for a MULTI-DOMAIN gated run
(no re-fit). Loads a run dir's dict-lambda + manifest + the persisted held-out
test set (test_docs/ + test_affinities/, written by multidomain_cloud.py), and
emits a per-rare-disease x domain-subset LR-AUC table plus the theta-mass
baseline. Self-contained: no BigQuery, no bundle cache (choice C).

The multi-domain LR score is the per-domain SUM of the single-domain
lr_placement_scores; a domain subset is the per-domain decomposition. Per-disease
detection is max-over-subtree(anchor) vs frontier-hits-subtree.

Only build_parser + the pure helpers (children_map, subtree_nodes,
build_domain_bows, per_disease_auc_row) are unit-tested; main() (Spark load +
parquet reads) is cluster-covered (make multidomain-lr-readout ID=N).
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
                        "test_docs/ + test_affinities/.")
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


def build_domain_bows(rows, feature_cols, vocab_sizes):
    """(bows {m: csr [n x V_m]}, frontiers list[list[int]], person_ids list) from
    collected test_docs rows. rows[i][feature_cols[m]] is a SparseVector-like
    (.indices/.values/.size); vocab_sizes[m] pins V_m."""
    n = len(rows)
    bows = {}
    for m, col in enumerate(feature_cols):
        V = int(vocab_sizes[m])
        indptr = np.zeros(n + 1, dtype=np.int64)
        idx_chunks, data_chunks = [], []
        for i, r in enumerate(rows):
            sv = r[col]
            idx = np.asarray(sv.indices, dtype=np.int64)
            val = np.asarray(sv.values, dtype=np.float64)
            idx_chunks.append(idx)
            data_chunks.append(val)
            indptr[i + 1] = indptr[i] + len(idx)
        indices = np.concatenate(idx_chunks) if idx_chunks else np.array([], np.int64)
        data = np.concatenate(data_chunks) if data_chunks else np.array([], np.float64)
        bows[m] = sp.csr_matrix((data, indices, indptr), shape=(n, V))
    frontiers = [[int(x) for x in r["frontier"]] for r in rows]
    person_ids = [r["person_id"] for r in rows]
    return bows, frontiers, person_ids


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


# ---- affinity (theta-mass) baseline ---------------------------------------
def affinity_matrix(aff_rows, n_nodes):
    """[n_docs x n_nodes] dense node-affinity matrix from collected
    test_affinities rows (r['nodeAffinity'] a DenseVector-like)."""
    out = np.zeros((len(aff_rows), n_nodes), dtype=float)
    for i, r in enumerate(aff_rows):
        out[i, :] = np.asarray(r["nodeAffinity"].toArray(), dtype=float)
    return out


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
    from _driver_common import make_spark_session
    from charmpheno.omop.cohorts import disease_anchors
    from spark_vi.models.topic.dag_placement import (
        DagLayout, lr_placement_scores_multidomain)

    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    lam_dict = load_lambda_dict(run_dir)                      # {m: [K x V_m]}
    n_dom = len(lam_dict)
    feature_cols = [f"features_{i}" for i in range(n_dom)]
    vocab_sizes = [lam_dict[m].shape[1] for m in range(n_dom)]
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

    if not (run_dir / "test_docs").exists():
        raise SystemExit(
            f"[lr] no test_docs/ under {run_dir} -- this run was fit before "
            "LR-readout persistence, or its test split was empty. Re-fit to "
            "produce a readable artifact.")

    try:
        with make_spark_session(app_name="multidomain-lr-readout") as spark:
            rows = spark.read.parquet(str(run_dir / "test_docs")).select(
                "person_id", *feature_cols, "frontier").collect()
            # test_affinities/ ALSO persists frontier alongside nodeAffinity
            # (multidomain_cloud.py's write). Read it from HERE, not from the
            # test_docs collect above: two separate spark.read.parquet(...)
            # .collect() calls are not guaranteed to return rows in the same
            # order, so pairing aff[i] with frontiers[i] (from the OTHER
            # collect) would silently score theta against the wrong patient's
            # label. Within a single collect, row order is fixed, so aff and
            # aff_frontiers stay aligned.
            aff_rows = spark.read.parquet(str(run_dir / "test_affinities")).select(
                "person_id", "nodeAffinity", "frontier").collect()
    except Exception as e:
        # run_dir may be a gs:// path, so the local .exists() guard above can't
        # see a missing/incomplete GCS run dir -- catch the Spark-side failure
        # (AnalysisException et al.) too and re-raise with the same clear message.
        raise SystemExit(
            f"[lr] failed to read test_docs/ or test_affinities/ under {run_dir} "
            f"-- this run may predate LR-readout persistence, or its test split "
            f"was empty. Re-fit to produce a readable artifact. ({e})") from e

    bows, frontiers, pids = build_domain_bows(rows, feature_cols, vocab_sizes)
    aff = affinity_matrix(aff_rows, len(lay.nodes))
    aff_frontiers = [[int(x) for x in r["frontier"]] for r in aff_rows]

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

    print(f"[lr] scored {len(pids)} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
