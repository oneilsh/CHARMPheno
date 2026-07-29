"""Post-hoc likelihood-ratio placement readout for a MULTI-DOMAIN gated run
(no re-fit). Loads a run dir's dict-lambda + manifest + the persisted held-out
test set (test_bow_<m>.npz + test_affinity.npy + test_meta.json, written
DRIVER-LOCAL by multidomain_cloud.py), and emits a per-rare-disease x
domain-subset LR-AUC table plus the theta-mass baseline. Self-contained and
Spark-FREE: no BigQuery, no bundle cache, no Spark session (choice C).

The multi-domain LR score is the per-domain SUM of the single-domain
lr_placement_scores; a domain subset is the per-domain decomposition. Per-disease
detection is max-over-subtree(anchor) vs frontier-hits-subtree. --normalize
applies a per-domain transform before that sum (none/std/length/length+std) so a
high-token-volume domain cannot dominate the ranking by magnitude alone; the
final two tables print PR-AUC under every rule for subset `all` and for subset
`drop:<last domain>` so both readings the design spec's acceptance criterion
needs (within-rule drag, and across-rule cost to the kept domains) are visible
(insight 0072; corrected acceptance criterion in
docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md).

Every function here is pure and unit-tested (build_parser, parse_alpha_grid,
children_map, subtree_nodes, per_disease_auc_row, load_lambda_dict,
load_test_set, normalize_arg, pr_by_normalization); main() wires them and is
cluster-run (make multidomain-lr-readout ID=N).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp


# CLI spellings of the per-domain normalization rules. 'none' maps to the
# library's None; see spark_vi.models.topic.dag_placement.NORMALIZE_MODES and
# docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md
NORMALIZE_RULES = ("none", "std", "length", "length+std")


def normalize_arg(rule):
    """CLI rule name -> the library `normalize` value ('none' -> None)."""
    if rule not in NORMALIZE_RULES:
        raise ValueError(f"unknown normalization rule {rule!r}")
    return None if rule == "none" else rule


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-domain per-domain LR placement-lift readout.")
    p.add_argument("--run-dir", required=True,
                   help="Run directory containing manifest.json + params/ + "
                        "test_bow_<m>.npz + test_affinity.npy + test_meta.json.")
    p.add_argument("--alpha-grid", default="0,1,10,100,inf",
                   help="Comma list of LR-shrinkage alphas (inf = the lift limit).")
    p.add_argument("--normalize", default="none", choices=list(NORMALIZE_RULES),
                   help="Per-domain score normalization applied before the "
                        "domain sum: none (raw), std (equalize per-domain scale), "
                        "length (per-doc mean log-LR per token), or length+std. "
                        "Governs the main tables; the comparison table always "
                        "shows every rule.")
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


def per_disease_pr(scores, frontiers, anchor, lay, parent_int, recalls=(0.5, 0.8)):
    """(pr_auc, {recall: precision}, n_pos) for detecting disease `anchor`.

    Same detection problem as `per_disease_auc_row` -- positive = the doc's
    frontier intersects subtree(anchor); per-disease score = max over that
    subtree's columns -- so PR and ROC numbers are directly comparable.

    pr_auc = `_average_precision` (step-wise AVERAGE PRECISION over distinct
    score thresholds, Davis & Goadrich 2006 / sklearn's average_precision_score
    convention): tied scores share a threshold, so AP is order-invariant and a
    constant (zero-information) scorer yields AP == prevalence regardless of
    which rows the positives sit in. At rare-disease base rates the
    random-classifier PR-AUC is the PREVALENCE (not 0.5), so the caller prints
    n_pos/n_docs beside it.

    prec_at[r] = precision at the smallest ACHIEVABLE threshold reaching
    recall >= r (nan if r is unreachable); see
    `spark_vi.models.topic.dag_placement._precision_at_recall`.
    One-class input -> nan, matching `_auc`.
    """
    from spark_vi.models.topic.dag_placement import (
        _average_precision, _precision_at_recall)
    sub = subtree_nodes(parent_int, anchor) & set(lay.nodes)
    if not sub:
        return float("nan"), {float(r): float("nan") for r in recalls}, 0
    cols = [lay.nodes.index(u) for u in sub]
    node_score = np.asarray(scores)[:, cols].max(axis=1)
    y = np.array([1 if (set(fr) & sub) else 0 for fr in frontiers], dtype=int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan"), {float(r): float("nan") for r in recalls}, n_pos

    ap = _average_precision(node_score, y)
    prec_at = _precision_at_recall(node_score, y, recalls)
    return ap, prec_at, n_pos


def pr_by_normalization(bows, lam_dict, lay, frontiers, anchors, parent_int, *,
                        alpha, domains=None, rules=NORMALIZE_RULES):
    """{rule: {anchor: pr_auc}} -- PR-AUC per anchor for ONE fixed domain subset,
    under each per-domain normalization rule.

    This is the A/B for insight 0072's finding that a high-volume low-signal
    domain costs most of the precision. The caller compares this called TWICE
    -- once with `domains=subsets["all"]`, once with `domains=subsets[ref_name]`
    (e.g. `drop:<domain>`) -- so that BOTH drag(rule) (all vs drop:X within a
    rule) and the rule's effect on the kept domains (drop:X across rules) are
    visible; a single rule's `all`-vs-`drop:X` difference under only rule
    'none' is NOT a valid acceptance criterion on its own (see the design
    spec's "Acceptance criterion" section for the refuted single-difference
    version and why). PR (not ROC) is the metric, because the damage is at the
    head of the ranking.
    """
    from spark_vi.models.topic.dag_placement import lr_domain_score_matrices
    out = {}
    for rule in rules:
        mats = lr_domain_score_matrices(bows, lam_dict, lay, alpha=alpha,
                                        domains=domains,
                                        normalize=normalize_arg(rule))
        scores = None
        for s in mats.values():
            scores = s if scores is None else scores + s
        out[rule] = {u: per_disease_pr(scores, frontiers, u, lay, parent_int)[0]
                     for u in anchors}
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
    from charmpheno.omop.cohorts import disease_anchors
    from spark_vi.models.topic.dag_placement import (
        DagLayout, lr_domain_score_matrices)

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
    norm = normalize_arg(args.normalize)

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
    sweep = lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg,
                                     alpha_grid=alpha_grid, normalize=norm)
    print(f"[lr] === overall detection LR-AUC(alpha), all domains, "
          f"max-over-nodes, normalize={args.normalize} ===", flush=True)
    for a in alpha_grid:
        print(f"[lr]   alpha={a}: {sweep[a]:.3f}", flush=True)

    # Per-disease x domain-subset table at the alpha=inf lift limit (headline).
    # Score ONCE per subset (scores do not depend on the anchor), then loop anchors.
    a_head = alpha_grid[-1]
    # Per-domain score matrices ONCE; every subset is the sum of its members.
    # Each domain's normalization is computed from that domain alone, so a
    # domain contributes the same in `all` as in `drop:x` and the decomposition
    # stays coherent across subsets.
    dom_mats = lr_domain_score_matrices(bows, lam_dict, lay, alpha=a_head,
                                        normalize=norm)
    subset_scores = {}
    for name, doms in subsets.items():
        total = None
        for i in doms:
            total = dom_mats[i] if total is None else total + dom_mats[i]
        subset_scores[name] = total
    print(f"[lr] === per-disease x domain-subset LR-AUC (alpha={a_head}, "
          f"normalize={args.normalize}) ===", flush=True)
    header = "disease".ljust(26) + "n+".rjust(5) + "  theta"
    for name in subsets:
        header += "  " + name[:12].rjust(12)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        # theta_auc alone uses the theta-mass collect (aff / aff_frontiers) --
        # a deliberate, previously-reviewed alignment guarantee that must not
        # change. Its OWN n_pos is discarded here: the printed "n+" column uses
        # the frontiers-based n_pos below instead, so n+ means one thing across
        # every column of this table AND matches the PR table / prev.
        theta_auc, _ = per_disease_auc_row(aff, aff_frontiers, u, lay, parent_int)
        n_pos, cells = None, ""
        for name in subsets:
            auc, n_pos_sub = per_disease_auc_row(subset_scores[name], frontiers, u,
                                                 lay, parent_int)
            if name == "all":
                n_pos = n_pos_sub          # same positive set for every subset
            cells += "  " + f"{auc:12.3f}"
        line = dname.ljust(26) + str(n_pos).rjust(5) + f"  {theta_auc:5.3f}" + cells
        print("[lr] " + line, flush=True)

    # --- PR-AUC table (same subsets as the AUC table). `prev` = n_pos/n_docs is
    # the random-classifier PR-AUC, the baseline that makes PR-AUC readable at
    # rare-disease base rates. ---
    print(f"[lr] === per-disease x domain-subset PR-AUC (avg precision, "
          f"alpha={a_head}, normalize={args.normalize}) ===", flush=True)
    header = "disease".ljust(26) + "n+".rjust(5) + "  prev".rjust(7)
    for name in subsets:
        header += "  " + name[:12].rjust(12)
    print("[lr] " + header, flush=True)
    # Captured for reuse by the domain-normalization comparison blocks below --
    # the positive set for a disease depends only on frontiers/anchor, never on
    # which domains or normalization rule produced the score matrix, so this one
    # n_pos serves every rule x subset combination those blocks print.
    n_pos_by_anchor = {}
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        n_pos = None
        line = dname.ljust(26)
        cells = ""
        for name in subsets:
            pr_auc, _, n_pos_sub = per_disease_pr(subset_scores[name], frontiers, u,
                                                  lay, parent_int)
            if name == "all":
                n_pos = n_pos_sub          # same positive set for every subset
            cells += "  " + f"{pr_auc:12.3f}"
        n_pos_by_anchor[u] = n_pos
        prev = (n_pos / n_docs) if n_docs else float("nan")
        line += str(n_pos).rjust(5) + f"{prev:7.4f}" + cells
        print("[lr] " + line, flush=True)

    # --- Precision@recall: the deployability read ("flag enough patients to catch
    # 80% of true cases -- what fraction of the flagged list is real?") for the
    # three headline subsets, incl. the cond vs cond+drug operational comparison.
    # drop:<last domain> is cond+drug when observation is the last domain; fall
    # back to whatever exists so this never KeyErrors on a 1- or 2-domain run
    # (and never IndexErrors when domain_names is empty). ---
    last_domain = domain_names[-1] if domain_names else None
    headline = [n for n in ("all", "only:condition",
                            f"drop:{last_domain}") if n in subsets]
    print("[lr] === precision @ recall (headline subsets) ===", flush=True)
    header = "disease".ljust(26) + "n+".rjust(5)
    for name in headline:
        header += "  " + f"{name[:10]}@50%".rjust(16) + f"{name[:10]}@80%".rjust(16)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        cells, n_pos_seen = "", 0
        for name in headline:
            _, prec_at, n_pos = per_disease_pr(subset_scores[name], frontiers, u,
                                               lay, parent_int, recalls=(0.5, 0.8))
            n_pos_seen = n_pos          # same positive set for every subset
            cells += "  " + f"{prec_at[0.5]:16.3f}" + f"{prec_at[0.8]:16.3f}"
        print("[lr] " + f"{dname:<26}{n_pos_seen:>5}" + cells, flush=True)

    # --- Domain-normalization comparison: PR-AUC under every rule, printed as
    # TWO STACKED BLOCKS of identical shape (subset `all`, then subset
    # `drop:<last_domain>`) rather than paired columns in one table, so no rule
    # name needs truncating (`length+std` is 10 chars).
    #
    # This is the corrected acceptance criterion (see the design spec's
    # "Acceptance criterion" section -- the original single cross-rule
    # difference conflated two independent quantities and could invert the
    # ranking of rules). Two readings, neither sufficient alone:
    #   reading 1 (within a rule, across these two blocks): drop:X minus all is
    #     drag(rule) -- what keeping domain X costs under that rule.
    #   reading 2 (across rules, within the drop:X block): shows what the rule
    #     does to the domains you KEEP, since every non-`none` rule also
    #     re-weights the retained domains against each other, not only against
    #     the dropped one -- so a rule can shrink drag(rule) while degrading
    #     the kept domains. ---
    ref_name = f"drop:{last_domain}" if f"drop:{last_domain}" in subsets else None
    all_by_rule = pr_by_normalization(bows, lam_dict, lay, frontiers, anchors,
                                      parent_int, alpha=a_head)
    ref_by_rule = {}
    if ref_name:
        ref_by_rule = pr_by_normalization(bows, lam_dict, lay, frontiers, anchors,
                                          parent_int, alpha=a_head,
                                          domains=subsets[ref_name])

    def _print_normalization_block(title, by_rule):
        print(f"[lr] === PR-AUC by domain-normalization rule ({title}, "
              f"alpha={a_head}) ===", flush=True)
        header = "disease".ljust(26) + "n+".rjust(5) + "  prev".rjust(7)
        for rule in NORMALIZE_RULES:
            header += "  " + rule.rjust(12)
        print("[lr] " + header, flush=True)
        for u in anchors:
            dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
            n_pos = n_pos_by_anchor.get(u)
            prev = (n_pos / n_docs) if n_docs else float("nan")
            line = dname.ljust(26) + str(n_pos).rjust(5) + f"{prev:7.4f}"
            for rule in NORMALIZE_RULES:
                line += "  " + f"{by_rule[rule][u]:12.3f}"
            print("[lr] " + line, flush=True)

    _print_normalization_block("subset=all", all_by_rule)
    if ref_name:
        _print_normalization_block(f"subset={ref_name}", ref_by_rule)
        print(f"[lr]   reading 1 (within a rule, across the two blocks above): "
              f"{ref_name} minus all is what keeping {last_domain} costs under "
              f"that rule.", flush=True)
        print(f"[lr]   reading 2 (across rules, within the {ref_name} block): "
              f"shows what the rule does to the domains you keep -- a rule can "
              f"shrink reading 1's cost while degrading these. Neither block "
              f"alone proves a rule is better; see the design spec's "
              f"acceptance criterion.", flush=True)
    else:
        print(f"[lr]   (no drop:<domain> reference block -- only {n_dom} "
              f"domain(s) in this run)", flush=True)

    print(f"[lr] scored {n_docs} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
