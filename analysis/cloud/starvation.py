"""Which node topics starve in a DAG-gated fit, and what they have in common.

A node topic is STARVED (the census's word, `inspect_topics.classify_block_topics`)
when its word distribution in its dominant domain spreads its effective support
(exp of the entropy) over more than half that domain's vocabulary: it sits near the
flat η prior, so it describes nothing. The measure is spark-vi's
(`spark_vi.eval.topic.sharpness.support_fraction`, re-exported here); the fitted
model's `nodeStatistics()` reports it per node, `flat` there being "starved" here. Under SVI a block whose documents stop sending
it words decays back toward η, so a topic can be fed at one iteration and starved later
(exp 0169: none at 60 iterations, 189 of 2,845 at 150).

Two uses:

* `python analysis/cloud/starvation.py RUN [RUN ...]` reads each run's fitted model
  (`<run>/model/result`, no Spark) and prints, per run, the starved rate by bins of the
  node's document support, its share of its parent's documents, its kept children, its
  largest child's share of its documents, its depth, and its λ mass relative to its
  parent's. With two or more runs it also follows the nodes starved in the LAST run
  back through the earlier ones (their support fraction and mass then).
* `flat_topics` / `trace_line` back the fit driver's `--trace-flat-every N`, which
  prints the starved count every N iterations, so a fit shows when starvation starts.

Egress: every output line is a count of nodes or a pooled statistic (median,
quantiles) over a bin; no per-node values are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from spark_vi.eval.topic.sharpness import support_fraction  # noqa: F401 (re-export)

STARVED_FRAC = 0.5      # the census's threshold (inspect_topics, classify_block_topics)
# Bands of a node's document support, shared by the report and the fit driver's
# node-statistics log line.
SUPPORT_BANDS = [0, 100, 200, 500, 1000, 5000, np.inf]


# --------------------------------------------------------------------------- #
# Per-topic flatness (the measure, `support_fraction`, is spark-vi's: R14.1)   #
# --------------------------------------------------------------------------- #

def flat_topics(lam, n_bg: int, *, frac: float = STARVED_FRAC):
    """`(n_flat_node_topics, n_node_topics, node_mass_share)` for the node topics
    (indices >= n_bg): how many sit at the flat prior, and the share of all λ mass the
    node topics hold (the rest is the background's)."""
    sf, mass = support_fraction(lam)
    node = slice(n_bg, None)
    tot = float(mass.sum())
    return (int(np.sum(sf[node] > frac)), int(sf[node].size),
            float(mass[node].sum() / tot) if tot > 0 else float("nan"))


def trace_line(it: int, lam, n_bg: int) -> str:
    n_flat, n_node, share = flat_topics(lam, n_bg)
    return (f"[trace] iter {it}: starved node topics {n_flat}/{n_node} "
            f"({100 * n_flat / max(n_node, 1):.1f}%) · node topics hold "
            f"{100 * share:.1f}% of the lambda mass")


# --------------------------------------------------------------------------- #
# The offline report                                                           #
# --------------------------------------------------------------------------- #
def load_run(run_dir):
    """`(lambda, node_set, n_bg, tpn)` from `<run>/model/result` (no Spark)."""
    from spark_vi.dag.nodeset import NodeSet
    from spark_vi.io.export import load_result
    res = load_result(str(Path(run_dir) / "model" / "result"))
    md = res.metadata["dag_node_set"]
    return (res.global_params["lambda"], NodeSet.from_dict(md),
            int(md["n_bg"]), int(md["tpn"]))


def node_table(lam, ns, n_bg, tpn) -> dict:
    """Per kept node (engine order 1..N): starved flag, depth, support, parent share,
    kept children, largest child share, mass ratio to the parent. A node with tpn > 1
    counts as starved when every topic of its block is."""
    from spark_vi.dag.layout import DagLayout
    sf, mass = support_fraction(lam)
    lay = DagLayout(ns.engine_parents, n_bg=n_bg, tpn=tpn)
    N = len(ns.user_ids) - 1
    sup = {e: (ns.n_docs if e == 0 else ns.support[ns.user_ids[e]])
           for e in range(N + 1)}
    node_mass = {e: float(sum(mass[t] for t in lay.block[e])) for e in range(1, N + 1)}
    kids = {e: [] for e in range(N + 1)}
    for c, ps in ns.engine_parents.items():
        for p in (ps if isinstance(ps, (list, tuple)) else [ps]):
            kids[int(p)].append(int(c))
    out = {k: [] for k in ("starved", "depth", "support", "parent_share", "n_kids",
                           "child_share", "mass_vs_parent", "frac", "mass")}
    for e in range(1, N + 1):
        ps = ns.engine_parents.get(e, [0])
        ps = [int(p) for p in (ps if isinstance(ps, (list, tuple)) else [ps])]
        out["starved"].append(all(sf[t] > STARVED_FRAC for t in lay.block[e]))
        out["depth"].append(lay.depth(e))
        out["support"].append(sup[e])
        out["parent_share"].append(max(sup[e] / max(sup[p], 1) for p in ps))
        out["n_kids"].append(len(kids[e]))
        out["child_share"].append(max((sup[c] / sup[e] for c in kids[e]), default=0.0))
        pm = [node_mass[p] for p in ps if p != 0]
        out["mass_vs_parent"].append(node_mass[e] / max(pm) if pm else float("nan"))
        out["frac"].append(float(np.mean([sf[t] for t in lay.block[e]])))
        out["mass"].append(node_mass[e])
    return {k: np.asarray(v) for k, v in out.items()}


_BINS = {
    "support": (SUPPORT_BANDS, "documents reaching the node"),
    "parent_share": ([0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0001],
                     "node's documents / its parent's (max over parents)"),
    "n_kids": ([0, 1, 2, 3, 6, np.inf], "kept children"),
    "child_share": ([0, 0.01, 0.3, 0.5, 0.8, 0.95, 1.0001],
                    "largest kept child's documents / the node's"),
    "depth": ([1, 4, 6, 8, 10, np.inf], "depth"),
}


def _binned(tab, key) -> list[str]:
    edges, what = _BINS[key]
    x, st = tab[key], tab["starved"]
    lines = [f"  by {what}:"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        hi_s = "inf" if np.isinf(hi) else ("1" if 1 < hi < 1.001 else f"{hi:g}")
        lines.append(f"    [{lo:g}, {hi_s}): {n:5d} nodes · starved {int(st[sel].sum()):4d}"
                     f" ({100 * st[sel].mean():5.1f}%)")
    return lines


def report(run_dirs) -> str:
    runs = [(Path(r).name, *load_run(r)) for r in run_dirs]
    lines = []
    tabs = []
    for name, lam, ns, n_bg, tpn in runs:
        tab = node_table(lam, ns, n_bg, tpn)
        tabs.append((name, ns, tab))
        st = tab["starved"]
        lines.append(f"## {name}: {int(st.sum())} of {st.size} node blocks starved "
                     f"({100 * st.mean():.1f}%)")
        if st.any() and (~st).any():
            for key, lab in (("support", "support"), ("parent_share", "parent share"),
                             ("child_share", "largest child share"),
                             ("mass_vs_parent", "mass / parent's")):
                a, b = tab[key][st], tab[key][~st]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if a.size and b.size:
                    lines.append(f"  {lab}: starved median {np.median(a):.3g} "
                                 f"(p10 {np.quantile(a, .1):.3g}, p90 "
                                 f"{np.quantile(a, .9):.3g}) · fed median "
                                 f"{np.median(b):.3g}")
        for key in _BINS:
            lines += _binned(tab, key)
    if len(tabs) > 1:
        last_name, last_ns, last_tab = tabs[-1]
        starved_ids = {last_ns.user_ids[i + 1] for i in np.flatnonzero(last_tab["starved"])}
        fed_ids = {last_ns.user_ids[i + 1] for i in np.flatnonzero(~last_tab["starved"])}
        lines.append(f"## the {len(starved_ids)} nodes starved in {last_name}, "
                     "in the earlier runs (shared nodes only)")
        for name, ns, tab in tabs[:-1]:
            idx = {ns.user_ids[i + 1]: i for i in range(tab["starved"].size)}
            for lab, ids in (("starved-later", starved_ids), ("fed-later", fed_ids)):
                rows = [idx[u] for u in ids if u in idx]
                if not rows:
                    continue
                r = np.asarray(rows)
                lines.append(f"  in {name}, the {lab} nodes ({r.size}): starved then "
                             f"{int(tab['starved'][r].sum())} · support fraction median "
                             f"{np.median(tab['frac'][r]):.3f} · mass / parent's median "
                             f"{np.nanmedian(tab['mass_vs_parent'][r]):.3g}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: starvation.py RUN_DIR [RUN_DIR ...] (oldest first)")
    print(report(sys.argv[1:]))
