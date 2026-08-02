"""Labeled post-fit readout for the per-node effective-rank probe.

Joins the fit's effrank sidecar (``effrank.json``, written by the gated spectral
init when CHARM_PROBE_EFFRANK is set) with the run manifest's node names and each
node's training-doc count, producing a labeled table sorted by participation
ratio (the data-driven K_v estimate). Pure stdlib + numpy so it runs on the
driver with no Spark/BQ; reads only the self-describing run-dir artifacts.

Two things it answers that the raw log table cannot:
  1. **Labels + counts** — engine node id -> concept name -> training-doc count,
     so "node 1 PR=36" becomes "Disorder of cardiovascular system, 41k docs, K~36".
  2. **Is effrank just volume?** — the Pearson correlation between participation
     and log10(n_docs). A high correlation means the (parent-deflated) rank still
     tracks population size, i.e. it is a volume proxy, not a diversity one; a low
     correlation means the deflation isolated genuine per-node phenotype increment.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
from pathlib import Path


def node_names(manifest: dict) -> dict[int, str]:
    """Engine node id -> concept display name (mirrors summarize_fit)."""
    cm = manifest.get("corpus_manifest", {})
    int2cid = {int(e): int(c) for e, c in cm.get("int2cid", {}).items()}
    name_by_id = {int(c): str(n) for c, n in cm.get("name_by_id", {}).items()}
    return {e: name_by_id.get(c, str(c)) for e, c in int2cid.items()}


def node_depths(parent_int: dict[int, list[int]]) -> dict[int, int]:
    """Longest-path depth from root (0) for each node; root-children = depth 1.

    Memoized DFS over the multi-parent map (acyclic by construction). A node with
    no parents, or only the root as parent, is depth 1 here (root itself is 0).
    """
    depth: dict[int, int] = {}

    def d(v, stack=()):
        if v == 0:
            return 0
        if v in depth:
            return depth[v]
        ps = [p for p in parent_int.get(v, []) if p != v and p not in stack]
        depth[v] = 1 if not ps else 1 + max(d(p, stack + (v,)) for p in ps)
        return depth[v]

    for v in parent_int:
        d(v)
    return depth


def pearson(xs, ys) -> float:
    """Pearson correlation of two equal-length sequences; 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def build_rows(sidecar: dict, names: dict[int, str],
               depths: dict[int, int]) -> list[dict]:
    """Join sidecar reports with names + depth; sorted by participation desc."""
    rows = []
    for k, rep in sidecar.items():
        u = int(k)
        rows.append({
            "node": u,
            "name": names.get(u, str(u)),
            "depth": depths.get(u, -1),
            "n_docs": int(rep.get("n_docs", 0)),
            "participation": float(rep.get("participation", 0.0)),
            "threshold": int(rep.get("threshold", 0)),
            "eigengap": int(rep.get("eigengap", 0)),
            "n_probed": int(rep.get("n_probed", 0)),
        })
    rows.sort(key=lambda r: r["participation"], reverse=True)
    return rows


def pr_volume_correlation(rows: list[dict]) -> float:
    """Pearson(participation, log10(n_docs)) over rows with n_docs > 0."""
    xs, ys = [], []
    for r in rows:
        if r["n_docs"] > 0:
            xs.append(r["participation"])
            ys.append(math.log10(r["n_docs"]))
    return pearson(xs, ys)


def render(rows: list[dict], *, k_uniform: int | None = None) -> str:
    """Render the labeled table + a diversity-vs-volume summary as text."""
    corr = pr_volume_correlation(rows)
    k_div = sum(max(1, round(r["participation"])) for r in rows)
    max_n = max((r["n_probed"] for r in rows), default=0)
    saturated = sum(1 for r in rows if r["n_probed"] == max_n and max_n > 0)
    lines = []
    lines.append("# Per-node effective rank (labeled)")
    lines.append("")
    lines.append(f"nodes: {len(rows)}  |  "
                 f"Σround(PR) [diversity-driven K]: {k_div}"
                 + (f"  vs current foreground K: {k_uniform}"
                    if k_uniform is not None else ""))
    lines.append(f"corr(PR, log10 n_docs): {corr:+.2f}  "
                 "(high => rank tracks volume, not diversity)")
    if saturated:
        lines.append(f"NOTE: {saturated}/{len(rows)} nodes saturate at "
                     f"n_probed={max_n} (raise CHARM_PROBE_EFFRANK_MAX to see "
                     "their true rank).")
    lines.append("")
    lines.append(f"{'PR':>6}  {'thr':>4} {'gap':>4} {'n':>4}  "
                 f"{'depth':>5} {'n_docs':>8}  node  name")
    for r in rows:
        lines.append(
            f"{r['participation']:6.1f}  {r['threshold']:>4} {r['eigengap']:>4} "
            f"{r['n_probed']:>4}  {r['depth']:>5} {r['n_docs']:>8}  "
            f"{r['node']:>4}  {r['name']}"
        )
    return "\n".join(lines)


def _run_dir_glob(pattern: str) -> Path:
    """Resolve a possibly-globbed --run-dir to a single existing directory.

    Uses ``glob.glob`` (not ``Path.glob``) so ABSOLUTE globbed patterns work --
    the runs dir is an absolute path and Path.glob rejects non-relative patterns.
    """
    matches = (sorted(_glob.glob(pattern)) if any(c in pattern for c in "*?[")
               else [pattern])
    dirs = [Path(p) for p in matches if Path(p).is_dir()]
    if not dirs:
        raise SystemExit(f"no run dir matched: {pattern}")
    return dirs[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="the fit's run dir (has manifest.json + effrank.json)")
    args = p.parse_args(argv)

    run_dir = _run_dir_glob(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    side_path = run_dir / "effrank.json"
    if not side_path.exists():
        raise SystemExit(
            f"no effrank.json under {run_dir} -- re-run the fit with "
            "CHARM_PROBE_EFFRANK=1 (spectral init) to produce it.")
    sidecar = json.loads(side_path.read_text())

    names = node_names(manifest)
    cm = manifest.get("corpus_manifest", {})
    parent_int = {int(n): [int(x) for x in ps]
                  for n, ps in cm.get("parent_int", {}).items()}
    depths = node_depths(parent_int)
    rows = build_rows(sidecar, names, depths)
    k_uniform = len(parent_int) * int(manifest.get("tpn", 0)) or None
    print(render(rows, k_uniform=k_uniform))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
