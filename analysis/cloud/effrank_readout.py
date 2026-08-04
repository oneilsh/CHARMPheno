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
        row = {
            "node": u,
            "name": names.get(u, str(u)),
            "depth": depths.get(u, -1),
            "n_docs": int(rep.get("n_docs", 0)),
            "participation": float(rep.get("participation", 0.0)),
            "threshold": int(rep.get("threshold", 0)),
            "eigengap": int(rep.get("eigengap", 0)),
            "n_probed": int(rep.get("n_probed", 0)),
        }
        # Parallel-analysis fields, present only when the pa probe ran.
        if "pa_k" in rep:
            row["pa_k"] = int(rep["pa_k"])
        if "pa_k_all" in rep:
            row["pa_k_all"] = int(rep["pa_k_all"])
        if "pa_pr_raw" in rep:
            row["pa_pr_raw"] = float(rep["pa_pr_raw"])
        if "pa_spec" in rep and "pa_floor" in rep:
            row["pa_spec"] = [float(x) for x in rep["pa_spec"]]
            row["pa_floor"] = [float(x) for x in rep["pa_floor"]]
        if "pa_k_rec" in rep:
            row["pa_k_rec"] = int(rep["pa_k_rec"])
        if "pa_spec_rec" in rep and "pa_floor_rec" in rep:
            row["pa_spec_rec"] = [float(x) for x in rep["pa_spec_rec"]]
            row["pa_floor_rec"] = [float(x) for x in rep["pa_floor_rec"]]
        rows.append(row)
    # Sort by pa_k when present (the live estimator), else participation.
    if any("pa_k" in r for r in rows):
        rows.sort(key=lambda r: r.get("pa_k", -1), reverse=True)
    else:
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


def pa_volume_correlation(rows: list[dict]) -> float:
    """Pearson(pa_k, log10(n_docs)) over rows with pa_k present and n_docs > 0.

    The acceptance signal for the parallel-analysis estimator: this should be FAR
    below the raw ``corr(PR, log n_docs)`` (~0.4-0.5 for effective rank). A low
    correlation means pa_k reflects per-node phenotype dimensionality, not just how
    many patients the node has. Returns 0.0 if no row carries pa_k.
    """
    xs, ys = [], []
    for r in rows:
        if "pa_k" in r and r["n_docs"] > 0:
            xs.append(r["pa_k"])
            ys.append(math.log10(r["n_docs"]))
    return pearson(xs, ys)


def pa_bucket_correlations(rows: list[dict], key="pa_k"):
    """corr(``key``, log10 n_docs) within support buckets -> [(label, n, corr)].

    The aggregate correlation can read ~0 while a low-support INVERSION (fewer docs
    -> higher count) cancels a positive trend among well-supported nodes. Reporting
    per bucket makes that cancellation visible instead of hiding it.
    """
    buckets = [("tiny <50", 0, 50), ("small 50-300", 50, 300),
               ("big >=300", 300, 10 ** 12)]
    out = []
    for label, lo, hi in buckets:
        xs, ys = [], []
        for r in rows:
            if key in r and lo <= r["n_docs"] < hi:
                xs.append(r[key])
                ys.append(math.log10(r["n_docs"]))
        out.append((label, len(xs), pearson(xs, ys) if len(xs) >= 2 else 0.0))
    return out


def pa_rank_from_spectrum(spec, floor, *, margin=2.0, tau=0.01, bg_skip=1) -> int:
    """Leading contiguous block above the null AND above a variance floor.

    Dependency-free twin of ``effective_rank.parallel_analysis_rank`` (kept inline
    so the readout needs no pyspark to import). Re-derives pa_k from the stored real
    spectrum + null floor, so margin/tau/bg_skip are readout-time knobs -- sweep the
    cutoff without a re-fit. See that function for the rationale.
    """
    m = min(len(spec), len(floor))
    if m == 0:
        return 0
    clear = [spec[k] > margin * floor[k] for k in range(m)]
    start = -1
    for k in range(min(int(bg_skip) + 1, m)):
        if clear[k]:
            start = k
            break
    if start < 0:
        return 0
    var_floor = tau * spec[start]
    run = 0
    k = start
    while k < m and clear[k] and spec[k] >= var_floor:
        run += 1
        k += 1
    return run


def recompute_pa_k(rows: list[dict], *, margin=2.0, tau=0.01, bg_skip=1) -> None:
    """Overwrite each row's ``pa_k`` from its stored spectra at the given cutoff.

    Rows without stored spectra (older sidecars) keep their fitted ``pa_k``.
    """
    for r in rows:
        if "pa_spec" in r and "pa_floor" in r:
            r["pa_k"] = pa_rank_from_spectrum(
                r["pa_spec"], r["pa_floor"], margin=margin, tau=tau,
                bg_skip=bg_skip)
        if "pa_spec_rec" in r and "pa_floor_rec" in r:
            r["pa_k_rec"] = pa_rank_from_spectrum(
                r["pa_spec_rec"], r["pa_floor_rec"], margin=margin, tau=tau,
                bg_skip=bg_skip)


def pa_tau_sweep(rows, taus, *, margin=2.0, bg_skip=1):
    """[(tau, Σpa_k, #nodes with pa_k > n_docs)] over a list of tau cutoffs.

    Only rows carrying stored spectra participate. Lets the reader see the whole
    cutoff curve at once -- where Σ stabilizes and where the impossible-count
    (pa_k > n_docs, the under-support tell) drops to zero.
    """
    have = [r for r in rows if "pa_spec" in r and "pa_floor" in r]
    out = []
    for t in taus:
        ks = [pa_rank_from_spectrum(r["pa_spec"], r["pa_floor"], margin=margin,
                                    tau=t, bg_skip=bg_skip) for r in have]
        imp = sum(1 for r, k in zip(have, ks) if k > r["n_docs"] > 0)
        out.append((t, sum(ks), imp))
    return out


def render(rows: list[dict], *, k_uniform: int | None = None) -> str:
    """Render the labeled table + a diversity-vs-volume summary as text.

    When the parallel-analysis probe ran (rows carry ``pa_k``), the table leads with
    the sample-size-aware ``pa_k`` and its own volume correlation -- the live
    per-node K estimate -- alongside the raw participation ratio (``PR``, the
    closed-negative effective-rank number) for contrast.
    """
    has_pa = any("pa_k" in r for r in rows)
    corr = pr_volume_correlation(rows)
    k_div = sum(max(1, round(r["participation"])) for r in rows)
    max_n = max((r["n_probed"] for r in rows), default=0)
    saturated = sum(1 for r in rows if r["n_probed"] == max_n and max_n > 0)
    lines = []
    lines.append("# Per-node K probe (labeled)")
    lines.append("")
    if has_pa:
        pa_total = sum(r.get("pa_k", 0) for r in rows)
        pa_all_total = sum(r.get("pa_k_all", 0) for r in rows)
        pa_corr = pa_volume_correlation(rows)
        has_all = any("pa_k_all" in r for r in rows)
        lines.append(f"nodes: {len(rows)}  |  Σpa_k [parallel-analysis K, "
                     f"leading-run]: {pa_total}"
                     + (f"  vs current foreground K: {k_uniform}"
                        if k_uniform is not None else ""))
        if has_all:
            impossible = sum(1 for r in rows
                             if r.get("pa_k_all", 0) > r["n_docs"] > 0)
            lines.append(f"  (count-all diagnostic Σpa_k_all: {pa_all_total}; "
                         f"{impossible} nodes had pa_k_all > n_docs -- the "
                         "tail-noise inflation the leading-run rule removes)")
        lines.append(f"corr(pa_k, log10 n_docs): {pa_corr:+.2f}  "
                     f"(vs raw corr(PR, log n_docs): {corr:+.2f})")
        if any("pa_k_rec" in r for r in rows):
            rec_total = sum(r.get("pa_k_rec", 0) for r in rows)
            rec_xy = [(r["pa_k_rec"], math.log10(r["n_docs"])) for r in rows
                      if "pa_k_rec" in r and r["n_docs"] > 0]
            rec_corr = pearson([x for x, _ in rec_xy], [y for _, y in rec_xy])
            lines.append(f"  recurrence-floored (per-node df) Σpa_k_rec: "
                         f"{rec_total}  |  corr(pa_k_rec, log10 n_docs): "
                         f"{rec_corr:+.2f}")
        # By-support-bucket correlation, so a low-n inversion can't hide in the
        # aggregate (the count-all estimator's headline ~0 was two effects cancelling).
        bkt = pa_bucket_correlations(rows, key="pa_k")
        lines.append("  by support: " + "  ".join(
            f"{lab} (n={n}) {c:+.2f}" for lab, n, c in bkt))
    else:
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
    if has_pa:
        has_rec = any("pa_k_rec" in r for r in rows)
        rec_hdr = f"{'rec':>5} " if has_rec else ""
        lines.append(f"{'pa_k':>5} {rec_hdr}{'p_all':>5}  {'PR':>6}  "
                     f"{'depth':>5} {'n_docs':>8}  node  name")
        for r in rows:
            rec_cell = f"{r.get('pa_k_rec', 0):>5} " if has_rec else ""
            lines.append(
                f"{r.get('pa_k', 0):>5} {rec_cell}{r.get('pa_k_all', 0):>5}  "
                f"{r['participation']:6.1f}  {r['depth']:>5} {r['n_docs']:>8}  "
                f"{r['node']:>4}  {r['name']}"
            )
    else:
        lines.append(f"{'PR':>6}  {'thr':>4} {'gap':>4} {'n':>4}  "
                     f"{'depth':>5} {'n_docs':>8}  node  name")
        for r in rows:
            lines.append(
                f"{r['participation']:6.1f}  {r['threshold']:>4} "
                f"{r['eigengap']:>4} {r['n_probed']:>4}  {r['depth']:>5} "
                f"{r['n_docs']:>8}  {r['node']:>4}  {r['name']}"
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
    p.add_argument("--pa-tau", type=float, default=0.01,
                   help="proportion-of-variance floor for pa_k (default 0.01); "
                        "re-derived from stored spectra, no re-fit needed")
    p.add_argument("--pa-margin", type=float, default=2.0,
                   help="null-clearance margin for pa_k (default 2.0)")
    p.add_argument("--pa-bg-skip", type=int, default=1,
                   help="leading background directions to skip (default 1)")
    p.add_argument("--pa-sweep", action="store_true",
                   help="print a tau sweep (Σpa_k + impossible-count per cutoff)")
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
    # Re-derive pa_k at the requested cutoff from stored spectra (no re-fit), then
    # re-sort by the fresh pa_k so the table reflects the chosen knobs.
    if any("pa_spec" in r for r in rows):
        recompute_pa_k(rows, margin=args.pa_margin, tau=args.pa_tau,
                       bg_skip=args.pa_bg_skip)
        rows.sort(key=lambda r: r.get("pa_k", -1), reverse=True)
    k_uniform = len(parent_int) * int(manifest.get("tpn", 0)) or None
    print(render(rows, k_uniform=k_uniform))
    if args.pa_sweep and any("pa_spec" in r for r in rows):
        print(f"\n# tau sweep (margin={args.pa_margin}, bg_skip={args.pa_bg_skip})")
        print(f"{'tau':>7} {'Σpa_k':>7} {'#impossible(pa_k>n_docs)':>26}")
        for t, tot, imp in pa_tau_sweep(
                rows, [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
                margin=args.pa_margin, bg_skip=args.pa_bg_skip):
            print(f"{t:>7.3f} {tot:>7} {imp:>26}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
