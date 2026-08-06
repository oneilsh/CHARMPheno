"""Labeled per-node readout for the covariate-adjusted prediction axis.

Reads the fit's ``manifest.json`` -> ``metrics.covariate_adjusted`` block (written
when the driver ran ``--pred-cov on``) and joins it with node names, printing a
per-node table of the out-of-fold-CV AUC/AP under [placement_score] alone
(``score_cv``) vs [placement_score, x_d] (``adj``), sorted by AUC lift.

The question it answers that the macro numbers cannot: **is a macro-AUC lift real
or a small-node artifact?** A per-node AUC gain that is NOT accompanied by an AP
gain is the signature of a weak marginal covariate (it nudges the all-pairs
ranking but does not concentrate cases at the top, where precision -- and
case-finding -- live). The summary counts AUC gains that ARE corroborated by AP
vs those that are not, and (when the run carries per-node ``npos``) flags gains
that sit on tiny positive counts. Pure stdlib -- runs on the driver, no Spark/BQ.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
from pathlib import Path


def node_names(manifest: dict) -> dict[int, str]:
    """Engine node id -> concept display name (mirrors effrank_readout/summarize_fit)."""
    cm = manifest.get("corpus_manifest", {})
    int2cid = {int(e): int(c) for e, c in cm.get("int2cid", {}).items()}
    name_by_id = {int(c): str(n) for c, n in cm.get("name_by_id", {}).items()}
    return {e: name_by_id.get(c, str(c)) for e, c in int2cid.items()}


def _run_dir_glob(pattern: str) -> Path:
    """Resolve a possibly-globbed --run-dir to a single existing directory."""
    matches = (sorted(_glob.glob(pattern)) if any(c in pattern for c in "*?[")
               else [pattern])
    dirs = [Path(p) for p in matches if Path(p).is_dir()]
    if not dirs:
        raise SystemExit(f"no run dir matched: {pattern}")
    return dirs[-1]


def build_rows(ca: dict, names: dict[int, str]) -> list[dict]:
    """Join the covariate_adjusted per-node dicts into labeled rows.

    A node whose CV could not run (either class < 2) reads nan and is carried with
    nan deltas so the reader sees it was skipped, not silently dropped. Sorted by
    AUC lift (adj - score_cv) descending; nan lifts sink to the bottom.
    """
    auc_adj = ca.get("node_auc_adj", {})
    auc_sc = ca.get("node_auc_score_cv", {})
    ap_adj = ca.get("node_ap_adj", {})
    ap_sc = ca.get("node_ap_score_cv", {})
    npos = ca.get("node_npos", {})            # present only in runs fit after this was added
    rows = []
    for k in auc_adj:
        u = int(k)
        a_adj, a_sc = auc_adj.get(k), auc_sc.get(k)
        p_adj, p_sc = ap_adj.get(k), ap_sc.get(k)
        auc_delta = (a_adj - a_sc) if _num(a_adj) and _num(a_sc) else float("nan")
        ap_delta = (p_adj - p_sc) if _num(p_adj) and _num(p_sc) else float("nan")
        rows.append({
            "node": u, "name": names.get(u, str(u)),
            "npos": int(npos[k]) if k in npos else None,
            "auc_sc": a_sc, "auc_adj": a_adj, "auc_delta": auc_delta,
            "ap_sc": p_sc, "ap_adj": p_adj, "ap_delta": ap_delta,
        })
    rows.sort(key=lambda r: (-1e9 if math.isnan(r["auc_delta"]) else r["auc_delta"]),
              reverse=True)
    return rows


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x)


def _fmt(x, w=6, p=3) -> str:
    return f"{'  nan':>{w}}" if not _num(x) else f"{x:>{w}.{p}f}"


def render(ca: dict, rows: list[dict], manifest: dict, *,
           auc_eps: float = 0.05, ap_eps: float = 0.005) -> str:
    """Per-node table + a corroboration summary.

    `auc_eps` is the "material AUC lift" cut; `ap_eps` the "AP corroborates it" cut.
    A lift that clears auc_eps but not ap_eps is deployment-marginal (ranking, not
    precision) -- the summary tallies both so the macro number can be believed or
    discounted.
    """
    cov = manifest.get("covariates", {})
    names_str = ",".join(cov.get("names") or []) or "?"
    has_npos = any(r["npos"] is not None for r in rows)
    lines = ["# Covariate-adjusted prediction readout"]
    lines.append(f"covariates: P={ca.get('n_covariates', '?')} ({names_str})")
    lines.append(
        f"detection AUC: score_cv {_f(ca.get('detection_auc_score_cv'))} -> "
        f"adj {_f(ca.get('detection_auc_adj'))}  "
        f"(delta {_signed(ca.get('detection_auc_adj'), ca.get('detection_auc_score_cv'))})")
    lines.append(
        f"macro AUC:     score_cv {_f(ca.get('auc_score_cv_macro'))} -> "
        f"adj {_f(ca.get('auc_adj_macro'))}  "
        f"(delta {_signed(ca.get('auc_adj_macro'), ca.get('auc_score_cv_macro'))})")
    lines.append(
        f"macro AP:      score_cv {_f(ca.get('ap_score_cv_macro'))} -> "
        f"adj {_f(ca.get('ap_adj_macro'))}  "
        f"(delta {_signed(ca.get('ap_adj_macro'), ca.get('ap_score_cv_macro'))})")
    lines.append("")
    npos_hdr = f"{'npos':>6} " if has_npos else ""
    lines.append(f"{'node':>4}  {'name':<34} {npos_hdr}"
                 f"{'auc_sc':>7} {'auc_adj':>7} {'dAUC':>7}   "
                 f"{'ap_sc':>7} {'ap_adj':>7} {'dAP':>7}")
    for r in rows:
        npos_cell = (f"{(r['npos'] if r['npos'] is not None else '-'):>6} "
                     if has_npos else "")
        lines.append(
            f"{r['node']:>4}  {r['name'][:34]:<34} {npos_cell}"
            f"{_fmt(r['auc_sc'],7)} {_fmt(r['auc_adj'],7)} {_signed_v(r['auc_delta']):>7}   "
            f"{_fmt(r['ap_sc'],7)} {_fmt(r['ap_adj'],7)} {_signed_v(r['ap_delta']):>7}")
    # --- corroboration summary --------------------------------------------
    scored = [r for r in rows if _num(r["auc_delta"])]
    big = [r for r in scored if r["auc_delta"] > auc_eps]
    corrob = [r for r in big if _num(r["ap_delta"]) and r["ap_delta"] > ap_eps]
    tiny = [r for r in big if r["npos"] is not None and r["npos"] < 20]
    lines.append("")
    lines.append(f"summary ({len(scored)} nodes scored):")
    lines.append(f"  AUC lift > {auc_eps:.02f}: {len(big)} nodes")
    lines.append(f"  ...of those ALSO AP lift > {ap_eps:.03f} (corroborated, real): "
                 f"{len(corrob)}")
    if has_npos:
        lines.append(f"  ...of those on npos < 20 (small-node, suspect): {len(tiny)}")
    verdict = ("macro-AUC lift is CORROBORATED by AP -> plausibly real"
               if len(corrob) >= max(1, len(big) // 2)
               else "macro-AUC lift is NOT corroborated by AP -> ranking-only / "
                    "deployment-marginal (weak covariate signature, insight 0026)")
    lines.append(f"  => {verdict}")
    return "\n".join(lines)


def _f(x) -> str:
    return "nan" if not _num(x) else f"{x:.3f}"


def _signed(a, b) -> str:
    return "nan" if not (_num(a) and _num(b)) else f"{a - b:+.3f}"


def _signed_v(d) -> str:
    return "nan" if not _num(d) else f"{d:+.3f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="the fit's run dir (has manifest.json with a --pred-cov run)")
    p.add_argument("--auc-eps", type=float, default=0.05,
                   help="material per-node AUC lift threshold (default 0.05)")
    p.add_argument("--ap-eps", type=float, default=0.005,
                   help="AP-corroboration threshold (default 0.005)")
    args = p.parse_args(argv)

    run_dir = _run_dir_glob(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    ca = (manifest.get("metrics", {}) or {}).get("covariate_adjusted")
    if ca is None:
        raise SystemExit(
            f"no metrics.covariate_adjusted in {run_dir}/manifest.json -- re-run "
            "the fit with --pred-cov on (and --covariate-formula) to produce it.")
    rows = build_rows(ca, node_names(manifest))
    print(render(ca, rows, manifest, auc_eps=args.auc_eps, ap_eps=args.ap_eps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
