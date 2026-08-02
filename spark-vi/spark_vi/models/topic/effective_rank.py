"""Data-driven per-node topic-count (K_v) estimation by rank revelation.

The DAG layout gives every node a fixed ``tpn`` topic block. Uniform ``tpn`` is
wrong in both directions: a tight leaf anchor (a few dominating concepts) wants
~2 topics, while a broad class node (e.g. a roll-up-flooded "Disorder of nervous
system" spanning diabetes, back pain, MS, ...) wants many. Patient *count* is a
bad allocator because it tracks volume, not phenotypic diversity: 100k patients
with one phenotype still need one topic. What we want K_v to track is the
**intrinsic dimensionality** of the node's patient population.

That dimensionality is already computed, unused, inside the anchor-word spectral
init. ``spectral_init_scalable.find_anchors_projected`` is greedy pivoted-QR: at
each step it picks the word whose row has the largest residual norm after
projecting out the orthonormal basis of already-chosen directions. Because
projecting out more directions can only shrink residuals and we take the max each
step, the selected residual norms form a NON-INCREASING sequence -- the
rank-revealing spectrum of the node's normalized co-occurrence sketch. Anchor
finding stops at a fixed n; run it out and the point where the spectrum collapses
to noise is the numerical rank = the number of distinct phenotype directions the
node actually holds = K_v.

This module is the pure core: rank-revealing pivoted-QR on a row-set, plus three
parameter-light estimators for the collapse point. It is domain-agnostic (float
matrices only); the cluster caller feeds it each node's ``group_QR`` /
``group_p_w`` normalized sketch.

Estimator choice. The three are reported side by side; the DEFAULT is the
participation ratio, because it is the only one that is both parameter-free AND
scale-invariant -- residual magnitudes differ across nodes, so a scale-invariant
rule applies uniformly with no per-node calibration.
"""
from __future__ import annotations

import json

import numpy as np

_EPS = 1e-12


def pivoted_qr_residual_spectrum(M, max_probe, *, seed_rows=None, eps=_EPS,
                                 return_pivots=False):
    """Rank-revealing pivoted-QR spectrum of the rows of ``M``.

    ``M`` is ``(V, d)`` -- one row per candidate direction (a word's normalized
    co-occurrence sketch row). Mirrors the geometry of
    ``spectral_init_scalable.find_anchors_projected`` exactly: keep an orthonormal
    basis of chosen residual directions; each step picks the row with the largest
    residual norm after projecting that basis out, and records that residual's
    SQUARED norm (a variance-like quantity, the eigenvalue analog). ``seed_rows``
    (optional row ids) pre-deflate the basis without contributing to the spectrum
    -- pass the background anchors so a node's spectrum measures diversity BEYOND
    the shared background, matching the fit's per-group deflation.

    Returns the list of squared residual norms in selection (non-increasing)
    order, length ``<= max_probe`` (shorter if distinct directions run out). The
    length of this list caps K_v at ``max_probe``; set it comfortably above any
    plausible node rank (the estimators saturate well below it for real data).

    Vectorized: instead of re-projecting every candidate against a growing basis
    in a Python loop (O(V·d·k) per step with per-row overhead), it maintains the
    residual matrix ``R`` in place and deflates all rows at once via one matmul
    per step (``R -= (R @ b)[:,None] · b``). The revealed spectrum and pivot
    choices are identical to the Gram-Schmidt form of
    ``find_anchors_projected`` -- picking the largest-residual row, orthonormalizing
    its residual, projecting it out of the rest -- but at BLAS speed, which the
    ~O(#nodes) driver-side probe needs at V in the thousands.

    ``return_pivots``: also return the row ids selected as pivots (in order),
    EXCLUDING seed_rows -- the node's own newly-claimed directions. Used by the
    hierarchical probe to accumulate a node's claim and deflate its descendants
    against the ancestors' FULL claimed set (not just their fit anchors), so a
    descendant measures its INCREMENT over its ancestors and shared structure is
    counted once, not re-counted per node.
    """
    M = np.asarray(M, dtype=np.float64)
    V = M.shape[0]
    R = M.copy()                      # residuals, deflated in place
    chosen: list[int] = []

    def deflate(row):
        b = R[row].copy()
        nrm = np.sqrt(b @ b)
        if nrm > eps:
            b /= nrm
            R[:] -= np.outer(R @ b, b)

    if seed_rows is not None:
        for s in seed_rows:
            si = int(s)
            deflate(si)
            chosen.append(si)         # seeds deflate but do not enter the spectrum
    n_seed = len(chosen)

    spectrum: list[float] = []
    for _ in range(int(max_probe)):
        norms2 = np.einsum("ij,ij->i", R, R)
        if chosen:
            norms2[chosen] = -np.inf   # never re-pick a chosen/seed row
        i = int(np.argmax(norms2))
        best = float(norms2[i])
        if best <= eps:
            break
        spectrum.append(best)
        chosen.append(i)
        deflate(i)
    if return_pivots:
        return spectrum, chosen[n_seed:]   # newly-claimed pivots, seeds excluded
    return spectrum


def participation_ratio(spectrum):
    """Effective rank = ``(Σ λ)² / Σ λ²`` over a variance-like ``spectrum``.

    Parameter-free and scale-invariant (multiplying every λ by a constant leaves
    the ratio unchanged), so one rule applies to every node with no per-node
    threshold. Equals the true rank r for a flat spectrum (r equal values),
    approaches 1 for a spectrum dominated by a single direction, and interpolates
    smoothly between -- the "effective number of significant directions". Empty or
    all-zero spectrum -> 0.0.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    total = s.sum()
    return float(total * total / (s * s).sum())


def threshold_rank(spectrum, tau=0.01):
    """Count of spectrum entries at least ``tau`` times the leading entry.

    A relative (not absolute) cut, so it is scale-invariant per node but needs the
    one knob ``tau``. Reported alongside the participation ratio as a sanity check;
    ``tau=0.01`` keeps any direction carrying >=1% of the leading variance.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    if s.size == 0 or s[0] <= 0:
        return 0
    return int((s >= tau * s[0]).sum())


def eigengap_rank(spectrum, *, min_rank=1):
    """Rank at the largest relative drop ``spectrum[k] / spectrum[k+1]``.

    Parameter-free but brittle when the spectrum is smooth (no clean gap), which
    is why it is a cross-check, not the default. Returns the number of directions
    ABOVE the biggest gap (so a gap between index k and k+1 -> rank k+1). With no
    interior gap (length <= 1) returns ``min(min_rank, len)``.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    n = s.size
    if n <= 1:
        return int(min(min_rank, n))
    ratios = s[:-1] / np.maximum(s[1:], _EPS)
    k = int(np.argmax(ratios))  # gap between k and k+1
    return max(int(min_rank), k + 1)


def effective_rank(M, max_probe, *, method="participation", seed_rows=None,
                   tau=0.01):
    """Convenience: pivoted-QR spectrum of ``M`` then the chosen estimator.

    ``method`` in {"participation", "threshold", "eigengap"}. Returns a float
    (participation) or int (threshold/eigengap). For the full picture use
    ``effective_rank_report``.
    """
    spec = pivoted_qr_residual_spectrum(M, max_probe, seed_rows=seed_rows)
    if method == "participation":
        return participation_ratio(spec)
    if method == "threshold":
        return threshold_rank(spec, tau=tau)
    if method == "eigengap":
        return eigengap_rank(spec)
    raise ValueError(f"unknown method: {method!r}")


def report_from_spectrum(spec, *, tau=0.01):
    """All three estimators + the raw spectrum, from an already-computed spectrum.

    Split out so a caller that already ran ``pivoted_qr_residual_spectrum`` (e.g.
    to also collect pivots for hierarchical deflation) can build the report
    without re-running the pivoted-QR.
    """
    return {
        "participation": participation_ratio(spec),
        "threshold": threshold_rank(spec, tau=tau),
        "eigengap": eigengap_rank(spec),
        "n_probed": len(spec),
        "spectrum": spec,
    }


def effective_rank_report(M, max_probe, *, seed_rows=None, tau=0.01):
    """All three estimators plus the raw spectrum, for side-by-side node dumps.

    Returns a dict with ``participation`` (float), ``threshold`` (int),
    ``eigengap`` (int), ``n_probed`` (int, spectrum length), and ``spectrum``
    (list) so the caller can eyeball the decay and pick an allocation rule.
    """
    spec = pivoted_qr_residual_spectrum(M, max_probe, seed_rows=seed_rows)
    return report_from_spectrum(spec, tau=tau)


def allocate_topics(effranks, *, floor=1, cap=None, round_fn=None):
    """Turn per-node effective ranks into integer topic-block sizes ``K_v``.

    ``effranks`` maps node id -> effective rank (float). Each is rounded (default
    ``round`` then int), clamped to ``[floor, cap]`` (``cap=None`` = no cap). The
    total ``Σ K_v`` is the layout's foreground K -- which grows with the corpus's
    intrinsic diversity, NOT with node count times a fixed tpn. Returns a dict
    node id -> K_v (int).
    """
    rf = round_fn or (lambda x: int(round(x)))
    out = {}
    for node, er in effranks.items():
        k = rf(er)
        k = max(int(floor), k)
        if cap is not None:
            k = min(int(cap), k)
        out[node] = k
    return out


def log_effrank_table(reports, *, n_nodes, k_uniform, printer=print,
                      prefix="[effrank]"):
    """Print a per-node effective-rank table + the diversity-vs-uniform K summary.

    ``reports`` maps node id -> an ``effective_rank_report`` dict. Rows are sorted
    by participation ratio (the default estimator) descending. The final line
    contrasts a diversity-driven foreground K (``Σ round(participation)``, floored
    at 1) against ``k_uniform`` (the current layout's foreground topic count), so
    the reader sees at a glance whether data-driven K_v would shrink or blow up
    the topic budget. ``printer`` is injectable for testing; defaults to ``print``.
    """
    if not reports:
        printer(f"{prefix} no nodes probed")
        return
    ordered = sorted(
        reports.items(), key=lambda kv: kv[1]["participation"], reverse=True
    )
    printer(
        f"{prefix} per-node effective rank; PR=participation (default), "
        "thr=threshold, gap=eigengap, n=collapse point"
    )
    printer(f"{prefix} node\tPR\tthr\tgap\tn")
    for node, rep in ordered:
        printer(
            f"{prefix} {node}\t{rep['participation']:.1f}\t"
            f"{rep['threshold']}\t{rep['eigengap']}\t{rep['n_probed']}"
        )
    effranks = {node: rep["participation"] for node, rep in reports.items()}
    k_diversity = sum(allocate_topics(effranks, floor=1).values())
    printer(
        f"{prefix} foreground K: diversity-driven Σround(PR)={k_diversity} "
        f"vs current foreground K={int(k_uniform)} "
        f"(nodes probed={len(reports)}/{n_nodes})"
    )


def save_effrank_sidecar(reports, path):
    """Write a compact JSON sidecar of per-node effective-rank reports.

    ``reports`` maps node id -> an ``effective_rank_report`` dict (optionally with
    an ``n_docs`` field the caller added). The bulky raw ``spectrum`` is dropped;
    the scalar estimators (participation/threshold/eigengap/n_probed) plus n_docs
    are kept, JSON-coerced (numpy scalars -> Python). Keys are stringified node
    ids (JSON object keys must be strings); a reader casts them back to int. A
    post-fit readout joins this with the run's manifest (int2cid -> name_by_id)
    and each node's n_docs for a labeled, count-annotated table.
    """
    out = {}
    for node, rep in reports.items():
        out[str(int(node))] = {
            "participation": float(rep["participation"]),
            "threshold": int(rep["threshold"]),
            "eigengap": int(rep["eigengap"]),
            "n_probed": int(rep["n_probed"]),
            "n_docs": int(rep.get("n_docs", 0)),
        }
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out
