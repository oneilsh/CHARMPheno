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
                                 return_pivots=False, eligible=None):
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

    ``eligible`` (optional boolean array over rows): only eligible rows may be
    SELECTED as pivots; ineligible rows still get deflated (they carry real
    structure to remove) but never enter the spectrum. This is the probe's
    document-frequency floor -- mirroring the fit's ``find_anchors`` min_doc_freq
    candidate mask -- so a low-count node's rank counts only directions supported
    by >= floor documents, NOT single-patient idiosyncratic words (which would
    otherwise inflate a 26-doc node to a spurious rank of >100). Without it every
    word row is eligible (the old behavior).
    """
    M = np.asarray(M, dtype=np.float64)
    V = M.shape[0]
    R = M.copy()                      # residuals, deflated in place
    chosen: list[int] = []
    ineligible = None if eligible is None else ~np.asarray(eligible, dtype=bool)

    def deflate(row):
        b = R[row].copy()
        nrm = np.sqrt(b @ b)
        if nrm > eps:
            b /= nrm
            R[:] -= np.outer(R @ b, b)

    if seed_rows is not None and len(seed_rows) > 0:
        # BATCHED seed pre-deflation: project R onto the orthogonal complement of
        # the seed rows' span in ONE BLAS-3 shot (QR + two matmuls) instead of k
        # memory-bound rank-1 updates. Orthogonal projection onto a subspace is
        # basis-independent, so this is numerically equivalent to the sequential
        # Gram-Schmidt deflate() above -- but at BLAS-3 throughput, which the
        # hierarchical probe needs when a node deflates against thousands of
        # accumulated ancestor pivots (the seed loop was the overnight bottleneck).
        seed_ids = sorted({int(s) for s in seed_rows})
        Q, _ = np.linalg.qr(M[seed_ids].T)     # (d, r): orthonormal basis of span
        R -= (R @ Q) @ Q.T
        chosen.extend(seed_ids)                # excluded from selection + spectrum
    n_seed = len(chosen)

    spectrum: list[float] = []
    for _ in range(int(max_probe)):
        norms2 = np.einsum("ij,ij->i", R, R)
        if chosen:
            norms2[chosen] = -np.inf   # never re-pick a chosen/seed row
        if ineligible is not None:
            norms2[ineligible] = -np.inf   # df floor: never SELECT a sub-floor word
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


def singular_value_spectrum(M, max_probe):
    """Squared singular values of ``M`` (V, d), descending, top ``max_probe``.

    The spectrum parallel analysis actually compares. (The effective-rank estimators
    above use the greedy pivoted-QR RESIDUAL sequence, which reuses the anchor-finding
    geometry -- but that greedy picks extreme rows and its residual sequence does NOT
    have the clean real-vs-null crossing that Horn's method relies on. The singular
    values are the variance-per-direction quantities Horn's null was designed for, and
    empirically give a stable, sample-size-aware count where the residual sequence
    does not.) Computed via the eigenvalues of the smaller Gram matrix (``d x d`` when
    V >= d, the usual sketch shape) -- squared singular values -- which is far cheaper
    than a full SVD at V in the thousands. Returns a plain list, length
    ``min(max_probe, min(V, d))``; empty for a degenerate ``M``.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or min(M.shape) == 0:
        return []
    V, d = M.shape
    gram = M.T @ M if V >= d else M @ M.T
    ev = np.linalg.eigvalsh(gram)          # ascending, real (Gram is symmetric PSD)
    ev = np.maximum(ev[::-1], 0.0)         # descending, clip tiny negatives
    return list(ev[: int(max_probe)])


def parallel_analysis_rank(spec_real, floor, *, margin=2.0):
    """Parallel-analysis per-node K: directions whose real variance clears the null.

    ``spec_real`` and ``floor`` are squared-singular-value spectra on the SAME
    projection scale (same ``d``, same projection seed): ``spec_real`` from the
    node's real co-occurrence sketch (``singular_value_spectrum``), ``floor`` from
    ``build_null_spectrum`` (a high percentile of null sketches drawn from the node's
    OWN marginal at the node's OWN sample size). Returns the COUNT of positions where
    ``spec_real[k] > margin * floor[k]``.

    This is the fix for effective rank's missing noise model. Effective rank counts
    every linear direction as signal, so it reads a low-support node's token-space
    richness (~min(#words, d)) as a phenotype count -- a 26-doc node reads ~90.
    Parallel analysis compares each direction to a null built at the node's ACTUAL
    ``n_docs``: as support shrinks, finite-sample fluctuation (and hence the floor)
    rises, so a small node clears only a few directions while a large node with
    genuine structure clears more. Empirically (offline planted-topic sweeps) this
    recovers a count that is STABLE across ``n_docs`` for a fixed structure and
    collapses toward 0 for under-supported nodes -- the sample-size awareness
    effective rank lacked.

    COUNT-ALL, not contiguous-from-top: the leading singular direction (position 0)
    is the shared marginal / background co-occurrence pervading every doc, and real
    data spreads variance OUT of it into the phenotype directions, so ``spec_real[0]``
    is actually BELOW the null there (ratio < 1). A contiguous Horn rule would stop
    at position 0 and always return 0; counting all positions above the floor instead
    yields the node's phenotype directions BEYOND that shared background -- which is
    exactly the per-node FOREGROUND K in CHARM's design (the n_bg background block
    already models the shared direction). So a K-topic node reads ~K-1 here (its
    deviations beyond the shared mean), not K.

    ``margin`` (default 2.0): a direction counts only if it clears ``margin x`` the
    null percentile. The projected co-occurrence null slightly UNDER-states the real
    tail (real docs carry within-doc concentration a marginal-i.i.d. null lacks, a
    ~1.2-1.5x multiplicative tail bias), while genuine phenotype directions clear the
    null by 5-50x; a factor-2 margin cleanly separates the two and also excludes the
    below-null position-0 marginal. At strong signal the count is insensitive to the
    exact margin (1.5/2/3 agree); the margin mainly sets sensitivity for weak/small
    nodes. Comparison runs to ``min(len(spec_real), len(floor))``.
    """
    a = np.asarray(spec_real, dtype=np.float64)
    b = np.asarray(floor, dtype=np.float64)
    m = min(a.size, b.size)
    if m == 0:
        return 0
    return int(np.count_nonzero(a[:m] > float(margin) * b[:m]))


def null_percentile_spectrum(null_specs, q=95):
    """Per-position ``q``-th percentile across a set of null spectra -> the floor.

    ``null_specs`` is a list of squared-singular-value spectra (each non-increasing,
    possibly ragged). Pads every spectrum to the longest length with 0.0, then takes
    the ``q``-th percentile at each position. The result is itself non-increasing
    (the per-position percentile of non-increasing columns is non-increasing) and is
    the ``floor`` argument to ``parallel_analysis_rank``. Empty input -> empty list.
    """
    specs = [np.asarray(s, dtype=np.float64) for s in null_specs if len(s) > 0]
    if not specs:
        return []
    length = max(s.size for s in specs)
    padded = np.zeros((len(specs), length), dtype=np.float64)
    for i, s in enumerate(specs):
        padded[i, : s.size] = s
    return list(np.percentile(padded, q, axis=0))


def build_null_spectrum(marginal, lengths, n_docs, V, d, seed, *,
                        reps=5, cap=2000, max_probe=None, q=95, R_rows=None):
    """Driver-side null spectrum for parallel analysis, from a node's OWN marginal.

    Draws ``reps`` null co-occurrence sketches that share the node's sample size and
    token character but carry NO real co-occurrence structure, then returns the
    per-position ``q``-th-percentile SINGULAR-VALUE spectrum (the noise floor for
    ``parallel_analysis_rank``). Each null sketch is built from
    ``n_sim = min(n_docs, cap)`` synthetic docs: a doc's length is drawn from the
    node's own ``lengths`` sample and its tokens i.i.d. from the node's own unigram
    ``marginal`` -- so any co-occurrence in the null is pure finite-sample
    coincidence at the node's actual scale. Projecting each null doc with the SAME
    projection rows (``R_rows``, same ``d``/seed as the real sketch) and normalizing
    exactly as the real path (``_row_normalize_projected``) puts the null and real
    spectra on one scale, so ``parallel_analysis_rank`` can compare them
    position-by-position.

    PER-NODE, not global: the null is drawn from THIS node's marginal + lengths, so
    a node whose tokens are a few dominating concepts gets a very different floor
    from a node spanning many -- preserving the per-node character a global-marginal
    calibration would wash out (the whole point of a per-node K).

    ``cap`` bounds the only cost that grows with the corpus: the finite-sample noise
    floor SATURATES in ``n_docs`` (fluctuation stabilizes within a few thousand
    docs), so simulating past ``cap`` docs buys nothing -- keeping the null to
    bounded driver arithmetic even for a 100k-patient node. Cost is ``reps`` null
    sketches + ``reps`` Gram-eigensolves per node; ``reps=5``/``cap=2000`` is the
    tested default. ``R_rows`` (V, d), if given, is reused across reps and nodes
    (precompute once via ``spectral_init_scalable.precompute_projection_rows``);
    otherwise it is built here from ``seed``. ``marginal`` is a per-token
    count/propensity vector (V,), normalized to a probability here; ``lengths`` is a
    sample of doc lengths (only L >= 2 is drawn, matching the co-occurrence
    contributors). Returns ``[]`` if the node has no usable support (no docs, empty
    marginal, or no length >= 2).

    The projection helpers are imported lazily so the pure estimators above stay
    importable without the sketch module (or its scipy dependency).
    """
    from spark_vi.models.topic.spectral_init_scalable import (
        _project_doc, _r_rows, _row_normalize_projected,
    )
    V = int(V)
    d = int(d)
    n_sim = int(min(int(n_docs), int(cap)))
    if max_probe is None:
        max_probe = d
    marg = np.asarray(marginal, dtype=np.float64)
    marg = np.where(marg > 0, marg, 0.0)
    total = float(marg.sum())
    lens = np.asarray([int(x) for x in np.asarray(lengths).ravel() if int(x) >= 2],
                      dtype=np.int64)
    if n_sim <= 0 or total <= 0.0 or lens.size == 0:
        return []
    probs = marg / total
    if R_rows is None:
        R_rows = _r_rows(np.arange(V), int(seed), d)
    R_rows = np.asarray(R_rows, dtype=np.float64)

    specs = []
    for rep in range(int(reps)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(rep)]))
        QR = np.zeros((V, d), dtype=np.float64)
        p_w = np.zeros(V, dtype=np.float64)
        draw_lens = lens[rng.integers(0, lens.size, size=n_sim)]
        for L in draw_lens:
            Li = int(L)
            if Li < 2:
                continue
            toks = rng.choice(V, size=Li, p=probs)
            idx, cnt = np.unique(toks, return_counts=True)
            qr, pwc = _project_doc(idx, cnt, R_rows[idx])
            QR[idx] += qr
            p_w[idx] += pwc
        Qbar = _row_normalize_projected(QR, p_w)
        specs.append(singular_value_spectrum(Qbar, max_probe))
    return null_percentile_spectrum(specs, q=q)


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
        entry = {
            "participation": float(rep["participation"]),
            "threshold": int(rep["threshold"]),
            "eigengap": int(rep["eigengap"]),
            "n_probed": int(rep["n_probed"]),
            "n_docs": int(rep.get("n_docs", 0)),
        }
        # Parallel-analysis fields (present only when CHARM_PROBE_PARALLEL_ANALYSIS
        # ran): pa_k is the sample-size-aware per-node K; pa_pr_raw is the raw
        # (un-deflated) participation of the SAME spectrum, kept beside it so the
        # readout can show how far the null floor pulled the estimate down.
        if "pa_k" in rep:
            entry["pa_k"] = int(rep["pa_k"])
        if "pa_pr_raw" in rep:
            entry["pa_pr_raw"] = float(rep["pa_pr_raw"])
        out[str(int(node))] = entry
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out
