"""Dashboard bundle export. Writes a four-file JSON bundle consumed by
the static Svelte dashboard. Schema defined in
docs/superpowers/specs/2026-05-13-dashboard-design.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _none_if_nan(x: float) -> float | None:
    """Convert NaN to None for JSON serialization (json.dumps emits
    'NaN' literally otherwise, which fails JSON.parse)."""
    x = float(x)
    return None if math.isnan(x) else x


def _none_if_nan_or_none(x: float | None) -> float | None:
    """Like ``_none_if_nan`` but also passes an already-None entry through
    as None, instead of raising. Callers (the dashboard build phases) may
    pre-convert NaN -> None upstream (the same convention used for
    theta_histogram/prominence_hist rows), so predictive-gain per-topic
    scalars can arrive either as raw NaN floats or as already-None entries;
    this accepts both."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _round_floats(arr: np.ndarray, *, decimals: int = 6) -> list:
    return np.round(arr.astype(np.float64), decimals=decimals).tolist()


def select_top_n_by_marginal(
    code_marginals: list[float],
    *,
    top_n: int,
) -> list[int]:
    """Return indices of the top-N codes by marginal token frequency.

    Small-cell suppression is enforced upstream via CountVectorizer.minDF
    at vocab-build time, so the codes reaching this function have already
    cleared the privacy threshold structurally. This function is therefore
    a pure ranking step: sort by marginal descending, take the first top_n.
    """
    if top_n <= 0:
        return []
    marginals = np.asarray(code_marginals, dtype=float)
    order = np.argsort(-marginals)
    return order[:top_n].tolist()


def write_model_and_vocab_bundles(
    *,
    out_dir: Path,
    beta: np.ndarray,           # K × V_full (row-stochastic)
    alpha: np.ndarray,          # length K
    vocab_ids: list[int],       # length V_full; vocab_ids[i] = concept_id at index i
    descriptions: dict[int, str],
    domains: dict[int, str],
    code_marginals: list[float],
    top_n: int,
) -> int:
    """Write model.json and vocab.json. Returns the displayed-vocab width.

    Accepts a row-stochastic β matrix (K × V_full) where each row sums to 1.
    Callers must normalize before passing; this function raises ValueError
    if any row sum deviates from 1.0 by more than 1e-6.

    Trims β columns and vocab metadata to the top-N codes ranked by
    corpus frequency (token marginal). After column trimming (which breaks
    row-stochasticity), β rows are renormalized so each row sums to 1 over
    the surviving trimmed columns.

    Small-cell suppression is enforced upstream via CountVectorizer.minDF
    at vocab-build time; this function applies only top-N ranking.
    """
    beta = np.asarray(beta, dtype=float)
    row_sums = beta.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"write_model_and_vocab_bundles: beta must be row-stochastic; "
            f"got row sums in [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    K, V_full = beta.shape
    keep = select_top_n_by_marginal(
        code_marginals,
        top_n=top_n,
    )
    V_disp = len(keep)
    beta_trimmed = beta[:, keep]
    # Column trim breaks row-stochasticity; renormalize the surviving columns.
    row_sums = beta_trimmed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    beta_trimmed = beta_trimmed / row_sums

    model_payload = {
        "K": int(K),
        "V": int(V_disp),
        "alpha": _round_floats(np.asarray(alpha)),
        "beta": _round_floats(beta_trimmed),
    }
    (out_dir / "model.json").write_text(json.dumps(model_payload, allow_nan=False))

    codes = []
    for new_idx, orig_idx in enumerate(keep):
        cid = vocab_ids[orig_idx]
        codes.append({
            "id": new_idx,
            "code": str(cid),
            "description": descriptions.get(cid, ""),
            "domain": domains.get(cid, "unknown"),
            "corpus_freq": float(code_marginals[orig_idx]),
        })
    (out_dir / "vocab.json").write_text(json.dumps({"codes": codes}, allow_nan=False))

    return V_disp


def write_phenotypes_bundle(
    out_path: Path,
    *,
    npmi: list[float],
    pair_coverage: list[float],
    corpus_prevalence: list[float],
    theta_histogram: list[list[float | None]] | None = None,
    theta_percentiles: list[dict[str, float]] | None = None,
    n_bins: int = 50,
    min_count: int = 20,
    topic_indices: list[int] | None = None,
    labels: list[str] | None = None,
    presence: list[float] | None = None,
    mean_gain: list[float] | None = None,
    depth: list[float] | None = None,
    prominence_hist: list[list[float | None]] | None = None,
    length_corr: list[float] | None = None,
    dedup_gain: list[float] | None = None,
    prominence_bin_edges: list[float] | None = None,
    null_band: dict | None = None,
    observed_delta_range: list[float] | None = None,
    predictive_gain_downdate_audit: dict | None = None,
    predictive_gain_scale: float | None = None,
    predictive_gain_n_docs: int | None = None,
    predictive_gain_smoothing: dict | None = None,
) -> None:
    """Write phenotypes.json.

    pair_coverage[k] is the fraction of top-N pairs that contributed to
    the NPMI calculation for topic k (cleared the joint-count threshold
    in the reference corpus). NaN-valued NPMI topics — those with zero
    scored pairs — should be passed in as NaN and pair_coverage as 0.0;
    downstream readers can use the pair_coverage=0 case to distinguish
    "unrated" from "rated and incoherent".

    topic_indices[k] is the original model-side topic id for displayed
    phenotype k. For LDA the adapter passes 0..K-1; for HDP it passes
    the mask-filtered truncation indices so the advanced view can
    surface them.

    Per-phenotype `label`, `description`, and `quality` start empty and
    are populated by the post-fit labeling step
    (scripts/label_phenotypes.py).

    theta_histogram[k] is the empirical θ histogram for topic k: a list
    of length n_bins where each entry is either a float (bin count) or
    None (bin suppressed due to small-cell, i.e., fewer than min_count
    documents). theta_percentiles[k] is a dict with keys p5, p25, p50,
    p75, p95 giving the corresponding θ percentiles for topic k. When
    theta_histogram is provided, the top-level payload includes
    theta_histogram_bin_edges (n_bins+1 evenly-spaced values from 0.0
    to 1.0) and theta_histogram_min_count (the suppression threshold
    used during histogram construction). None entries in histogram rows
    serialize to JSON null and round-trip cleanly.

    Predictive-gain aggregates (PROVISIONAL schema — Phase-2 of the
    predictive-gain metric, spark_vi.mllib.topic.predictive_gain; Phase-2
    will finalize prominence_range/prominence_bin_edges from a real fitted
    corpus's observed_delta_range rather than the module's placeholder
    default). All of ``presence``, ``mean_gain``, ``depth``,
    ``prominence_hist``, ``length_corr``, ``dedup_gain``,
    ``prominence_bin_edges``, ``null_band``, ``observed_delta_range``,
    ``predictive_gain_downdate_audit``, ``predictive_gain_scale``, and
    ``predictive_gain_n_docs`` default to None; when EVERY one is None
    (LDA/HDP bundles, or an STM build where the enhancement-only
    predictive-gain phase failed) the output carries no "predictive_gain"
    key at all, so existing bundles are byte-unchanged. Otherwise a single
    top-level ``predictive_gain`` object is written, nesting only the
    pieces actually supplied:
      presence[k], mean_gain[k], depth[k], length_corr[k], dedup_gain[k]
        length-K per-topic floats (NaN -> JSON null), aligned with
        ``npmi``/``corpus_prevalence`` (topic_indices order).
      prominence_hist[k]
        length-n_bins per-topic histogram of Delta_k (NaN/None entries ->
        JSON null), the aggregate Delta distribution replacing a theta-hat
        histogram for this view.
      prominence_bin_edges
        length n_bins+1 edges shared by every topic's prominence_hist.
      null_band
        pooled corpus null-band summary dict (mean, std, n, hist, p95),
        passed through unchanged — descriptive only, NOT what presence is
        tested against (presence is a per-document paired test; see
        ``corpus_predictive_gain_gated``'s docstring).
      observed_delta_range
        [min, max] Delta_k actually observed in the corpus that produced
        this bundle — the real-numbers basis for recalibrating
        prominence_range/prominence_bin_edges in a later Phase-2 pass.
      downdate_audit
        {"max_abs_overall": float, "mean_abs_overall": float,
        "n_docs_audited": int} — the cold-vs-fast (``fast=True`` downdate)
        reliability check (``predictive_gain_downdate_audit``), passed
        through unchanged. ``max_abs_overall`` is the single worst-case
        per-document discrepancy; ``mean_abs_overall`` is the mean, over
        finite per-topic entries, of the audit's mean_abs_discrepancy —
        the certification signal for whether the fast downdate's
        aggregates are broadly trustworthy (mean small even if max is
        large -> only rare pathological documents disagree) or broadly
        suspect (mean itself large).
      scale
        the scalar generative-variance scale c the aggregates were computed
        at (``predictive_gain_scale``; the calibrated eta_scale, or the
        unit fallback).
      n_docs
        number of documents that actually contributed to the aggregates
        (``predictive_gain_n_docs``).
    """
    K = len(npmi)
    if theta_percentiles is not None and theta_histogram is None:
        raise ValueError(
            "theta_percentiles requires theta_histogram to be provided",
        )
    if len(pair_coverage) != K:
        raise ValueError(
            f"pair_coverage length {len(pair_coverage)} != npmi length {K}",
        )
    if len(corpus_prevalence) != K:
        raise ValueError(
            f"corpus_prevalence length {len(corpus_prevalence)} != npmi length {K}",
        )
    if theta_histogram is not None:
        if len(theta_histogram) != K:
            raise ValueError(
                f"theta_histogram length {len(theta_histogram)} != npmi length {K}",
            )
        for row_idx, row in enumerate(theta_histogram):
            if len(row) != n_bins:
                raise ValueError(
                    f"theta_histogram row {row_idx} length {len(row)} != n_bins {n_bins}",
                )
            for entry in row:
                if entry is not None and not isinstance(entry, float):
                    raise ValueError(
                        f"theta_histogram row {row_idx} contains non-float, non-None entry: {entry!r}",
                    )
    if theta_percentiles is not None:
        if len(theta_percentiles) != K:
            raise ValueError(
                f"theta_percentiles length {len(theta_percentiles)} != npmi length {K}",
            )
    for _name, _vals in (
        ("presence", presence), ("mean_gain", mean_gain), ("depth", depth),
        ("length_corr", length_corr), ("dedup_gain", dedup_gain),
    ):
        if _vals is not None and len(_vals) != K:
            raise ValueError(
                f"{_name} length {len(_vals)} != npmi length {K}",
            )
    if prominence_hist is not None:
        if len(prominence_hist) != K:
            raise ValueError(
                f"prominence_hist length {len(prominence_hist)} != npmi length {K}",
            )
        for row_idx, row in enumerate(prominence_hist):
            if len(row) != n_bins:
                raise ValueError(
                    f"prominence_hist row {row_idx} length {len(row)} != n_bins {n_bins}",
                )
    labels = labels or [""] * K
    if topic_indices is None:
        topic_indices = list(range(K))
    phenotypes = []
    for k in range(K):
        entry: dict = {
            "id": k,
            "label": labels[k],
            "description": "",
            "quality": None,
            "npmi": _none_if_nan(npmi[k]),
            "pair_coverage": _none_if_nan(pair_coverage[k]),
            "corpus_prevalence": float(corpus_prevalence[k]),
            "original_topic_id": int(topic_indices[k]),
        }
        if theta_histogram is not None:
            entry["theta_histogram"] = theta_histogram[k]
        if theta_percentiles is not None:
            entry["theta_percentiles"] = theta_percentiles[k]
        phenotypes.append(entry)
    payload: dict = {"phenotypes": phenotypes}
    if theta_histogram is not None:
        payload["theta_histogram_bin_edges"] = np.linspace(0, 1, n_bins + 1).tolist()
        payload["theta_histogram_min_count"] = int(min_count)

    # Predictive-gain aggregates (PROVISIONAL — see docstring above): a
    # single nested "predictive_gain" object, present ONLY if at least one
    # of the params below was supplied, so an all-None call (LDA/HDP,
    # or an STM build whose enhancement-only phase failed) leaves the
    # payload byte-unchanged from the pre-existing schema.
    pg_supplied = (
        presence, mean_gain, depth, prominence_hist, length_corr, dedup_gain,
        prominence_bin_edges, null_band, observed_delta_range,
        predictive_gain_downdate_audit, predictive_gain_scale,
        predictive_gain_n_docs, predictive_gain_smoothing,
    )
    if any(v is not None for v in pg_supplied):
        pg: dict = {}
        if presence is not None:
            pg["presence"] = [_none_if_nan_or_none(v) for v in presence]
        if mean_gain is not None:
            pg["mean_gain"] = [_none_if_nan_or_none(v) for v in mean_gain]
        if depth is not None:
            pg["depth"] = [_none_if_nan_or_none(v) for v in depth]
        if length_corr is not None:
            pg["length_corr"] = [_none_if_nan_or_none(v) for v in length_corr]
        if dedup_gain is not None:
            pg["dedup_gain"] = [_none_if_nan_or_none(v) for v in dedup_gain]
        if prominence_hist is not None:
            pg["prominence_hist"] = [
                [None if (v is None or math.isnan(v)) else float(v) for v in row]
                for row in prominence_hist
            ]
        if prominence_bin_edges is not None:
            pg["prominence_bin_edges"] = [float(v) for v in prominence_bin_edges]
        if null_band is not None:
            pg["null_band"] = null_band
        if observed_delta_range is not None:
            pg["observed_delta_range"] = [float(v) for v in observed_delta_range]
        if predictive_gain_downdate_audit is not None:
            pg["downdate_audit"] = predictive_gain_downdate_audit
        if predictive_gain_scale is not None:
            pg["scale"] = float(predictive_gain_scale)
        if predictive_gain_n_docs is not None:
            pg["n_docs"] = int(predictive_gain_n_docs)
        if predictive_gain_smoothing is not None:
            # Self-describing provenance: whether the background-unigram smoother
            # was active for THIS bundle (marginal supplied), its lambda, and the
            # uniform backoff floor. Lets a reader confirm smoothing from the
            # bundle alone -- no log line / mtime archaeology needed.
            pg["smoothing"] = predictive_gain_smoothing
        payload["predictive_gain"] = pg

    out_path.write_text(json.dumps(payload, allow_nan=False))


def write_covariate_effects(
    *,
    out_dir: Path,
    Gamma: np.ndarray,
    covariate_names: list[str],
    K: int,
    P: int,
) -> None:
    """Write STM-specific bundle artifact: per-covariate effect matrix Γ̂.

    (Formerly named ``adapt_stm`` — renamed to end the collision with
    ``model_adapter.adapt_stm``, which builds the uniform DashboardExport. This
    function only writes the Γ sidecar that powers the dashboard's client-side
    covariate conditioning, per ADR 0028.)

    Schema for covariate_effects.json:
        [{"covariate": "<name>", "per_topic": [γ_0, γ_1, ..., γ_{K-1}]}, ...]

    One row per covariate (length P); each row carries K topic-effect values.
    Companion bundle artifacts (vocab, β, α-equivalent) come from the existing
    write_model_and_vocab_bundles path; this function only adds the Γ piece.
    """
    if Gamma.shape != (P, K):
        raise ValueError(
            f"write_covariate_effects: Gamma shape mismatch — got {Gamma.shape}, expected ({P}, {K})"
        )
    if len(covariate_names) != P:
        raise ValueError(
            f"write_covariate_effects: covariate_names length {len(covariate_names)} != P={P}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"covariate": name, "per_topic": Gamma[p].tolist()}
        for p, name in enumerate(covariate_names)
    ]
    (out_dir / "covariate_effects.json").write_text(json.dumps(payload, indent=2))
