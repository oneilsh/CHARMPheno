"""LOCAL (pure numpy) gated concentration-recovery experiment (CR-4).

CR-1/2/3 (insight 0038, docs/experiments/0038-concentration-recovery/) validated
that held-out predictive-LL recovers the planted topic concentration for a
NON-gated STM (mean-0 prior, every topic allowed for every document). The real
STM is GATED: each document may only express background topics plus its own
group's foreground topics (hard topic-masking, TopicBlockPartition). This
script asks: does gating change (a) whether held-out-LL still recovers the
planted generative scale, and (b) how faithfully the recovered concentration
matches the planted one, versus running the SAME documents through a
non-gated (trivial all-allowed) sweep?

HS-1 (commit e22bcae) added ``corpus_heldout_scale_sweep_gated`` to
spark_vi.mllib.topic.stm -- the gated analog of the non-gated sweep CR-2/3
used. Its key validation test
(spark-vi/tests/test_heldout_scale_sweep.py::TestArgmaxRecoversPlantedScale)
plants a GATED corpus at a known scalar generative scale s and confirms the
sweep's argmax_c recovers it; THIS script reuses that test's exact planting
recipe (draw eta ~ N(0, s*I) over the doc's allowed topic set, softmax within
that set, sample tokens from theta @ beta) and its exact global_params
construction (lambda = beta * (500*V) + 0.01, so both expElogbeta/inference
and beta_prob/scoring are near-identical to the planted beta -- isolating the
scale-recovery question from beta-estimation error), parametrized over a grid
of planted scales instead of a single s=5.

Sigma is fixed to IDENTITY (R = I), so the swept c IS the generative eta
variance directly and no correlation structure enters the comparison -- this
experiment isolates GATING, not correlation recovery (that is CR-1/2/3's and
the corpus_eta_scale_gated work's territory).

For each planted scale, the SAME planted documents are run through TWO
sweeps: a GATED one (the real TopicBlockPartition, background + per-group
foreground blocks) and a NON-GATED one (a trivial partition whose
background_k == K and no foreground block, so allowed_indices always returns
all K topics regardless of the document's group -- the CR-1/2/3 regime,
replicated here on gated-planted data). Comparing the two argmax_c values and
their recovered concentrations answers whether hard gating helps, hurts, or
is neutral for concentration recovery, and whether non-gated inference leaks
posterior mass onto topics the document was never allowed to express.

Runnable directly: `python scripts/concentration_recovery_gated_experiment.py`.
spark_vi is installed editable (see spark-vi/pyproject.toml) so no sys.path
shim is required for it. ``tests._stm_synth`` (the ``synthetic_gated_corpus``
beta/partition builder) is importable the same way: spark-vi/tests is a
regular package (has __init__.py) and the spark-vi repo root -- where that
package lives -- is already on sys.path via the editable install, exactly as
spark-vi/tests/test_heldout_scale_sweep.py imports it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import digamma

from spark_vi.eval.topic.concentration import doc_concentration
from spark_vi.eval.topic.concentration_recovery import make_shared_beta
from spark_vi.mllib.topic.stm import _gated_mode_theta, corpus_heldout_scale_sweep_gated
from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.stm import _stm_doc_inference
from spark_vi.models.topic.types import STMDocument
from tests._stm_synth import synthetic_gated_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sibling of docs/experiments/0038-concentration-recovery/{results.json,results.md}
# (the non-gated CR-3 results) -- NOT under a "data" path component, so it is
# not swept up by the repo-wide "Generated data" gitignore rule (see the
# non-gated script's DEFAULT_OUT_DIR comment for the same reasoning).
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "0038-concentration-recovery" / "gated"

DEFAULT_SCALES = [1.0, 2.0, 5.0, 10.0]
DEFAULT_C_GRID = [1, 2, 3, 5, 8, 12, 20]
BETA_MODES = ("disjoint", "shared")


def build_structure(*, groups, fg_per_group, bg_k, V, doc_len, seed, beta_mode="disjoint"):
    """Build the gated beta + TopicBlockPartition ONCE (shared across all
    planted scales). Also builds the TRIVIAL non-gated partition
    (background_k = K, no foreground) so the same documents can be run through
    an all-allowed sweep for comparison.

    beta_mode == "disjoint": beta from synthetic_gated_corpus -- each topic
      owns a DISJOINT signature vocabulary block (its own docs are discarded;
      only planted_beta/partition are kept, documents are RE-PLANTED per scale
      by plant_gated_docs). With disjoint vocab, a document's tokens carry ~0
      likelihood for any topic outside its true allowed set, so hard gating is
      expected to matter little for concentration recovery.

    beta_mode == "shared": beta from make_shared_beta -- topics SHARE a common
      term pool (spark_vi.eval.topic.concentration_recovery), so a non-allowed
      topic CAN steal shared-term mass during non-gated inference. The
      TopicBlockPartition is built separately (same K topics, same background +
      per-group foreground layout) so the ONLY thing that changes vs disjoint
      is the vocabulary overlap. This is the realistic regime where gating
      should actually matter.
    """
    if beta_mode not in BETA_MODES:
        raise ValueError(f"build_structure: unknown beta_mode {beta_mode!r} (want one of {BETA_MODES})")

    if beta_mode == "disjoint":
        _throwaway_docs, planted_beta, gated_partition = synthetic_gated_corpus(
            groups=groups, fg_per_group=fg_per_group, bg_k=bg_k, V=V, D=2,
            doc_len=doc_len, bg_frac=0.5, seed=seed,
        )
    else:  # shared
        gated_partition = TopicBlockPartition(
            group_var="g", background_k=bg_k,
            foreground=tuple((g, fg_per_group) for g in groups),
        )
        planted_beta = make_shared_beta(gated_partition.K, V, seed=seed)

    K = gated_partition.K
    nongated_partition = TopicBlockPartition(
        group_var=gated_partition.group_var, background_k=K, foreground=(),
    )
    assert nongated_partition.K == K
    for g in (frozenset(), *(frozenset({g}) for g in groups)):
        assert set(nongated_partition.allowed_indices(g).tolist()) == set(range(K)), (
            "trivial non-gated partition must allow all K topics for any group"
        )
    return planted_beta, gated_partition, nongated_partition


def make_global_params(planted_beta: np.ndarray) -> dict:
    """global_params with an INFORMATIVE lambda (verbatim construction from
    test_heldout_scale_sweep.py::_make_global_params): lambda = beta *
    (500*V) + 0.01 so expElogbeta (inference) and beta_prob (scoring, E[beta])
    are both near-identical to the planted beta. Sigma = I (identity) is
    intentional: R = Sigma / sqrt(outer(diag,diag)) = I, so the swept c IS the
    generative eta-variance and no correlation structure enters -- isolating
    GATING from correlation recovery."""
    K, V = planted_beta.shape
    lam = planted_beta * (500.0 * V) + 0.01
    return {"lambda": lam, "Gamma": np.zeros((1, K)), "Sigma": np.eye(K)}


def plant_gated_docs(planted_beta, gated_partition, *, groups, D, doc_len, s, seed):
    """Plant D gated documents at generative scale s: draw eta ~ N(0, s*I)
    over the doc's allowed (background + own group's foreground) topic set,
    softmax within that set (zeros elsewhere), sample doc_len tokens from
    theta @ planted_beta. Verbatim per-doc recipe from
    test_heldout_scale_sweep.py::TestArgmaxRecoversPlantedScale, parametrized
    by s and by an arbitrary group tuple (that test hardcodes groups A, B).

    Documents cycle through background-only (frozenset()) and each foreground
    group in turn, so the corpus has a mix of doc types like the KEY test.
    Returns (docs, planted_thetas) -- planted_thetas is the (D, K) ground
    truth, used only to measure the planted concentration (never fed to any
    inference/scoring call).
    """
    rng = np.random.default_rng(seed)
    K = gated_partition.K
    V = planted_beta.shape[1]
    groups_cycle = [frozenset()] + [frozenset({g}) for g in groups]

    docs = []
    planted_thetas = np.zeros((D, K))
    for i in range(D):
        g = groups_cycle[i % len(groups_cycle)]
        allowed = np.sort(gated_partition.allowed_indices(g))
        draw = rng.normal(scale=np.sqrt(s), size=allowed.shape[0])
        z = draw - draw.max()
        w = np.exp(z)
        theta = np.zeros(K)
        theta[allowed] = w / w.sum()
        planted_thetas[i] = theta
        toks = rng.choice(V, size=doc_len, p=theta @ planted_beta)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(
            indices=u.astype(np.int32), counts=c.astype(np.float64),
            length=int(c.sum()), x=np.array([1.0]), groups=g,
        ))
    return docs, planted_thetas


def recover_gated(docs, global_params, partition, c: float) -> np.ndarray:
    """Recover (D, K) theta_hat via per-document GATED STM MAP inference at a
    FIXED scale c, on the FULL document (no held-out split -- mirrors
    concentration_recovery.stm_recover_theta's "recover on the whole doc"
    role, as opposed to corpus_heldout_scale_sweep_gated's visible/held split
    which exists only to SCORE c, not to produce a final theta_hat).

    Uses the SAME R-normalization + (1/c)*safe_inverse(R[allowed]) +
    expElogbeta inference + _gated_mode_theta softmax-within-allowed as
    corpus_heldout_scale_sweep_gated (spark_vi.mllib.topic.stm) -- this is
    the only new local inference helper in this experiment; it does not
    re-derive gated inference or scoring, only re-runs the sweep's own E-step
    at one fixed c instead of sweeping and scoring held-out tokens.
    """
    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))

    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))

    rinv_cache: dict[tuple, np.ndarray] = {}
    thetas = np.zeros((len(docs), K))
    for i, doc in enumerate(docs):
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            rinv_cache[key] = Rinv_allowed
        Sigma_inv_allowed = (1.0 / c) * Rinv_allowed

        eta_hat, _, _ = _stm_doc_inference(
            indices=doc.indices, counts=doc.counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
            allowed=allowed, reference=None,
        )
        thetas[i] = _gated_mode_theta(eta_hat, allowed, K)
    return thetas


def _median_concentration(thetas: np.ndarray) -> tuple[float, float]:
    """(median top_mass, median eff_topics) across docs, skipping any
    degenerate (nan) doc_concentration rows."""
    tops, effs = zip(*(doc_concentration(theta) for theta in thetas))
    return float(np.nanmedian(tops)), float(np.nanmedian(effs))


def _leaked_mass(thetas: np.ndarray, docs, gated_partition) -> float:
    """Median mass a recovered theta_hat places on topics OUTSIDE the
    document's TRUE gated-allowed set (background + own group's foreground,
    per gated_partition -- regardless of which partition produced thetas).
    Zero by construction for gated recovery (the gated softmax never assigns
    non-allowed topics any mass); a positive value for non-gated recovery
    means inference spread posterior mass onto topics the document was never
    actually allowed to express."""
    K = gated_partition.K
    leaked = np.zeros(len(docs))
    for i, doc in enumerate(docs):
        true_allowed = set(gated_partition.allowed_indices(doc.groups).tolist())
        not_allowed = [k for k in range(K) if k not in true_allowed]
        leaked[i] = thetas[i, not_allowed].sum() if not_allowed else 0.0
    return float(np.median(leaked))


def _sweep_with_boundary_widen(
    docs, global_params, partition, c_grid, *, holdout_frac, seed, max_extensions=4,
):
    """Run corpus_heldout_scale_sweep_gated, and if argmax_c lands on a grid
    BOUNDARY, widen the grid (double the top / halve the bottom) and retry --
    mirrors the safeguard in
    test_heldout_scale_sweep.py::TestArgmaxRecoversPlantedScale (a boundary
    argmax is "not a validated peak": the true optimum may lie outside the
    grid, or the held-out-LL curve may simply be too flat near the boundary
    for the visible grid to distinguish an interior peak from noise).

    Returns (result, grid_used, boundary_hit) so the caller can report
    whether widening was needed -- a flat/boundary-seeking curve is itself
    diagnostic (see CR-4's non-gated-vs-gated comparison)."""
    grid = list(c_grid)
    result = corpus_heldout_scale_sweep_gated(
        docs, global_params, partition, c_grid=grid,
        holdout_frac=holdout_frac, reference=None, seed=seed,
    )
    boundary_hit = False
    extensions = 0
    while result["argmax_c"] == grid[-1] and extensions < max_extensions:
        boundary_hit = True
        grid = grid + [grid[-1] * 2]
        result = corpus_heldout_scale_sweep_gated(
            docs, global_params, partition, c_grid=grid,
            holdout_frac=holdout_frac, reference=None, seed=seed,
        )
        extensions += 1
    while result["argmax_c"] == grid[0] and extensions < max_extensions:
        boundary_hit = True
        grid = [grid[0] / 2.0] + grid
        result = corpus_heldout_scale_sweep_gated(
            docs, global_params, partition, c_grid=grid,
            holdout_frac=holdout_frac, reference=None, seed=seed,
        )
        extensions += 1
    return result, grid, boundary_hit


def run_cell(
    planted_beta, gated_partition, nongated_partition, global_params, *,
    groups, D, doc_len, s, c_grid, holdout_frac, seed,
) -> dict:
    """Run one planted scale s: plant gated docs, run both the GATED and
    NON-GATED held-out sweeps on the SAME docs, recover theta_hat at each
    argmax, and measure concentration + leaked mass. Returns a
    JSON-serializable dict."""
    docs, planted_thetas = plant_gated_docs(
        planted_beta, gated_partition, groups=groups, D=D, doc_len=doc_len,
        s=s, seed=seed,
    )
    planted_top_mass, planted_eff_topics = _median_concentration(planted_thetas)

    gated_ho, gated_grid, gated_boundary = _sweep_with_boundary_widen(
        docs, global_params, gated_partition, c_grid,
        holdout_frac=holdout_frac, seed=seed,
    )
    nongated_ho, nongated_grid, nongated_boundary = _sweep_with_boundary_widen(
        docs, global_params, nongated_partition, c_grid,
        holdout_frac=holdout_frac, seed=seed,
    )
    gated_argmax_c = gated_ho["argmax_c"]
    nongated_argmax_c = nongated_ho["argmax_c"]

    gated_thetas = recover_gated(docs, global_params, gated_partition, gated_argmax_c)
    nongated_thetas = recover_gated(docs, global_params, nongated_partition, nongated_argmax_c)

    gated_top_mass, gated_eff_topics = _median_concentration(gated_thetas)
    nongated_top_mass, nongated_eff_topics = _median_concentration(nongated_thetas)

    gated_leaked = _leaked_mass(gated_thetas, docs, gated_partition)
    nongated_leaked = _leaked_mass(nongated_thetas, docs, gated_partition)

    return {
        "s": s,
        "planted": {"top_mass": planted_top_mass, "eff_topics": planted_eff_topics},
        "gated": {
            "lls": {str(c): v for c, v in gated_ho["lls"].items()},
            "argmax_c": gated_argmax_c,
            "grid_used": gated_grid,
            "boundary_widened": gated_boundary,
            "recovered_top_mass": gated_top_mass,
            "recovered_eff_topics": gated_eff_topics,
            "abs_err": abs(gated_top_mass - planted_top_mass),
            "leaked_mass": gated_leaked,
            "n_docs": gated_ho["n_docs"],
        },
        "nongated": {
            "lls": {str(c): v for c, v in nongated_ho["lls"].items()},
            "argmax_c": nongated_argmax_c,
            "grid_used": nongated_grid,
            "boundary_widened": nongated_boundary,
            "recovered_top_mass": nongated_top_mass,
            "recovered_eff_topics": nongated_eff_topics,
            "abs_err": abs(nongated_top_mass - planted_top_mass),
            "leaked_mass": nongated_leaked,
            "n_docs": nongated_ho["n_docs"],
        },
    }


def run(
    *,
    groups=("A", "B"), fg_per_group=1, bg_k=2, V=100, D=300, doc_len=55,
    holdout_frac=0.3, seed=0, scales=None, c_grid=None, beta_mode="disjoint",
) -> dict:
    """Run the full sweep (one cell per planted scale s) and return a
    self-describing results dict: {"config": {...}, "cells": [one dict per
    scale, see run_cell]}. A single shared (beta, gated_partition,
    nongated_partition) is used for every cell so the topic structure is held
    constant across the whole sweep -- only the planted scale and, within a
    cell, the gating regime vary. beta_mode selects disjoint vs shared-term
    vocabulary (see build_structure)."""
    from tests._stm_synth import topic_support_jaccard

    scales = DEFAULT_SCALES if scales is None else scales
    c_grid = DEFAULT_C_GRID if c_grid is None else c_grid

    planted_beta, gated_partition, nongated_partition = build_structure(
        groups=groups, fg_per_group=fg_per_group, bg_k=bg_k, V=V,
        doc_len=doc_len, seed=seed, beta_mode=beta_mode,
    )
    global_params = make_global_params(planted_beta)
    # Mean pairwise topic-support Jaccard: 0 = fully disjoint vocab, ->1 = full
    # overlap. Quantifies "how much do topics share terms" so the disjoint vs
    # shared regimes are described, not just asserted.
    beta_jaccard = topic_support_jaccard(planted_beta)

    cells = [
        run_cell(
            planted_beta, gated_partition, nongated_partition, global_params,
            groups=groups, D=D, doc_len=doc_len, s=s, c_grid=c_grid,
            holdout_frac=holdout_frac, seed=seed,
        )
        for s in scales
    ]
    return {
        "config": {
            "beta_mode": beta_mode,
            "beta_support_jaccard": beta_jaccard,
            "groups": list(groups), "fg_per_group": fg_per_group, "bg_k": bg_k,
            "K": gated_partition.K, "V": V, "D": D, "doc_len": doc_len,
            "holdout_frac": holdout_frac, "seed": seed,
            "scales": scales, "c_grid": c_grid,
            "sigma": "identity (R = I; c IS the generative eta-variance)",
        },
        "cells": cells,
    }


def render_markdown_table(results: dict) -> str:
    header = (
        "| s | planted_top_mass | GATED_argmax_c | GATED_recovered_top_mass | GATED_abs_err "
        "| NONGATED_argmax_c | NONGATED_recovered_top_mass | NONGATED_abs_err |"
    )
    sep = "|" + "---|" * 8
    lines = [header, sep]
    for cell in results["cells"]:
        lines.append(
            "| {s} | {planted:.4f} | {gc} | {gr:.4f} | {ge:.4f} "
            "| {nc} | {nr:.4f} | {ne:.4f} |".format(
                s=cell["s"],
                planted=cell["planted"]["top_mass"],
                gc=cell["gated"]["argmax_c"],
                gr=cell["gated"]["recovered_top_mass"],
                ge=cell["gated"]["abs_err"],
                nc=cell["nongated"]["argmax_c"],
                nr=cell["nongated"]["recovered_top_mass"],
                ne=cell["nongated"]["abs_err"],
            )
        )

    footnotes = []
    for cell in results["cells"]:
        for regime in ("gated", "nongated"):
            block = cell[regime]
            if block["boundary_widened"]:
                footnotes.append(
                    f"- s={cell['s']}, {regime}: argmax hit a grid boundary and the grid was "
                    f"widened to {block['grid_used']} (final argmax_c={block['argmax_c']})"
                )
    if footnotes:
        lines.append("")
        lines.append("Boundary-widened cells (argmax was not an interior peak on the base grid):")
        lines.extend(footnotes)

    return "\n".join(lines)


def build_summary(results: dict, *, recover_tol: float = 0.08) -> str:
    """One-paragraph summary answering: (a) does held-out-LL recover the
    planted scale UNDER gating (gated argmax_c near the grid value nearest
    s, and gated abs_err small)? (b) how does GATED recovery error compare to
    NON-GATED recovery error on the SAME documents? (c) does non-gated
    inference spread mass onto topics the document was never allowed to
    express (leaked_mass, eff_topics)?"""
    cells = results["cells"]
    c_grid = results["config"]["c_grid"]

    nearest_hits = []
    for cell in cells:
        nearest = min(c_grid, key=lambda c: abs(c - cell["s"]))
        nearest_hits.append(cell["gated"]["argmax_c"] == nearest)
    all_nearest = all(nearest_hits)

    gated_errs = [c["gated"]["abs_err"] for c in cells]
    nongated_errs = [c["nongated"]["abs_err"] for c in cells]
    mean_gated_err = float(np.mean(gated_errs))
    mean_nongated_err = float(np.mean(nongated_errs))
    all_gated_recover = all(e < recover_tol for e in gated_errs)

    if mean_gated_err < mean_nongated_err - 0.005:
        gating_verdict = (
            f"gating HELPS recovery (lower mean abs error: gated={mean_gated_err:.4f} "
            f"vs non-gated={mean_nongated_err:.4f} on the SAME documents)"
        )
    elif mean_nongated_err < mean_gated_err - 0.005:
        gating_verdict = (
            f"gating HURTS recovery relative to non-gated on these same documents "
            f"(gated={mean_gated_err:.4f} vs non-gated={mean_nongated_err:.4f})"
        )
    else:
        gating_verdict = (
            f"gating does NOT materially change recovery error "
            f"(gated={mean_gated_err:.4f} vs non-gated={mean_nongated_err:.4f})"
        )

    n_gated_widened = sum(1 for c in cells if c["gated"]["boundary_widened"])
    n_nongated_widened = sum(1 for c in cells if c["nongated"]["boundary_widened"])
    if n_nongated_widened > n_gated_widened:
        flatness_bit = (
            f" The non-gated sweep's argmax hit a grid boundary (needing the grid widened) in "
            f"{n_nongated_widened}/{len(cells)} cells vs {n_gated_widened}/{len(cells)} for "
            f"gated -- the non-gated held-out-LL curve is comparatively FLAT across a wide range "
            f"of c, so its argmax is a much less sharply-identified optimum than the gated one, "
            f"even where the two regimes' RECOVERED concentrations end up close."
        )
    else:
        flatness_bit = ""

    mean_gated_leak = float(np.mean([c["gated"]["leaked_mass"] for c in cells]))
    mean_nongated_leak = float(np.mean([c["nongated"]["leaked_mass"] for c in cells]))
    mean_gated_eff = float(np.mean([c["gated"]["recovered_eff_topics"] for c in cells]))
    mean_nongated_eff = float(np.mean([c["nongated"]["recovered_eff_topics"] for c in cells]))
    leaks = mean_nongated_leak > 0.01

    recover_bit = "DOES" if all_gated_recover else "does NOT"
    argmax_bit = (
        "the gated argmax_c lands on the grid value nearest each planted s in every cell"
        if all_nearest else
        "the gated argmax_c does NOT always land on the grid value nearest the planted s"
    )

    leak_bit = (
        f"non-gated inference DOES spread posterior mass onto topics the document was never "
        f"allowed to express (median leaked mass={mean_nongated_leak:.4f} vs "
        f"{mean_gated_leak:.4f} for gated, which is exactly 0 by construction; "
        f"mean recovered eff_topics non-gated={mean_nongated_eff:.3f} vs gated={mean_gated_eff:.3f})"
        if leaks else
        f"non-gated inference does NOT meaningfully leak mass onto non-allowed topics here "
        f"(median leaked mass={mean_nongated_leak:.4f}; eff_topics non-gated="
        f"{mean_nongated_eff:.3f} vs gated={mean_gated_eff:.3f})"
    )

    return (
        f"[beta_mode={results['config']['beta_mode']}, mean topic-support Jaccard="
        f"{results['config']['beta_support_jaccard']:.3f}] Under GATING, held-out predictive-LL "
        f"{recover_bit} recover the planted generative scale across all {len(cells)} planted "
        f"scales (worst-case gated abs error {max(gated_errs):.4f}, tolerance {recover_tol}); "
        f"{argmax_bit}.{flatness_bit} Comparing the SAME documents run through both regimes, "
        f"{gating_verdict}. Finally, {leak_bit}. Sigma was fixed to identity (R = I) throughout, "
        f"so this isolates GATING from correlation-structure recovery."
    )


def render_markdown_doc(results: dict) -> str:
    """Full results.md body (header + table + summary) for one beta_mode."""
    cfg = results["config"]
    table = render_markdown_table(results)
    summary = build_summary(results)
    return (
        f"# Gated concentration-recovery experiment (CR-4) results -- {cfg['beta_mode']} vocabulary\n\n"
        f"Seed: {cfg['seed']}. beta_mode={cfg['beta_mode']} (mean topic-support "
        f"Jaccard={cfg['beta_support_jaccard']:.3f}; 0=disjoint vocab, ->1=full overlap). "
        f"Config: K={cfg['K']} (bg_k={cfg['bg_k']}, groups={cfg['groups']}, "
        f"fg_per_group={cfg['fg_per_group']}), V={cfg['V']}, D={cfg['D']}, doc_len={cfg['doc_len']}, "
        f"holdout_frac={cfg['holdout_frac']}, Sigma={cfg['sigma']}, c_grid={cfg['c_grid']}.\n\n"
        + table + "\n\n## Summary\n\n" + summary + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CR-4: LOCAL gated concentration-recovery experiment -- "
        "does hard topic-gating change held-out-LL's ability to recover the "
        "planted generative scale, vs a non-gated sweep on the same documents? "
        "Runs over disjoint and/or shared-term topic vocabularies.",
    )
    parser.add_argument("--fg-per-group", type=int, default=1)
    parser.add_argument("--bg-k", type=int, default=2)
    parser.add_argument("--V", type=int, default=100)
    parser.add_argument("--D", type=int, default=300)
    parser.add_argument("--doc-len", type=int, default=55)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--beta", choices=(*BETA_MODES, "both"), default="disjoint",
        help="topic-word vocabulary regime: 'disjoint' (each topic owns a "
        "signature block; gating expected to not matter), 'shared' (topics "
        "share a term pool; gating expected to matter), or 'both' (run each "
        "and write a per-mode artifact). Default: disjoint.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"output directory for results-{{mode}}.json/.md (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    modes = list(BETA_MODES) if args.beta == "both" else [args.beta]
    args.out.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        results = run(
            fg_per_group=args.fg_per_group, bg_k=args.bg_k, V=args.V, D=args.D,
            doc_len=args.doc_len, holdout_frac=args.holdout_frac, seed=args.seed,
            beta_mode=mode,
        )
        print(f"===== beta_mode={mode} =====")
        print(render_markdown_table(results))
        print()
        print(build_summary(results))
        print()

        (args.out / f"results-{mode}.json").write_text(json.dumps(results, indent=2) + "\n")
        (args.out / f"results-{mode}.md").write_text(render_markdown_doc(results))


if __name__ == "__main__":
    main()
