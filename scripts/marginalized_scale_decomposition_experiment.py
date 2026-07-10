"""Synthetic MAP-vs-marginalized η-scale decomposition experiment (exp 0046).

The production held-out scale calibration currently selects the generative
variance scale c (Σ = c·R) by a MAP plug-in: for each held-out fraction it
infers the per-document MAP η at each c and scores the visible→held predictive
LL, then takes the smoothed argmax c* (smooth_scale_log_quadratic). A MAP plug-in
under-propagates posterior uncertainty in η, and that regularization bias GROWS
as fewer tokens are visible (larger held-out fraction) -- so the selected c*
DRIFTS with the held-out fraction even though the true generative scale is fixed.
The marginalized estimator (Task 2's sweep_heldout_marginalized) instead
integrates the held-token likelihood over a Laplace posterior on η (n_samples
draws), which should remove that plug-in bias and PIN c* at the true scale
regardless of the held-out fraction.

This script is the SCIENTIFIC EVIDENCE GATE for that claim. On synthetic data
planted at a KNOWN generative η-scale (logistic_normal level = the true η
variance), it sweeps c under BOTH estimators at holdout ∈ {0.5, 0.7, 0.95} and
reports, per regime:

  - the continuous c*(estimator, holdout) via smooth_scale_log_quadratic (NOT
    the quantized grid argmax, which is jittery on the flat LL shelf),
  - residual_drift = max_h c* - min_h c* for each estimator (the headline: MAP
    drift should exceed marginalized drift), and
  - marg_scale_error = mean_h(marginalized c*) - LEVEL (does the marginalized
    estimator recover the planted scale?).

It adds NO inference of its own -- pure numpy orchestration over committed,
unit-tested primitives (make_shared_beta, plant_corpus, sweep_heldout,
sweep_heldout_marginalized, corpus_concentration_summary,
smooth_scale_log_quadratic). Runnable directly:
`python scripts/marginalized_scale_decomposition_experiment.py`. spark_vi is
installed editable (spark-vi/pyproject.toml) so no sys.path shim is required.

Interpretation is written into the results .md WITHOUT overclaiming: if MAP
drift > marginalized drift and marginalized c* ≈ LEVEL, the drift is confirmed
a MAP artifact that the marginalized estimator fixes; if the marginalized
estimator ALSO drifts (Laplace under-dispersion is a known second-order
residual), the residual magnitude is reported plainly -- that number decides
whether a later importance-sampling refinement is needed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spark_vi.eval.topic.concentration_recovery import (
    corpus_concentration_summary,
    make_shared_beta,
    plant_corpus,
    sweep_heldout,
    sweep_heldout_marginalized,
)
from spark_vi.mllib.topic.stm import smooth_scale_log_quadratic

REPO_ROOT = Path(__file__).resolve().parent.parent

# Committed deliverable -> no "data" path component (repo-wide .gitignore
# excludes docs/experiments/data/**), matching concentration_recovery_experiment.
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "0046-marginalized-scale-decomposition"

# The known generative η-scale: logistic_normal η ~ N(0, LEVEL·I). This IS the
# scale the marginalized estimator should recover. 5.0 is a realistic η-scale
# (the project's natural scale is ~7.6; insight 0030).
DEFAULT_LEVEL = 5.0

# Wide, fine, log-spaced c grid so smooth_scale_log_quadratic has a good window
# on the flat LL shelf. Rounded to keep it a clean JSON-serializable grid.
DEFAULT_C_GRID = np.round(np.geomspace(0.5, 20.0, 15), 4).tolist()

# The 0.95 cell (few visible tokens) is where the MAP regularization artifact
# is largest -- the decisive holdout.
DEFAULT_HOLDOUTS = [0.5, 0.7, 0.95]

DEFAULT_N_SAMPLES = 128

# Two cells. clean: fast, the primary evidence. real: matches the production
# corpus shape (K=60, V=5000, doc_len=44), external validity, compute-heavy.
DEFAULT_REGIMES: dict[str, dict] = {
    "clean": dict(K=8, V=400, doc_len=60, D=1000),
    "real": dict(K=60, V=5000, doc_len=44, D=1500),
}


def _cstar(sweep_result: dict) -> tuple[float, float | None]:
    """Continuous c* (and its log-c SE) from a sweep's flat LL shelf via the
    log-quadratic reducer -- never the quantized grid argmax."""
    smooth = smooth_scale_log_quadratic(sweep_result["lls"])
    return float(smooth["c_star"]), smooth["se_log_c"]


def run_regime(
    *,
    K: int,
    V: int,
    doc_len: int,
    D: int,
    level: float,
    c_grid: list,
    holdouts: list,
    n_samples: int,
    seed: int,
) -> dict:
    """Run one regime: plant at a known η-scale, then for each holdout fraction
    sweep c under BOTH the MAP plug-in (sweep_heldout, method="stm") and the
    marginalized estimator (sweep_heldout_marginalized), extracting continuous
    c* for each. Returns a JSON-serializable dict with per-estimator c* (and
    log-c SE) keyed by holdout, per-estimator residual_drift across holdouts,
    and marg_scale_error = mean_h(marginalized c*) - level.

    docs are shared across holdouts and estimators (single plant per regime);
    only the holdout fraction and the estimator vary, so the sweep is a
    controlled comparison. wall_seconds records the regime's compute cost so the
    real-regime time-box is auditable in the results md.
    """
    t0 = time.perf_counter()
    beta = make_shared_beta(K, V, seed=seed)
    docs, theta_true = plant_corpus(
        beta, D=D, doc_len=doc_len, mechanism="logistic_normal", level=level, seed=seed,
    )
    planted = corpus_concentration_summary(theta_true)

    map_cstar: dict[float, float] = {}
    map_se: dict[float, float | None] = {}
    marg_cstar: dict[float, float] = {}
    marg_se: dict[float, float | None] = {}

    for h in holdouts:
        map_sweep = sweep_heldout(
            docs, beta, method="stm", knobs=c_grid, holdout_frac=h, seed=0,
        )
        marg_sweep = sweep_heldout_marginalized(
            docs, beta, knobs=c_grid, holdout_frac=h, seed=0, n_samples=n_samples,
        )
        map_cstar[h], map_se[h] = _cstar(map_sweep)
        marg_cstar[h], marg_se[h] = _cstar(marg_sweep)

    map_residual_drift = float(max(map_cstar.values()) - min(map_cstar.values()))
    marg_residual_drift = float(max(marg_cstar.values()) - min(marg_cstar.values()))
    marg_scale_error = float(np.mean(list(marg_cstar.values())) - level)

    wall_seconds = time.perf_counter() - t0

    return {
        "config": {"K": K, "V": V, "doc_len": doc_len, "D": D, "level": level,
                   "n_samples": n_samples, "seed": seed},
        "planted_top_mass_p50": float(planted["top_mass"]["p50"]),
        "planted_eff_topics_p50": float(planted["eff_topics"]["p50"]),
        "map_cstar": map_cstar,
        "map_se_log_c": map_se,
        "marg_cstar": marg_cstar,
        "marg_se_log_c": marg_se,
        "map_residual_drift": map_residual_drift,
        "marg_residual_drift": marg_residual_drift,
        "marg_scale_error": marg_scale_error,
        "wall_seconds": float(wall_seconds),
    }


def run(
    *,
    regimes: dict[str, dict],
    level: float,
    c_grid: list,
    holdouts: list,
    n_samples: int,
    seed: int,
) -> dict:
    """Run every regime and return {"config": {...}, "regimes": {name: result}}.
    The shared LEVEL, c_grid, holdouts, n_samples, seed are held constant across
    regimes; only the corpus shape (K, V, doc_len, D) varies."""
    return {
        "config": {
            "level": level, "c_grid": c_grid, "holdouts": holdouts,
            "n_samples": n_samples, "seed": seed, "regimes": regimes,
        },
        "regimes": {
            name: run_regime(
                level=level, c_grid=c_grid, holdouts=holdouts,
                n_samples=n_samples, seed=seed, **shape,
            )
            for name, shape in regimes.items()
        },
    }


def render_markdown_table(name: str, cell: dict) -> str:
    """c*-vs-holdout table for one regime: one row per estimator, one column
    per holdout fraction (c* with log-c SE), plus residual_drift."""
    holdouts = sorted(cell["map_cstar"].keys())
    head = "| estimator | " + " | ".join(f"c* @ h={h}" for h in holdouts) + " | residual_drift |"
    sep = "|" + "---|" * (len(holdouts) + 2)

    def row(label: str, cstar: dict, se: dict, drift: float) -> str:
        cells = []
        for h in holdouts:
            s = se[h]
            se_str = f"±{s:.3f}" if s is not None else "±n/a"
            cells.append(f"{cstar[h]:.3f} ({se_str})")
        return f"| {label} | " + " | ".join(cells) + f" | {drift:.3f} |"

    lines = [
        f"### Regime: {name}",
        "",
        f"Planted η-scale (LEVEL) = {cell['config']['level']}; "
        f"planted top_mass p50 = {cell['planted_top_mass_p50']:.4f}; "
        f"corpus K={cell['config']['K']}, V={cell['config']['V']}, "
        f"doc_len={cell['config']['doc_len']}, D={cell['config']['D']}; "
        f"n_samples={cell['config']['n_samples']}; wall={cell['wall_seconds']:.1f}s.",
        "",
        head, sep,
        row("MAP plug-in", cell["map_cstar"], cell["map_se_log_c"], cell["map_residual_drift"]),
        row("marginalized", cell["marg_cstar"], cell["marg_se_log_c"], cell["marg_residual_drift"]),
        "",
        f"marg_scale_error = mean_h(marginalized c*) - LEVEL = {cell['marg_scale_error']:+.3f} "
        f"(marginalized recovers the planted scale iff this ≈ 0).",
    ]
    return "\n".join(lines)


def build_summary(results: dict) -> str:
    """Plain-language read across regimes: does MAP drift exceed marginalized
    drift, and does marginalized c* ≈ LEVEL? States what the data shows without
    overclaiming; flags a material marginalized residual (the Laplace
    under-dispersion second-order term) if present."""
    level = results["config"]["level"]
    bits = []
    for name, cell in results["regimes"].items():
        map_d = cell["map_residual_drift"]
        marg_d = cell["marg_residual_drift"]
        err = cell["marg_scale_error"]
        artifact = "confirmed" if map_d > marg_d else "NOT confirmed"
        bits.append(
            f"[{name}] MAP residual_drift={map_d:.3f} vs marginalized "
            f"residual_drift={marg_d:.3f} (MAP-as-artifact {artifact}); "
            f"marg_scale_error={err:+.3f} against LEVEL={level}"
        )
    return (
        "Headline (per regime): the MAP plug-in c* should drift more across the "
        "holdout fractions than the marginalized c*, and the marginalized c* "
        "should sit near the planted LEVEL. "
        + " || ".join(bits)
        + ". A material marginalized residual_drift (or a nonzero marg_scale_error) "
        "is the known Laplace under-dispersion second-order term and is the number "
        "that decides whether a later importance-sampling refinement is warranted."
    )


def _jsonify(obj):
    """Recursively stringify float dict keys (holdout fractions) for JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp 0046: synthetic MAP-vs-marginalized η-scale decomposition. "
        "Plants at a known generative scale and shows the MAP plug-in c* drifts "
        "with the held-out fraction while the marginalized c* stays pinned.",
    )
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--regime", choices=["clean", "real", "both"], default="both",
        help="which regime(s) to run (default: both).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="fast tiny config (overrides shapes/grid/holdouts) for a quick run.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"output directory for results{{,-real-regime}}.{{json,md}} (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    if args.smoke:
        regimes = {"smoke": dict(K=6, V=120, doc_len=44, D=40)}
        c_grid = np.round(np.geomspace(0.5, 20.0, 6), 4).tolist()
        holdouts = [0.5, 0.9]
        n_samples = 16
    else:
        if args.regime == "both":
            regimes = dict(DEFAULT_REGIMES)
        else:
            regimes = {args.regime: DEFAULT_REGIMES[args.regime]}
        c_grid = DEFAULT_C_GRID
        holdouts = DEFAULT_HOLDOUTS
        n_samples = args.n_samples

    args.out.mkdir(parents=True, exist_ok=True)

    # Run and write per regime so the compute-heavy real regime is persisted the
    # moment it finishes (and so clean-only reruns don't clobber real results).
    file_for = {"clean": "results", "real": "results-real-regime"}

    for name, shape in regimes.items():
        single = run(
            regimes={name: shape}, level=args.level, c_grid=c_grid,
            holdouts=holdouts, n_samples=n_samples, seed=args.seed,
        )
        cell = single["regimes"][name]
        table = render_markdown_table(name, cell)
        summary = build_summary(single)

        print(table)
        print()
        print(summary)
        print()

        stem = file_for.get(name, f"results-{name}")
        (args.out / f"{stem}.json").write_text(json.dumps(_jsonify(single), indent=2) + "\n")
        (args.out / f"{stem}.md").write_text(
            "# exp 0046: MAP-vs-marginalized η-scale decomposition "
            f"({name} regime)\n\n"
            f"Seed: {args.seed}. LEVEL (planted η-scale): {args.level}. "
            f"n_samples: {n_samples}. c-grid: {c_grid}. holdouts: {holdouts}.\n\n"
            + table + "\n\n## Summary\n\n" + summary + "\n"
        )


if __name__ == "__main__":
    main()
