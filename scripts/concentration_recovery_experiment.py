"""Factorial driver for the local (pure numpy) concentration-recovery
diagnostic (CR-3).

CR-1 (spark_vi.eval.topic.concentration_recovery: make_shared_beta,
plant_corpus, stm_recover_theta, lda_recover_theta, lda_optimize_alpha,
corpus_concentration_summary) and CR-2 (sweep_heldout / the held-out
predictive-LL gold standard) built and unit-validated the planting/recovery
primitives. This script adds NO inference of its own -- it is pure
orchestration: for every (planting mechanism, concentration level) cell it

  1. plants a corpus and measures the PLANTED concentration (top_mass p50),
  2. sweeps STM recovery over a Sigma-scale grid `c` and LDA recovery over a
     Dirichlet `alpha` grid, recording the recovered top_mass at each knob,
  3. runs the held-out predictive-LL gold standard for both families (argmax
     knob over the SAME grids) and reports the recovered top_mass + absolute
     error against the planted value at that knob,
  4. runs LDA's own alpha-optimization (beta frozen) as a reference point:
     does LDA's internal M-step pick a MORE concentrated alpha than the
     held-out-LL optimum ("reads hot")?

The two planting mechanisms are Bayesian-dual priors: logistic_normal
(STM's own generative form, eta ~ N(0, level*I), theta = softmax(eta)) and
dirichlet (LDA's own generative form, theta ~ Dirichlet(level*ones(K))). The
factorial therefore also answers a "matched prior" question: does the family
whose prior matches the planting mechanism recover more faithfully?

Runnable directly: `python scripts/concentration_recovery_experiment.py`.
spark_vi is installed editable (see spark-vi/pyproject.toml) so no sys.path
shim is required, unlike the spark-submit cloud drivers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spark_vi.eval.topic.concentration_recovery import (
    corpus_concentration_summary,
    lda_optimize_alpha,
    lda_recover_theta,
    make_shared_beta,
    plant_corpus,
    stm_recover_theta,
    sweep_heldout,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# NOTE: the brief's suggested output path
# (docs/experiments/data/0038-concentration-recovery/) sits under a
# directory literally named "data", which .gitignore excludes repo-wide
# ("Generated data -- never in repo"). This diagnostic's results ARE meant
# to be committed (they are the deliverable), so the default output lives
# one level up, without a "data" path component, to avoid being silently
# gitignored.
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "0038-concentration-recovery"

DEFAULT_MECHANISM_LEVELS: dict[str, list[float]] = {
    # logistic_normal: eta ~ N(0, level*I); LARGER level -> peakier.
    "logistic_normal": [1, 3, 5, 9],
    # dirichlet: theta ~ Dirichlet(level*ones(K)); SMALLER level -> peakier.
    "dirichlet": [3.0, 1.0, 0.3, 0.1],
}
DEFAULT_C_GRID = [1, 2, 3, 5, 8, 12]
DEFAULT_ALPHA_GRID = [0.02, 0.05, 0.1, 0.3, 1.0, 3.0]


def _top_mass_p50(theta: np.ndarray) -> float:
    """Median top_mass (max_k theta_k) across documents, via the shared
    corpus_concentration_summary readout (CR-1)."""
    return corpus_concentration_summary(theta)["top_mass"]["p50"]


def _eff_topics_p50(theta: np.ndarray) -> float:
    return corpus_concentration_summary(theta)["eff_topics"]["p50"]


def run_cell(
    beta: np.ndarray,
    K: int,
    *,
    mechanism: str,
    level: float,
    D: int,
    doc_len: int,
    c_grid: list,
    alpha_grid: list,
    holdout_frac: float,
    seed: int,
) -> dict:
    """Run one (mechanism, level) cell of the factorial.

    Plants a corpus, sweeps STM (c_grid) and LDA (alpha_grid) recovery,
    scores both families' held-out-LL gold standard over the SAME grids,
    and runs LDA's own alpha optimization (beta frozen). Returns a
    JSON-serializable dict; see the module docstring for what each step
    measures.
    """
    docs, theta_true = plant_corpus(
        beta, D=D, doc_len=doc_len, mechanism=mechanism, level=level, seed=seed,
    )
    planted_top_mass = _top_mass_p50(theta_true)
    planted_eff_topics = _eff_topics_p50(theta_true)

    stm_curve = {c: _top_mass_p50(stm_recover_theta(docs, beta, c=c)) for c in c_grid}
    lda_curve = {
        alpha: _top_mass_p50(lda_recover_theta(docs, beta, alpha=alpha)) for alpha in alpha_grid
    }

    stm_ho = sweep_heldout(
        docs, beta, method="stm", knobs=c_grid, holdout_frac=holdout_frac, seed=seed,
    )
    stm_argmax_c = stm_ho["argmax_knob"]
    stm_recovered = stm_curve[stm_argmax_c]
    stm_abs_err = abs(stm_recovered - planted_top_mass)

    lda_ho = sweep_heldout(
        docs, beta, method="lda", knobs=alpha_grid, holdout_frac=holdout_frac, seed=seed,
    )
    lda_argmax_alpha = lda_ho["argmax_knob"]
    lda_recovered = lda_curve[lda_argmax_alpha]
    lda_abs_err = abs(lda_recovered - planted_top_mass)

    alpha_opt = lda_optimize_alpha(docs, beta, K)
    theta_hat_opt = lda_recover_theta(docs, beta, alpha=alpha_opt)
    lda_opt_recovered = _top_mass_p50(theta_hat_opt)
    lda_opt_err = abs(lda_opt_recovered - planted_top_mass)

    return {
        "mechanism": mechanism,
        "level": level,
        "planted": {"top_mass_p50": planted_top_mass, "eff_topics_p50": planted_eff_topics},
        "stm_curve": {str(c): v for c, v in stm_curve.items()},
        "lda_curve": {str(a): v for a, v in lda_curve.items()},
        "stm_heldout": {
            "lls": {str(c): v for c, v in stm_ho["lls"].items()},
            "argmax_c": stm_argmax_c,
            "recovered_top_mass_p50": stm_recovered,
            "abs_err": stm_abs_err,
        },
        "lda_heldout": {
            "lls": {str(a): v for a, v in lda_ho["lls"].items()},
            "argmax_alpha": lda_argmax_alpha,
            "recovered_top_mass_p50": lda_recovered,
            "abs_err": lda_abs_err,
        },
        "lda_opt_alpha": {
            "alpha": alpha_opt.tolist(),
            "alpha_mean": float(np.mean(alpha_opt)),
            "recovered_top_mass_p50": lda_opt_recovered,
            "abs_err": lda_opt_err,
        },
    }


def run(
    *,
    K: int,
    V: int,
    D: int,
    doc_len: int,
    holdout_frac: float,
    seed: int,
    mechanism_levels: dict[str, list[float]],
    c_grid: list,
    alpha_grid: list,
) -> dict:
    """Run the full factorial (all mechanisms x all levels) and return a
    self-describing results dict: {"config": {...}, "cells": [one dict per
    (mechanism, level), see run_cell]}. A single shared beta (make_shared_beta)
    is used for every cell so the topic-term matrix is held constant across
    the whole factorial -- only the planting mechanism/level and the
    recovery knob vary.
    """
    beta = make_shared_beta(K, V, seed=seed)
    cells = [
        run_cell(
            beta, K, mechanism=mechanism, level=level, D=D, doc_len=doc_len,
            c_grid=c_grid, alpha_grid=alpha_grid, holdout_frac=holdout_frac, seed=seed,
        )
        for mechanism, levels in mechanism_levels.items()
        for level in levels
    ]
    return {
        "config": {
            "K": K, "V": V, "D": D, "doc_len": doc_len,
            "holdout_frac": holdout_frac, "seed": seed,
            "mechanism_levels": mechanism_levels,
            "c_grid": c_grid, "alpha_grid": alpha_grid,
        },
        "cells": cells,
    }


def render_markdown_table(results: dict) -> str:
    """One row per (mechanism, level) cell, columns per the CR-3 brief."""
    header = (
        "| mechanism | level | planted_top_mass | STM_argmax_c | STM_top_mass | STM_abs_err "
        "| LDA_argmax_alpha | LDA_top_mass | LDA_abs_err | LDA_opt_top_mass | LDA_opt_err |"
    )
    sep = "|" + "---|" * 11
    lines = [header, sep]
    for cell in results["cells"]:
        lines.append(
            "| {mech} | {level} | {planted:.4f} | {sc} | {sr:.4f} | {se:.4f} "
            "| {la} | {lr:.4f} | {le:.4f} | {lor:.4f} | {loe:.4f} |".format(
                mech=cell["mechanism"],
                level=cell["level"],
                planted=cell["planted"]["top_mass_p50"],
                sc=cell["stm_heldout"]["argmax_c"],
                sr=cell["stm_heldout"]["recovered_top_mass_p50"],
                se=cell["stm_heldout"]["abs_err"],
                la=cell["lda_heldout"]["argmax_alpha"],
                lr=cell["lda_heldout"]["recovered_top_mass_p50"],
                le=cell["lda_heldout"]["abs_err"],
                lor=cell["lda_opt_alpha"]["recovered_top_mass_p50"],
                loe=cell["lda_opt_alpha"]["abs_err"],
            )
        )
    return "\n".join(lines)


def build_summary(results: dict, *, recover_tol: float = 0.08) -> str:
    """One-paragraph summary answering the three CR-3 questions: (1) does
    held-out-LL recover planted concentration across all mechanisms/levels
    (abs_err < recover_tol for every cell, both families)? (2) which family
    has the lower mean abs error overall, and per planting mechanism (the
    "matched prior" question: STM better on logistic_normal-planted,
    LDA better on dirichlet-planted)? (3) does LDA's own alpha-optimization
    over-concentrate (higher recovered top_mass) relative to the held-out-LL
    optimum?
    """
    cells = results["cells"]
    stm_errs = [c["stm_heldout"]["abs_err"] for c in cells]
    lda_errs = [c["lda_heldout"]["abs_err"] for c in cells]

    all_recover = all(e < recover_tol for e in stm_errs) and all(e < recover_tol for e in lda_errs)
    max_err = max(stm_errs + lda_errs)

    mean_stm = float(np.mean(stm_errs))
    mean_lda = float(np.mean(lda_errs))
    overall_winner = "STM" if mean_stm < mean_lda else "LDA"

    by_mech: dict[str, dict[str, float]] = {}
    for mech in results["config"]["mechanism_levels"]:
        mech_cells = [c for c in cells if c["mechanism"] == mech]
        m_stm = float(np.mean([c["stm_heldout"]["abs_err"] for c in mech_cells]))
        m_lda = float(np.mean([c["lda_heldout"]["abs_err"] for c in mech_cells]))
        by_mech[mech] = {"stm": m_stm, "lda": m_lda}

    matched_prior_holds = (
        by_mech.get("logistic_normal", {}).get("stm", float("inf"))
        < by_mech.get("logistic_normal", {}).get("lda", float("-inf"))
        and by_mech.get("dirichlet", {}).get("lda", float("inf"))
        < by_mech.get("dirichlet", {}).get("stm", float("-inf"))
    )

    opt_hotter = [
        c["lda_opt_alpha"]["recovered_top_mass_p50"] - c["lda_heldout"]["recovered_top_mass_p50"]
        for c in cells
    ]
    mean_opt_delta = float(np.mean(opt_hotter))
    opt_reads_hot = mean_opt_delta > 0.01
    opt_reads_cold = mean_opt_delta < -0.01

    mech_bits = "; ".join(
        f"{mech}-planted: STM err={vals['stm']:.4f}, LDA err={vals['lda']:.4f}"
        for mech, vals in by_mech.items()
    )

    if opt_reads_hot:
        opt_verdict = (
            f"LDA's own alpha-optimization over-concentrates relative to the held-out-LL "
            f"optimum (mean top_mass delta +{mean_opt_delta:.4f}, i.e. reads hotter/peakier)"
        )
    elif opt_reads_cold:
        opt_verdict = (
            f"LDA's own alpha-optimization UNDER-concentrates relative to the held-out-LL "
            f"optimum (mean top_mass delta {mean_opt_delta:.4f}, i.e. reads cooler/more diffuse)"
        )
    else:
        opt_verdict = (
            f"LDA's own alpha-optimization tracks the held-out-LL optimum closely "
            f"(mean top_mass delta {mean_opt_delta:+.4f})"
        )

    return (
        f"Held-out predictive-LL {'DOES' if all_recover else 'does NOT'} recover the planted "
        f"concentration across all {len(cells)} (mechanism, level) cells for both families "
        f"(worst-case abs error {max_err:.4f}, tolerance {recover_tol}). "
        f"Overall, {overall_winner} has the lower mean absolute error "
        f"(STM mean={mean_stm:.4f} vs LDA mean={mean_lda:.4f}). Split by planting mechanism, "
        f"{mech_bits} -- the 'matched prior' effect (STM wins on logistic_normal-planted data, "
        f"LDA wins on dirichlet-planted data) {'HOLDS' if matched_prior_holds else 'does NOT hold'} "
        f"in this run. Finally, {opt_verdict}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CR-3: factorial concentration-recovery experiment "
        "(mechanism x level x {STM Sigma-scale sweep, LDA alpha sweep} "
        "+ held-out-LL gold standard + LDA's own alpha-optimization).",
    )
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--V", type=int, default=400)
    parser.add_argument("--D", type=int, default=300)
    parser.add_argument("--doc-len", type=int, default=60)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"output directory for results.json/results.md (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    results = run(
        K=args.K, V=args.V, D=args.D, doc_len=args.doc_len,
        holdout_frac=args.holdout_frac, seed=args.seed,
        mechanism_levels=DEFAULT_MECHANISM_LEVELS,
        c_grid=DEFAULT_C_GRID, alpha_grid=DEFAULT_ALPHA_GRID,
    )

    table = render_markdown_table(results)
    summary = build_summary(results)

    print(table)
    print()
    print(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "results.md").write_text(
        "# Concentration-recovery experiment (CR-3) results\n\n"
        f"Seed: {args.seed}. Config: K={args.K}, V={args.V}, D={args.D}, "
        f"doc_len={args.doc_len}, holdout_frac={args.holdout_frac}.\n\n"
        + table + "\n\n## Summary\n\n" + summary + "\n"
    )


if __name__ == "__main__":
    main()
