"""Factorial driver for the CO-FIT-beta concentration-recovery diagnostic (CR-4).

The frozen-beta diagnostic (scripts/concentration_recovery_experiment.py,
docs/experiments/0038-concentration-recovery) planted documents at a KNOWN
per-document concentration over a KNOWN shared-vocab beta and recovered the
concentration with beta FROZEN at truth. It confirmed insight 0038: held-out
predictive-LL recovers the true concentration, and LDA's alpha-optimization
does NOT read hot -- so the real-data STM-vs-LDA peakiness gap (STM top_mass
0.269 vs LDA 0.513, exps 0033/0034) is NOT an alpha-inference artifact.

0038 explicitly left one test un-run ("What this does NOT claim"): a synthetic
run where LDA and STM each CO-FIT beta rather than freezing it at truth. That
is this experiment. The hypothesis (0038's proposed mechanism): when beta is
learned, LDA's Dirichlet document-sparsity pressure carves a SHARPER, more
document-specific beta, raising per-document top_mass (peakier patients), while
STM's logistic-normal stays more blended -- reproducing the real-data gap via
beta-co-adaptation.

For every (planting mechanism, concentration level) cell and each regime this
script:

  1. plants a TRAIN corpus and a disjoint TEST corpus at the same
     mechanism/level over one shared beta_true, and measures the planted
     concentration (top_mass p50 on the test thetas);
  2. FROZEN baseline (reuses the 0038 machinery): sweeps the concentration knob
     with beta frozen at truth on the test docs, held-out-LL argmax picks the
     knob; records recovered top_mass. beta-recovery error is 0 by construction
     and beta-sharpness is that of beta_true;
  3. CO-FIT STM: for each Sigma scale c, learns beta on TRAIN under N(0, c*I),
     scores document-completion held-out-LL on TEST under the learned beta;
     argmax c is the held-out-LL-calibrated knob (document-completion
     evaluation, Wallach et al. 2009). At the argmax: beta-recovery error vs
     beta_true (Hungarian match), beta-sharpness, and the resulting test-doc
     theta concentration;
  4. CO-FIT LDA: the same held-out-LL sweep over Dirichlet alpha (apples-to-
     apples with STM), PLUS LDA's own alpha-optimization co-fit (learn beta and
     empirical-Bayes optimize alpha jointly) as the real-data-matching
     reference.

The headline comparison is CO-FIT vs FROZEN top_mass and beta-sharpness, STM
vs LDA, at the held-out-LL-calibrated knob: does co-fitting beta open a
peakiness gap (LDA sharper beta + peakier theta) that frozen beta did not?

Runnable directly: `python scripts/cofit_beta_concentration_experiment.py`
(clean regime) or `--regime real`. spark_vi is installed editable so no
sys.path shim is required.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spark_vi.eval.topic.concentration_recovery import (
    beta_recovery_error,
    beta_sharpness,
    corpus_concentration_summary,
    lda_cofit_beta,
    lda_recover_theta,
    make_shared_beta,
    plant_corpus,
    stm_recover_theta,
    sweep_heldout,
    sweep_heldout_cofit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "0045-cofit-beta-concentration"

# Same mechanism/level grid as the frozen-beta 0038 experiment so the FROZEN
# baseline columns line up exactly with docs/experiments/0038-*.
DEFAULT_MECHANISM_LEVELS: dict[str, list[float]] = {
    "logistic_normal": [1, 3, 5, 9],
    "dirichlet": [3.0, 1.0, 0.3, 0.1],
}
DEFAULT_C_GRID = [1, 2, 3, 5, 8]
DEFAULT_ALPHA_GRID = [0.05, 0.1, 0.3, 1.0, 3.0]

REGIMES = {
    # (K, V, D_train, D_test, doc_len, n_em_iter)
    "clean": dict(K=8, V=400, D_train=300, D_test=200, doc_len=60, n_em_iter=60),
    "real": dict(K=60, V=5000, D_train=600, D_test=300, doc_len=44, n_em_iter=50),
    # Large-D control: same real K/V/doc_len but 5x the training corpus, to test
    # whether the co-fit concentration collapse is a small-sample beta-
    # identifiability failure (curable by data) or a fundamental co-fit effect.
    "real_bigD": dict(K=60, V=5000, D_train=3000, D_test=1000, doc_len=44, n_em_iter=50),
}

# --subset runs only two representative cells (one peaky per mechanism) so the
# expensive large-D control finishes in a couple of cells rather than eight.
SUBSET_MECHANISM_LEVELS: dict[str, list[float]] = {
    "logistic_normal": [5],
    "dirichlet": [0.1],
}


def _top_mass_p50(theta: np.ndarray) -> float:
    return corpus_concentration_summary(theta)["top_mass"]["p50"]


def _eff_topics_p50(theta: np.ndarray) -> float:
    return corpus_concentration_summary(theta)["eff_topics"]["p50"]


def run_cell(
    beta_true: np.ndarray, K: int, V: int, *, mechanism: str, level: float,
    D_train: int, D_test: int, doc_len: int, n_em_iter: int,
    c_grid: list, alpha_grid: list, top_k: int, holdout_frac: float, seed: int,
) -> dict:
    """One (mechanism, level) cell: frozen baseline + co-fit STM/LDA. See the
    module docstring for what each block measures."""
    train_docs, _ = plant_corpus(
        beta_true, D=D_train, doc_len=doc_len, mechanism=mechanism, level=level, seed=seed,
    )
    test_docs, theta_true_test = plant_corpus(
        beta_true, D=D_test, doc_len=doc_len, mechanism=mechanism, level=level,
        seed=seed + 10_000,
    )
    planted_top_mass = _top_mass_p50(theta_true_test)
    planted_eff_topics = _eff_topics_p50(theta_true_test)
    true_sharp = beta_sharpness(beta_true, top_k=top_k)

    # --- FROZEN baseline (beta frozen at truth on the TEST docs). ---
    frz_stm = sweep_heldout(test_docs, beta_true, method="stm", knobs=c_grid,
                            holdout_frac=holdout_frac, seed=seed)
    frz_stm_c = frz_stm["argmax_knob"]
    frz_stm_tm = _top_mass_p50(stm_recover_theta(test_docs, beta_true, c=frz_stm_c))
    frz_lda = sweep_heldout(test_docs, beta_true, method="lda", knobs=alpha_grid,
                            holdout_frac=holdout_frac, seed=seed)
    frz_lda_a = frz_lda["argmax_knob"]
    frz_lda_tm = _top_mass_p50(lda_recover_theta(test_docs, beta_true, alpha=frz_lda_a))

    # --- CO-FIT STM (learn beta on train, calibrate c by held-out-LL on test). ---
    stm_sweep = sweep_heldout_cofit(train_docs, test_docs, K, V, method="stm",
                                    knobs=c_grid, n_em_iter=n_em_iter,
                                    holdout_frac=holdout_frac, seed=seed)
    stm_c = stm_sweep["argmax_knob"]
    stm_beta = stm_sweep["beta_hat"][stm_c]
    stm_tm = _top_mass_p50(stm_recover_theta(test_docs, stm_beta, c=stm_c))
    stm_rec = beta_recovery_error(beta_true, stm_beta)
    stm_sharp = beta_sharpness(stm_beta, top_k=top_k)

    # --- CO-FIT LDA held-out-LL sweep (apples-to-apples with STM). ---
    lda_sweep = sweep_heldout_cofit(train_docs, test_docs, K, V, method="lda",
                                    knobs=alpha_grid, n_em_iter=n_em_iter,
                                    holdout_frac=holdout_frac, seed=seed)
    lda_a = lda_sweep["argmax_knob"]
    lda_beta = lda_sweep["beta_hat"][lda_a]
    lda_tm = _top_mass_p50(lda_recover_theta(test_docs, lda_beta, alpha=lda_a))
    lda_rec = beta_recovery_error(beta_true, lda_beta)
    lda_sharp = beta_sharpness(lda_beta, top_k=top_k)

    # --- CO-FIT LDA with its own alpha-optimization (real-data-matching). ---
    lda_opt_beta, alpha_opt = lda_cofit_beta(
        train_docs, K, V, alpha=1.0, n_em_iter=n_em_iter, seed=seed, optimize_alpha=True,
    )
    lda_opt_tm = _top_mass_p50(lda_recover_theta(test_docs, lda_opt_beta, alpha=alpha_opt))
    lda_opt_rec = beta_recovery_error(beta_true, lda_opt_beta)
    lda_opt_sharp = beta_sharpness(lda_opt_beta, top_k=top_k)

    return {
        "mechanism": mechanism,
        "level": level,
        "planted": {"top_mass_p50": planted_top_mass, "eff_topics_p50": planted_eff_topics},
        "true_beta_sharpness": true_sharp,
        "frozen": {
            "stm": {"argmax_c": frz_stm_c, "top_mass_p50": frz_stm_tm},
            "lda": {"argmax_alpha": frz_lda_a, "top_mass_p50": frz_lda_tm},
        },
        "cofit_stm": {
            "lls": {str(k): v for k, v in stm_sweep["lls"].items()},
            "argmax_c": stm_c,
            "top_mass_p50": stm_tm,
            "beta_recovery": {"mean_l1": stm_rec["mean_l1"], "mean_cos_dist": stm_rec["mean_cos_dist"]},
            "beta_sharpness": stm_sharp,
        },
        "cofit_lda_heldout": {
            "lls": {str(k): v for k, v in lda_sweep["lls"].items()},
            "argmax_alpha": lda_a,
            "top_mass_p50": lda_tm,
            "beta_recovery": {"mean_l1": lda_rec["mean_l1"], "mean_cos_dist": lda_rec["mean_cos_dist"]},
            "beta_sharpness": lda_sharp,
        },
        "cofit_lda_alpha_opt": {
            "alpha_mean": float(np.mean(alpha_opt)),
            "top_mass_p50": lda_opt_tm,
            "beta_recovery": {"mean_l1": lda_opt_rec["mean_l1"], "mean_cos_dist": lda_opt_rec["mean_cos_dist"]},
            "beta_sharpness": lda_opt_sharp,
        },
    }


def run(*, regime: str, mechanism_levels: dict[str, list[float]], c_grid: list,
        alpha_grid: list, top_k: int, holdout_frac: float, seed: int) -> dict:
    cfg = REGIMES[regime]
    K, V = cfg["K"], cfg["V"]
    beta_true = make_shared_beta(K, V, seed=seed)
    cells = []
    for mechanism, levels in mechanism_levels.items():
        for level in levels:
            t0 = time.time()
            cell = run_cell(
                beta_true, K, V, mechanism=mechanism, level=level,
                D_train=cfg["D_train"], D_test=cfg["D_test"], doc_len=cfg["doc_len"],
                n_em_iter=cfg["n_em_iter"], c_grid=c_grid, alpha_grid=alpha_grid,
                top_k=top_k, holdout_frac=holdout_frac, seed=seed,
            )
            cell["_seconds"] = round(time.time() - t0, 1)
            print(f"  {mechanism} level={level}: {cell['_seconds']}s", flush=True)
            cells.append(cell)
    return {
        "config": {
            "regime": regime, **cfg, "top_k": top_k,
            "holdout_frac": holdout_frac, "seed": seed,
            "mechanism_levels": mechanism_levels, "c_grid": c_grid, "alpha_grid": alpha_grid,
        },
        "cells": cells,
    }


def render_markdown_table(results: dict) -> str:
    header = (
        "| mechanism | level | planted | FROZEN STM tm | FROZEN LDA tm "
        "| COFIT STM tm | COFIT LDA(HO) tm | COFIT LDA(aopt) tm "
        "| STM betasharp topk | LDA(HO) betasharp topk | LDA(aopt) betasharp topk "
        "| true betasharp topk | STM betacos | LDA(HO) betacos |"
    )
    sep = "|" + "---|" * 15
    lines = [header, sep]
    for c in results["cells"]:
        lines.append(
            "| {m} | {lv} | {pl:.3f} | {fs:.3f} | {fl:.3f} | {cs:.3f} | {cl:.3f} | {co:.3f} "
            "| {ss:.3f} | {ls:.3f} | {los:.3f} | {ts:.3f} | {scos:.3f} | {lcos:.3f} |".format(
                m=c["mechanism"], lv=c["level"], pl=c["planted"]["top_mass_p50"],
                fs=c["frozen"]["stm"]["top_mass_p50"], fl=c["frozen"]["lda"]["top_mass_p50"],
                cs=c["cofit_stm"]["top_mass_p50"], cl=c["cofit_lda_heldout"]["top_mass_p50"],
                co=c["cofit_lda_alpha_opt"]["top_mass_p50"],
                ss=c["cofit_stm"]["beta_sharpness"]["top_k_mass"],
                ls=c["cofit_lda_heldout"]["beta_sharpness"]["top_k_mass"],
                los=c["cofit_lda_alpha_opt"]["beta_sharpness"]["top_k_mass"],
                ts=c["true_beta_sharpness"]["top_k_mass"],
                scos=c["cofit_stm"]["beta_recovery"]["mean_cos_dist"],
                lcos=c["cofit_lda_heldout"]["beta_recovery"]["mean_cos_dist"],
            )
        )
    return "\n".join(lines)


def build_summary(results: dict) -> str:
    """Verdict: does co-fitting beta open an STM-vs-LDA peakiness gap (LDA
    sharper beta + peakier theta) at the held-out-LL-calibrated knob, that the
    frozen baseline does not?"""
    cells = results["cells"]

    def mean(key_fn):
        return float(np.mean([key_fn(c) for c in cells]))

    # theta peakiness at the calibrated knob.
    frz_stm_tm = mean(lambda c: c["frozen"]["stm"]["top_mass_p50"])
    frz_lda_tm = mean(lambda c: c["frozen"]["lda"]["top_mass_p50"])
    cof_stm_tm = mean(lambda c: c["cofit_stm"]["top_mass_p50"])
    cof_lda_tm = mean(lambda c: c["cofit_lda_heldout"]["top_mass_p50"])
    cof_lda_opt_tm = mean(lambda c: c["cofit_lda_alpha_opt"]["top_mass_p50"])
    planted_tm = mean(lambda c: c["planted"]["top_mass_p50"])

    # beta sharpness (top_k mass; higher = sharper).
    stm_sharp = mean(lambda c: c["cofit_stm"]["beta_sharpness"]["top_k_mass"])
    lda_sharp = mean(lambda c: c["cofit_lda_heldout"]["beta_sharpness"]["top_k_mass"])
    lda_opt_sharp = mean(lambda c: c["cofit_lda_alpha_opt"]["beta_sharpness"]["top_k_mass"])
    true_sharp = mean(lambda c: c["true_beta_sharpness"]["top_k_mass"])

    # beta sharpness (eff_vocab; lower = sharper).
    stm_ev = mean(lambda c: c["cofit_stm"]["beta_sharpness"]["eff_vocab"])
    lda_ev = mean(lambda c: c["cofit_lda_heldout"]["beta_sharpness"]["eff_vocab"])

    frozen_gap = frz_lda_tm - frz_stm_tm
    cofit_gap = cof_lda_tm - cof_stm_tm
    lda_beta_sharper = lda_sharp > stm_sharp and lda_ev < stm_ev
    gap_widens = cofit_gap > frozen_gap + 0.02

    verdict = (
        "CONFIRMED" if (gap_widens and lda_beta_sharper)
        else "PARTIAL" if (gap_widens or lda_beta_sharper)
        else "REFUTED"
    )
    return (
        f"[{results['config']['regime']} regime] VERDICT: {verdict}. "
        f"theta top_mass at the held-out-LL-calibrated knob (mean over "
        f"{len(cells)} cells; planted {planted_tm:.3f}): FROZEN STM {frz_stm_tm:.3f} / "
        f"LDA {frz_lda_tm:.3f} (gap {frozen_gap:+.3f}); CO-FIT STM {cof_stm_tm:.3f} / "
        f"LDA-heldout {cof_lda_tm:.3f} (gap {cofit_gap:+.3f}) / LDA-alpha-opt "
        f"{cof_lda_opt_tm:.3f}. beta-sharpness top_k mass (true {true_sharp:.3f}): "
        f"STM {stm_sharp:.3f} vs LDA {lda_sharp:.3f} (eff_vocab STM {stm_ev:.0f} vs "
        f"LDA {lda_ev:.0f}); LDA beta {'IS' if lda_beta_sharper else 'is NOT'} sharper "
        f"than STM. Co-fitting beta {'WIDENS' if gap_widens else 'does NOT widen'} the "
        f"STM-vs-LDA peakiness gap vs frozen beta."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=list(REGIMES), default="clean")
    parser.add_argument("--subset", action="store_true",
                        help="run only the two representative peaky cells "
                        "(logistic_normal level 5, dirichlet level 0.1)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    mechanism_levels = SUBSET_MECHANISM_LEVELS if args.subset else DEFAULT_MECHANISM_LEVELS
    print(f"[{args.regime}{' subset' if args.subset else ''}] running...", flush=True)
    results = run(
        regime=args.regime, mechanism_levels=mechanism_levels,
        c_grid=DEFAULT_C_GRID, alpha_grid=DEFAULT_ALPHA_GRID,
        top_k=args.top_k, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    table = render_markdown_table(results)
    summary = build_summary(results)
    print(table)
    print()
    print(summary)

    suffix = "" if args.regime == "clean" else f"-{args.regime}-regime"
    if args.subset:
        suffix += "-subset"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / f"results{suffix}.md").write_text(
        f"# Co-fit-beta concentration-recovery (CR-4) results -- {args.regime} regime\n\n"
        f"Seed {args.seed}. Config: {results['config']}.\n\n"
        + table + "\n\n## Summary\n\n" + summary + "\n"
    )


if __name__ == "__main__":
    main()
