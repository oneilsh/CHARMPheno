"""Task B2: refit-loop dynamics experiment (synthetic, LOCAL, no Spark/cluster).

We calibrate a generative concentration scale c* by held-out predictive-LL
(insight 0038; the gated sweep is ``corpus_heldout_scale_sweep_gated``, ADR
0034/0036), then fold it back into the model: pin Sigma_ii = c* (the engine
option ``sigma_diagonal_pin``, commit 8fd40cc, Sigma_gen = c*R) and REFIT so
beta re-sharpens under the calibrated prior, then RE-calibrate. A reviewer's
question (Q3): does iterating fit -> calibrate -> refit CONVERGE to a fixed
point, or RATCHET upward (refit at higher c -> beta sharpens -> docs more
identifiable -> next calibration returns higher c -> sharper still, with no
settle)? The prediction under test is a CONTRACTION: the calibration
objective is held-out, so a beta that over-sharpens by fitting visible-token
noise pays on held-out tokens -- the same cross-validation that makes the
single-round sweep runaway-proof (insight 0038) should also bound this OUTER
loop. This script tests that prediction on synthetic data where the planted
scale (S_TRUE) is KNOWN, so "did it converge near the truth" is directly
checkable, not just "did it stop moving."

Loop (see module docstring items below for exact mechanics):
  1. Plant a gated logistic-normal corpus at a KNOWN generative scale
     S_TRUE via ``gated_ln_corpus(..., eta_scale=S_TRUE)`` (the new
     eta_scale kwarg on the TEST synthetic helper, spark-vi/tests/_stm_synth.py
     -- default-preserving, see test_refit_dynamics.py).
  2. Round 0 (baseline): fit at the unit pin (default sigma_diagonal_pin=1.0),
     calibrate c0 via the held-out sweep.
  3. Rounds 1..N: refit at the PREVIOUS round's c* (sigma_diagonal_pin=c_prev),
     recalibrate with a FRESH held-out split each round (seed=round number --
     Fable guardrail b, so no split-specific overfitting can explain a settle).
  4. Stopping rule (Fable guardrail a): stop when |c_n - c_prev| <= one local
     grid step of C_GRID near c_n. Monotonic-rise alarm: if c* strictly rises
     three rounds running, flag RATCHET SUSPECTED (does not abort -- it is
     itself a finding).
  5. Report the trajectory (c*, median top_mass/eff_topics, planted_recovery)
     and a verdict: converge vs ratchet, rounds-to-fixed-point, landed near
     S_TRUE?, did beta sharpen (co-fit-beta benefit)?

Runnable directly: `python scripts/refit_dynamics_synthetic.py`. spark_vi is
installed editable and spark-vi/tests is a regular package (has __init__.py),
so ``tests._stm_synth`` imports the same way the spark-vi test suite does
(see scripts/concentration_recovery_gated_experiment.py for the identical
sys.path-free import pattern).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spark_vi.mllib.topic.stm import (
    corpus_concentration_stm,
    corpus_heldout_scale_sweep_gated,
)
from tests._stm_synth import (
    fit_stm,
    gated_ln_corpus_overlap,
    planted_recovery,
    topic_support_jaccard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "0040-refit-dynamics"

S_TRUE = 5.0
# Realistic vocabulary-overlap regime (feedback_synthetic_vocab_overlap): half of
# every topic's mass on a shared pool (support Jaccard ~1/3, matching the real HF
# beta ~0.35), so beta CANNOT absorb the concentration and the scale must climb --
# unlike the disjoint-vocab plant (exp 0040 v1), which converged trivially because
# the unit fit was already at the planted concentration.
SHARED_FRAC = 0.5
C_GRID = [1, 2, 3, 4, 5, 6, 8, 12, 20]   # finer near 5 so the fixed point isn't grid-locked
N_ROUNDS = 5
GROUP_WEIGHTS = {"A": 0.7, "B": 0.3}
FG_PER_GROUP = 4
BG_K = 3
V = 400
D = 2000
DOC_LEN = 45
N_ITER = 200
HOLDOUT_FRAC = 0.5


def local_grid_step(c: float, grid: list) -> float:
    """Local spacing of ``grid`` at the grid value ``c``: the larger of the
    gaps to its immediate left/right neighbors (falling back to the single
    available neighbor at either boundary). Used as the stopping-rule
    tolerance so "settled within one grid step" is well-defined even on a
    non-uniform grid (C_GRID is finer near 5, coarser toward 20)."""
    g = sorted(grid)
    i = g.index(c)
    left = g[i] - g[i - 1] if i > 0 else None
    right = g[i + 1] - g[i] if i < len(g) - 1 else None
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _median(summary: dict, field: str) -> float:
    return float(summary[field]["p50"])


def approximate_planted_target(docs, beta_true, sigma_true, part) -> tuple[float, float]:
    """Rough (labeled APPROXIMATE) target line: what would the concentration
    readout show if beta and Sigma were exactly the planted truth? Builds a
    global_params dict directly from the ground truth (lambda = beta_true
    scaled sharp, Gamma = 0, Sigma = the planted eta_scale*Sigma_true
    covariance -- NOT the fit's unit-diagonal correlation R, since this is
    the true generative covariance, not a correlation to be rescaled) and
    runs the SAME corpus_concentration_stm readout the fit rounds use.
    Because beta_true is scaled sharp (near a delta on its support) and the
    prior is exactly the true generative one, the recovered posterior mode is
    close to (but not exactly) the actual per-doc draw used to generate that
    document's tokens -- an approximation, not the planted values themselves
    (gated_ln_corpus does not return the per-doc eta draws)."""
    K = beta_true.shape[0]
    lam_true = beta_true * (500.0 * beta_true.shape[1]) + 0.01
    gp_true = {"lambda": lam_true, "Gamma": np.zeros((1, K)), "Sigma": sigma_true}
    summary = corpus_concentration_stm(docs, gp_true, part, reference=None)
    return _median(summary, "top_mass"), _median(summary, "eff_topics")


def run(*, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    print(f"[plant] gated_ln_corpus_overlap: groups={GROUP_WEIGHTS} fg_per_group={FG_PER_GROUP} "
          f"bg_k={BG_K} V={V} D={D} doc_len={DOC_LEN} shared_frac={SHARED_FRAC} "
          f"eta_scale(S_TRUE)={S_TRUE} seed=0")
    docs, part, sigma_true, beta_true = gated_ln_corpus_overlap(
        group_weights=GROUP_WEIGHTS, fg_per_group=FG_PER_GROUP, bg_k=BG_K,
        V=V, D=D, doc_len=DOC_LEN, shared_frac=SHARED_FRAC, eta_scale=S_TRUE, seed=0,
    )
    K = part.K
    jac = topic_support_jaccard(beta_true)
    print(f"[plant] K={K} (bg_k={BG_K} + 2*fg_per_group={FG_PER_GROUP}); "
          f"topic-support Jaccard={jac:.3f} (realistic overlap, real HF beta ~0.35)")

    planted_top, planted_eff = approximate_planted_target(docs, beta_true, sigma_true, part)
    print(f"[planted target, APPROXIMATE] median top_mass={planted_top:.4f} "
          f"eff_topics={planted_eff:.4f} (from true beta/Sigma, not exact per-doc draws)")

    trajectory: list[dict] = []

    print("[round 0] fitting at unit pin (sigma_diagonal_pin default=1.0) ...")
    gp = fit_stm(docs, K=K, V=V, sigma_init=1.0, n_iter=N_ITER,
                 partition=part, reference_topic=True)
    sweep = corpus_heldout_scale_sweep_gated(
        docs, gp, part, c_grid=C_GRID, holdout_frac=HOLDOUT_FRAC, reference=0, seed=0,
    )
    c0 = sweep["argmax_c"]
    lls0 = {str(k): float(v) for k, v in sweep["lls"].items()}
    conc = corpus_concentration_stm(docs, gp, part, reference=0)
    top_mass0 = _median(conc, "top_mass")
    eff_topics0 = _median(conc, "eff_topics")
    beta_hat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
    recovery0 = planted_recovery(beta_hat, beta_true)
    print(f"[round 0] c*={c0} top_mass={top_mass0:.4f} eff_topics={eff_topics0:.4f} "
          f"planted_recovery={recovery0}/{K}")
    print(f"[round 0] held-out LL curve: "
          f"{ {k: round(v, 4) for k, v in lls0.items()} }")
    trajectory.append({
        "round": 0, "pin": 1.0, "c_star": c0,
        "top_mass": top_mass0, "eff_topics": eff_topics0, "recovery": recovery0,
        "lls": lls0,
    })

    c_prev = c0
    rises = [c0]
    ratchet_suspected = False
    stopped_round = None

    for n in range(1, N_ROUNDS + 1):
        print(f"[round {n}] refitting at pin=sigma_diagonal_pin={c_prev} ...")
        gp = fit_stm(docs, K=K, V=V, sigma_init=1.0, n_iter=N_ITER,
                     partition=part, reference_topic=True,
                     sigma_diagonal_pin=c_prev)
        sweep = corpus_heldout_scale_sweep_gated(
            docs, gp, part, c_grid=C_GRID, holdout_frac=HOLDOUT_FRAC,
            reference=0, seed=n,   # FRESH holdout split each round (guardrail b)
        )
        c_n = sweep["argmax_c"]
        lls_n = {str(k): float(v) for k, v in sweep["lls"].items()}
        conc = corpus_concentration_stm(docs, gp, part, reference=0)
        top_mass = _median(conc, "top_mass")
        eff_topics = _median(conc, "eff_topics")
        beta_hat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recovery = planted_recovery(beta_hat, beta_true)

        delta = abs(c_n - c_prev)
        step = local_grid_step(c_n, C_GRID)
        print(f"[round {n}] c*={c_n} (prev={c_prev}, |delta|={delta}, local grid step={step}) "
              f"top_mass={top_mass:.4f} eff_topics={eff_topics:.4f} "
              f"planted_recovery={recovery}/{K}")
        print(f"[round {n}] held-out LL curve: "
              f"{ {k: round(v, 4) for k, v in lls_n.items()} }")
        trajectory.append({
            "round": n, "pin": c_prev, "c_star": c_n,
            "top_mass": top_mass, "eff_topics": eff_topics, "recovery": recovery,
            "lls": lls_n,
        })

        rises.append(c_n)
        if len(rises) >= 4 and rises[-1] > rises[-2] > rises[-3] > rises[-4]:
            ratchet_suspected = True
            print(f"[round {n}] ALARM: c* has strictly increased 3 rounds running "
                  f"({rises[-4]} < {rises[-3]} < {rises[-2]} < {rises[-1]}) -- RATCHET SUSPECTED")

        if delta <= step:
            print(f"[round {n}] STOPPING: |delta c*|={delta} <= local grid step {step} -- settled.")
            stopped_round = n
            c_prev = c_n
            break
        c_prev = c_n

    settled = stopped_round is not None
    result = {
        "config": {
            "S_TRUE": S_TRUE, "C_GRID": C_GRID, "N_ROUNDS": N_ROUNDS,
            "group_weights": GROUP_WEIGHTS, "fg_per_group": FG_PER_GROUP,
            "bg_k": BG_K, "K": K, "V": V, "D": D, "doc_len": DOC_LEN,
            "n_iter": N_ITER, "holdout_frac": HOLDOUT_FRAC,
            "shared_frac": SHARED_FRAC, "topic_support_jaccard": round(jac, 4),
        },
        "planted_target_approx": {"top_mass": planted_top, "eff_topics": planted_eff},
        "trajectory": trajectory,
        "settled": settled,
        "stopped_round": stopped_round,
        "ratchet_suspected": ratchet_suspected,
    }
    return result


def render_table(result: dict) -> str:
    header = "| round | pin used | recalibrated c* | median top_mass | median eff_topics | planted_recovery |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    K = result["config"]["K"]
    for row in result["trajectory"]:
        lines.append(
            f"| {row['round']} | {row['pin']} | {row['c_star']} | "
            f"{row['top_mass']:.4f} | {row['eff_topics']:.4f} | {row['recovery']}/{K} |"
        )
    return "\n".join(lines)


def build_verdict(result: dict) -> str:
    traj = result["trajectory"]
    r0, r_last = traj[0], traj[-1]
    settled = result["settled"]
    ratchet = result["ratchet_suspected"]
    S_TRUE_local = result["config"]["S_TRUE"]

    if settled:
        conv_bit = (
            f"CONVERGED: c* settled at round {result['stopped_round']} "
            f"(c*={r_last['c_star']}, within one grid step of the previous round's pin={r_last['pin']})."
        )
        if ratchet:
            conv_bit += (
                " (Note: the monotonic-rise alarm fired transiently earlier in the "
                "trajectory before settling -- see trajectory column.)"
            )
    elif ratchet:
        conv_bit = (
            f"RATCHET SUSPECTED: c* rose monotonically for 3+ rounds running without settling "
            f"within {result['config']['N_ROUNDS']} rounds (trajectory: "
            f"{[row['c_star'] for row in traj]})."
        )
    else:
        conv_bit = (
            f"DID NOT SETTLE within {result['config']['N_ROUNDS']} rounds, but no monotonic "
            f"3-round rise was observed either (trajectory: {[row['c_star'] for row in traj]}) "
            f"-- neither a clean contraction nor a clean ratchet on this run."
        )

    dist0 = abs(r0["c_star"] - S_TRUE_local)
    dist_last = abs(r_last["c_star"] - S_TRUE_local)
    landed_near = dist_last <= local_grid_step(
        min(result["config"]["C_GRID"], key=lambda c: abs(c - r_last["c_star"])),
        result["config"]["C_GRID"],
    )
    truth_bit = (
        f"Landed at c*={r_last['c_star']} vs S_TRUE={S_TRUE_local} (|error|={dist_last}, "
        f"round-0 baseline error was {dist0}) -- "
        + ("close to the planted scale." if landed_near else "NOT close to the planted scale.")
    )

    top_delta = r_last["top_mass"] - r0["top_mass"]
    eff_delta = r_last["eff_topics"] - r0["eff_topics"]
    recov_delta = r_last["recovery"] - r0["recovery"]
    sharpen_bit = (
        f"beta {'SHARPENED' if top_delta > 0 else 'did NOT sharpen'} from round 0 to the final round: "
        f"median top_mass {r0['top_mass']:.4f} -> {r_last['top_mass']:.4f} (delta={top_delta:+.4f}), "
        f"median eff_topics {r0['eff_topics']:.4f} -> {r_last['eff_topics']:.4f} (delta={eff_delta:+.4f}), "
        f"planted_recovery {r0['recovery']} -> {r_last['recovery']} "
        f"(delta={recov_delta:+d}, out of K={result['config']['K']}). "
        f"Approximate planted target: top_mass={result['planted_target_approx']['top_mass']:.4f}, "
        f"eff_topics={result['planted_target_approx']['eff_topics']:.4f}."
    )

    rounds_bit = (
        f"Rounds to fixed point: {result['stopped_round']} (round-{result['stopped_round']} moved "
        f"|delta|={abs(r_last['c_star'] - traj[-2]['c_star'])} vs round-{result['stopped_round']-1}'s "
        f"|delta|={abs(traj[-2]['c_star'] - traj[-3]['c_star']) if len(traj) >= 3 else 'n/a'})."
        if settled and result["stopped_round"] >= 1 else
        "No fixed point reached within the round budget."
    )

    return f"{conv_bit} {rounds_bit} {truth_bit} {sharpen_bit}"


def render_markdown_doc(result: dict) -> str:
    cfg = result["config"]
    table = render_table(result)
    verdict = build_verdict(result)
    fable_held = (
        "Fable's contraction prediction HELD on this run."
        if result["settled"] and not result["ratchet_suspected"] else
        "Fable's contraction prediction did NOT clearly hold on this run (see verdict above)."
    )
    return (
        "# Experiment 0040: refit-loop dynamics (synthetic, LOCAL)\n\n"
        "Task B2. Tests reviewer Fable's Q3 prediction: iterating "
        "fit -> calibrate (held-out predictive-LL, insight 0038) -> refit at the "
        "calibrated scale (`sigma_diagonal_pin`, commit 8fd40cc, Sigma_gen = c*R, "
        "ADR 0034/0036) is a CONTRACTION (settles near a fixed point) rather than a "
        "RATCHET (monotonic runaway). Synthetic corpus with a KNOWN planted scale "
        f"S_TRUE={cfg['S_TRUE']} (via the new `eta_scale` kwarg on "
        "`gated_ln_corpus`, spark-vi/tests/_stm_synth.py) so 'did it converge near "
        "the truth' is directly checkable.\n\n"
        f"Config: K={cfg['K']} (bg_k={cfg['bg_k']}, groups={cfg['group_weights']}, "
        f"fg_per_group={cfg['fg_per_group']}), V={cfg['V']}, D={cfg['D']}, "
        f"doc_len={cfg['doc_len']}, n_iter={cfg['n_iter']}, "
        f"holdout_frac={cfg['holdout_frac']}, C_GRID={cfg['C_GRID']}, "
        f"N_ROUNDS budget={cfg['N_ROUNDS']}. Fresh held-out split each round "
        "(seed=round number). Plant seed=0.\n\n"
        f"Approximate planted target (from true beta/Sigma, not exact per-doc draws -- "
        "gated_ln_corpus does not return the per-doc eta draws): median top_mass="
        f"{result['planted_target_approx']['top_mass']:.4f}, median eff_topics="
        f"{result['planted_target_approx']['eff_topics']:.4f}.\n\n"
        "## Trajectory\n\n" + table + "\n\n"
        "## Verdict\n\n" + verdict + "\n\n" + fable_held + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B2: LOCAL synthetic refit-loop dynamics experiment "
        "(fit -> calibrate -> refit convergence vs ratchet).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = run(out_dir=args.out)

    print()
    print(render_table(result))
    print()
    verdict = build_verdict(result)
    print("VERDICT:", verdict)

    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "results.md").write_text(render_markdown_doc(result))
    print(f"\nWrote {args.out / 'results.json'} and {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
