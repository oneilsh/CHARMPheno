# Experiment 0040: refit-loop dynamics (synthetic, LOCAL)

Task B2. Tests reviewer Fable's Q3 prediction: iterating fit -> calibrate (held-out predictive-LL, insight 0038) -> refit at the calibrated scale (`sigma_diagonal_pin`, commit 8fd40cc, Sigma_gen = c*R, ADR 0034/0036) is a CONTRACTION (settles near a fixed point) rather than a RATCHET (monotonic runaway). Synthetic corpus with a KNOWN planted scale S_TRUE=5.0 (via the new `eta_scale` kwarg on `gated_ln_corpus`, spark-vi/tests/_stm_synth.py) so 'did it converge near the truth' is directly checkable.

Config: K=11 (bg_k=3, groups={'A': 0.7, 'B': 0.3}, fg_per_group=4), V=400, D=2000, doc_len=45, n_iter=200, holdout_frac=0.5, C_GRID=[1, 2, 3, 4, 5, 6, 8, 12, 20], N_ROUNDS budget=5. Fresh held-out split each round (seed=round number). Plant seed=0.

Approximate planted target (from true beta/Sigma, not exact per-doc draws -- gated_ln_corpus does not return the per-doc eta draws): median top_mass=0.5457, median eff_topics=2.6035.

## Trajectory

| round | pin used | recalibrated c* | median top_mass | median eff_topics | planted_recovery |
|---|---|---|---|---|---|
| 0 | 1.0 | 3 | 0.4897 | 3.2044 | 11/11 |
| 1 | 3 | 2 | 0.4346 | 3.3597 | 11/11 |

## Verdict

CONVERGED: c* settled at round 1 (c*=2, within one grid step of the previous round's pin=3). Rounds to fixed point: 1 (round-1 moved |delta|=1 vs round-0's |delta|=n/a). Landed at c*=2 vs S_TRUE=5.0 (|error|=3.0, round-0 baseline error was 2.0) -- NOT close to the planted scale. beta did NOT sharpen from round 0 to the final round: median top_mass 0.4897 -> 0.4346 (delta=-0.0551), median eff_topics 3.2044 -> 3.3597 (delta=+0.1552), planted_recovery 11 -> 11 (delta=+0, out of K=11). Approximate planted target: top_mass=0.5457, eff_topics=2.6035.

Fable's contraction prediction HELD on this run.
