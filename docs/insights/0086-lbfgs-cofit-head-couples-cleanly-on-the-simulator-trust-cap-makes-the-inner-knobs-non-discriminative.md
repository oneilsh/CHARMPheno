# 0086 — The L-BFGS co-fit head couples cleanly on the simulator; under the trust cap the inner knobs are non-discriminative

**Date:** 2026-09-07
**Status:** Tentative (one simulator, one seed family; the cluster read is exp 0120)
**Tags:** svi, pc, head-optimizer, prediction-constrained, lbfgs, co-fit, coupling, local-simulator

**Setting context:** The local-simulator gate for the matrix-free amortized
L-BFGS co-fit head (`head_optimizer="lbfgs"`, plan
`2026-09-07-lbfgs-cofit-head-plan.md` WP1(b)), run in
`spark-vi/tests/test_pc_lbfgs_cofit.py`. Planted PC corpus: D=180, V=48, K=6,
C=3, each node's label a logistic of its two signal topics' θ-mass; flat
`OnlinePCLDA` with `head_standardize`/`head_intercept`, `head_l2=1e-2`,
`weight_y=12` / 5-iter warmup, `head_trust_move=0.03`, 22 outer SVI iters. The
head re-uses the FROZEN-θ readout solver (`batched_lr.solve_batched_lr`) against
the within-iter-fixed θ, warm-started from the current `w_CK`/`b_CK`, run
`head_inner_iters` passes; the in-memory re-scoring seam re-derives θ each outer
iter (θ MOVES — the one new problem vs the readout's frozen θ).

**The coupling is clean — all three gate criteria pass, and by wide margins.**
(1) `corr_relΔλ` REACHES the healthy band and stays BOUNDED: it climbs to the
`head_trust_move=0.03` cap, sits there while shaping is active, and then DECAYS
as the head converges (RM: the natural move drops below the cap, steps pass
through un-capped and decaying) — never an explosion. (2) `grad_y` (the incoming
head-gradient inf-norm, summed-scale) SHRINKS ~5.2 → 0.12 over the fit — the head
CONVERGES on the moving θ rather than chasing it, the exact opposite of exp
0119's SGD head (`grad_y` 988 → 10k, `corr` 6e-6). (3) the co-fit head TRACKS the
from-scratch sklearn oracle on the final θ: heldout macro-AUC ~0.92 for both,
gap |oracle − head| ≈ 0.004–0.005 (the head slightly EDGES the oracle — it fits
the same standardized ridge objective the oracle does, warm-started).

**Under the trust cap the two new knobs are non-discriminative.** Sweeping
`head_inner_iters ∈ {1,3,5}` × `head_history_reset ∈ {True,False}` moves nothing
that matters: all six satisfy (1)-(3) with corr peak 0.030, `grad_y` last
0.10–0.12, oracle gap ≤ 0.005. The mechanism: with the EG λ-move capped at 0.03,
θ drifts slowly enough that the head CONVERGES WITHIN each outer iter (grad_y
falls to ~0.1), so (a) more inner passes buy almost nothing past the first, and
(b) carried curvature ≈ fresh curvature — the near-converged head takes
near-zero steps, whose (s,y) pairs fail the curvature guard and are not stored,
so `history_reset=False` collapses onto `=True`. The `state`-carry plumbing IS
wired and verified independently (on a frozen-θ two-solve fixture, carrying the
history genuinely changes the second solve; `state=None`/`{}` is byte-identical
to the readout path), it just has nothing to bite on here.

**Decision (finalizes D3 for WP2/WP3):** ship `head_inner_iters=3` (a cheap
amortization margin — marginally lower end-of-fit `grad_y` than 1, negligible
cost at O(C·K) shuffle/pass) and `head_history_reset=True` (fresh curvature each
iter — zero stale-curvature risk when θ moves FASTER at cluster scale, where the
cap may not fully contain the per-iter drift and this simulator's "converges
within-iter" premise can break). The knobs stay exposed precisely so exp 0120 /
whole-Mondo can revisit them if the cluster coupling is looser than the sim's.

**What this does NOT establish:** that controlled shaping helps the READOUT (the
0120 primary — corr healthy but readout flat is still on the table, plan failure
read #3), nor cluster-scale coupling at K≈1498 (moving θ under a much larger head
and looser effective cap). This is the local greenlight the plan gated the build
on, not the PC verdict. `test_pc_lbfgs_cofit.py::test_lbfgs_cofit_coupling_gate`.
