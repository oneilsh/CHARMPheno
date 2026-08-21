# 0074 — Warm-starting the batched readout L-BFGS supplies a point, not curvature: it does NOT cut iterations-to-convergence, DOES improve any budget-capped iterate, and without a curvature-stall guard it is a 4× regression

**Date:** 2026-08-21
**Topic:** distributed readout, optimization, batched L-BFGS, dev loop

**Status:** Confirmed on the batched-LR fixtures (unit + local-Spark integration, 6 seeds,
D up to 20k) while building warm starts for the ADR 0046 readout; motivated by the real
exp 0103 cost profile (C=437: ~200 iterations / ~1,228 distributed passes / ~35 min per
cold solve, three cold solves per supervised arm).

**Setting.** `solve_batched_lr` (sklearn-oracle objective: summed log-loss + C=1 ridge,
per-node standardization folded in) warm-started from a related fit's raw-space `(V,
b_raw)` mapped through the new fit's own moments (`unfold_standardization` — mapping
another fit's `W_std` directly is a silent bug, the standardization coordinates differ).
Warm-start sources: the same rows (re-fit), a 75% hash split (the calibration fit), a 0.3
row sample (the A/B harness refit).

**The three findings:**

1. **No iterations-to-convergence win — the endgame is history-bound, not
   distance-bound.** Warm is ahead at every early iteration (max|grad| 3.0 vs 9.1 after
   one iteration, 0.10 vs 0.31 after five) yet reaches `gtol` no sooner (cold 14 data
   passes vs warm 20 on the integration fixture; a wash at D=20k). A cold run enters the
   flat bottom carrying curvature pairs spanning its whole descent — which solve the
   ridge-flat directions of a simplex-featured design — while a warm run has only sampled
   its starting neighborhood. Also `gtol` bounds a SUMMED gradient, so a subset-fit's
   warm start is never near it at iteration 0 (instant freezes fire only when re-fitting
   the same rows).

2. **The reliable win is the iterate at a fixed budget.** At caps of 2–5 iterations the
   warm iterate is 3–6× closer to the converged answer in score space (6 seeds). Since
   the production solves already spend the full `max_iter=200` (exp 0103), warm starting
   cannot cost wall-clock there and strictly improves the returned point; the wall-clock
   cut itself comes from capping iterations (`CHARM_DEV` now caps `readout_max_iter` at
   60 — at ~iter 60 the macro-AUC ranking is stable to ~1e-3 while the gradient still has
   decades to fall, exp 0103 trajectory). Warm starts are what make an aggressive cap
   safe.

3. **Naive warm starting is a regression, not a wash.** A node warm-started into the flat
   bottom takes steps whose `s·y` falls below the curvature guard, so no pair is ever
   stored, γ freezes stale, and the model proposes the same ~1e-7 step forever — while F
   still decreases ~1e-13 relative per step, so the function-decrease stall rule never
   fires. Measured: **205 passes (hits max_iter) warm vs 48 cold** for a node already at
   its optimum (loss agreeing to 10 significant figures). Fix: two consecutive
   curvature-free iterations on a warm-started run ⇒ `stalled` (the existing "as good as
   this arithmetic gets" verdict); that case drops to 46 passes, and same-rows re-warm to
   37 vs cold's 77. The guard is gated on `warm_started` — cold behavior stays
   byte-identical (verified against the pre-change solver over 4 seeds × 3 gtols: params,
   n_iter, converged, stalled, n_stats_calls all identical).

**Rule of thumb going forward:** warm-start every related solve (calibration split, A/B
sample refit, any future co-fit amortization) for capped-point quality, budget the loop
with the iteration cap, and never expect warm starts to buy convergence speed on these
summed, ill-conditioned objectives. The masked-degenerate interaction is load-bearing:
a masked node's objective is the bare ridge (gradient `l2·w`, zero only at w=0), so the
driver must zero masked rows of the warm start or masked nodes iterate; both the
property and its contrapositive are pinned by tests.
