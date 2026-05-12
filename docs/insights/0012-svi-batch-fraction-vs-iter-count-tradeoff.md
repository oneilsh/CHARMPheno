# 0012 — SVI batch-fraction and iter count must be tuned together
**Date:** 2026-05-12
**Topic:** svi
**Status:** Observed

Smaller mini-batch fractions (e.g. 0.1 of the corpus) give *more stable*
per-iter ELBO trajectories — less variance in each natural-gradient
step, less catastrophic catch-all-grabbing during warm-up. This is what
the Hoffman et al. SVI paper recommends for stability.

But smaller batches only help if you actually run enough iters to let
the Robbins-Monro learning-rate schedule ρ_t = (τ + t)^(−κ) descend.
With ~20 iters at batch-fraction=0.1, ρ stays in the 0.05–0.08 range
the entire run and never anneals into the regime where small updates
accumulate into convergence.

So: small batch + few iters is the worst of both worlds — slow
convergence per iter AND no benefit from the stability. Either:
- Small batch fraction + many iters (slow but stable), or
- Larger batch fraction + few iters (faster, more variance, OK if the
  data isn't pathological).

For exploratory doc-unit experiments where you want fast turnaround,
the larger-batch route is usually correct.

**Implications.** When setting `batch_fraction` and `--num-iters`
together, pick them as a pair, not independently. The current
patient-year HDP/LDA runs at `batch_fraction=0.1` and `iters=20` are
in the bad regime — we'd see more from either bumping iters to 50+
or bumping batch fraction to 0.3+.

**Setting context.** Discussed during HDP cluster runs where bumping
executor count didn't reduce iter time (see
[0013](0013-spark-scaling-driver-bottleneck.md)); the natural next
question was whether smaller batches would help, leading to this
trade-off being articulated. No fresh run yet confirming the
prescription — flagged for the next round of experiments.
