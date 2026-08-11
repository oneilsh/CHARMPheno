# 0065 — A co-fit logistic head in minibatch SVI does not converge with one gradient step per iteration; an aggregatable Newton (IRLS) step fixes it

**Date:** 2026-08-11
**Topic:** svi
**Status:** Confirmed

The prediction-constrained model (`OnlinePCLDA`) co-fits a logistic head `w_CK`
jointly with the topics under minibatch SVI. For a long time the head read heldout
AUC ≈ chance (0.52) while a batch LogisticRegression on the *same* final topics
(`pc_topics_lr`) reached 0.61–0.62. A sequence of hypotheses were all **refuted**:
a ridge/L2 issue, a train/eval θ representation mismatch (the differentiable
`_cavi_theta_anp` unroll vs the converged `infer_local` θ — measured cosine 0.95, an
AUC cost of ~0.001, negligible), the choice of optimizer (Adam ≈ SGD), and the
learning rate (an aggressively hot Adam only made `|w_CK|` oscillate).

**The real mechanism.** The head takes exactly **one gradient step per SVI global
iteration**, against a θ computed from *continuously moving* topics (the supervised
correction reshapes them every step). One noisy gradient step per iteration cannot
converge a logistic head — it wanders and lands ~orthogonal to the batch-LR
direction (cos +0.09; `head_vs_lr_direction_cosine` localizer). This is a
non-convergence, not a mis-specification: the head never reaches the logistic
optimum of *any* representation it sees.

**The fix that fits distributed SVI: Newton/IRLS, not more SGD.** A driver-side SGD
inner-loop is blocked by two infra facts — the runner scales every sufficient-stat by
`corpus/batch` (would corrupt raw per-doc θ shipped through the stats dict), and the
corpus is over-partitioned (a per-partition inner-loop averages over ~3 docs). But
**Newton needs only aggregatable sufficient statistics**: the per-label gradient
`g_c = Σ_d (p−y)π_d` (K-vector) and Fisher information `H_c = Σ_d p(1−p)·π_dπ_dᵀ`
(K×K), both additive doc-sums. They ride the existing `treeReduce`, and because the
runner scales *both* by the same factor, the solve `H⁻¹g` is **scale-invariant** — the
scaling cancels, no raw θ ever reaches the driver, no runner change. One ridge-Newton
step per iteration converges the logistic head (Newton converges logistic in a handful
of steps). Result: head 0.52 → 0.60, direction cos 0.09 → 0.35.

**Two residual cautions.** (1) Per-minibatch Newton chases each minibatch's own
optimum, so a large step (`head_lr` 0.7) oscillates and a near-singular minibatch
`H_c` can spike `|w_CK|` (saw 26.7). Damping (`head_lr` ~0.3, an EMA of per-minibatch
optima) + a relative ridge (fraction of `mean(diag(H))`) stabilizes it. (2) Even
stabilized, the head plateaus below the batch-LR ceiling when the **topics have not
converged** within the iteration budget — the head then chases a still-moving target
and ends calibrated to the topic *trajectory*, not the final topics. The remaining gap
closes with more supervised iterations (let topics settle) or Polyak-averaging the head.

**Implications.** For any coupled non-conjugate parameter jointly optimized in
minibatch SVI (a regression head here; the STM covariance is the sibling, insight
0029/0034), a single first-order step per global iteration is not enough — the
parameter must be *converged* per iteration relative to the moving representation. When
the parameter's objective is (conditionally) convex with an aggregatable Hessian, an
IRLS/Newton step is the scale-invariant way to do that without collecting per-doc
quantities to the driver. This is the same lesson Hughes et al. (AISTATS 2018) encode
by optimizing `{φ, η}` with Adam and re-solving the local π to its MAP each step.

**Setting context.** AoU OMOP `mdd_stable_treatment` cohort, 10-drug fully-observed
multi-label, K=50, weight_y=1000, distributed VI-native `OnlinePCLDA`/`VIRunner`,
subsampling 0.1 (~3.45k-doc minibatches), 25 unsupervised warm-up + 100 supervised
iters. Experiments 0072 (sgd) → 0073 (hot adam) → 0074/0075 (newton, `head_optimizer`
param). Localized cluster-free with the `head_vs_lr_cosine` / `theta_mismatch`
diagnostics (`analysis/pc/diagnostics/`, run under `--eval-only`) and
`spark-vi/tests/manual_pc_head_direction_repro.py`.
