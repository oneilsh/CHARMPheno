# 0043 — Drop the Firth head and its inner-loop (Path B); the fixed ridge IS the head regularizer

**Status:** Accepted
**Date:** 2026-08-17

> **Numbering note.** Shared-counter caveat as ADR 0038–0042; renumber if a sibling
> claimed 0043 first.

## Context

The PC co-fit logistic head (`w_CK`) can run away to `|w|→∞` because PC's supervised
shaping makes the topics separable (`p(1−p)→0`), leaving the logistic MLE at infinity.
Two regularizers were built to bound it:

1. **A fixed absolute L2 ridge** (`head_l2`, = Hughes's `lambda_w`), applied on the
   corpus-summed head gradient so `|w| ~ |g|/head_l2` — a constant, leverage-independent
   floor (ADR 0040/0041, insight 0067/0068).
2. **A Firth / Jeffreys-prior penalty** (`head_penalty='firth'`), the "parameter-free"
   cure: the `+½·log det I(w)` term pulls `p` back toward ½ as `|w|→∞`, bounding `w` with
   no ridge to tune. Firth needed per-doc logistic leverages, which only the **driver-side
   inner-loop IRLS path** ("Path B", `head_inner_iters>0`) collects — so requesting Firth
   force-activated Path B, which ships a bounded subsample of raw θ to the driver each SVI
   iteration (unlike the fully-aggregatable one-step Newton).

Firth was attractive precisely because it promised to remove the one tunable head
hyperparameter. But two findings undercut it:

- **Firth structurally weakens in our regime (exp 0080).** Its regularization strength ∝
  the Fisher leverage ≈ `p/n`. Its unit tests pass in the small-sample regime (`p=2`,
  `n≈40` → strong pull). The scaled closure-mask head is the opposite: well-powered nodes
  have **large n** (leverage→0), and rare leaves are **p≫n / rank-deficient** (the `pinv`
  `rcond` truncation zeroes the Firth term in exactly the near-singular directions `w`
  escapes through). Both erode the pull where `|w|` grows. On the 41-anchor closure run
  Firth only got `|w|` to ~4.9e3 (vs 1.3e4 unregularized) — nowhere near its `O(1–10)`
  fixed point. The parameter-free property breaks down at large-n + rank-deficient design.

- **Path B is fixed-point-redundant with the fixed ridge (insight 0067).** With `head_l2>0`,
  the aggregated one-step Newton accumulates over the ~60 iterations the topics take to
  settle to the *same* regularized fixed point Path B reaches within one iteration
  (verified `INNER==one-step` byte-for-byte). So the non-Firth inner loop earns nothing,
  and Firth — the inner loop's only remaining unique consumer — does not work here.

- **The fixed ridge already delivers the goal (exp 0081/0082, insight 0069).** At
  41-anchor scale the `head_l2=0.01` co-fit head is bounded (`|w|max~2126`, no Firth,
  `head_penalty='none'`) AND well-calibrated on `P(child|parent)` (pooled ECE 0.0098,
  competitive with the two-stage readout LR). The unified conditional-probability model —
  the thing Firth's calibration story was supposed to enable — works on the ridge alone.

## Decision

**Remove the Firth head and the entire inner-loop path (Path B) from the engine and its
whole plumbing chain.** The fixed absolute L2 ridge (`head_l2`) is the single, sufficient
head regularizer.

Concretely, deleted from `OnlinePCLDA` and its shim/driver/config:
`head_penalty`, `head_inner_iters`, `head_sample_cap`; the `_FIRTH_FLOOR`/`_FIRTH_TRUST`
constants and the `_firth_score` closure; the `head_theta`/`head_s`/`head_obs` per-doc
design emission, `_HEAD_DESIGN`, and the `combine_stats` concatenate-not-sum special case;
the `headPenalty`/`headInnerIters` MLlib Params; the `--head-penalty`/`--head-inner-iters`
driver CLI and `run_experiment` passthrough; and `test_pc_firth_head.py`.

The head update is now exactly two branches: the one-step ridge-Newton
(`head_optimizer='newton'`, the shipped gated-PC path) and the RM-damped SGD step.

## Alternatives considered

- **Keep Firth parked-but-present** (the pre-decision state): a unit-test-green engine
  path, gated behind `head_penalty='firth'`, for a hypothetical future small-sample
  deployment. Rejected: it kept a large, self-described-redundant subsystem (Path B) alive
  solely to host a feature that does not work in the regime we actually operate in, and a
  code walkthrough would spend effort on dead machinery. The git history preserves the
  implementation if the small-sample regime ever becomes real.
- **Firth + ridge** (to rescue Firth at large n): reintroduces the very tunable parameter
  Firth existed to remove — no advantage over ridge alone.
- **Drop only Firth, keep the (now Firth-less) inner loop as an off-by-default option.**
  Rejected: insight 0067 proved it byte-for-byte redundant with the one-step head under a
  fixed ridge, and it uniquely carries the raw-θ-to-driver cost. With Firth gone it has no
  consumer.

## Consequences

- **One head hyperparameter remains** (`head_l2`), and it is a *fixed* prior — Hughes's
  `lambda_w=1e-3` default, data-independent, wide good basin (~1e-4..1e-2). It is set, not
  swept per-run. If removing even that one constant is ever desired, the principled route
  is **empirical-Bayes hierarchical shrinkage** (shrink each child head toward its parent,
  shrinkage strength *estimated* from data — genuinely parameter-free), which would also
  address the residual small-cohort discrimination tax (insight 0069); Firth is not that
  route.
- **The engine loses its only raw-θ-to-driver path**, restoring the fully-aggregatable,
  scale-invariant one-step Newton (ADR 0039) as the sole head fit — a real architectural
  simplification, not just a Firth removal.
- No shipped run changes behavior: every experiment to date ran `head_penalty='none'` /
  `head_inner_iters=0`, i.e. the retained one-step path.
- The multi-domain per-domain λ correction, the gate, and the DAG-closure head are
  untouched.
