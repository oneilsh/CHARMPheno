# 0031 — Opt-in K−1 reference-topic parameterization for STM

**Status:** Accepted
**Date:** 2026-06-27

## Context

STM's per-document prior is logistic-normal: θ_d = softmax(η_d), η_d ~ N(Γᵀx_d, Σ).
softmax is invariant to adding a constant to every coordinate of η
(softmax(η) = softmax(η + c·1)), so the overall "level" of η is not identified by
the likelihood — only the weak Gaussian prior (covariance Σ) pins it. Insight
0029 traced our online STM's σ_init-selected collapse↔Σ-blowup to this
translation degeneracy (plus a random init and an under-regularized Σ): the
likelihood-flat all-ones direction lets η drift to the softmax-saturation
boundary (Σ runs to ~10^10 on real data) or, with a tight prior, collapse to
near-uniform θ. The published CTM/STM remove the degeneracy by fixing one
topic's η ≡ 0 and working in K−1 dimensions. Ablation 1 in insight 0029 showed
spectral init alone does not close the σ_init=1 gap (0/8 recovery even with a
good β seed), because a tight Σ≈1 prior re-collapses η regardless of where β
starts. The K−1 reference parameterization is the missing third stabilizer.

## Decision

Add an opt-in `reference_topic: bool = False` to `OnlineSTM`. When on, pin
**topic 0**'s η ≡ 0 (the always-on "baseline" topic) and optimize the other K−1
topics' η relative to it; θ = softmax([0, ν]). Default-off is byte-identical to
the prior engine.

Two design choices:

- **Reference = topic 0, a single global background topic.** Γ and Σ are shared
  across all documents, so a single global reference is the only consistent
  choice — and it must be in *every* document's allowed set. Topic 0 is always
  the first background topic (`TopicBlockPartition` lays background out first and
  enforces `background_k >= 1`), so it is in every allowed set, gated or not.
  Non-gated STM is the degenerate "allowed = all K" case, so one mechanism
  covers both. The reduced gradient/Hessian are exactly the free×free sub-blocks
  of the full ones (fixing a coordinate to a constant makes the reduced
  derivatives the corresponding sub-block), so the existing analytic
  gradient/Hessian are reused — sliced, not rederived.

- **"Clamped-K" implementation.** Γ (P×K) and Σ (K-diagonal) keep full shape; β/λ
  stay full-K because the reference is a real topic (its exp(0)=1 sits in the
  softmax denominator and it accumulates word statistics — only its η leaves the
  optimizer and the prior). The reference is dropped only from `local_update`'s
  prior-side accumulators: `XtMu`, `residual_diag`, `n_docs_per_topic`, and the
  η-KL sub-space. That makes `update_global` need **no changes**: its background
  Γ solve produces the reference column as A⁻¹·0 = 0, and its lazy-block rule
  (`present = n_docs_per_topic > 0`) leaves the reference's Σ entry at its initial
  value. The reference's η-KL row/col is excluded before the `slogdet`, so the
  Laplace sub-covariance stays non-singular.

## Alternatives considered

- **True K−1 re-indexing (Γ/Σ shaped K−1).** Mathematically tidier (no inert
  reference column) but threads an off-by-one mapping (free param j ↔ topic j+1)
  through `initialize_global`, the M-step, diagnostics, and export — a much
  larger blast radius for the same behavior. Clamped-K keeps every K-indexed
  array's shape and makes the default path provably untouched.
- **A per-block reference for gated STM.** Inconsistent with Γ/Σ being shared
  across documents (each block's docs would anchor on a different topic). The
  single global background reference is forced by the shared-prior structure.
- **Keep hunting σ_init / raise sigma_ridge.** A band-aid: insight 0029 shows
  the knife-edge is structural. The reference removes the degeneracy at its root.

## Consequences

- **Closes the σ_init=1 gap (insight 0029, Ablation 2).** Spectral+reference
  recovers 2/8 at σ_init=1 where spectral-alone gave 0/8; spectral+reference+Σ-prior
  reaches 3/8 across σ_init ∈ {1, 5, 20}, effectively removing the σ_init knob,
  and Σ stays bounded (no ~10^10 blowup). Reference alone is necessary-not-sufficient
  (0/8 non-gated) — like spectral alone — but in the gated rare-arm scenario it
  recovers the minority foreground on its own.
- **Dashboard prevalence path stays consistent by construction.** The corpus-mean
  proportion helpers (`corpus_mean_topic_proportions` /
  `corpus_mean_topic_proportions_gated`) compute softmax(Γᵀx) over the full K.
  Under a reference fit Γ[:, 0] = 0, so column 0 enters that softmax as exp(0)=1 —
  exactly the pinned reference the per-doc inference uses. The prevalence numbers
  are therefore correct for a reference fit with no change to those helpers; this
  is a load-bearing coupling (do not "simplify" the softmax to drop a zero column).
- **Inference consistency.** `infer_local` pins the same reference, so exported
  per-document θ matches training. (`infer_local` not applying gating `allowed`
  is a pre-existing condition, unrelated to this change.)
- **Not yet on the cluster.** `reference_topic` is reachable through the
  `OnlineSTM` API and the local ablation, but is not threaded through the
  `StreamingSTM` mllib estimator, so the production Spark fit path cannot use it
  yet — the same posture as spectral init. Wiring it (constructor param +
  metadata persistence so a reloaded model re-pins) is a separate decision, taken
  when promoting beyond the local ablation.
- **Dirichlet-family models are unaffected.** LDA/HDP/PLDA have no logistic-normal
  prior and no translation degeneracy; this guard is intrinsic to STM (insight 0028).
