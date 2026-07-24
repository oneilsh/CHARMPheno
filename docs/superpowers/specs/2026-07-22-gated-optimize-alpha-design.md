# Gated `optimize_alpha` — learned per-node asymmetric doc-concentration

**Status:** design (brainstormed 2026-07-22, branch `case-finding`)

## Problem

`GatedOnlineLDA` fixes the Dirichlet doc-concentration α for the whole fit. The
only asymmetry available is `nodeAlphaScale` — a single fixed multiplier that
makes every node block a priori rarer than background. Vanilla `OnlineLDA`, by
contrast, can *learn* an asymmetric α from data (`optimize_alpha`, Wallach et al.
2009), which is the recommended setting for topic quality. The gated engine
raises `NotImplementedError` on `optimize_alpha=True` today, because its gated
`local_update` never emits the sufficient statistic the Newton step needs.

Goal: let the gated engine **learn** an asymmetric α, so per-node rarity is
estimated from the data instead of set by one hand-tuned scalar. Different
disease nodes have different prevalence; a learned per-node α should reflect that.

## Design decisions (locked in brainstorming)

1. **Parameterization: per-node tied α.** α is a tied vector of length
   `1 + n_nodes`: one shared background α_bg (the `n_bg` background topics are
   exchangeable) plus one α_u per DAG node (shared across that node's `tpn`
   topics). The length-K α vector the E-step and transform consume is *expanded*
   from this tied vector. This is the direct generalization of the current
   1-parameter `nodeAlphaScale`.

2. **Exact gated Newton, not the vanilla Sherman-Morrison reuse.** Because tying
   shrinks the parameter space to `1 + n_nodes` (≤ ~130 in the real DAGs), the
   exact dense Newton solve is trivial and unbiased. We write a purpose-built
   `gated_alpha_newton_step` rather than bending vanilla's O(K) structured-inverse
   trick (which exists only to stay linear in a large free-per-topic K).

3. **Frontier histogram computed once from labels.** The allowed-set group
   structure is static (frontiers are observed training labels, not re-estimated),
   so the prior-normalizer part of the gradient/Hessian is a driver-side closed
   form read from a one-time `{frontier: count}` histogram — NOT a per-iteration
   distributed statistic.

4. **α only, not η.** The gate does not restructure the topic-word prior, so
   `optimize_eta` needs no gated changes and is out of scope (YAGNI).

5. **API mirrors vanilla.** The shim gains `optimizeDocConcentration` (bool,
   default `False`) — exact name/type/semantics of `OnlineLDAEstimator`.
   `nodeAlphaScale` remains the *initial* α; optimize refines from there.

## The math

Notation: ψ = digamma, ψ′ = trigamma, Γ = gamma function. A doc d's variational
topic posterior is q(θ_d) = Dir(γ_d) restricted to its allowed set
A_d = `lay.allowed_set(frontier_d)` (background ∪ closure-blocks of the frontier).
Groups g index the *distinct* allowed sets across the corpus; N_g is the corpus
doc-count of group g; block(u) is node u's topic block, |block(u)| = tpn
(and |block(bg)| = n_bg).

The part of the ELBO that depends on α, under gating:

    L(α) = Σ_g N_g · [ logΓ(Σ_{k∈A_g} α_k) − Σ_{k∈A_g} logΓ(α_k) ]
         + Σ_d Σ_{k∈A_d} (α_k − 1) · E[log θ_dk]

The first bracket is the Dirichlet log-normalizer of each group's *sub-simplex*
(this is where gating differs from vanilla: the normalizer is over A_g, not the
full K). The second is the data term.

**Gradient** wrt the tied parameter α_u (chain rule over the |block(u)| topics
that share α_u):

    g_u =  Σ_{g: u∈A_g} N_g · |block(u)| · [ ψ(Σ_{A_g} α) − ψ(α_u) ]      (PRIOR)
         + Σ_{d: u∈A_d} Σ_{k∈block(u)} E[log θ_dk]                        (DATA)

- PRIOR term: closed form on the driver from the static histogram {N_g} and the
  current α. Σ_{A_g} α = Σ over blocks in A_g of |block| · α_block.
- DATA term: the distributed E-step statistic S_u (below), corpus-scaled.

The same form gives g_bg (background block is in every allowed set, so the group
sum runs over all g).

**Hessian** (dense, `(1+n_nodes) × (1+n_nodes)`), from the second derivatives of
the same L. Writing m_u = |block(u)|:

    off-diagonal (u ≠ v):  H_uv = Σ_{g: u,v∈A_g} N_g · m_u · m_v · ψ′(Σ_{A_g} α)
    diagonal:              H_uu = Σ_{g: u∈A_g}   N_g · m_u² · ψ′(Σ_{A_g} α)
                                − Σ_{g: u∈A_g}   N_g · m_u  · ψ′(α_u)

H is the negative-definite ELBO Hessian; the Newton step is Δα = −H⁻¹ g solved
with `np.linalg.solve` (small dense system). The pure function returns the raw
Δα; the caller applies ρ_t damping and the 1e-3 floor (same contract as the
existing `alpha_newton_step`).

Reference: Blei, Ng, Jordan (2003) Appendix A.4.2 for the symmetric-simplex
Newton; this is its gated, block-tied generalization (per-group sub-simplex
normalizers, tying via the chain rule).

## Sufficient statistics & distributed flow

- **New per-iteration stat** emitted by the gated `local_update` when
  `optimize_alpha` is on: `e_log_theta_node_sum`, a length-`(1+n_nodes)` vector
  whose entry for block b is Σ over the batch's docs (with b in their allowed set)
  of Σ_{k∈block(b)} (ψ(γ_k) − ψ(γ_sum)). The runner corpus-scales it (ADR 0005)
  exactly like vanilla's `e_log_theta_sum`. When `optimize_alpha` is off, the stat
  is absent and behavior is byte-identical to today.

- **Static frontier histogram** `{frozenset(frontier): N_g}`: computed once from
  the training `labelCol` before the fit loop (a cheap groupBy/collect at
  foreground scale), and held on the model instance for `update_global` to read.
  Not emitted per iteration.

- **`update_global`** (when `optimize_alpha`): expand current tied α, assemble the
  dense gradient and Hessian from (static histogram, current α, corpus-scaled
  `e_log_theta_node_sum`), take the Newton step, ρ-damp, floor at 1e-3, and store
  the updated tied α (re-expanded to length-K for the next E-step and transform).

## File structure

- `spark_vi/inference/concentration_optimization.py` — add the pure
  `gated_alpha_newton_step(alpha_tied, block_sizes, e_log_theta_block_sum_scaled,
  group_counts, group_membership)` returning raw Δα_tied. Pure (no damping/floor),
  unit-testable in isolation, mirroring the existing `alpha_newton_step` contract.
- `spark_vi/models/topic/gated_lda.py` — drop the `optimize_alpha`
  `NotImplementedError`; store the tied-α layout + frontier histogram; emit
  `e_log_theta_node_sum` in `local_update`; call the gated step in `update_global`.
- `spark_vi/mllib/topic/gated_lda.py` — add the `optimizeDocConcentration` Param
  (bool, default False); compute the frontier histogram from `labelCol` at fit and
  pass it to the engine; thread the flag through.
- Tests: `test_concentration_optimization.py`, `test_gated_lda.py`,
  `test_gated_lda_shim.py`.

## Validation (acceptance)

**Primary gate — planted-α recovery.** Plant a synthetic gated corpus with known
per-node α_u (a mix of rare and common nodes); fit with `optimize_alpha=True`;
assert the learned α̂_u recovers the truth: Spearman correlation between α̂_u and
α_u above a pre-registered threshold, AND the rare-below-common ordering holds
(planted-rare nodes get smaller α̂ than planted-common ones). Threshold is set
from a first calibration run and NOT loosened after the fact (xfail with a
recorded reason if it fails, per repo test-honesty).

**Derivation guard (unit).** Test the pure `gated_alpha_newton_step` against a
finite-difference gradient of L(α) on a tiny synthetic (few groups, few nodes),
confirming the assembled gradient/Hessian match numerical differentiation. Locks
the derivation against sign/transcription errors.

## Interactions & risks

- **Mass-starved nodes self-regularize.** A node seen by few docs contributes a
  small N_u to both gradient and Hessian, so its Newton step is small and α_u
  stays near its `nodeAlphaScale` init. The 1e-3 floor prevents collapse.
- **SVI minibatch.** The data term is corpus-scaled and ρ-damped (as vanilla); the
  prior term is exact from the full static histogram (it is not sampled), so the
  two halves of the gradient are on the same corpus scale.
- **Transform unchanged.** `_transform` already consumes the fitted length-K α
  vector; it is simply asymmetric now.
- **`optimize_alpha` + spectral init** compose without interaction — init only
  seeds λ, not α.

## Out of scope

- Learning η (topic-concentration) for the gated engine.
- Per-topic (untied, free-K) α.
- Any change to the placement/readout or cloud driver beyond passing the new flag
  through experiment configs (that is thread A′, the fixed-asym sweep, and a later
  cluster experiment).
