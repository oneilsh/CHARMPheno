# ADR 0045 — PC scale-invariance: floor the gated α, per-doc-mean newton head, exponentiated-gradient mass-preserving λ-correction

**Date:** 2026-08-19
**Status:** Accepted (refines ADR 0044; ADR 0044's "ρ-damping suffices, no trust clip"
claim is superseded by decision 3 below)
**Context:** insight 0073 (the three serial scale bugs behind "PC neutral at whole-Mondo");
exps 0094–0098 (Mondo cardiovascular, C=437, K=444, full AoU population).

## Decision

Make the three PC controls that were latent scale bugs SCALE-INVARIANT. Each was `1/K`,
absolute, or a raw multiplier — fine on toy K/D, broken at whole-Mondo K/D.

### 1. The gated doc-concentration α is a floored prior constant, not `1/K`

The gated engine honors `docConcentration` (was hard-coded `alpha = 1/lay.K`). The default
`1/K` (0.0022 at K=444) collapses θ until the differentiable-CAVI Jacobian `∂θ/∂eb`
underflows (`2.7e-90`), killing the supervised shaping path. Set a scalar α **≥ 0.5** (the
Jeffreys prior; α=1.0 uniform is equally valid) so the Jacobian stays alive at any
`grad_cavi_iters`. Exposed as driver `--doc-concentration` → `docConcentration`. Do NOT
enable α co-fitting on this data: Blei's Newton-Raphson fits α to the sparse empirical
doc-topic distribution and drives it DOWN into the collapse regime (and preferentially
starves the rare-node topics we most want to shape). If α adaptation is ever wanted, floor
it (≥0.5) or decouple a fixed shaping-unroll α from a sparse inference α.

### 2. The newton head step is per-doc mean (`head_l2` becomes a scale-invariant ridge)

`OnlinePCLDA.update_global`'s newton branch divides the collected head gradient `g` and
Fisher `H` by `n_docs` before the ridge solve. The unregularized step `H⁻¹g` is unchanged
(the 1/N cancels), but `head_l2` now applies to per-doc-mean stats, so `|w|` settles at
`~|g_mean|/head_l2` — corpus-independent. This is the reference's
`rescale_total_loss_by_n_tokens`, applied to the per-doc head. Fixes the `|w| → 2.4e5`
runaway an absolute ridge on the ~110k-doc summed gradient produced.

### 3. The λ-correction is an exponentiated-gradient, MASS-PRESERVING step

`OnlinePCLDA._corrected_lambda_block` applies the supervised correction MULTIPLICATIVELY
and conserves each topic-row's total pseudocount:

    rel   = ρ · weight_y · nat / λ_unsup            # per-cell relative move (dimensionless)
    λ'    = λ_unsup · exp(−clip(rel, ±_EG_CLIP))    # exponentiated gradient (stays > 0)
    λ'   *= Σλ_unsup_k / Σλ'_k                       # renormalize each topic to its mass

`nat = grad_eb · eb` is the natural gradient of ADR 0044 (scale-stable). The additive step
ADR 0044 shipped (`λ_unsup − ρ·wy·nat`) has no simplex constraint: at large `weight_y` it
drains a topic's mass to ~0, and the empty topic's `E[logβ] → −∞` detonates the ELBO
(exp 0098: `−4.5e27`, `Σλ_k min → 12.9`). The Hughes reference never steps raw params — it
puts topics on the simplex (row-softmax) and uses NEF exponentiated-gradient (mirror
descent) steps that stay on the simplex by construction. The mass-preserving EG step mirrors
that: each topic redistributes vocab mass toward the supervised gradient but keeps its total
pseudocount fixed. **`weight_y` is bounded BY CONSTRUCTION** — numerically verified: Σλ
drift 2e-16 at `weight_y` up to 1000; reduces to the additive step for small moves.

## Why (over the alternatives)

- **α floor vs co-fit:** co-fitting α is the "principled" move but is actively harmful here
  (drives α into collapse); a fixed prior constant is both simpler and correct for the
  sparse-patient regime.
- **per-doc-mean vs a bigger absolute `head_l2`:** a larger absolute ridge is a corpus-tuned
  band-aid (needs re-tuning at each D); per-doc-mean removes the corpus dependence entirely.
- **EG mass-preserving vs a trust-region cap on `corr_relΔλ`:** a trust cap is a knob and
  only bounds the GLOBAL move, not per-topic starvation. The EG step is knob-free (only a
  numerical `_EG_CLIP` overflow guard), per-topic mass-conserving, and reference-faithful —
  it makes `weight_y` safe by construction rather than by clamping. This is why ADR 0044's
  "the RM ρ step-damping bounds the step, no trust clip needed" is retired: ρ does NOT bound
  it (weight_y multiplies outside ρ's effect), but the simplex geometry does.

## Consequences

- `weight_y` is now a SAFE hyperparameter (stability is hyperparameter-free by construction);
  it still sets shaping STRENGTH (the PC Lagrange multiplier), so its VALUE is an empirical
  choice, not a stability risk. A fully knob-free version would use the PC constraint
  formulation (target prediction loss ε, auto-solve the multiplier) — not adopted.
- `_EG_CLIP = 4.0` is a numerical overflow guard, not a tuning knob (a single-cell factor of
  e^±4 is already extreme and per-topic mass is conserved regardless).
- Guarded by `test_correction_is_mass_preserving_and_bounded_at_high_weight_y` (Σλ drift ≈ 0
  and λ > 0 at weight_y up to 1000) and the per-iter `corr_relΔλ` / `||grad_y||` / `|w_CK|max`
  driver diagnostics.
- Cleanup deferred: `_grad_topics_to_lambda` (the trigamma transform) is fully dead on the
  fit path; the `topic_trust` param is dead. Remove in a cleanup pass.
