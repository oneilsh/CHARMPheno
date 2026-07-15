# Full-covariance Σ for STM — design spec

**Status:** Implemented (this branch, SDD arc — Tasks 1-11)
**Date:** 2026-06-30
**Branch:** stm

## Goal

Replace the current diagonal (mean-field) logistic-normal covariance with a full
(K−1)×(K−1) Σ — the genuine CTM/STM treatment (Blei & Lafferty 2007) — so the
model captures and exposes **topic correlations** (which phenotypes co-occur).
The correlation structure is the deliverable, not just a cleaner engine: success
is an interpretable, well-conditioned correlation matrix on real data with topic
quality maintained.

### Why

Today `OnlineSTM` stores Σ as a K-vector of per-topic variances
([stm.py "Sigma" init](../../../spark-vi/spark_vi/models/topic/stm.py)) and every
downstream term is elementwise-diagonal (the Gaussian prior `diff*diff/Sigma_diag`,
the prior Hessian `diag(1/Sigma_diag)`, the η-KL). This forgoes the signature
feature that distinguishes CTM/STM from LDA — modeled topic correlation — and is
why the reference `stm` package's `sigma.prior` (a shrink-toward-diagonal lever)
is a literal no-op for us: there are no off-diagonals to shrink. Moving to a full
Σ restores the feature and makes both Σ priors meaningful again.

## Design decisions (settled in brainstorm)

1. **Goal:** the scientific feature — recover and expose topic correlations;
   correlation extraction is in scope (dashboard *surfacing* is a separate arc).
2. **Replace, don't toggle:** full Σ becomes the only path; the diagonal
   representation is removed (no `covariance_type` knob). The whole stabilizer
   stack (reference, spectral, gating, Σ-prior) re-validates under full Σ; the
   diagonal results (exp 0015/0017, insight 0030) become the *prior* baseline.
3. **Scope:** one spec covers both non-gated and gated/blocked models.
4. **Gated prior = marginal sub-block:** a hard-gated doc's active-topic prior is
   the marginal Gaussian η_A ~ N(μ_A, Σ_AA) — the A,A sub-block of Σ, inactive
   topics integrated out (matches hard masking; not conditional on a clamped
   value).
5. **Multi-group membership:** a document may belong to a *set* of groups; its
   allowed set is background ∪ the union of its groups' foreground blocks. Cross-
   group covariance is informed exactly by co-membership (comorbid) documents.
6. **Regularizers (both opt-in, default off):** the conjugate inverse-Wishart
   prior on Σ (Blei & Lafferty 2007) AND stm's diagonal-shrink `sigma.prior` ∈
   `[0, 1]` (Roberts et al.). Defaults reduce to plain full-cov MLE.
7. **No backward compatibility:** clean break — legacy diagonal checkpoints do
   not reload; re-fit under full Σ. No promote-on-load shim.

## Component 1 — Core full-Σ inference engine

η_d ∈ ℝ^(K−1) over the free topics (reference pinned at 0); prior η_d ~ N(μ_d, Σ)
with μ_d = Γᵀx_d (per-doc mean from covariates) and a single shared (K−1)×(K−1)
SPD Σ.

**E-step (per doc).** The softmax data term is unchanged. The prior term changes
from diagonal to full:

- log-prior: −0.5 (η−μ)ᵀ Σ⁻¹ (η−μ)
- gradient: −Σ⁻¹ (η−μ)   (was −(η−μ)/σ²)
- prior Hessian block: Σ⁻¹   (was `diag(1/Sigma_diag)`,
  [stm.py:191](../../../spark-vi/spark_vi/models/topic/stm.py#L191))

The full K×K data Hessian and the full Laplace covariance ν_d = inv(H) are
**already computed** ([stm.py:183-192](../../../spark-vi/spark_vi/models/topic/stm.py#L183-L192),
[_spd_inverse](../../../spark-vi/spark_vi/models/topic/stm.py#L201-L226)) — the
diagonal model simply discards the off-diagonals. Σ⁻¹ is computed **once per
global update** (Cholesky) and broadcast, never per doc.

**M-step (full residual covariance — the CTM update).**

Σ = (1/D) Σ_d [ (η̂_d − μ_d)(η̂_d − μ_d)ᵀ + ν_d ]

the mean residual outer-product plus the mean Laplace covariance. Online:
Σ ← (1−ρ)Σ + ρ·Σ_batch — a convex blend of SPD matrices stays SPD — with a
`sigma_ridge`·I floor (the ridge already exists).

**Reference topic** unchanged in spirit: Σ lives over the K−1 free topics; the
pinned reference carries no row/column (as it carries no variance today).

**Cost.** At K=40 Σ is 39×39: one Cholesky per global iter, one K×K inverse per
doc (already paid). Negligible vs the V-sized work.

## Component 2 — Gated marginal sub-blocks + multi-group

**Data model.** Generalize the gating key from scalar `doc_group_col` → a *set of
groups per doc* (`doc_groups`). A doc's allowed set:

A_d = background_indices ∪ ⋃_{g ∈ groups(d)} block_indices(g)

(single-group is the |groups(d)|=1 case — backward-compatible at the data layer).
Threads through the corpus builder, partition/topic-block spec, and the per-doc
record.

**E-step prior (marginal).** A gated doc optimizes η over A_d with the marginal
prior η_{A_d} ~ N(μ_{A_d}, Σ_{A_d,A_d}); prior precision = inv(Σ_{A_d,A_d}).
Allowed sets are determined by the group-*combination*, so inv(Σ_{A_d,A_d}) is
precomputed **once per distinct combination present in the minibatch** (few) and
broadcast — not per doc.

**M-step (pairwise lazy update — the ADR-0027 generalization).** Each doc scatter-
adds its (η̂−μ)(η̂−μ)ᵀ + ν_d into the A_d×A_d block of the Σ accumulator and
increments a parallel **support matrix** N_ij for every active pair (i,j) ∈
A_d×A_d. Then:

Σ_ij = accumulator_ij / N_ij   for N_ij ≥ min_pair_support;
entries with N_ij < min_pair_support fall back to the prior (their scatter is
zeroed, see Component 3/4).

A cross-group covariance is thus estimated **only from comorbid docs** that co-
activate both groups; pairs no doc co-activates remain at prior. This is the per-
*pair* version of today's per-*block* lazy rule
([corpus_mean_topic_proportions_gated_rdd](../../../spark-vi/spark_vi/mllib/topic/stm.py),
ADR 0027).

**Non-gated** is the degenerate case: one allowed set = all K−1 free topics,
N_ij = D everywhere, no scatter — identical to Component 1.

**The SPD catch (the central numerical risk).** In the non-gated case Σ is a sum
of PSD outer-products + PSD Laplace covariances → PSD by construction, SPD after
ridge. In the gated case, entries are estimated from *different doc subsets* and
some cross-group cells are pinned to the prior, so the assembled matrix is not the
covariance of any single sample and **need not be SPD**. This is guaranteed to
arise, not hypothetical: background topics appear in every allowed set, so they
correlate with every foreground; "background↔B and background↔C strong but B↔C
pinned" is the default structure of every gated fit. Mitigation is three layered,
load-bearing (not optional):

1. the inverse-Wishart prior (Component 3) fills uninformed entries with a
   coherent SPD scale instead of a bald zero;
2. an eigenvalue-floor projection to the nearest SPD matrix (generalizing the
   per-doc Hessian repair [_spd_inverse](../../../spark-vi/spark_vi/models/topic/stm.py#L201-L226)
   to Σ) — which also *imputes* unobserved cross-group entries to the value most
   consistent with the observed correlations (a principled completion);
3. the `sigma_ridge`·I floor.

## Component 3 — Regularizers (both opt-in, default off)

**Inverse-Wishart prior** (Blei & Lafferty 2007, the conjugate Σ prior).
Parameters: scale matrix Ψ = `sigma_prior_scale`·I and dof/pseudo-count
ν = `sigma_prior_count`. The M-step becomes the MAP under IW — blend the data
scatter S with the prior scale:

Σ = (S + Ψ) / (D_eff + ν)

shrinking Σ toward Ψ/(·) and regularizing uninformed entries (the SPD filler).
This **cleanly replaces** today's diagonal inverse-gamma: `sigma_prior_scale` /
`sigma_prior_count` keep their names; the diagonal inverse-gamma was exactly the
Ψ=ψI special case. Default Ψ=0, ν=0 → plain MLE.

Interaction with the per-pair support (gated path): D_eff is **per-entry** — the
data weight on entry (i,j) is N_ij, and the prior pseudo-count ν acts per-entry,
so Σ_ij = (S_ij + Ψ_ij) / (N_ij + ν). Thin cross-group entries (small N_ij after
the min_pair_support floor zeroes their scatter) are therefore **prior-dominated**
— precisely the "really pseudo-counts" regime the floor is meant to enforce. The
non-gated path is the uniform case N_ij = D for all entries, recovering the
single-scalar (S + Ψ)/(D + ν).

**stm diagonal-shrink** (`sigma.prior` ∈ `[0, 1]`, now meaningful under full Σ).
Applied after the IW blend: Σ ← (1−w)·Σ + w·diag(diag(Σ)), a new scalar knob
`sigma_diag_shrink` (default 0). Shrinks off-diagonal correlations toward zero —
the de-correlation lever — and incidentally aids SPD (diagonal dominance).

**Pipeline order:** data scatter (with the min_pair_support floor) → IW blend →
diagonal-shrink → ridge + SPD-repair. Both knobs at their defaults reduce exactly
to the Component 1/2 MLE.

## Component 4 — Correlation extraction, ELBO, storage, support floor

**Deliverable — correlation matrix.** From Σ: R_ij = Σ_ij / sqrt(Σ_ii·Σ_jj), the
(K−1)×(K−1) topic correlation over free topics (reference topic has no entry),
computed at export and stored with Σ. The support matrix N_ij (and the per-entry
measured-vs-imputed annotation it enables) is **deferred to the dashboard-surfacing
arc** — not persisted in this arc.

**Small-support floor (robustness + privacy).** `min_pair_support` is a hard gate:
a covariance entry backed by fewer than `min_pair_support` co-activating documents
is statistically unreliable AND a small-cell disclosure risk. Below the floor the
few real docs' contribution is **zeroed out of the scatter** (S_ij → 0) and the
entry falls back to the prior / SPD-completion — never a raw few-patient estimate.
Background and within-group cells have massive support and never trip it; the
floor bites exactly the thin cross-group cells. The IW prior pseudo-counts own
those cells (the "really smeared / really pseudo-counts" regime). Default value to
be chosen in planning (small-cell convention, consistent with existing
suppression norms).

**ELBO / η-KL.** The Gaussian KL term
([stm.py:547-553](../../../spark-vi/spark_vi/models/topic/stm.py#L547-L553))
generalizes from the diagonal to the full form — trace(Σ⁻¹ ν_d),
(η̂−μ)ᵀ Σ⁻¹ (η̂−μ), and logdet Σ via the Cholesky already computed. A gated doc
uses its marginal sub-block Σ_{A_d,A_d}. Mechanical but real; gets golden tests.

**Storage / serialization.** `global_params["Sigma"]` goes from a K-vector to a
(K−1)×(K−1) matrix (`.npy` handles the shape change). New persisted artifacts: the
correlation matrix R and a free-topic↔topic-id map for labeling. No legacy migration
(clean break). The support matrix N_ij is **deferred** — not persisted in this arc.

**Diagnostics.** The `Σ[min…max]` trace (ADR 0030) generalizes to: eigenvalue
range + condition number of Σ, and max |off-diagonal correlation|. New per-iter
health signals. The `imputed_fraction` diagnostic (share of entries below the
support floor) is **deferred to the dashboard-surfacing arc** — it requires the
persisted N_ij, which is not stored in this arc.

**Downstream (noted, not built here):** a full Σ un-parks the ADR-0028-B logistic-
normal sampler (it can sample N(μ,Σ) properly); dashboard surfacing of R is the
separate dashboard arc.

## Component 5 — Validation

**Unit / synthetic tests.**
- Non-gated: recover a *planted* Σ on synthetic correlated-topic data.
- Reduces-to-MLE: non-gated, both regularizers off = plain full-cov.
- Marginal sub-block: gated doc E-step prior precision = inv(Σ_AA), checked on a
  hand-computed small case.
- Multi-group: a 2-group doc activates the union; a cross-group entry gains
  support only from comorbid docs.
- min_pair_support: thin cross-group cells fall back to prior, scatter zeroed,
  support matrix correct.
- SPD: deliberately inconsistent gated blocks → repair yields valid SPD = nearest
  completion.
- Regularizers: IW ν=0 ⇒ MLE, ψI shrinks toward scale; diagonal-shrink w=0 ⇒
  identity, w=1 ⇒ diagonal.
- ELBO golden values; reference topic carries no Σ row/column.

**Cluster experiments (100 iters).**
- **Non-gated full-Σ on the cancer cohort** — the feature run: recovers sensible
  phenotype correlations (comorbidity clusters), holds topic quality (NPMI ≈ exp
  0017), keeps Σ well-conditioned.
- **Gated multi-group on a comorbid cohort** (e.g. cancer + dementia) — validates
  cross-group covariance from comorbid patients + the small-support suppression
  end-to-end.

The diagonal results (exp 0015/0017, insight 0030) are the prior baseline. The
diagonal-Σ stability experiments 0018/0019 are **superseded** by this arc (the
full-Σ run reveals whether any blowup persists; the IW prior is the lever if so).

## Out of scope / downstream

- Dashboard surfacing of the correlation matrix (separate dashboard arc).
- Un-parking the ADR-0028-B logistic-normal sampler (enabled by full Σ, built
  later).
- The gated-LDA / PLDA separate-class redesign (this arc is the STM covariance;
  PLDA is tracked separately).

## References

- Blei & Lafferty (2007), "A Correlated Topic Model of Science," Annals of
  Applied Statistics — the logistic-normal full-Σ treatment and the conjugate
  inverse-Wishart prior on Σ.
- Roberts, Stewart & Tingley (2019), the `stm` R package — `sigma.prior` ∈ `[0, 1]`
  diagonal-shrink regularizer (default 0); source
  https://github.com/bstewart/stm .
- insight 0029 (the three missing stabilizers), insight 0030 (diagonal-Σ spectral
  result, the prior baseline), ADR 0027 (lazy block update — generalized here to
  per-pair), ADR 0028-B (the parked logistic-normal sampler), ADR 0030 (the Σ
  diagnostic trace), ADR 0031 (K−1 reference).
