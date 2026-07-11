# PG-STM Inference Core — Design (sub-project 1 of 3)

**Date:** 2026-07-11
**Branch:** pg-stm (off the clean, reviewed stm base 0eeb43e)
**Thread:** Pólya-Gamma / full-Bayes gated STM — keep the Gaussian latents, replace the estimator.
**Status:** design (this doc), pre-plan.
**Decomposition:** this is sub-project **1 of 3** of a production PG-SVI engine —
(1) **PG inference core** [this doc], (2) distributed SVI wrapping the core, (3) fit/
export/dashboard integration (Σ posterior in the bundle + LKJ prior refinement).

## Motivation — the bet

The entire scale-calibration scaffolding on the parked branch (pin Σ → held-out
sweep → MAP compression → bias maps → f-drift → t-prior) is a **workaround for one
root cause: point-estimate EM on a logistic-normal prior**. The Σ runaway that
started it (insight 0033) is an **estimator-class pathology** — a point estimate of a
weakly-identified variance feeding its own prior with no trust region diverges — not a
property of the model. The shipped fix (ADR 0034: pin Σ's diagonal to 1, discard the
variances) confirms the diagnosis by removing exactly that degree of freedom.

**The bet:** keep the Gaussian latents (they carry the deliverables the Dirichlet route
drops — correlation R, continuous covariates Γ, gating), and **replace Laplace/MAP with
Pólya-Gamma conjugate augmentation + a proper prior on Σ, inferring a posterior rather
than a point**. A weakly-identified topic's variance then shows up as a *wide posterior*,
not a divergent point estimate, and nothing re-enters its own prior as a point value. If a
proper prior + posterior cures the runaway, the whole pin-and-calibrate arc retires at the
root — and the MAP compression (hence the bias maps) goes with it.

This sub-project builds the minimum needed to **test that bet**, in a form that is the
production engine's kernel (not a throwaway prototype).

## Scope

Single-machine, full-batch **PG-VI** core + an exact **PG-Gibbs cross-check**, validated on
synthetic. **No Spark, no export, no dashboard** (sub-projects 2 and 3). The deliverable is
a go/no-go checkpoint (below), not a shippable model.

## The model

Generative, per document d with covariates x_d and allowed topic block A_d (background ∪
its group):

- **Nested (hierarchical) stick-breaking link** (Linderman, Johnson & Adams 2015 flat
  stick-breaking, composed into two levels — the composition is what makes gating consistent
  under stick-breaking; flat single-sequence stick-breaking is NOT closed under subsetting the
  allowed topic set, so a shared background Σ would be ill-defined). For a group-g doc:
  - **Level 0 — the gate** (one logit ψ_gate): `π_bg = σ(ψ_gate)`, `π_fg = 1 − σ(ψ_gate)` split
    mass between the shared background block and the doc's foreground block F_g.
  - **Level 1 — within each block** (flat stick-breaking): `θ_k = π_bg · sb(ψ_bg)_k` for k in
    background (B−1 sticks, SHARED across all docs), `θ_k = π_fg · sb(ψ_fg_g)_k` for k in F_g
    (m_g−1 sticks, g's block), and **θ_k = 0 exactly** for k in any other group (hard gating).
  Because the within-background break sits *under* π_bg, background stick j means the same
  thing for every doc regardless of group — so a single shared Σ is well-defined. This nested
  structure also generalizes to an ontology cascade (gate → level-1 categories → … → leaf
  phenotypes), which is why gating is the entry point for later ontology-guided fits.
- **Correlated Gaussian prior on the full stick vector** ψ_d = [ψ_gate, ψ_bg (B−1),
  ψ_fg_1 (m_1−1), …, ψ_fg_G] ~ Normal(Γᵀx_d, Σ). A group-g doc's E-step runs on the ACTIVE
  sub-vector [ψ_gate, ψ_bg, ψ_fg_g]; the other groups' foreground sticks are inactive. Γ
  carries continuous covariates; Σ is **block-structured** (a gate dim, a background block,
  one block per group, learned cross-terms gate↔background and gate↔group, and group↔group'
  never co-active → kept at prior). Σ is the covariance of the stick logits (an order-
  dependent reparameterization of the current STM's R — refit, not transferred).
- **Tokens:** z_{d,n} ~ Categorical(θ_d); w_{d,n} ~ Categorical(β_{z}). β is (K×V).
- **Pólya-Gamma augmentation** (Polson, Scott & Windle 2013): the counts factor into binomials
  — the **gate** (N_bg "successes" out of N total), plus flat within-block stick-breaking
  binomials in background and in F_g (at stick k, n_k successes out of trials-at-risk
  b_k = Σ_{j≥k} n_j *within the block*); augmenting each with ω ~ PG(b, ψ) makes ψ_d
  **conditionally Gaussian**.

Two consequences, both accepted:

- **No reference topic.** Nested stick-breaking is inherently identified (the gate + each
  block's flat stick-breaking are all bijective), so the softmax translation gauge — and the
  reference-topic pin that fixed it — disappears. The full stick vector has dimension
  **K − G** (1 gate + (B−1) background + Σ_g(m_g−1) foreground, for G groups); Σ and Γ are
  (K−G)-dimensional. No pinned coordinate, no `Γ[:,0]≈0` special-casing.
- **Σ prior = Inverse-Wishart** (this sub-project). Conjugate to the Gaussian → closed-form
  Σ update in both the VI kernel and the Gibbs cross-check; proper → a genuine trust region
  that tests the runaway-cure with the least new machinery. Block-structured to honor gating
  (background ∪ group; cross-group entries are never co-active, updated from within-block
  scatter only, mirroring today's marginal-precision logic). IW couples correlation and
  scale; the interpretable **LKJ(correlation) + half-normal(scale)** prior that decouples
  them (ADR 0034's intent) is a **deferred refinement for sub-project #3**, taken on only if
  the bet holds — it is not load-bearing for the mechanism test.

## Inference

**Full-batch mean-field PG-VI** — deliberately built as the Tier-3 SVI kernel at batch =
full corpus, so sub-project #2 is "the same updates, minibatched + `treeReduce`-d with a
Robbins-Monro ρ," no algorithm rework. Coordinate ascent over q(z)q(ψ)q(ω)q(β)q(Γ)q(Σ):

All operate on the doc's ACTIVE stick vector [ψ_gate, ψ_bg, ψ_fg_g] and the block-Σ marginal
over those dims (mirroring the current gated STM's marginal-precision E-step):

- ω_d | ψ_d → Pólya-Gamma (closed form) — per gate + per within-block stick
- ψ_d | ω_d, counts, Γ, Σ → Gaussian (the PG result), over the active sticks
- Σ | {ψ_d − Γᵀx_d} → Inverse-Wishart (conjugate; **the update that used to run away, now a
  proper posterior**), block-structured (gate dim, background block, per-group blocks, learned
  gate↔block cross-terms; group↔group' kept at prior)
- Γ | {ψ_d} → Gaussian / ridge (as current STM)
- β | topic-word stats → Dirichlet (as current STM)

**The one deliberate approximation:** the token-assignment update needs E_q[log θ_k], which
under a stick-breaking-Gaussian ψ is not closed form (unlike Dirichlet's digamma). The VI
kernel uses a **delta-method** approximation, composed per the nested structure:
E[log θ_k] = E[log σ(ψ_gate)] (background) or E[log(1−σ(ψ_gate))] (foreground) + the flat
per-block delta-method term. This is the mean-field bias that could in principle mask Σ
behavior.

**Gibbs cross-check** (exact): sample ψ | ω, counts (Gaussian); ω | ψ (PG); z | ψ, β
(Categorical with θ computed *exactly* from sampled ψ); Σ | ψ (IW); Γ, β conjugate. This
audits the delta-method and the Σ posterior — it is the instrument that distinguishes "the
bet fails" from "mean-field is the culprit."

## Success criterion — the milestone-1 checkpoint

The runaway is a mechanism and reproduces in controlled synthetic; real-corpus confirmation
defers to sub-project #2 (when the distributed engine can run exp-0027). Three synthetic
tests:

1. **Runaway reproduction + cure (decisive).** Plant a gated corpus with a known Σ
   containing a weakly-identified topic (doc-scarce, effective sample size ≈ 15 — the
   bias-map identifiability-audit regime). Run the current point-estimate EM with the
   **diagonal un-pinned** (the config that blew up) and PG-VI (+ Gibbs) on it. **Pass** =
   point-EM diverges (reproduces the ~10^10 blowup) while PG-VI's Σ posterior stays
   **bounded** and the weakly-identified topic shows a **wide posterior**, not a divergent
   point.
2. **Structure recovery.** On a well-identified planted corpus, PG-VI recovers β, Γ, Σ
   within tolerance: `planted_recovery` ≥ the STM baseline (minus noise margin), Σ-posterior-
   mean correlation ≈ planted, per-group foreground recovered (`foreground_recovers_group`).
3. **VI–Gibbs agreement.** PG-VI's Σ posterior (mean + spread) matches the exact Gibbs
   posterior on both regimes. Material disagreement is itself a finding (the delta-method
   bound needs work before Tier 3).

**Checkpoint decision:** 1+2+3 pass → the bet holds, proceed to sub-project #2. Test 1 fails
(PG-VI also diverges or over-shrinks) → **the bet is dead, and that is a publishable
headline** ("the gated runaway survives full Bayes → pinning is fundamental"); we stop at
days of cost. This asymmetry is the entire reason the core is built and gated first.

## Components, reuse, dependency

**New (single machine, no Spark/export):** `spark-vi/spark_vi/models/topic/pg_stm.py` — the
stick-breaking link + count factorization, the PG-VI coordinate updates (ω, Gaussian ψ, IW
Σ, delta-method assignment), and the PG-Gibbs cross-check (same module or a test sibling),
plus a runaway-reproduction harness that plants the doc-scarce topic.

**Dependency:** `polyagamma` (Bleki's maintained, numpy-based PG sampler; PyPI
`polyagamma==2.0.2`, ships arm64/x86 wheels). Verified installing cleanly and sampling in
the dev env — `random_polyagamma(h=b, z=ψ)` is the vectorized PG(b, ψ) draw the ω update
needs. Chosen over the older `pypolyagamma` (Linderman's C++ package), which fails to build
against modern setuptools (stale `pkg_resources` import) — verified broken here. Because
`polyagamma` ships wheels, there is no cluster-image build risk in sub-project #2 either, so
the vendor-a-sampler fallback is not needed.

**Reuse:** `tests/_stm_synth.py` (`synthetic_gated_corpus`, `planted_recovery`,
`foreground_recovers_group`), `TopicBlockPartition` (gating), and the Γ-ridge / β-Dirichlet
update shapes from the current STM. Note: most scale-calibration diagnostics (sweep, drift,
bias-map) become irrelevant if the bet holds; the survivors are `planted_recovery` and the
generate-then-reinfer adequacy check.

**Testing (TDD):** stick-breaking ↔ simplex bijection (round-trip); PG-augmented ψ
conditional matches a reference Gaussian; IW Σ update correctness; the runaway
reproduction (point-EM diverges) + cure (PG-VI bounded, wide posterior); planted recovery;
VI–Gibbs Σ agreement.

## Out of scope (later sub-projects)

- Distributed SVI (minibatch + `treeReduce` + `StreamingPGSTM`) — sub-project #2.
- Export/dashboard: carrying a Σ *posterior* (with uncertainty) into the bundle + the
  correlation view; retiring the pin/calibrate export blocks; the LKJ + half-normal prior —
  sub-project #3.
- Real exp-0027 runaway confirmation (needs the distributed engine to run at scale).

## References

- Polson, Scott & Windle 2013, "Bayesian inference for logistic models using Pólya-Gamma
  latent variables", JASA (the augmentation).
- Linderman, Johnson & Adams 2015, "Dependent multinomial models made easy: stick-breaking
  with the Pólya-Gamma augmentation", NeurIPS (the multinomial link this uses).
- Blei & Lafferty 2007, "A correlated topic model of Science" (the logistic-normal topic
  model this re-infers).
- Lewandowski, Kurowicka & Joe 2009 (LKJ prior — deferred to #3).
- Internal: insight 0033 (the per-topic-variance runaway), ADR 0034 (block-wise unit-
  diagonal — the pin this replaces), insights 0037/0038 (held-out calibration lineage the
  bet aims to retire). Project memory: `project_concentration_scale_thread`.
