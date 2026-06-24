# 0027 — Gated STM on imbalanced arms: the majority arm's foreground collapses into the shared background; the minority arm captures its anchor but not sub-phenotypes

**Date:** 2026-06-24
**Topic:** stm | gating | covariates | npmi | doc-units | diagnostics
**Status:** Observed

The first cluster run of gated background/foreground STM (experiment
[0004](../experiments/0004-gated-stm-cancer-dementia.md), ADR
[0026](../decisions/0026-gated-stm-hard-masking.md): 30 background + 10 cancer
foreground + 10 dementia foreground, K=50, `cancer_or_dementia` cohort) is the
**mirror image** of prevalence-only STM ([0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)).
Prevalence-only gave the *majority* arm (cancer) the crisp phenotype tail and the
*minority* arm (dementia) nothing. Gating flips which arm is represented in
dedicated topics — but for the dominant arm the flip backfires: its foreground
block collapses into degenerate near-empty slots, because its phenotypes are
better placed in the all-docs background.

Document split (NPMI references): full corpus 13,295 docs; cancer sub-corpus
10,819 (81.4%); dementia sub-corpus 2,476 (18.6%).

## The majority arm's phenotypes go to the background, not its foreground

The background block (shared by all docs) absorbs not just the universal
anchors but the **crisp cancer phenotypes**:

- 4 anchors hold ~92% of corpus mass (E[β] = 0.346 cardiometabolic +
  0.243 + 0.235 + 0.101 symptom anchors = 0.925) — the 0021/0026
  anchor-concentration signature.
- **Breast cancer** (topic 22, E[β]=0.034, NPMI 0.215, peak 0.080) and
  **skin/melanoma/derm** (topic 4, NPMI 0.298, peak 0.044) are *background*
  topics — exactly 0026's "majority-led crisp tail," now living in the shared
  block.
- ~24 remaining background slots are E[β]=0.0003, Σλ≈370–500, NPMI≈0.06, top-N
  = HTN/HLD/Obesity/Pain/GERD — the 0019/0026 "gracefully-unused slot echoing
  baseline" class.

The **cancer foreground block is entirely degenerate**: all 10 slots are
Σλ≈315–323, E[β]=0.0003, NPMI 0.027–0.045 (mean 0.033), with near-identical
top-N (HTN, HLD, breast-cancer 137809, Obesity, Actinic keratosis). No
differentiation, no distinct phenotype.

**Mechanism.** A background topic may fire on *every* document; a cancer
foreground topic only on cancer documents. With cancer at 81% of docs, the
dominant cancer phenotypes (breast, skin) explain the majority of the whole
corpus, so the optimizer places them in the background where they earn mass
across most documents. That leaves the cancer-only foreground block with no
residual distinctive signal — it collapses to baseline-echo. The fit-log block
masses confirm it is *occupied but contentless*: `blocks[bg=1.1e6
cancer=3.19e3 dementia=2.77e3]` — cancer foreground carries slightly *more*
mass than dementia, spread thin across 10 redundant slots. (Occupied ≠ used:
read Σλ/E[β] and top-N, not block mass alone.)

## The minority arm captures its anchor concept, but no sub-phenotypes

The dementia foreground block does something prevalence-only STM could not: the
**Dementia concept (4182210) reaches dedicated topics** — every one of the 10
dementia foreground topics is led by Dementia + Postconcussion syndrome
(372610). But:

- 9 of 10 are near-duplicates (Dementia + Postconcussion + baseline symptoms;
  Σλ≈200, E[β]=0.0002). No within-dementia differentiation (no Lewy-body /
  vascular / etc. subtype).
- 1 real-ish cluster: topic 41 (Σλ=968, the largest dementia slot) is
  musculoskeletal/aging (alopecia, knee/hip OA, fracture, carpal tunnel),
  cov 76% — a dementia-arm comorbidity cluster, not a dementia-distinctive
  phenotype.

Why dementia gets its anchor into the foreground while cancer does not: the
background, trained mostly on cancer, under-represents dementia, so the
dementia signal has nowhere to settle *except* its foreground block. Minority
status is what makes the foreground block useful.

## The cancer-vs-dementia NPMI gap is mostly a reference artifact

Naively the dementia foreground (NPMI 0.13–0.24) looks far healthier than the
cancer foreground (0.03). It largely is not — both blocks are mostly degenerate
duplicate slots. The gap is dominated by the **reference corpus**, per
[0010](0010-npmi-not-comparable-across-doc-units.md): each foreground block is
scored against its own group sub-corpus. Dementia's reference is 2,476
*homogeneous* docs where Dementia + the baseline symptoms co-occur in nearly
every document (→ high pairwise NPMI); cancer's is 10,819 *heterogeneous* docs
(many tumor types) where the same baseline terms co-occur less tightly (→ low
NPMI). The per-block reference is the right design (0007: scoring a rare arm
against the full corpus triggers the zero-pair penalty) — but it makes NPMI
**non-comparable across blocks**. Compare topics *within* a block, never the
block means against each other.

## Verdict on experiment 0004's success criterion 4

Criterion 4: "≥1 dementia foreground topic with NPMI > 0.10 **and** top-N not
dominated by universal anchor vocabulary." **Partially met / directional.**
The NPMI > 0.10 half passes for all dementia foreground topics, and the
Dementia concept reaching dedicated topics is genuine progress over
prevalence-only STM (0026, where it could not). But the top-N of those topics
*is* dominated by anchor vocabulary (Dementia is distinctive; everything after
it — Postconcussion, Pain, Chest pain, HTN, HLD, Nausea — is the universal
baseline). No crisp dementia-distinctive *phenotype* free of anchors emerged.
So gating buys the minority arm prevalence-and-anchor representation, not the
sub-phenotype resolution the criterion targeted.

## Implications and levers

1. **Don't allocate a foreground block to the majority arm.** On an imbalanced
   cohort it is wasted capacity — the majority's phenotypes settle in the
   all-docs background. Reserve foreground blocks for minority arms; let the
   background serve the majority.
2. **Balance the background if the majority must also be foregrounded.**
   Class-balanced / stratified SVI sampling (0026 lever 2) would stop the
   background from being a cancer model, freeing the majority foreground to
   carry distinct content — at the cost of a background that no longer reflects
   the true document mix.
3. **Gating gives presence, not resolution.** It guarantees a minority arm
   *appears* in dedicated topics (the prevalence-only failure mode it was
   designed to fix), but it does not by itself carve sub-phenotypes within an
   arm. Within-arm resolution needs either more foreground slots per arm with
   enough arm-specific mass, or content covariates / SAGE (0026 lever 1).
4. **Block mass is occupancy, not content.** A foreground block can be fully
   occupied (cancer here) yet contentless. Diagnose blocks by per-topic
   Σλ/E[β] spread and top-N distinctiveness, and never compare NPMI across
   blocks with different references (0010).

## Setting context

Gated STM (hard-masking, ADR 0026), K=50 = 30 background + 10 cancer + 10
dementia foreground, `group_var=source_cohort`. `cancer_or_dementia` cohort
(union of `first_cancer_year` + `first_dementia_year`, prior_obs_days=0, 365d
fully-observed follow-up), `patient_cohort` doc-unit, source_table
condition_era, person_mod=4 (~25% sample), prevalence formula
`~ C(sex) + age` (source_cohort deliberately excluded — it is the gating var,
ADR 0026), realized vocab 4,422, 100 SVI iters, batch_fraction≈0.2, τ_0=64,
κ=0.7, random_seed=42. NPMI eval: per-block reference (background → full corpus
13,295; cancer → 10,819; dementia → 2,476), min_pair_count=3, top_n=20. This is
the first on-cluster gated-STM run; prevalence-only baseline on the same cohort
is [0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md).
