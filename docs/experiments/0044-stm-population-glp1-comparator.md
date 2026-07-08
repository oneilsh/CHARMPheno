---
id: 44
slug: stm-population-glp1-comparator
status: done
model_class: stm
cohort: population_glp1
cohort_def: population_glp1
prior_obs_days: 365
person_mod: 1
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 110
background_k: 80
foreground: "glp1_ra:15,sglt2i:15"
group_var: source_cohort
max_iter: 300
subsampling_rate: 0.1
tau0: 256
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0044 — Population + GLP-1 & active-comparator gated STM

## Goal

What does the year after starting a GLP-1 receptor agonist look like — against
the general population AND against an active comparator (SGLT2 inhibitors)
started by a similar patient population — in one gated fit? The gated Σ yields
GLP-1↔comparator and GLP-1↔background topic (anti-)correlations. Contrasting
against a same-indication comparator controls for confounding-by-indication, so
what separates the drug foreground blocks is closer to drug-specific structure.

## Cohort

New drug-anchored `population_glp1` cohort (`apply_population_drug_cohort`),
one document per person, `source_cohort ∈ {glp1_ra, sglt2i, general}` — a clean
GLP-1-vs-SGLT2i active-comparator contrast:

- **glp1_ra / sglt2i** — incident new-users whose index year is monotherapy for
  that class (never the other, or the other started > 365d away); 365d prior
  coverage, 365d observed follow-up.
- **general** — no tracked drug exposure; random observed year with the SAME
  1yr-prior + 1yr-follow-up bracket.

A both-GLP1+SGLT2i user goes to the **earlier drug's single arm when the two
starts are > 365d apart** (the second drug is outside the index year, so that
year is genuine monotherapy) and is **excluded** otherwise (no combination arm).
**Tirzepatide (dual GIP/GLP-1) users are excluded entirely** — resolved (via its
own descendant set) only to keep them out of the GLP-1 arm and the background,
not fitted as an arm: at 128 docs it was too thin to model, and folding it into
glp1_ra would mix a dual-agonist into the pure-GLP-1-RA contrast.

Drug classes are resolved as `concept_ancestor` **descendants** of their seed
ingredients (RxNorm Ingredient names + any pinned ids; **tirzepatide pinned to
779705** — name-only resolution under-counted it to 128). The build logs each
class's resolved concept-set size + person count. Documents are the person's
conditions in the post-index year; drugs are the anchor only.

**Cache note:** the partition + concept-set logic is NOT part of the corpus cache
key, so a rebuild after these changes must be FORCED (clear the cache entry / use
the force flag) or the stale v1 corpus is reused.

## Configuration

Full population (`person_mod: 1`) — the drug arms are a small slice of the
population and need it. K=110 = 80 background + glp1_ra:15 + sglt2i:15. Incident
new-user (`prior_obs_days: 365`), 1-year windows. Otherwise the exp 0043 hardened
+ slowed stack (subsample 0.1, tau0 256, kappa 0.7, max_iter 300, reference +
dense spectral, sigma_init 1, min_pair_support 10, block-wise unit-diagonal Σ /
ADR 0034), `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- Build diagnostics report non-empty per-arm document counts and the both-user
  gap histogram (routing >365d gaps to the earlier monotherapy arm).
- Drug foreground arms recover recognizable, distinctive structure (e.g. GI /
  weight / appetite signal for GLP-1; cardiorenal signal for SGLT2i) on top of
  the shared T2DM indication.
- Σ variance bounded (no runaway); honest correlation report.

## Result

Ran on the cluster 2026-07-08. Fit converged at iter 300/300 (ELBO −5.53e6,
5708s ≈ 95 min) — the heaviest fit in the project (full population, K=110).
294,485 distinct persons / 8.76M condition rows → **177,359 documents**, frozen
vocab 10,000. `concentration readout: top_mass p50=0.132, eff_topics p50=27.5`.

**Descendant concept-set resolution confirmed** (the v2 fix): glp1_ra 1403
concepts / 20,722 persons, sglt2i 1006 / 13,528, tirzepatide 63 / 1,530 (resolved
only to exclude). Both-user |g−s| gap histogram: 0-7:422, 8-30:179, 31-90:438,
91-180:563, 181-365:944, **366+:3,848** — the 3,848 wide-gap both-users routed to
their earlier monotherapy arm; the ~2,546 within-365d excluded (no combination
arm).

Σ clean and bounded: block-wise unit-diagonal (`Σ_var[min=1 max=1]`), eig in
−0.675 to 38.3, `blocks[bg=7.7e6 glp1_ra=1.41e5 sglt2i=7.95e4]`. No runaway (the
eval's `runaway = topic 0 Σ_ii=1.000` is the argmax-over-constant-unit-diagonal
artifact, as in exp 0031, not a real blowup). Γ well-behaved (|Γ| max 3.62, mean
0.329). Background NPMI mean **+0.183** (median +0.173, stdev 0.088, min +0.063,
max +0.599, all 80 rated).

**All three success criteria met — and the drug-specific contrast is crisp and
clinically faithful.** The active-comparator design worked: both foreground
blocks carry the shared T2DM indication, and drug-specific structure separates
cleanly on top of it.

- **SGLT2i foreground → cardiorenal.** Two clean heart-failure topics (topic 99
  CHF / chronic diastolic HF / hypertensive HF / cardiomyopathy / systolic HF;
  topic 107 orthopnea / pericardial effusion / chronic systolic HF / syncope /
  cardiomyopathy), coronary disease (topic 109 coronary atherosclerosis /
  ischemic myocardium / old MI / aortocoronary bypass graft), and CKD /
  proteinuria (topics 96, 103). This *exceeds* the pre-fit guess ("genitourinary
  / volume") — it recovered the actual evidence-based HFrEF/HFpEF +
  cardiovascular-risk indication SGLT2i is prescribed for.
- **GLP-1 foreground → obesity + its signature GI adverse-effect footprint.**
  Topic 87 is the tell: morbid obesity co-occurring with nausea / vomiting /
  constipation / GERD — the classic GLP-1 GI side-effect cluster — alongside pure
  obesity/metabolic topics (topic 82 obesity / severe obesity / prediabetes;
  topic 83 obesity / renal disorder due to T2DM / proteinuria; topic 89 morbid
  obesity / OSA / steatosis / metabolic syndrome / PCOS).
- **Shared T2DM spine in both blocks** (glp1_ra topics 81/86/88; sglt2i topics
  95/102/108) confirms confounding-by-indication is controlled: the diabetes
  indication sits in both arms, and the HF/CAD-vs-obesity/GI split is what the
  gating actually separates.

See insight 0041. Next: export → ingest → `label_phenotypes` → add
`population_glp1` to the dashboard manifest as an additional cohort
(population_cancer stays default), same flow as population_eds/0043.

## Related

First cohort on the drug-anchor track (parallel to the disease track). Follows
exp 0043 (population_eds) on the same hardened + slowed stack. See insight 0041.
