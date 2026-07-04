---
id: 30
slug: stm-population-eds-gated
status: pending
model_class: stm
cohort: population_eds
cohort_def: population_eds
prior_obs_days: 0
person_mod: 1
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 100
background_k: 80
foreground: "eds:20"
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

# Experiment 0030 — Population-background + Ehlers-Danlos-foreground gated STM

## Goal

The rare-disease-on-a-background-population case: a gated STM whose background
is the whole population and whose single foreground group is Ehlers-Danlos
syndrome (EDS). Same asymmetric gated architecture as exp 0028 (population +
cancer), swapping the cancer anchor for a much rarer disease to test that the
foreground block recovers a recognizable EDS comorbidity signature (e.g. POTS /
dysautonomia, GI dysmotility, chronic pain, joint hypermobility / connective
tissue) rather than collapsing into the background.

## Cohort

New `population_eds` cohort (built on the generalized
`apply_population_disease_cohort`, disease registry key `eds`, OMOP ancestor
79145, no exclusions):

- **eds** (`source_cohort='eds'`): persons with a first EDS diagnosis, windowed
  to the 365 days after that diagnosis; carries the 20 EDS foreground topics.
- **general** (`source_cohort='general'`): every other person, windowed to a
  deterministic random 365-day event-anchored span; background-only.

`prior_obs_days: 0` admits prevalent EDS cases to maximize a rare arm.

## Configuration

Full population (`person_mod: 1`) — the heaviest fit in this batch — to maximize
the EDS foreground and give a rich background. **K=100 = 80 background + 20 EDS.**
The background was bumped from 40 to 80 topics: the first fit (below) resolved
191,872 background documents, far more than the 40-topic budget could resolve, so
the extra background capacity should split the general-population comorbidity
atlas more finely. The schedule was **slowed** (tau0 128→256, max_iter 200→300)
because the first fit converged early (iter 89) — a gentler Robbins-Monro warm-up
lets the larger corpus + larger K settle more gradually. Otherwise the exp 0028
hardened stack (subsample 0.1, kappa 0.7, reference + dense spectral, sigma_init
1, min_pair_support 10, block-wise unit-diagonal Σ / ADR 0034), `~ C(sex) + age`,
`known_sex_only`.

## Success criteria

- Covariate diagnostics show a realistic 2-level sex distribution.
- EDS foreground topics recover recognizable EDS-associated phenotypes; the EDS
  arm has enough documents (check corpus diagnostics — if thin, revisit
  `person_mod`).
- Σ variance bounded (no runaway); honest correlation report.

## Result (fit 1 — SUPERSEDED config: K=60 = 40 bg + 20 eds, tau0 128, max_iter 200)

> Refitting with K=100 (80 background) and a slower schedule (see Configuration).
> The finding below — that the EDS foreground carves faithful sub-phenotypes — is
> expected to hold; the background is what the larger K should sharpen.

Fit converged at iter 89/200 (ELBO −6.04e6, 2774s). Full population resolved to
332,502 persons / 191,872 fit documents; the **EDS foreground arm = 956 docs**
(≈0.5% of docs). Small but sufficient — at `person_mod: 4` this would have been
~240 docs, so the full-population choice was load-bearing. Σ bounded (all Σ_ii=1,
block-wise unit-diagonal; no runaway), `blocks[bg=7.85e6 eds=1.94e4]`. NPMI:
background mean +0.180 (max +0.587), EDS block mean +0.156 (max +0.286,
reference=956 docs).

All four success criteria met. The 20 EDS foreground topics recovered
clinically faithful EDS sub-phenotypes rather than collapsing into background:

- **POTS / dysautonomia** — topic 53 (tachycardia, orthostatic hypotension,
  palpitations), 41 (disorder of autonomic nervous system), 57 (hypermobile EDS
  type 3 + POTS + autonomic failure).
- **MCAS** — topic 45 (urticaria, anaphylaxis, systemic mast cell disease), 52
  (mast cell activation syndrome + joint derangement + TMJ + thoracic aortic
  ectasia).
- **Joint instability** — topic 48 (shoulder joint instability / dislocation),
  55 (hypermobility syndrome + connective-tissue disorder + mitral valve
  prolapse + Raynaud's).
- **Vascular EDS** — topic 58 (collagen disease + aortic aneurysm + migraine),
  59 (aortic dissection + vascular complication).
- **GI dysmotility / overlap** — topic 49 (EDS + fibromyalgia + gastroparesis +
  POTS — the classic overlap pentad).

Background 40 topics read as a clean general-population comorbidity atlas
(HTN/T2DM/HLD, CKD, low-back/knee/neck pain, anxiety/depression, COPD/dyspnea,
thyroid). A strong standalone demo model.

## Related

Follows exp 0028 (population + cancer gated). First cohort on the generalized
population+disease registry. Empirical finding logged as insight
[0035](../insights/0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md).
