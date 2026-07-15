---
id: 31
slug: stm-population-sparse-gated
status: done
model_class: stm
cohort: population_sparse
cohort_def: population_sparse
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 5
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 40
foreground: "sparse:10"
group_var: source_cohort
max_iter: 200
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0031 — Whole-population density-split gated STM (no disease anchor)

## Goal

The "better 0029": read what light-coder general years are made of, against a
clean whole-population background with NO cancer arm mixed in. The population is
windowed then split by in-window coding density — dense years form the
background, light-coder (5–19 code) years get their own `sparse` foreground
block. If the sparse foreground reads as wellness/screening/routine, the
short-doc floor is well-justified; if it shows structured conditions, short docs
carry real signal.

## Cohort

New `population_sparse` cohort (`apply_population_sparse_cohort`, outside the
disease framework — no concept set):

- **general** (`source_cohort='general'`): persons whose event-anchored 365-day
  window has >= 20 codes; background-only.
- **sparse** (`source_cohort='sparse'`): persons whose window has 5–19 codes;
  10-topic foreground block. Persons with < 5 codes dropped
  (`doc_min_length: 5`).

## Configuration

K=50 = 40 background + 10 sparse. 25% sample (`person_mod: 4`) — ample for a
whole-population light-coder read. Otherwise the exp 0028 gentle + hardened
stack, `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- Sparse foreground topics interpretable (wellness/screening vs structured
  conditions) — the answer to the short-doc-floor question.
- Σ variance bounded; honest correlation report.

## Result

Ran on the cluster 2026-07-03. Fit converged at iter 200/200 (ELBO −1.59e6,
925s). 62,565 persons / documents at `person_mod: 4`; the density split landed
near 50/50 — **31,269 light-coder docs in the `sparse` foreground arm** (5–19
codes) against ~31,300 dense general docs, frozen vocab 6115. Σ bounded and
proper (block-wise unit-diagonal, all Σ_ii=1; eig in 0.203–5.07;
`blocks[bg=2.13e6 sparse=5.42e4]`); no runaway (the eval's
`runaway = topic 49 Σ_ii=1.000` is an artifact of argmax over the constant
unit diagonal, not a real blowup). Γ well-behaved (|Γ| max 3.33, mean 0.34).
NPMI: background mean **+0.192** (median +0.175, max +0.627, all 40 rated);
sparse block mean **+0.100** (median +0.084, min +0.022, max +0.220, all 10
rated).

**Both success criteria met, and the short-doc-floor question is answered:
light-coder years are predominantly routine, with real structured pockets.**

The 10 sparse foreground topics read mostly as wellness / screening / routine /
acute-minor care — which justifies the `doc_min_length: 5` floor — but they are
not empty of signal:

- **Routine / screening / acute-minor** (the majority): metabolic screening
  (topic 45 HTN + hyperlipidemia + CAD + prediabetes; 44 T2DM + HTN + lipids),
  refraction (42 myopia / presbyopia / astigmatism), audiology (49 sensorineural
  hearing loss / impacted cerumen / tinnitus), vitamin-D / alcohol screening
  (40), acute URI (41 pharyngitis / sinusitis / bronchitis), obesity + snoring
  (48), acute cardiorespiratory symptoms (47 cough / palpitations / dyspnea /
  COVID-19).
- **Genuinely structured pockets**: MSK (topic 46 joint pain / knee OA / carpal
  tunnel — the single most coherent sparse topic at **NPMI +0.220**), and the
  metabolic pair carry real comorbidity signal rather than noise.

The sparse block's lower coherence (+0.100 vs the background's +0.192) **is
itself the result, not a defect**: light-coder documents have few codes, so few
within-doc co-occurring pairs, so intrinsically lower NPMI — and routine /
screening content is diffuse by construction. The spread across the block traces
the wellness-to-signal gradient directly: topic 43 (headache / abdominal pain /
acne / dysuria, +0.022) is the diffuse floor; topic 46 (MSK, +0.220) is the
coherent pocket.

Background 40 topics are a clean whole-population comorbidity atlas — CKD/CHF,
T2DM, allergic rhinitis/asthma, dermatology, knee OA/RA, cataract, OSA,
breast/bone, AFib/CAD, hypothyroid, BPH/prostate, HIV/HepC, seizure/epilepsy,
PTSD/bipolar, lupus/Sjögren — strong standalone demo material, consistent with
exp 0028/0030's background.

See insight 0040. Not exported to the dashboard: this is a methods/robustness
result (what short docs carry), not a headline demo cohort; population_cancer +
population_eds remain the two demo models.

## Related

Reframes exp 0029 (population + cancer + sparse) without the cancer arm.
