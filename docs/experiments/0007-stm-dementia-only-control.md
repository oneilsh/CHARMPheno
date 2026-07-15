---
id: 7
slug: stm-dementia-only-control
status: pending
model_class: stm
cohort: dementia
cohort_def: first_dementia_year
prior_obs_days: 0
person_mod: 4
doc_unit: patient
doc_min_length: 20
K: 40
max_iter: 100
vocab_size: 10000
min_df: 20
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
print_topics_every: 10
---

# Control: non-gated STM on the dementia arm alone (isolate model class vs. gating)

The decisive missing cell. We have:

| | LDA | STM |
|---|---|---|
| **dementia alone (mod=4)** | [0006](0006-lda-dementia-only-control.md) ✓ recovers Alzheimer's/HIV/seizure | **this run** |
| **combined (dementia = minority)** | (0021-family ✓) | [0005](0005-gated-stm-dementia-only-foreground.md) / [0026](../insights/0026-stm-prevalence-gives-prevalence-not-content-fidelity.md) ✗ |

0006 proved plain LDA recovers crisp dementia sub-phenotypes from this exact
corpus (person_mod=4, first_dementia_year). The gated/combined STM runs (0005,
0026) did not surface dementia sub-phenotypes. A local reproduction (real
`OnlineSTM` code, in-process) found **no gating bug** — the masking/M-step are
correct, degradation is smooth, and gating does not under-perform non-gated STM.
That points the open question at **model class** — STM's logistic-normal
document–topic prior vs. LDA's Dirichlet — rather than gating. This run isolates
exactly that variable.

## Design: change only the model class vs. 0006

Everything matched to 0006 — `first_dementia_year`, `person_mod=4`,
`prior_obs_days=0`, `patient` doc-unit (one lifetime doc per dementia patient),
`doc_min_length=20`, condition_era, `vocab_size=10000`, `min_df=20`, K=40,
`max_iter=100`, `random_seed=42` — **except** the model is non-gated STM
(prevalence-only). No `background_k`/`foreground`, so this is plain STM, not the
gated variant. `source_cohort` is absent (single cohort), so the covariate join
keys on `person_id` alone and the `patient` doc-spec is accepted.

The covariate formula `~ C(sex) + age` is the minimal STM-required prevalence
model (STM needs covariates; sex/age are incidental and were shown in
[0026](../insights/0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)
to leave the topic-word structure essentially LDA-like). So a difference vs.
0006 is attributable to the **prior family**, not the covariates — sex/age do
not cause topic collapse.

## What the result decides

Read the per-iter top-terms log (now emitted for STM every 10 iters) and the
per-topic **peak β** / Σλ spread / top-term diversity — not the NPMI mean
(0027: NPMI can reward degenerate uniform topics; [0010](../insights/0010-npmi-not-comparable-across-doc-units.md)).

1. **STM collapses like 0005** (no tail topic escapes the universal
   Dementia+Postconcussion+baseline signature; peak β stays ~uniform) → the
   cause is **STM's logistic-normal prior**, not gating. Implication: gating
   itself is sound; for minority *content* discovery the lever is **gated LDA**
   (Dirichlet) or content covariates / SAGE — not balanced-background SVI.
2. **STM recovers like 0006** (distinct dementia sub-phenotypes — Alzheimer's,
   HIV-dementia, seizure — at peak β well above uniform) → the prior is fine on
   a clean single-cohort corpus, and the 0005 collapse is specifically about the
   **combined / minority** setting (shared β estimated under 79–81% cancer
   token mass). Then the dig moves to class-balanced SVI / the combined-corpus
   estimation, and gating is exonerated.

Either branch turns the 0005/0027 thread into a correctly-attributed insight,
which is why it is worth one more run before writing anything up.
