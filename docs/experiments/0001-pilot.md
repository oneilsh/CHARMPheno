---
id: 1
slug: pilot
status: pending
model_class: lda
cohort: dementia
created: 2026-05-28
K: 5
max_iter: 5
vocab_size: 500
print_topics_every: 1
---

# Experiment 0001 — pilot

## Intent
First end-to-end validation of the experiment-tracking pipeline.
Tiny K (5) and max_iter (2) so the smoke is cheap. After this proves out,
mark `status: done` and the next real experiment starts at 0002.

## Fit history
- 2026-05-28 20:38 UTC — **Session 1**: failed at argparse. Frontmatter sent
  `--cohort dementia`; LDA driver expected the long form `first_dementia_year`.
  Fix: split `cohort` (display id) and `cohort_def` (driver value) in defaults
  files; `build_lda_args` reads `cohort_def`. Commit `3fe2eb6`.
- 2026-05-28 20:46 UTC — **Session 2**: failed with
  `FileNotFoundError: No manifest.json at resumeFrom path`. Wrapper auto-resumed
  because the save_dir existed (from session 1's pre-fit `mkdir`), but no
  checkpoint had been written. Fix: resume detection now keys on
  `(save_dir / "manifest.json").exists()` rather than just the directory.
  Commit `d971796`.
- 2026-05-28 20:52 UTC — **Session 3**: clean run. Fresh fit, 2 iters,
  aggregates written, eval ran, NPMI per-topic table printed.
- 2026-05-29 — Status flipped back to `pending` (and `max_iter` bumped 2→5) to
  validate the Increment 2 / 2.5 chain: `make next-exp NO_EVAL=1 && make eval-exp`.
  Expected: fit auto-resumes from the existing checkpoint, runs 3 more iters,
  writes `## Fit session 4` with `--no-eval` skipping eval; then `eval-exp`
  auto-discovers id=1 (most-recent manifest.json) and appends a timestamped
  `## Eval (NPMI) — <ts>` section.

## Results

Full summary as written by the cluster (three sessions accumulated; the
session-3 fit + eval portion is what actually represents the modeled run):

```
# Experiment 0001 — pilot

## Effective config
K: 5
cohort: dementia
created: 2026-05-28
doc_min_length: 20
doc_unit: patient_year
id: 1
kappa: 0.7
max_iter: 2
min_df: 20
min_patient_count: 20
model_class: lda
optimize_doc_concentration: True
optimize_topic_concentration: False
person_mod: 10
print_topics_every: 1
save_interval: 5
seed: 42
slug: pilot
source_table: condition_era
status: pending
subsampling_rate: 0.2
tau0: 64
top_n_tokens: 6
vocab_size: 500

## Fit session 3 (clean)
[driver] cdr=wb-affable-acorn-7941.R2024Q3R8, billing_project=wb-fresh-seed-6621, K=5, max_iter=2, person_mod=10
[driver] Spark 3.5.3, master=yarn, defaultParallelism=2
[driver] >>> BQ load + summary
[driver]   OMOP: 135125 rows, 1122 distinct persons
[driver] <<< BQ load + summary: 51.5s
[driver] >>> vectorize (CountVectorizer, doc_spec=patient_year, min_doc_length=20)
[driver] <<< vectorize: 9.7s
[driver] >>> concept-name lookup
[driver]   resolved 500 concept names
[driver] <<< concept-name lookup: 5.7s
[driver]   vocab size: 500 (cap 500, minDF 20), documents: 1374
[driver] >>> fit (K=5, maxIter=2, ...)
[driver]   iter 1/2: ELBO=-97881.5970, batch=245, rho=0.0538, 86.4s
[driver]   α[min=0.1959 max=0.1973 mean=0.1965], η=0.2, Σλ_k[min=1.1e+03 max=1.58e+03]
[driver]   --- topics @ iter 1 ---
[driver]    topic  3  α=0.1969  E[β]=0.2405  Σλ=1.58e+03  peak=0.013  | Chest pain, Constipation, Shortness of breath, Vomiting, Diarrhea, Nausea
[driver]    topic  0  α=0.1973  E[β]=0.2200  Σλ=1.45e+03  peak=0.012  | Hyperlipidemia, Chest pain, Shortness of breath, Anxiety, Dementia, Nausea
[driver]    topic  2  α=0.1964  E[β]=0.1897  Σλ=1.25e+03  peak=0.013  | Chest pain, Hyperlipidemia, Essential hypertension, Nausea, Insomnia, Pain
[driver]    topic  4  α=0.1961  E[β]=0.1831  Σλ=1.21e+03  peak=0.021  | Essential hypertension, Hyperlipidemia, Chronic pain, Pain, Anemia, T2D
[driver]    topic  1  α=0.1959  E[β]=0.1667  Σλ=1.1e+03  peak=0.015  | Dementia, Pain, Chest pain, Nausea, Vomiting, Shortness of breath
[driver]   iter 2/2: ELBO=-106384.8872, batch=285, rho=0.0532, 85.9s
[driver]   α[min=0.1878 max=0.1945 mean=0.1916], η=0.2, Σλ_k[min=1.23e+03 max=3.4e+03]
[driver]   --- topics @ iter 2 ---
[driver]    topic  3  α=0.1945  E[β]=0.3184  Σλ=3.4e+03  peak=0.016  | Constipation, Chest pain, Shortness of breath, Nausea, Vomiting, Diarrhea
[driver]    topic  0  α=0.1938  E[β]=0.2472  Σλ=2.64e+03  peak=0.015  | Pain, Anxiety, Chest pain, Nausea, Shortness of breath, Dementia
[driver]    topic  4  α=0.1926  E[β]=0.1744  Σλ=1.86e+03  peak=0.029  | Essential hypertension, Hyperlipidemia, T2D, OSA, GERD, Chronic pain
[driver]    topic  2  α=0.1893  E[β]=0.1445  Σλ=1.54e+03  peak=0.014  | Hyperlipidemia, Essential hypertension, Chest pain, Nausea, Pain, Vomiting
[driver]    topic  1  α=0.1878  E[β]=0.1155  Σλ=1.23e+03  peak=0.018  | Dementia, Pain, Chest pain, Nausea, Vomiting, Shortness of breath
[driver]   elbo trace tail: [-97881.60, -106384.89]
[driver] <<< fit: 260.0s
[driver] computing theta aggregates via model.transform(fit_df)...
[driver]   wrote theta aggregates to metadata (1374 patients, K=5).
[driver] re-saved augmented VIResult to .../runs/0001-pilot
[driver] fit complete

### Session complete (exit 0)

## Eval (NPMI)
[driver] checkpoint=.../runs/0001-pilot, top_n=20, model_class=lda
[driver]   corpus_manifest: cdr=wb-affable-acorn-7941.R2024Q3R8, source_table=condition_era, person_mod=10
[driver]   doc_spec: patient_year, min_doc_length=20, replicate_eras=True
[driver]   frozen vocab: 500 terms
[driver] >>> npmi reference (full corpus)
[driver]   reference: 101758 docs
[driver] <<< npmi reference: 36.0s
[driver] >>> NPMI coherence (top_n=20, min_pair_count=3): 148.2s

  per-topic stats (reference=full corpus, reference_size=101758, top_n=20, min_pair_count=3, unrated=0/5):
  mean=+0.2979  median=+0.3333  stdev=0.1169  min=+0.0835  max=+0.4185

   topic 3  NPMI=+0.4185  cov=100%  E[β]=0.3184  Σλ=3.4e+03  α=0.194  top: Constipation, Chest pain, Shortness of breath, Nausea, Vomiting, Diarrhea, Pain, Abdominal pain
   topic 0  NPMI=+0.3763  cov=100%  E[β]=0.2472  Σλ=2.64e+03  α=0.194  top: Pain, Anxiety, Chest pain, Nausea, Shortness of breath, Dementia, Wheezing, Hyperlipidemia
   topic 4  NPMI=+0.0835  cov=100%  E[β]=0.1744  Σλ=1.86e+03  α=0.193  top: Essential hypertension, Hyperlipidemia, T2D, OSA, GERD, Chronic pain, Major depression, COPD
   topic 2  NPMI=+0.2777  cov=100%  E[β]=0.1445  Σλ=1.54e+03  α=0.189  top: Hyperlipidemia, Essential hypertension, Chest pain, Nausea, Pain, Vomiting, Shortness of breath, Insomnia
   topic 1  NPMI=+0.3333  cov=100%  E[β]=0.1155  Σλ=1.23e+03  α=0.188  top: Dementia, Pain, Chest pain, Nausea, Vomiting, Shortness of breath, Wheezing, Anxiety
[driver] EVAL COHERENCE CLOUD PASSED
```

(Sessions 1–2 stack traces omitted from this embed for brevity; they're
captured in the **Fit history** section above and live in full on the
cluster at `$RUNS_DIR/0001-pilot/summary.md`.)

## Interpretation

**Pipeline works end-to-end.** Three fit sessions in one accumulating
`summary.md`, two failures cleanly recorded with exit codes, one clean run
with full per-iter trend + NPMI eval, no patient identifiers leaked, copy/paste
round-trip from cluster terminal works.

**Two bugs caught and fixed inline** (see Fit history). Both were structural
mismatches between the wrapper's assumptions and the existing driver/checkpoint
contracts:
1. **Cohort naming**: the wrapper conflated frontend display id (`dementia`)
   with driver argparse choice (`first_dementia_year`). Two-field split
   (`cohort` + `cohort_def`) cleanly separates them.
2. **Resume detection**: dir-existence ≠ checkpoint-existence. Now keys on
   `manifest.json` (the actual marker the driver looks for).

**Modeling output is sensible for K=5 on a 1374-patient × 500-vocab dementia
cohort, 2 iters.** Topics already separating into reasonable themes — Topic 1
anchored on Dementia/Pain (the cohort-defining concept), Topic 4 on the
classic cardiometabolic cluster (Hypertension/Hyperlipidemia/T2D/OSA/GERD),
Topic 3 on GI/respiratory symptoms. Topic 2 looks redundant with 0/3 (mixed
symptoms) and Topic 4's mean NPMI is the lowest (0.08) — both expected at K=5
for a cohort this size. Real exploration starts at K≥40 in subsequent
experiments. Mean NPMI=0.30 is in the expected range for a 2-iter fit on
this corpus.

**No follow-up bugs to track on the pipeline itself.** Verbose Spark INFO
logs in `summary.md` are cosmetic (Increment 2's structured per-iter parser
will compress them). No `Eval complete (exit 0)` marker after the eval
section — minor inconsistency; could be added in Increment 2 alongside the
SIGTERM trap.

## Links
- Spec: docs/superpowers/specs/2026-05-28-experiment-tracking-design.md
- Plan: docs/superpowers/plans/2026-05-28-experiment-tracking-increment-1.md
- Fixes during pilot: `3fe2eb6` (cohort_def split), `d971796` (resume manifest check)
