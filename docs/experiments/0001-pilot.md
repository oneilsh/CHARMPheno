---
id: 1
slug: pilot
status: done
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
- 2026-05-29 13:59 UTC — **Session 4** (chain validation): `make next-exp NO_EVAL=1`
  auto-resumed from the session-3 checkpoint (`n_iterations=2, converged=False`),
  ran 5 fresh iters (driver treats `max_iter` as iters-to-run-this-call, not cumulative;
  see Interpretation), aggregates re-written. `--no-eval` correctly skipped the
  eval block inside `make next-exp`. **Spark/Hadoop INFO logs absent from Session 4**
  output (NOISE_PATTERNS filter shipped in Inc 2 working as designed — compare to
  the noisy Sessions 2/3 above). `### Session complete (exit 0)` marker present.
- 2026-05-29 14:15 UTC — **Eval (separate invocation)**: `make eval-exp` with no
  argument auto-discovered id=1 via most-recent-manifest mtime (Inc 2.5), ran the
  reference corpus + NPMI on the augmented checkpoint, appended a timestamped
  `## Eval (NPMI) — 2026-05-29 14:15:42 UTC` block + `### Eval complete (exit 0)`
  marker. NPMI mean +0.289 (vs +0.298 after 2 iters) — topic 1 lost a touch of
  coherence as it specialized further on dementia/osteopenia, topics 0 and 3
  gained.

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

### Session 4 + second eval (Inc 2 / 2.5 chain validation, 2026-05-29)

```
## Fit session 4
Started: 2026-05-29 13:59:47 UTC

[driver] cdr=wb-affable-acorn-7941.R2024Q3R8, K=5, max_iter=5, person_mod=10
[driver] Spark 3.5.3, master=yarn, defaultParallelism=2
[driver] >>> BQ load + summary
[driver]   OMOP: 135125 rows, 1122 distinct persons
[driver] <<< BQ load + summary: 54.4s
[driver] >>> vectorize (CountVectorizer, doc_spec=patient_year, min_doc_length=20)
[driver] <<< vectorize: 9.5s
[driver]   vocab size: 500, documents: 1374
[driver] >>> fit (K=5, maxIter=5, ...)
[driver]   Resuming from .../runs/0001-pilot (n_iterations=2, converged=False)
[driver]   iter 1/5: ELBO=-76965.1919, batch=245, rho=0.0527, 86.0s
[driver]   iter 2/5: ELBO=-102301.1319, batch=285, rho=0.0521, 85.5s
[driver]   iter 3/5: ELBO=-103083.4967, batch=311, rho=0.0516, 85.1s
[driver]   iter 4/5: ELBO=-95219.3898,  batch=289, rho=0.0511, 84.5s
[driver]   iter 5/5: ELBO=-92602.0360,  batch=281, rho=0.0506, 84.5s
[driver]   --- topics @ iter 5 ---
[driver]    topic  3  α=0.182  E[β]=0.3634  Σλ=9.49e+03  | Chest pain, SoB, Nausea, Diarrhea, Vomiting, Pain
[driver]    topic  0  α=0.178  E[β]=0.2942  Σλ=7.68e+03  | Pain, Anxiety, Chest pain, Nausea, SoB, Wheezing
[driver]    topic  4  α=0.182  E[β]=0.1896  Σλ=4.95e+03  | Hypertension, Hyperlipidemia, GERD, T2D, Chronic pain, OSA
[driver]    topic  2  α=0.158  E[β]=0.0877  Σλ=2.29e+03  | Hyperlipidemia, Hypertension, Chest pain, AFib, SoB, Nausea
[driver]    topic  1  α=0.156  E[β]=0.0652  Σλ=1.7e+03   | Dementia, Pain, Vomiting, Nausea, Chest pain, Osteopenia
[driver]   elbo trace tail: [-103083.50, -95219.39, -92602.04]
[driver] <<< fit: 514.9s
[driver] computing theta aggregates via model.transform(fit_df)...
[driver]   wrote theta aggregates to metadata (1374 patients, K=5).
[driver] re-saved augmented VIResult to .../runs/0001-pilot
[driver] fit complete

### Session complete (exit 0)

## Eval (NPMI) — 2026-05-29 14:15:42 UTC
[driver] checkpoint=.../runs/0001-pilot, top_n=20, model_class=lda
[driver] >>> npmi reference (full corpus)
[driver]   reference: 101758 docs
[driver] <<< npmi reference: 36.0s
[driver] >>> NPMI coherence: 152.0s

  per-topic stats (top_n=20, min_pair_count=3, unrated=0/5):
  mean=+0.2894  median=+0.2661  stdev=0.1332  min=+0.0849  max=+0.4723

   topic 3  NPMI=+0.4723  cov=100%  Σλ=9.49e+03  α=0.182  top: Chest pain, SoB, Nausea, Diarrhea, Vomiting, Pain, Constipation, Abdominal pain
   topic 0  NPMI=+0.3891  cov=100%  Σλ=7.68e+03  α=0.178  top: Pain, Anxiety, Chest pain, Nausea, SoB, Wheezing, Vomiting, Dementia
   topic 4  NPMI=+0.0849  cov=100%  Σλ=4.95e+03  α=0.182  top: Hypertension, Hyperlipidemia, GERD, T2D, Chronic pain, OSA, Anxiety disorder, Major depression
   topic 2  NPMI=+0.2661  cov=100%  Σλ=2.29e+03  α=0.158  top: Hyperlipidemia, Hypertension, Chest pain, AFib, SoB, Nausea, Pain, Vomiting
   topic 1  NPMI=+0.2346  cov=100%  Σλ=1.7e+03   α=0.156  top: Dementia, Pain, Vomiting, Nausea, Chest pain, Osteopenia, Anxiety, Hypothyroidism
[driver] EVAL COHERENCE CLOUD PASSED

### Eval complete (exit 0)
```

Note the contrast with Sessions 2/3 above: zero Spark/Hadoop INFO lines in the
Session 4 transcript — the `NOISE_PATTERNS` filter shipped in Inc 2 is doing its
job. Eval has its own `### Eval complete` marker now and a timestamp in the
header (Inc 2.5 ergonomics).

## Interpretation

**Pipeline works end-to-end across Inc 1, Inc 2, and Inc 2.5.** One accumulating
`summary.md` now contains: two failed sessions (with exit codes and stack
traces), two clean fit sessions on the same checkpoint (the second of which
auto-resumed), and two evals (one integrated into the original fit, one
separately re-run via `make eval-exp` with no argument). Copy/paste round-trip
from cluster terminal works for all of it. No patient identifiers leaked at any
point.

**Inc 2 / 2.5 chain validation (Session 4 + second eval) confirms:**
- `NO_EVAL=1` cleanly skips the eval block inside `make next-exp` — only the
  `### Session complete` marker appears, no eval section.
- `NOISE_PATTERNS` regex strips all Spark/Hadoop INFO lines from Session 4
  (visible by comparing Session 2/3 transcripts above, which still carry the
  full noise).
- `make eval-exp` with no argument correctly auto-discovers id=1 by
  most-recent-manifest mtime (Inc 2.5 helper).
- Eval block carries a timestamp in the header (`## Eval (NPMI) — <UTC ts>`)
  and a matching `### Eval complete (exit 0)` marker. Both Inc 2 additions
  address the "no marker after eval" gap noted from Session 3.
- Resume path read `n_iterations=2, converged=False` from the prior checkpoint
  and continued from there.

**Two bugs caught and fixed inline during the original 3-session run** (see
Fit history). Both were structural mismatches between the wrapper's assumptions
and the existing driver/checkpoint contracts:
1. **Cohort naming**: wrapper conflated frontend display id (`dementia`) with
   driver argparse choice (`first_dementia_year`). Two-field split
   (`cohort` + `cohort_def`) cleanly separates them. Commit `3fe2eb6`.
2. **Resume detection**: dir-existence ≠ checkpoint-existence. Now keys on
   `manifest.json` (the actual marker the driver looks for). Commit `d971796`.

**One semantic curiosity, not a bug:** `max_iter` is iters-to-run-this-call,
not cumulative. Session 4 resumed at `n_iterations=2` then ran `iter 1/5` …
`iter 5/5`, so the checkpoint now reflects 7 total iters of VI, not 5. This is
how the underlying `spark_vi.mllib.topic.lda` driver has always behaved — the
experiment-tracking wrapper inherits it as-is. Worth a separate ADR if we ever
want cumulative semantics, but not blocking.

**Modeling output remains sensible at K=5 on a 1374-patient × 500-vocab dementia
cohort.** With 5 more iters of VI, topics sharpened in roughly the directions
expected:
- Topic 1 (dementia anchor) shed Shortness of Breath and absorbed Osteopenia
  and Hypothyroidism — moving toward a "frailty-adjacent" signature. Its NPMI
  dipped from 0.33 → 0.23 as it specialized away from the corpus-mean symptom
  bag.
- Topic 4 (cardiometabolic) stayed coherent thematically but its mean NPMI
  stayed at the floor (0.08) — these conditions co-occur in the cohort but
  rarely in the *full* reference corpus at the 20-doc window, so this is
  expected and tracks with the producer-consumer-units insight (0023).
- Topics 0, 2, 3 all gained or held coherence; Topic 3 (GI/respiratory symptom
  hub) climbed to 0.47, the highest in the run.

Mean NPMI=0.29 (vs 0.30 after 2 iters). The slight drop with more iters is
real: Topic 1's specialization toward dementia-adjacent rare-ish concepts trades
coherence-vs-the-corpus for coherence-vs-the-cohort. Both are valid; NPMI just
measures one of them. Real exploration starts at K≥40 in subsequent experiments.

**No follow-up bugs to track on the pipeline itself.** The remaining items from
the original write-up (Spark INFO noise; missing `Eval complete` marker) are
both resolved as of Inc 2.

## Links
- Spec: docs/superpowers/specs/2026-05-28-experiment-tracking-design.md
- Plans: Inc 1: docs/superpowers/plans/2026-05-28-experiment-tracking-increment-1.md;
  Inc 2: docs/superpowers/plans/2026-05-29-experiment-tracking-increment-2.md;
  Inc 2.5: docs/superpowers/plans/2026-05-29-experiment-tracking-increment-2-5.md
- Fixes during pilot: `3fe2eb6` (cohort_def split), `d971796` (resume manifest check)
