# Precision/recall readout + PPI vocabulary strip — Design

**Date:** 2026-07-28
**Status:** Approved (user, 2026-07-28). Ready for a plan.
**Arc:** the multi-domain gated LDA arc (`docs/superpowers/specs/2026-07-24-multidomain-gated-lda-arc-design.md`), following the LR readout (`2026-07-28-multidomain-lr-readout-design.md`) and its first result (insight 0071).

## Goal

Two focused additions, landing together so ONE re-fit of exp 0071/0072 produces
both answers:

- **Piece A — precision/recall readout.** Insight 0071 measured ranking (AUC) only.
  At rare-disease base rates (scleroderma 79/39,463 ≈ 0.2%) a high AUC need not
  mean deployable precision, and the marginal AUC gain of condition+drug over
  condition-alone (+0.02–0.045) cannot tell us whether that gain is operationally
  real. Add PR-AUC + precision-at-recall per disease per domain subset. POST-HOC:
  reads the same persisted test set, no re-fit needed on its own.
- **Piece B — PPI vocabulary strip.** Insight 0071 found the observation domain is
  net-NEGATIVE for all six rare diseases (`drop:observation ≥ all` everywhere;
  `only:observation` ≈ chance), consistent with its vocabulary being dominated by
  All of Us survey/SDOH tokens (PMI, "DNA Quiz", "Can't Afford Care", "Are you
  still seeing…"). All of Us stores survey/SDOH data as observations with
  `vocabulary_id = 'PPI'` (Participant-Provided Information), so one principled
  filter removes the bulk. Requires a re-fit (it changes the vocabulary).

The test this sets up: does a PPI-stripped observation domain flip from drag toward
neutral/positive, and does condition+drug buy deployable precision over
condition-alone?

## Piece A — Precision/recall readout (`analysis/cloud/multidomain_lr_readout.py`)

### New pure helper

```
per_disease_pr(scores, frontiers, anchor, lay, parent_int, recalls=(0.5, 0.8))
    -> (pr_auc, {recall: precision}, n_pos)
```

Positive set and per-disease score are IDENTICAL to the existing
`per_disease_auc_row` (positive = doc's frontier ∩ `subtree(anchor)`; score = max
over `subtree(anchor) ∩ lay.nodes` columns), so the PR numbers describe exactly
the same detection problem as the AUC table — the two are directly comparable.

Computation is plain numpy (no new dependency): sort docs by score descending,
cumulative true/false positives, `precision = tp/(tp+fp)`, `recall = tp/n_pos`.

- **`pr_auc`** = average precision, `Σ_i (recall_i − recall_{i−1}) · precision_i`
  (the standard step-wise AP estimator; no trapezoidal interpolation, which is
  optimistically biased for PR curves — Davis & Goadrich 2006).
- **`{recall: precision}`** = for each requested recall level r, the precision at
  the smallest threshold achieving recall ≥ r (i.e. the first index where
  `recall_i ≥ r`); `nan` if that recall is unreachable.
- **`n_pos`** = positive count, reported so a small-n row is not over-read.
- One-class input (`n_pos == 0` or all positive) → `pr_auc = nan`, precisions
  `nan`, mirroring `_auc`'s one-class convention.

### Two new prints in `main()`

Both come after the existing AUC table, reusing the already-computed
`subset_scores` (no re-scoring):

1. **PR-AUC table** — rows = the six rare6 anchors, columns = the same 8 domain
   subsets (`all`, `only:<domain>` ×3, `drop:<domain>` ×3), plus `n+` and a
   **`prev`** column (`n_pos / n_docs`). Prevalence is the random-classifier
   PR-AUC, so `prev` is the baseline that makes PR-AUC interpretable (unlike ROC,
   PR's baseline moves with the base rate).
2. **Precision@recall summary** — rows = the six anchors, columns = precision at
   50% and 80% recall for the three headline subsets: `all`, `only:condition`,
   and `drop:observation` (= condition + drug, the best-scoring subset in insight
   0071). This is the deployability read ("flag enough patients to catch 80% of
   true cases — what fraction of the flagged list is real?") and the direct
   cond-vs-cond+drug operational comparison.

The θ-mass baseline keeps its existing alignment discipline: θ rows are scored
with `aff` + `aff_frontiers` (their own collect), LR rows with `frontiers`.

## Piece B — PPI vocabulary strip

### `charmpheno/charmpheno/omop/bigquery.py`

`load_omop_bigquery` gains a keyword-only parameter:

```
exclude_vocabularies: tuple[str, ...] = ()
```

Default `()` = today's behavior, byte-identical (no extra column, no filter). When
non-empty, the EXISTING left-join to `concept` (already present for
`concept_name`) also selects `vocabulary_id`, and rows whose `vocabulary_id` is in
the set are dropped. The filter is **NULL-safe**: a concept absent from the
`concept` table has a null `vocabulary_id` from the left join and is KEPT (an
unmapped code is not evidence of being a survey item). `vocabulary_id` is dropped
from the output projection, so the canonical output schema is unchanged.

General mechanism, not observation-specific — any domain can exclude any
vocabulary. Only observation uses it now.

### `analysis/cloud/multidomain_cloud.py`

New CLI knob `--obs-exclude-vocab` (comma-separated, default empty string →
`()`). It is applied ONLY to the observation domain's `load_omop_bigquery` call;
condition and drug loads are unaffected. Recorded in the manifest's
`corpus_manifest` so a run's artifact says what was stripped.

### `scripts/run_experiment.py` + experiment definitions

`build_multidomain_args` emits `--obs-exclude-vocab` from
`effective.get("obs_exclude_vocab", "")`. Exp **0071** and **0072** frontmatter
gain `obs_exclude_vocab: PPI`. Exp 0070 (2-domain, no observation) and any other
multidomain fit are unchanged unless they opt in — **the knob defaults to empty**
(user decision).

## Validation / acceptance

1. **PR helper unit tests (pure numpy, no Spark):**
   - Perfect ranker (all positives above all negatives) → `pr_auc == 1.0` and
     precision@0.5 == precision@0.8 == 1.0.
   - Uninformative ranker (constant score, or positives spread uniformly) →
     `pr_auc ≈ prevalence` (the random baseline), within tolerance.
   - Precision@recall reads the correct operating point on a hand-computed small
     example (known TP/FP ordering), including an unreachable recall → `nan`.
   - One-class input → `nan`, no crash.
2. **PPI plumbing unit tests:** `--obs-exclude-vocab` parses to a tuple (empty
   default); `build_multidomain_args` emits it; exp 0071/0072 frontmatter
   validates and carries `obs_exclude_vocab: PPI`; the NULL-safe exclusion
   predicate keeps null-vocabulary rows and drops PPI rows (tiny local-Spark
   frame). The live BigQuery read stays cluster-covered.
3. **Cluster smoke (user-run):** re-fit 0071/0072 with `obs_exclude_vocab: PPI`,
   then the readout. Expected: the observation topic dump no longer shows
   PMI/"DNA Quiz"/"Can't Afford Care"/"Are you still seeing…"; the readout prints
   the AUC table plus BOTH new PR tables; `manifest.dead_nodes` still empty.
   The scientific read: does `drop:observation ≥ all` still hold, and by how much
   less (did the strip move observation from drag toward neutral)?

## Out of scope

- **`max_df` / document-frequency cap.** The other de-noising lever discussed
  (drop codes present in >X% of patients, a statistical stopword filter that would
  also catch clinical junk like "History of event" and "Long-term current use
  of…"). Deferred by user decision this round — noted as a future one-knob add to
  the per-domain vocab spec, applicable to every domain.
- **The measurement (labs) domain.** High-value next add and now a config line
  thanks to SP3c, but it needs design thought (concept presence vs value
  awareness), deferred by user decision.
- **The ω sweep (SP4).** Largely retired by the readout (insight 0071): ω
  re-weights domains at FITTING time, while the θ-free readout selects domains
  post-hoc for free.
- **Re-fitting exp 0070.** Two-domain, no observation; unaffected.

## Risks

- **The strip may not be enough.** PPI removes the AoU survey bulk but not the
  clinical junk (SNOMED "History of event", "Long-term current use of…", generic
  encounter/vitals concepts) visible in the exp 0071 topic dump. Observation may
  improve from "drag" to "still mildly negative" rather than to positive; the
  α→∞ LR lift already provides idf-for-free, which bounds the achievable gain.
  This is an informative outcome either way, and `max_df` is the follow-up lever.
- **PPI coverage assumption.** That All of Us survey/SDOH observations carry
  `vocabulary_id = 'PPI'` is an AoU CDM convention, verified against the cluster
  only when the fit runs. If the topic dump still shows survey items afterward,
  the filter set needs widening (e.g. an additional vocabulary or a concept-class
  filter) — the mechanism is general enough to absorb that without redesign.
- **Small positive counts.** Scleroderma (79) and myasthenia gravis (80) give
  noisy PR estimates, and precision at high recall is especially unstable at
  those counts; `n+` and `prev` are printed beside every row to keep this visible.
