# Pre-diagnosis (lookback) document window — design

**Status:** approved (brainstorm 2026-07-18). Exploratory research build; no
production target.

## Motivation

The case-finding corpus currently documents each patient by the year *after* their
first disease diagnosis (`[index_date, index_date + window_days)`, a forward
window). For finding *undiagnosed* cases this is the wrong signal: the post-index
window is contaminated with diagnostic workup, subtype coding, and treatment
follow-up that exist *because* the patient was diagnosed — none of which an
undiagnosed patient has. The clinically correct framing is prediction from
**pre-diagnosis history**: what did this patient look like *before* anyone put the
label on them?

## Key structural consequence: features and label decouple

`index_date` = the first code in the disease concept set (min over inclusion
descendants). So **no disease code exists before it** — a pre-index window
contains zero DAG-node codes. Two consequences:

1. **Leakage-free by construction.** The disease codes are temporally excluded
   from the features, so the `strip_mode` machinery becomes a no-op (kept, but it
   strips nothing).
2. **The label must come from a separate window.** Today the frontier (which
   subtype[s]) is read from in-window node codes; a pre-index window has none. So
   the subtype/comorbid label is read from a forward **label window**
   `[index_date, index_date + label_window_days)` — which is exactly the window
   0060 used as its features, now repurposed to read the multi-node diagnostic
   picture (subtypes + comorbid rare diseases → a set-valued frontier).

So per patient: **features = pre-index conditions; label = disease nodes attested
in the forward label window; background = empty label + pre-index features.**

## Windows

- **Feature window (lookback):** `[greatest(op_start, index − lookback_days), index)`.
  - exp 0061: `lookback_days = 365` → exactly 1 year (the prior gate guarantees it
    is fully observed).
  - exp 0062: `lookback_days = 1825` → up to 5 years, clipped to whatever history
    exists (1–5 years); `doc_min_length` drops the too-sparse.
- **Label window (forward):** `[index, index + label_window_days)`,
  `label_window_days = 365` (a 1-year lookahead to capture the full multi-node
  frontier). Same for both exps.

## Cohort gate (both foreground and background)

Require the symmetric bracket `[index − 365, index + label_window_days)` observed:

- **Prior**: `index_date ≥ observation_period_start_date + 365` — at least one year
  of pre-index history (the user's requirement, for the features).
- **Forward**: `index_date + label_window_days ≤ observation_period_end_date` —
  already present in the forward cohort; now covers the label window.

Applied to the random-index background too, so foreground and background have
matched observation depth (avoids a "cases observed longer than controls"
confound). The lookback cohort is therefore "the forward cohort ∩ {≥1yr prior}".

Survivorship caveat (documented, accepted): both arms now condition on patients
who *have* ≥1yr of pre-index observation — a different (better-recorded)
population than "diagnosed at record start". Inherent to the pre-diagnosis
framing.

## Components

### 1. Cohort layer (`charmpheno/omop/cohorts.py`)

- `apply_first_diagnosis_year_cohort` and the background
  `_random_event_windows` / `_random_observed_year_cohort`: add a `window_mode`
  ("forward" | "lookback") + `lookback_days` + `label_window_days`.
- In **lookback** mode these functions expose per-person `index_date` +
  `source_cohort` and produce TWO windowed event frames from the raw events:
  `feature_events` = the pre-index lookback window, `label_events` = the forward
  label window. The prior gate is added; the forward gate is retained for the
  label window.
- **forward** mode is unchanged (single window = both features and label).

### 2. Assembly (`charmpheno/omop/case_finding_assembly.py`)

- `assemble_from_events(events_df, ..., label_events=None)`: when `label_events`
  is given, derive the frontier (`doc_attested_nodes` → `attach_frontiers`) from
  `label_events` and the features (`to_bow_dataframe`) from `events_df`; the doc
  roster/patient split stays consistent (split by person_id, both frames). When
  `label_events is None`, behavior is unchanged (features and frontier both from
  `events_df` — forward mode).
- `assemble_case_finding_corpus`: plumb `window_mode` / `lookback_days` /
  `label_window_days`; pass the two frames through in lookback mode.

### 3. Driver / config / cache

- `dag_placement_cloud.py`: `--window-mode`, `--lookback-days`,
  `--label-window-days`; record in manifest.
- `_case_finding_cache.py`: fold the new windowing params into the cache key
  (bump version) so a lookback corpus caches under its own key.
- `run_experiment.build_dag_placement_args`: emit the new flags.
- `_base.yaml`: `window_mode: forward`, `lookback_days: 365`,
  `label_window_days: 365` (defaults preserve existing behavior).

### 4. Experiments

- **0061** — lookback, `lookback_days: 365`, else identical to 0060 (rare6,
  frontier anchors, sym α, strip_both moot, scalable-spectral). The apples-to-apples
  "1yr pre vs 1yr post" test.
- **0062** — as 0061 but `lookback_days: 1825` (up-to-5yr history).

## Validation

- Unit tests (charmpheno): lookback windowing selects only pre-index events;
  label window selects only forward events; the ≥1yr-prior gate drops
  short-history patients; background gets matched lookback + empty frontier; the
  two-frame assembly derives features from pre-index and frontier from the label
  window (a planted patient whose pre-index conditions differ from post-index
  disease codes ends up with the pre-index tokens as features and the post-index
  disease node as its frontier).
- Config-parse tests: 0061/0062 build the right argv; `_base` defaults preserve
  forward behavior.
- The real read is the cluster A/B: 0061 (1yr lookback) vs 0060 (1yr lookahead),
  and 0062 (5yr) vs 0061 — does pre-diagnosis history detect/route cases, and does
  more history help?

## Out of scope

- Richer features (meds/labs) — a separate, larger arc (the other big lever).
- Temporal/sequence modeling within the window (still a bag).
- Changing the model, gate, or anchor_scope (all orthogonal; 0061/0062 inherit
  0060's frontier-anchor + scalable-spectral setup).
