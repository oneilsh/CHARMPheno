# Design: GLP-1 + active-comparator gated STM (drug-anchored cohort)

Date: 2026-07-08
Status: approved (design), pending implementation plan

## Goal

Fit a gated STM that reads what the year after starting a GLP-1 receptor agonist
looks like, contrasted both against the general population and against an
active-comparator drug (SGLT2 inhibitors) started by a similar patient
population. The gated Σ then yields GLP-1↔comparator and GLP-1↔background topic
correlations (and anti-correlations) in one model.

Scientific framing: contrasting GLP-1 initiators against the whole population
shows *who gets prescribed GLP-1* (largely their T2DM/obesity/cardiometabolic
**indication**); contrasting against an active comparator with the same
indication (new-user SGLT2i) controls for confounding-by-indication, so what
separates the two foreground blocks is closer to **drug-specific** structure.
Doing both in one gated fit gives population context and indication-controlled
contrast together.

## Cohort architecture — a drug-anchored track parallel to the disease track

Everything built so far anchors on a **diagnosis** (a condition concept) and
fills documents with **conditions**. A GLP-1 cohort anchors on a **drug** but
keeps condition documents. So this adds a drug-anchor track alongside the
existing disease track in `charmpheno/charmpheno/omop/cohorts.py`; the corpus
content path (conditions) is unchanged — the drug read happens inside the cohort,
exactly as the disease cohorts read `concept_ancestor`/`observation_period`.

### Drug registry

A drug registry mapping a class key to its concept spec, parallel to
`_DISEASE_REGISTRY`:

```
_DRUG_REGISTRY = {
    "glp1_ra":     {ingredients: semaglutide, liraglutide, dulaglutide,
                    exenatide, lixisenatide},
    "sglt2i":      {ingredients: empagliflozin, dapagliflozin, canagliflozin,
                    ertugliflozin},
    "tirzepatide": {ingredients: tirzepatide},
}
```

Concept sets are resolved from RxNorm-ingredient / ATC-class ancestors (target
ATC: A10BJ "GLP-1 analogues" for glp1_ra, A10BK "SGLT2 inhibitors" for sglt2i;
tirzepatide as its own ingredient). **Resolution + descendant/patient counts are
verified on the cluster before fitting** — the same gate used for the EDS
ancestor `79145`. The exact OMOP concept_ids are an implementation-time lookup,
not hard-coded from memory.

### Drug-anchored primitive

```
apply_first_drug_year_cohort(
    cond_df, *, inclusion_ingredients, window_days=365,
    spark, cdr_dataset, billing_project, date_col, prior_obs_days=365,
) -> DataFrame
```

Reads `drug_era` (one era per person per RxNorm ingredient; `drug_era_start_date`
is the exposure start), takes the **first** era-start across the class's
ingredient set as the person's index date, applies the same observation-period
bracketing as the disease primitive (`_window_observed_cohort`: `prior_obs_days`
prior coverage so the index is a true new-user initiation, plus a fully-observed
`window_days` forward window), and returns the person's **condition** rows in
`[index_date, index_date + window_days)`. Mirrors
`apply_first_diagnosis_year_cohort` with the anchor moved to the drug domain.

### Five-way population partition

```
apply_population_drug_cohort(
    cond_df, *, window_days=365, prior_obs_days=365, combo_max_gap_days=90, ...
)
```

Each person is assigned to exactly one group (one document) from their new-user
first-era dates `g` (glp1_ra), `s` (sglt2i), `t` (tirzepatide) — each requiring
`prior_obs_days` prior coverage with no earlier use of that class, and a
fully-observed `window_days` forward window from the eventual index. Groups are
mutually exclusive and one-document-per-person:

| Group | Membership rule | Index (window start) |
|---|---|---|
| `tirzepatide` | `t` present | first tirzepatide era |
| `glp1_sglt2_combo` | `g` and `s` present, no `t`, and **`|g − s| ≤ combo_max_gap_days`** | earlier of `g`, `s` |
| `glp1_ra` | `g` present, **never** `s` or `t` | `g` |
| `sglt2i` | `s` present, **never** `g` or `t` | `s` |
| `general` | none of `g`, `s`, `t` | event-anchored random year (below) |
| *(excluded)* | `g` and `s` present, no `t`, but **`|g − s| > combo_max_gap_days`** | — dropped from the cohort |

Precedence checked tirzepatide → combo → single-class → general. The two design
rules that keep the arms clean:

1. **Combo = co-initiation.** `glp1_sglt2_combo` requires the two initiations
   within `combo_max_gap_days` (default 90) so both drugs genuinely start inside
   the observed document year — a transition-to-dual-therapy year, not "took both
   at some point years apart."
2. **Non-combo both-users are excluded, not reclassified.** A person who
   initiated both but outside the combo gap is dropped entirely — NOT placed in a
   single-drug arm (their index year would be contaminated by the second drug)
   and NOT placed in `general` (they are treated patients and would pollute the
   background). This makes the single-drug arms **"only ever that class"**:
   `glp1_ra` = GLP-1 new-user who never initiated SGLT2i or tirzepatide, and
   likewise for `sglt2i`. The result is a maximally clean contrast — pure-GLP-1
   vs pure-SGLT2i vs co-initiated-combo vs tirzepatide vs untreated background.

`source_cohort ∈ {tirzepatide, glp1_sglt2_combo, glp1_ra, sglt2i, general}`; the
four drug groups are foreground blocks, `general` is background-only.

**`combo_max_gap_days` is data-set from a diagnostic, not guessed.** GLP-1 +
SGLT2i combination is usually reached by *sequential* intensification (start one,
add the other weeks-to-months later), not same-day dual starts, so the `|g − s|`
gap distribution is expected to show a co-initiation spike near 0 plus a
sequential tail. The corpus build emits a **`|g − s|` histogram for all
both-class users** (a no-fit diagnostic, like the sparse cohort's coding-density
histogram) so the threshold is set from where the co-initiation cluster actually
ends in this CDR; 90 is the starting default. **v1 scoping:**
`combo_max_gap_days` is a **cohort-code parameter (default 90)**, not yet a
frontmatter field — threading it through the driver/loader/cache-key stack is a
fast-follow, taken only if the histogram shows 90 is wrong. The diagnostic logs
regardless of the chosen value, so the data-driven decision does not depend on
the plumbing.

Splitting tirzepatide and combo into their own arms (rather than folding into
glp1_ra) is deliberate: merging arms post-hoc is trivial, un-merging is not, so
we keep the information now and collapse thin/uninteresting arms after seeing the
counts.

*Open item:* the `tirzepatide` arm is "any tirzepatide new-user" and is NOT
further purity-filtered for concurrent SGLT2i (tirzepatide precedence wins) —
kept simple because it is the thinnest arm; revisit if it turns out large enough
to split a tirzepatide-only vs tirzepatide+SGLT2i sub-contrast.

### Symmetric observability for the general arm

The drug arms are incident new-users: 365d prior coverage + a fully-observed
365d follow-up. The `general` arm must carry the **same** bracket, or the
contrast bakes in an observability confound (well-observed drug patients vs
sparsely-observed general patients). Today `_random_event_windows`
([cohorts.py](../../charmpheno/charmpheno/omop/cohorts.py)) enforces only the
forward window (`event_date + window_days ≤ period_end`). Add a **prior-coverage
predicate** — `event_date − prior_obs_days ≥ period_start` within the same
covering observation period — so an eligible random anchor has a fully-observed
year both before and after. `_random_observed_year_cohort` /
`apply_population_drug_cohort` thread `prior_obs_days` into it. All five groups
then share the 1yr-prior + 1yr-follow-up observed window.

### Scope / reuse

- `_window_observed_cohort` (prior+forward bracketing) is reused as-is for the
  drug primitive.
- The disease track (`apply_first_diagnosis_year_cohort`,
  `apply_population_disease_cohort`) is **not** modified. A future unification of
  disease+drug into one "population + N foreground arms by anchor spec" core is
  possible but deferred (YAGNI) — build the drug track cleanly now.
- New cohort `population_glp1` registered in `SUPPORTED_COHORTS`,
  `COHORT_METADATA`, `apply_cohort`; `experiments/defaults/population_glp1.yaml`.
- Cohort tests on synthetic data: the five-way partition incl. the
  `combo_max_gap_days` boundary (co-init → combo; over-gap both-user → excluded,
  landing in neither a single arm nor `general`), tirzepatide precedence,
  single-arm "only ever that class" purity, and the general-arm prior-coverage
  predicate. Plus a test that the `|g − s|` gap-histogram diagnostic emits.

## Experiment / fit parameters

New experiment record (id assigned at plan time), on the 0043 hardened + slowed
stack:

| field | value | note |
|---|---|---|
| cohort / cohort_def | `population_glp1` | new |
| model_class | stm | |
| K | 140 | 80 background + glp1_ra:15 + sglt2i:15 + tirzepatide:15 + glp1_sglt2_combo:15 |
| background_k | 80 | |
| foreground | `glp1_ra:15,sglt2i:15,tirzepatide:15,glp1_sglt2_combo:15` | |
| group_var | source_cohort | |
| person_mod | 1 | full population — thin tirzepatide/combo arms need it |
| prior_obs_days | 365 | incident new-user (drug arms + general) |
| window_days | 365 | 1-year documents |
| combo_max_gap_days | 90 | cohort-code default (v1: not a frontmatter field); re-cut from the diagnostic histogram |
| doc_min_length | 10 | |
| covariate_formula | `~ C(sex) + age` | known_sex_only |
| schedule / hardening | subsample 0.1, tau0 256, kappa 0.7, max_iter 300, sigma_init 1, reference + dense spectral, min_pair_support 10, block-wise unit-diagonal Σ (ADR 0034) | |

## Downstream

Fit on the Dataproc master (`make build-covariates` → `make exp`), verify the
per-arm document counts in corpus diagnostics (especially tirzepatide + combo),
then export (`make build-dashboard-exp`), annotate via `scripts/label_phenotypes.py`,
and add `population_glp1` to `dashboard/public/data/manifest.json` as an
additional cohort (population_cancer stays default).

## Risks / open items

- **Tirzepatide (and possibly combo) thinness.** FDA-2022 drug + new-user +
  fully-observed follow-up year may leave a small arm. Mitigated by person_mod 1;
  if still too thin, merge tirzepatide into glp1_ra (or drop) — the split-now
  design anticipates this. Log the per-arm counts.
- **Concept-set resolution.** Confirm ATC A10BJ / A10BK (and tirzepatide) resolve
  to the intended ingredient/drug descendants and workable patient counts on the
  cluster before the fit.
- **Drug content is anchor-only.** Documents contain conditions, not the drugs —
  the model reads the *comorbidity* structure around initiation, not co-prescribing.
- **Heaviest fit in the project** (full population, K=140). Confirm it fits the
  cluster time/memory envelope; fall back to person_mod 2 or K=120 if needed.

## Out of scope

- Modifying the disease track or unifying disease+drug cohorts.
- Adding drugs to document content (conditions-only documents stay).
- A separate pre-vs-post-initiation design (this is post-initiation only).
