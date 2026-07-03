# Population+EDS and Population+Sparse Demo Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new gated-STM demo models — Population+Ehlers-Danlos (exp 0030) and Population+Sparse (exp 0031) — on a generalized `population+disease` cohort primitive, then fit, export, and list them in the dashboard alongside the existing `population_cancer` default.

**Architecture:** Generalize the population+disease cohort in `cohorts.py` into a concept-ancestor-driven registry (parameterized window, multi inclusion-ancestor includes-then-excludes) so EDS is a registry entry, not copy-paste. The density/sparse cohort stays a separate function (no concept set). Then two new experiment records + config defaults drive cluster fits via the existing `run_experiment.py` / Makefile flow; bundles export into `dashboard/public/data/<cohort>/` and get hand-listed in `manifest.json`.

**Tech Stack:** PySpark (cohort DataFrame transforms), pytest (local unit tests with a `spark` fixture), the project's Dataproc/BigQuery fit + dashboard-bundle Makefile targets, Svelte dashboard (manifest-driven cohort selector).

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-03-population-eds-and-sparse-demo-models-design.md` — the source of truth for all decisions below.
- **EDS OMOP top-level concept id:** `79145` (inclusion ancestor; no exclusions). Verify descendant + patient counts on the cluster before the fit.
- **Do NOT touch** `apply_first_cancer_year_cohort` or `apply_first_dementia_year_cohort` — the "full generalize" of the first-year functions is deferred. Only add the generic primitive + population wrapper.
- **`population_cancer` (exp 0028) is NOT refit** — its concept set must stay identical; it remains the manifest `default`.
- **Shared fit stack** (both new experiments, from exp 0028): `subsampling_rate: 0.1`, `tau0: 128`, `kappa: 0.7`, `max_iter: 200`, `sigma_init: 1.0`, `reference_topic: true`, `spectral_init: true`, `spectral_method: dense`, `min_pair_support: 10`, `covariate_formula: "~ C(sex) + age"`, `categorical_cols: [sex]`, `continuous_cols: [age]`, `known_sex_only: true`, `random_seed: 42`, `cache_uri: hdfs:///user/dataproc/charm/covariates_cache`, `group_var: source_cohort`, block-wise unit-diagonal Σ (ADR 0034).
- **Per-model knobs:** EDS → K=60 (bg 40 + `eds:20`), `person_mod: 1`, `prior_obs_days: 0`, `doc_min_length: 10`. Sparse → K=50 (bg 40 + `sparse:10`), `person_mod: 4`, `prior_obs_days: 0`, `doc_min_length: 5`.
- **Local test command:** from the `charmpheno/` directory, `python -m pytest tests/test_cohorts.py -v` (a `spark` fixture lives in `charmpheno/tests/conftest.py`; `testpaths = ["tests"]`).
- **Cluster commands** (Tasks 6–7) run from `analysis/cloud/` on the Dataproc master, not locally.

---

### Task 1: Pure concept-set helper (multi-inclusion, includes-then-excludes)

Extract the concept-ancestor set logic into a pure DataFrame-in/DataFrame-out helper so the generalized inclusion/exclusion semantics are unit-testable without a live CDR (mirrors how `_bucket_general_by_density` / `_combine_cohorts` are tested).

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py` (add helper + import)
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Produces: `_concept_set_from_ancestors(ca_df, *, inclusion_ancestors: Sequence[int], exclusion_ancestors: Sequence[int] = ()) -> DataFrame` — takes a `concept_ancestor` DataFrame (`ancestor_concept_id`, `descendant_concept_id`), returns a distinct single-column `concept_id` DataFrame equal to (⋃ descendants of inclusions) − (⋃ descendants of exclusions).

- [ ] **Step 1: Write the failing tests**

Add to `charmpheno/tests/test_cohorts.py`:

```python
def test_concept_set_from_ancestors_unions_inclusions_and_subtracts_exclusions(spark):
    from charmpheno.omop.cohorts import _concept_set_from_ancestors
    ca = spark.createDataFrame(
        [
            (100, 1), (100, 2), (100, 3),   # inclusion ancestor A -> {1,2,3}
            (200, 3), (200, 4),             # inclusion ancestor B -> {3,4}
            (900, 2),                       # exclusion ancestor   -> {2}
        ],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    out = _concept_set_from_ancestors(
        ca, inclusion_ancestors=[100, 200], exclusion_ancestors=[900],
    )
    # union {1,2,3,4} minus {2} = {1,3,4}; 3 is in both inclusions, not excluded
    assert {r["concept_id"] for r in out.collect()} == {1, 3, 4}


def test_concept_set_from_ancestors_no_exclusions_dedups(spark):
    from charmpheno.omop.cohorts import _concept_set_from_ancestors
    ca = spark.createDataFrame(
        [(79145, 10), (79145, 11), (79145, 11)],
        ["ancestor_concept_id", "descendant_concept_id"],
    )
    out = _concept_set_from_ancestors(ca, inclusion_ancestors=[79145])
    assert {r["concept_id"] for r in out.collect()} == {10, 11}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_concept_set_from_ancestors_unions_inclusions_and_subtracts_exclusions tests/test_cohorts.py::test_concept_set_from_ancestors_no_exclusions_dedups -v`
Expected: FAIL with `ImportError: cannot import name '_concept_set_from_ancestors'`.

- [ ] **Step 3: Add the import and the helper**

At the top of `cohorts.py`, add below `from __future__ import annotations`:

```python
from collections.abc import Sequence
```

Add the helper (place it just above `apply_first_cancer_year_cohort`):

```python
def _concept_set_from_ancestors(
    ca_df: DataFrame,
    *,
    inclusion_ancestors: Sequence[int],
    exclusion_ancestors: Sequence[int] = (),
) -> DataFrame:
    """Build a concept-id set from a concept_ancestor DataFrame.

    Includes-then-excludes: (⋃ descendants of ``inclusion_ancestors``) −
    (⋃ descendants of ``exclusion_ancestors``). A concept reachable from both
    an inclusion and an exclusion ancestor is excluded. ``ca_df`` must have
    ``ancestor_concept_id`` and ``descendant_concept_id`` columns; returns a
    distinct single-column ``concept_id`` DataFrame. Predicates on
    ``ancestor_concept_id`` push down to BQ, so only the ~thousands of relevant
    concept ids materialize, not the full concept_ancestor table.
    """
    included = (
        ca_df.where(F.col("ancestor_concept_id").isin(list(inclusion_ancestors)))
        .select(F.col("descendant_concept_id").alias("concept_id"))
        .distinct()
    )
    if exclusion_ancestors:
        excluded = (
            ca_df.where(F.col("ancestor_concept_id").isin(list(exclusion_ancestors)))
            .select(F.col("descendant_concept_id").alias("concept_id"))
        )
        included = included.subtract(excluded).distinct()
    return included
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k concept_set_from_ancestors -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): pure concept-set helper (multi-inclusion, includes-then-excludes)"
```

---

### Task 2: Generic first-diagnosis primitive + EDS constant

Add the reusable "first qualifying dx + window" cohort primitive, built on Task 1's helper and parameterized by ancestors + `window_days`. This is the code path future rare diseases (and EDS) use; the existing cancer/dementia functions are left untouched.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: `_concept_set_from_ancestors` (Task 1), existing `_window_observed_cohort`.
- Produces:
  - `_EDS_ANCESTOR = 79145` (module constant).
  - `apply_first_diagnosis_year_cohort(cond_df, *, inclusion_ancestors: Sequence[int], exclusion_ancestors: Sequence[int] = (), window_days: int = _WINDOW_DAYS, spark, cdr_dataset, billing_project, date_col, prior_obs_days: int = _WINDOW_DAYS) -> DataFrame` — filters `cond_df` to persons with a first qualifying dx and rows in `[index_date, index_date + window_days)`.

- [ ] **Step 1: Write the failing test (validation surface)**

The body reads `concept_ancestor` / `observation_period` from BigQuery, so (like the existing data-path functions) it is exercised on the cluster, not locally. Locally assert it is importable with the right signature:

```python
def test_apply_first_diagnosis_year_cohort_is_importable_with_ancestor_params():
    import inspect
    from charmpheno.omop.cohorts import (
        apply_first_diagnosis_year_cohort, _EDS_ANCESTOR,
    )
    assert _EDS_ANCESTOR == 79145
    params = inspect.signature(apply_first_diagnosis_year_cohort).parameters
    assert "inclusion_ancestors" in params
    assert "exclusion_ancestors" in params
    assert "window_days" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_apply_first_diagnosis_year_cohort_is_importable_with_ancestor_params -v`
Expected: FAIL with `ImportError: cannot import name 'apply_first_diagnosis_year_cohort'`.

- [ ] **Step 3: Add the EDS constant and the primitive**

Add the constant near `_DEMENTIA_ANCESTOR` (after its exclusion tuple):

```python
# Top-level OMOP concept whose descendants define Ehlers-Danlos syndrome.
# Provided by the domain owner (2026-07-03). No exclusions.
# VERIFY ON FIRST RUN (as with dementia):
#   SELECT COUNT(*) FROM concept_ancestor WHERE ancestor_concept_id = 79145;
_EDS_ANCESTOR = 79145
```

Add the primitive just below `_concept_set_from_ancestors`:

```python
def apply_first_diagnosis_year_cohort(
    cond_df: DataFrame,
    *,
    inclusion_ancestors: Sequence[int],
    exclusion_ancestors: Sequence[int] = (),
    window_days: int = _WINDOW_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Filter to persons with a first qualifying dx + a ``window_days`` window.

    Generalizes the per-disease first-year cohorts: the concept set is
    (⋃ descendants of ``inclusion_ancestors``) − (⋃ descendants of
    ``exclusion_ancestors``) via :func:`_concept_set_from_ancestors`; the
    document window is ``[index_date, index_date + window_days)``. Same
    observation-period bracketing as the cancer/dementia cohorts
    (:func:`_window_observed_cohort`, ``prior_obs_days`` lookback). Returns
    ``cond_df``'s schema, filtered.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    concepts = _concept_set_from_ancestors(
        ca,
        inclusion_ancestors=inclusion_ancestors,
        exclusion_ancestors=exclusion_ancestors,
    )

    first_dx = (
        cond_df.join(F.broadcast(concepts), on="concept_id", how="inner")
        .groupBy("person_id")
        .agg(F.min(date_col).alias("index_date"))
    )

    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_dx, op, prior_obs_days=prior_obs_days,
    )

    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_apply_first_diagnosis_year_cohort_is_importable_with_ancestor_params -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): generic first-diagnosis primitive + EDS ancestor constant"
```

---

### Task 3: Disease registry + generalized population wrapper + register population_eds

Add the disease registry, the generalized `apply_population_disease_cohort`, reroute `population_cancer` through it (identical concept set → no model change), and register the new `population_eds` cohort.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py` (registry, wrapper, `apply_population_cancer_cohort` delegation, `SUPPORTED_COHORTS`, `COHORT_METADATA`, `apply_cohort`)
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: `apply_first_diagnosis_year_cohort` (Task 2), `_CANCER_ANCESTOR`, `_CANCER_EXCLUSION_ANCESTORS`, `_EDS_ANCESTOR`, `_random_observed_year_cohort`.
- Produces:
  - `_DISEASE_REGISTRY: dict[str, dict]` — keys `"cancer"`, `"eds"`, each `{"inclusion_ancestors": tuple[int, ...], "exclusion_ancestors": tuple[int, ...]}`.
  - `apply_population_disease_cohort(cond_df, *, disease: str, window_days: int = _WINDOW_DAYS, spark, cdr_dataset, billing_project, date_col, prior_obs_days: int = _WINDOW_DAYS) -> DataFrame` — disease arm tagged `source_cohort=<disease>`, general arm `source_cohort='general'`, disjoint by person.
  - `"population_eds"` present in `SUPPORTED_COHORTS`, `COHORT_METADATA`, and `apply_cohort` dispatch.

- [ ] **Step 1: Write the failing tests**

```python
def test_disease_registry_has_cancer_and_eds_with_expected_ancestors():
    from charmpheno.omop.cohorts import (
        _DISEASE_REGISTRY, _CANCER_ANCESTOR, _CANCER_EXCLUSION_ANCESTORS,
        _EDS_ANCESTOR,
    )
    assert _DISEASE_REGISTRY["cancer"]["inclusion_ancestors"] == (_CANCER_ANCESTOR,)
    assert _DISEASE_REGISTRY["cancer"]["exclusion_ancestors"] == _CANCER_EXCLUSION_ANCESTORS
    assert _DISEASE_REGISTRY["eds"]["inclusion_ancestors"] == (_EDS_ANCESTOR,)
    assert _DISEASE_REGISTRY["eds"]["exclusion_ancestors"] == ()


def test_supported_cohorts_includes_population_eds():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_eds" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_eds():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_eds"]
    assert m["id"] == "population_eds"
    assert m["label"] and m["description"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k "population_eds or disease_registry" -v`
Expected: FAIL (ImportError on `_DISEASE_REGISTRY` / KeyError on metadata).

- [ ] **Step 3: Add the registry and generalized wrapper**

Add the registry near the other module constants (after `_EDS_ANCESTOR`):

```python
# Disease registry for the generalized population+disease cohort. Each entry is
# fully described by concept ancestors; adding a rare disease is a new entry
# here + a SUPPORTED_COHORTS/COHORT_METADATA/apply_cohort line, no new function.
_DISEASE_REGISTRY: dict[str, dict] = {
    "cancer": {
        "inclusion_ancestors": (_CANCER_ANCESTOR,),
        "exclusion_ancestors": _CANCER_EXCLUSION_ANCESTORS,
    },
    "eds": {
        "inclusion_ancestors": (_EDS_ANCESTOR,),
        "exclusion_ancestors": (),
    },
}
```

Add the generalized wrapper (place it just above the existing `apply_population_cancer_cohort`):

```python
def apply_population_disease_cohort(
    cond_df: DataFrame,
    *,
    disease: str,
    window_days: int = _WINDOW_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole-population background + a single disease foreground, disjoint.

    Generalizes :func:`apply_population_cancer_cohort` to any registered
    ``disease`` (see ``_DISEASE_REGISTRY``). One document per person:

    - disease arm — persons with a first qualifying dx (via
      :func:`apply_first_diagnosis_year_cohort` on the registry's ancestors),
      windowed to ``window_days`` post-dx, tagged ``source_cohort=<disease>``.
    - ``general`` arm — every OTHER person, on a deterministic random
      event-anchored ``window_days`` window (:func:`_random_observed_year_cohort`),
      tagged ``source_cohort='general'`` (background-only).

    ``window_days`` threads to BOTH arms so foreground and background windows
    match. Arms are disjoint by person (general = ``left_anti`` of the disease
    arm's persons). Returns ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    try:
        spec = _DISEASE_REGISTRY[disease]
    except KeyError:
        raise ValueError(
            f"disease {disease!r} not in registry "
            f"(known: {tuple(_DISEASE_REGISTRY)})"
        )

    diseased = apply_first_diagnosis_year_cohort(
        cond_df,
        inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"],
        window_days=window_days,
        spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    diseased_persons = diseased.select("person_id").distinct()

    non_diseased = cond_df.join(diseased_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_diseased, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days,
    )

    return (
        diseased.withColumn("source_cohort", F.lit(disease))
        .unionByName(general.withColumn("source_cohort", F.lit("general")))
    )
```

Replace the body of the existing `apply_population_cancer_cohort` so it delegates (keep its signature + docstring; the concept set is unchanged, so the exported 0028 model is unaffected):

```python
    return apply_population_disease_cohort(
        cond_df, disease="cancer", window_days=_WINDOW_DAYS,
        spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
```

- [ ] **Step 4: Register `population_eds`**

In `SUPPORTED_COHORTS`, add `"population_eds"`:

```python
SUPPORTED_COHORTS: tuple[str, ...] = (
    "first_cancer_year",
    "first_dementia_year",
    "cancer_or_dementia",
    "population_cancer",
    "population_cancer_sparse",
    "population_eds",
)
```

In `COHORT_METADATA`, add:

```python
    "population_eds": {
        "id": "population_eds",
        "label": "Population + Ehlers-Danlos (gated)",
        "description": (
            "The whole population as a shared background, with an "
            "Ehlers-Danlos syndrome (EDS) subcohort carrying its own foreground "
            "topics — a rare-disease-on-a-background-population example. Disjoint, "
            "one document per person: persons with a first EDS diagnosis (OMOP "
            "79145 and descendants) get the 365-day post-diagnosis window and "
            "source_cohort='eds'; every other person gets a deterministic random "
            "365-day window anchored on one of their own condition-eras and "
            "source_cohort='general' (background-only). K=60 (40 shared "
            "background + 20 EDS foreground), fit on the full population as a "
            "gated block-wise correlated STM with a sex-at-birth (M/F) and age "
            "prevalence covariate."
        ),
    },
```

In `apply_cohort`, add a dispatch branch before the final `raise`:

```python
    if cohort == "population_eds":
        return apply_population_disease_cohort(
            cond_df, disease="eds", spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -v`
Expected: all pass (new registry/registration tests + all pre-existing tests, including `test_cohort_metadata_has_population_cancer`, still green — the `population_cancer` delegation preserves behavior).

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): disease registry + generalized population wrapper; register population_eds"
```

---

### Task 4: population_sparse cohort (outside the disease framework)

Add the whole-population density-split cohort — no disease arm — and register it.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py` (`apply_population_sparse_cohort`, `SUPPORTED_COHORTS`, `COHORT_METADATA`, `apply_cohort`)
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: `_random_observed_year_cohort`, `_bucket_general_by_density`.
- Produces:
  - `apply_population_sparse_cohort(cond_df, *, window_days: int = _WINDOW_DAYS, sparse_min: int = 5, dense_min: int = 20, spark, cdr_dataset, billing_project, date_col, prior_obs_days: int = _WINDOW_DAYS) -> DataFrame` — whole population, `source_cohort ∈ {'general','sparse'}`.
  - `"population_sparse"` in `SUPPORTED_COHORTS`, `COHORT_METADATA`, `apply_cohort`.

- [ ] **Step 1: Write the failing tests**

```python
def test_supported_cohorts_includes_population_sparse():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_sparse" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_sparse():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_sparse"]
    assert m["id"] == "population_sparse"
    assert m["label"] and m["description"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k population_sparse -v`
Expected: FAIL (`population_sparse` not in `SUPPORTED_COHORTS` / KeyError on metadata).

- [ ] **Step 3: Add the cohort function**

Place it just below `apply_population_cancer_sparse_cohort`:

```python
def apply_population_sparse_cohort(
    cond_df: DataFrame,
    *,
    window_days: int = _WINDOW_DAYS,
    sparse_min: int = 5,
    dense_min: int = 20,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole population split by in-window coding density — no disease arm.

    Every person is windowed to one deterministic random event-anchored
    ``window_days`` span (:func:`_random_observed_year_cohort`), then split by
    per-person in-window event count (:func:`_bucket_general_by_density`):
    ``>= dense_min`` -> ``source_cohort='general'`` (dense background),
    ``sparse_min..dense_min-1`` -> ``source_cohort='sparse'`` (a light-coder
    foreground block), ``< sparse_min`` dropped. Unlike
    :func:`apply_population_cancer_sparse_cohort`, there is NO disease foreground
    — cancer/EDS patients simply fold into the population, giving a clean
    whole-population reference for reading what light-coder years contain.

    ``prior_obs_days`` is accepted for a uniform :func:`apply_cohort` signature
    but unused (there is no disease index event to be "first" of). Returns
    ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    general = _random_observed_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days,
    )
    return _bucket_general_by_density(
        general, sparse_min=sparse_min, dense_min=dense_min,
    )
```

- [ ] **Step 4: Register `population_sparse`**

Add `"population_sparse"` to `SUPPORTED_COHORTS`. Add to `COHORT_METADATA`:

```python
    "population_sparse": {
        "id": "population_sparse",
        "label": "Population + Sparse (gated)",
        "description": (
            "The whole population split by in-window coding density — no disease "
            "anchor. Each person is windowed to one deterministic random "
            "event-anchored 365-day span, then split by event count: heavily-coded "
            "years (>= 20 events) are source_cohort='general' (background-only), "
            "light-coder years (5-19 events) become source_cohort='sparse' — their "
            "own foreground block. K=50 (40 background + 10 sparse). A gated "
            "block-wise correlated STM reads the sparse foreground topics to show "
            "what light-coder general years are made of, against a clean "
            "whole-population background."
        ),
    },
```

Add the `apply_cohort` dispatch branch (before the final `raise`):

```python
    if cohort == "population_sparse":
        return apply_population_sparse_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): population_sparse whole-population density-split cohort"
```

---

### Task 5: Config defaults + experiment records (0030 EDS, 0031 sparse)

Add the `<cohort>.yaml` defaults and the `docs/experiments/` records so `run_experiment.py` can resolve and dispatch each fit. Validate frontmatter + defaults locally.

**Files:**
- Create: `experiments/defaults/population_eds.yaml`
- Create: `experiments/defaults/population_sparse.yaml`
- Create: `docs/experiments/0030-stm-population-eds-gated.md`
- Create: `docs/experiments/0031-stm-population-sparse-gated.md`
- Test: `charmpheno/tests/test_experiment_records.py` (new) — OR run the inline validation in Step 4 if a test module for records does not fit the repo's layout; the inline validation is the authoritative check.

**Interfaces:**
- Consumes: `run_experiment.read_frontmatter`, `run_experiment.validate_frontmatter`, `run_experiment.load_defaults` (in `scripts/run_experiment.py`; `EXPERIMENTS_DIR = repo/docs/experiments`, `DEFAULTS_DIR = repo/experiments/defaults`).
- Produces: experiment ids 30 and 31 resolvable by `run_experiment.py --id`.

- [ ] **Step 1: Create the defaults files**

`experiments/defaults/population_eds.yaml`:

```yaml
# Whole-population background + Ehlers-Danlos foreground gated cohort (exp 0030).
# Like population_cancer, but the disease arm is EDS (OMOP 79145). The
# experiment frontmatter supplies the STM covariate + gating fields (K,
# background_k, foreground: eds:N, group_var, covariate_formula); this file
# exists so load_defaults finds a <cohort>.yaml to merge over _base.yaml.
#
# `cohort` is the display/lookup id (matches filename); `cohort_def` is the
# value passed to the driver's --cohort.
cohort: population_eds
cohort_def: population_eds
```

`experiments/defaults/population_sparse.yaml`:

```yaml
# Whole-population density-split gated cohort (exp 0031). No disease anchor:
# the population is windowed then split into a dense 'general' (background-only)
# group and a light-coder 'sparse' foreground group. The experiment frontmatter
# supplies the STM covariate + gating fields (K, background_k, foreground:
# sparse:N, group_var, covariate_formula); this file exists so load_defaults
# finds a <cohort>.yaml to merge over _base.yaml.
#
# `cohort` is the display/lookup id (matches filename); `cohort_def` is the
# value passed to the driver's --cohort.
cohort: population_sparse
cohort_def: population_sparse
```

- [ ] **Step 2: Create the experiment record for exp 0030 (EDS)**

`docs/experiments/0030-stm-population-eds-gated.md` — mirror 0028's frontmatter with the EDS knobs from Global Constraints:

```markdown
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
K: 60
background_k: 40
foreground: "eds:20"
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
the EDS foreground and give a rich background. K=60 = 40 background + 20 EDS.
Otherwise the exp 0028 gentle + hardened stack (subsample 0.1, tau0 128, 200
iter, reference + dense spectral, sigma_init 1, min_pair_support 10, block-wise
unit-diagonal Σ / ADR 0034), `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- Covariate diagnostics show a realistic 2-level sex distribution.
- EDS foreground topics recover recognizable EDS-associated phenotypes; the EDS
  arm has enough documents (check corpus diagnostics — if thin, revisit
  `person_mod`).
- Σ variance bounded (no runaway); honest correlation report.

## Related

Follows exp 0028 (population + cancer gated). First cohort on the generalized
population+disease registry.
```

- [ ] **Step 3: Create the experiment record for exp 0031 (sparse)**

`docs/experiments/0031-stm-population-sparse-gated.md`:

```markdown
---
id: 31
slug: stm-population-sparse-gated
status: pending
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

## Related

Reframes exp 0029 (population + cancer + sparse) without the cancer arm.
```

- [ ] **Step 4: Validate frontmatter + defaults resolve locally**

Run:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -c "
import sys; sys.path.insert(0, 'scripts')
import run_experiment as r
for eid in (30, 31):
    p = r.find_by_id(r.EXPERIMENTS_DIR, eid)
    fm = r.read_frontmatter(p)
    r.validate_frontmatter(fm)
    d = r.load_defaults(fm['cohort'], r.DEFAULTS_DIR)
    print(eid, p.name, 'cohort=', fm['cohort'], 'OK')
"
```

Expected: prints `30 0030-stm-population-eds-gated.md cohort= population_eds OK` and the analogous line for 31, no exceptions. (If `validate_frontmatter` flags a missing required field, add it to the frontmatter to match what it requires, then re-run.)

- [ ] **Step 5: Commit**

```bash
git add experiments/defaults/population_eds.yaml experiments/defaults/population_sparse.yaml docs/experiments/0030-stm-population-eds-gated.md docs/experiments/0031-stm-population-sparse-gated.md
git commit -m "feat(experiments): add exp 0030 population_eds and 0031 population_sparse records"
```

---

### Task 6: Fit exp 0030 and 0031 on the cluster

Operational (not local TDD). Run from `analysis/cloud/` on the Dataproc master, after Tasks 1–5 are merged/available on the cluster checkout. The `known_sex_only` covariate join means each fit needs a fresh covariate cache build.

**Files:**
- Produces: fit artifacts under the cluster runs dir; `summary.md` per experiment.

- [ ] **Step 1: Sanity-check the EDS concept set on the cluster**

Before the full fit, confirm 79145 resolves to a non-empty EDS descendant set and a workable patient count (BigQuery, in the CDR the fit uses):

```sql
SELECT COUNT(*) AS n_descendants
FROM concept_ancestor WHERE ancestor_concept_id = 79145;
```

Expected: a non-zero descendant count. If 0, stop and re-confirm the EDS ancestor id before fitting.

- [ ] **Step 2: Build covariates for exp 0030 (EDS)**

Run: `cd analysis/cloud && make build-covariates EXP=30`
Expected: builds the STM covariate cache for cohort `population_eds`; the covariate-level diagnostic reports `sex_at_birth_concept_id` {8507, 8532} and a 2-level `sex` {M, F} (not F-only).

- [ ] **Step 3: Fit exp 0030**

Run: `cd analysis/cloud && make exp ID=30`
Expected: fit runs to `max_iter: 200`; `make summary ID=30` shows the merged effective config, a non-trivial `eds` arm doc count in corpus diagnostics, and bounded Σ (no runaway). If the EDS arm is thin, note it and consult before adjusting `person_mod`.

- [ ] **Step 4: Build covariates + fit exp 0031 (sparse)**

Run: `cd analysis/cloud && make build-covariates EXP=31 && make exp ID=31`
Expected: fit runs to completion; `make summary ID=31` shows both `general` and `sparse` groups present and bounded Σ.

- [ ] **Step 5: Record outcomes**

Update `status:` in each experiment record (`pending` → the project's post-fit status convention) and note the arm doc counts + any Σ observations in the record's body. Commit:

```bash
git add docs/experiments/0030-stm-population-eds-gated.md docs/experiments/0031-stm-population-sparse-gated.md
git commit -m "chore(experiments): record 0030/0031 fit outcomes"
```

---

### Task 7: Export dashboard bundles + list in manifest

Operational. Export each fit's bundle into `dashboard/public/data/<cohort>/` and add both to the manifest (hand-maintained; `population_cancer` stays `default`).

**Files:**
- Produces: `dashboard/public/data/population_eds/` and `dashboard/public/data/population_sparse/` bundles.
- Modify: `dashboard/public/data/manifest.json`

- [ ] **Step 1: Build the dashboard bundle for each experiment**

Run: `cd analysis/cloud && make build-dashboard-exp ID=30 && make build-dashboard-exp ID=31`
Expected: writes the 8-file bundles (`corpus_stats.json`, `phenotypes.json`, `gating.json`, `correlation.json`, `covariate_effects.json`, `covariate_schema.json`, `model.json`, `vocab.json`) into `dashboard/public/data/population_eds/` and `.../population_sparse/`.

- [ ] **Step 2: Verify bundle completeness**

Run: `cd analysis/cloud && python -m pytest tests/test_bundle_completeness.py -v`
Expected: PASS for the new bundles (all required files present + well-formed). Also spot-check `gating.json` for each: `population_eds` groups == `["eds"]`; `population_sparse` groups == `["sparse"]`.

- [ ] **Step 3: Add both cohorts to the manifest**

Edit `dashboard/public/data/manifest.json` — keep `"default": "population_cancer"`, append two entries to `cohorts`. Use each cohort's `label`/`description` from `COHORT_METADATA` (Tasks 3–4) so the selector text matches the bundle. Resulting shape:

```json
{
  "default": "population_cancer",
  "cohorts": [
    { "id": "population_cancer", "label": "Population + Cancer (gated)", "description": "..." },
    { "id": "population_eds", "label": "Population + Ehlers-Danlos (gated)", "description": "<COHORT_METADATA['population_eds'].description>" },
    { "id": "population_sparse", "label": "Population + Sparse (gated)", "description": "<COHORT_METADATA['population_sparse'].description>" }
  ]
}
```

(Keep the existing `population_cancer` entry's description verbatim.)

- [ ] **Step 4: Verify the dashboard loads all three cohorts**

Run: `cd dashboard && npm run dev` (or the project's dev command) and open the app. In the cohort selector, confirm all three cohorts appear, `Population + Cancer` is the default, and selecting `Population + Ehlers-Danlos` and `Population + Sparse` each loads phenotypes + gating without console errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/public/data/population_eds dashboard/public/data/population_sparse dashboard/public/data/manifest.json
git commit -m "feat(dashboard): export + list population_eds and population_sparse cohorts"
```

---

## Self-Review

**Spec coverage:**
- Generalized `apply_first_diagnosis_year_cohort` (multi-inclusion, includes-then-excludes, `window_days`) → Tasks 1–2. ✓
- Disease registry; `population_cancer`/`population_eds` as entries; first-year functions untouched; `population_cancer` byte-identical → Task 3. ✓
- `population_sparse` outside the disease framework → Task 4. ✓
- Registration + cohort tests → Tasks 3–4. ✓
- exp 0030 (K=60, person_mod 1, EDS knobs) + exp 0031 (K=50, person_mod 4, sparse knobs) records + defaults → Task 5. ✓
- Fit → Task 6; export + manifest (population_cancer stays default) + bundle-completeness → Task 7. ✓
- Out of scope (no 0028 refit, no first-year collapse, no comorbid re-add) respected. ✓

**Placeholder scan:** No TBD/TODO. The `<COHORT_METADATA[...].description>` markers in Task 7 Step 3 are explicit "copy the text produced in Tasks 3–4" instructions, not gaps. Cluster commands (Tasks 6–7) are exact Makefile targets; their outputs are environment-dependent by nature and are verified via `summary`/`test_bundle_completeness`.

**Type consistency:** `_concept_set_from_ancestors` (Task 1) is consumed by `apply_first_diagnosis_year_cohort` (Task 2), consumed by `apply_population_disease_cohort` (Task 3); `Sequence` imported in Task 1. `_DISEASE_REGISTRY` keys (`cancer`,`eds`) match the `disease=` args used in `apply_cohort` and `apply_population_cancer_cohort`. `source_cohort` tag values (`eds`/`general`/`sparse`) match the `foreground`/`group_var` in the experiment records and the `gating.json` group checks in Task 7.
