# GLP-1 + comparator gated cohort — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a drug-anchored gated cohort `population_glp1` — whole-population background + four drug foreground arms (glp1_ra, sglt2i, tirzepatide, glp1_sglt2_combo) — anchored on first `drug_era`, with incident new-user + symmetric-observability windowing.

**Architecture:** A drug-anchor track parallel to the existing disease track in `charmpheno/charmpheno/omop/cohorts.py`. Anchoring reads `drug_era` (resolving ingredient sets by RxNorm ingredient NAME, no hard-coded concept_ids); document content stays condition-based (unchanged corpus path). A pure partition core (`_assign_drug_groups`) implements the five-way precedence + combo-gap + exclusion logic and is fully unit-tested; the BQ-reading orchestration wrapper is validation-surface only, matching how the disease cohorts are tested.

**Tech Stack:** PySpark (cohort DataFrame transforms), pytest with a `spark` fixture (`charmpheno/tests/conftest.py`, `testpaths=["tests"]`), OMOP CDR tables (`drug_era`, `concept`, `observation_period`, `condition_era`).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-08-glp1-comparator-gated-cohort-design.md` — source of truth.
- **Do NOT modify** the disease track (`apply_first_diagnosis_year_cohort`, `apply_population_disease_cohort`, `_DISEASE_REGISTRY`) or the `first_*`/`population_cancer`/`population_eds`/`population_sparse` cohorts.
- **Document content is conditions; drugs are anchor-only.** The cohort reads `drug_era` for index dates but returns the person's `cond_df` (condition) rows.
- **Five groups**, `source_cohort ∈ {tirzepatide, glp1_sglt2_combo, glp1_ra, sglt2i, general}`; non-combo both-users are **excluded** (in neither a drug arm nor `general`).
- **Precedence:** tirzepatide → glp1_sglt2_combo (|g−s| ≤ `combo_max_gap_days`) → glp1_ra (g only) → sglt2i (s only) → general (none). Single-drug arms are "only ever that class".
- **`combo_max_gap_days` = 90**, a cohort-code default (v1: NOT a frontmatter field). `_COMBO_MAX_GAP_DAYS = 90`.
- **Incident new-user + symmetric observability:** all arms use `prior_obs_days=365` prior coverage + fully-observed `window_days=365` follow-up, including the `general` arm.
- **Ingredient names** (RxNorm `Ingredient`, matched case-insensitively): glp1_ra = {semaglutide, liraglutide, dulaglutide, exenatide, lixisenatide}; sglt2i = {empagliflozin, dapagliflozin, canagliflozin, ertugliflozin}; tirzepatide = {tirzepatide}.
- **Local test command:** from `charmpheno/`, `python -m pytest tests/test_cohorts.py -v` (activate the repo venv first if needed: `source /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/.venv/bin/activate`). Spark startup makes these tests take a few seconds — not a hang.
- **Commits:** no `Co-Authored-By` or extra trailers. The `stm` branch auto-pushes to origin — expected.
- **Cluster steps (Task 7)** run on the Dataproc master from `analysis/cloud/`, not locally.

## File structure

- Modify `charmpheno/charmpheno/omop/cohorts.py` — all cohort code (Tasks 1–5).
- Modify `charmpheno/tests/test_cohorts.py` — unit tests (Tasks 1–5).
- Create `experiments/defaults/population_glp1.yaml` (Task 5).
- Create `docs/experiments/0044-stm-population-glp1-comparator.md` (Task 6).

---

### Task 1: Prior-coverage predicate for the general arm

Give `_random_event_windows` (and `_random_observed_year_cohort`) an optional `prior_obs_days` so the general-arm random anchor requires a fully-observed year *before* it too — symmetric with the drug arms. Default 0 preserves every existing caller's behavior.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Produces: `_random_event_windows(cond_df, observation_period, *, date_col, window_days=_WINDOW_DAYS, prior_obs_days=0)` and `_random_observed_year_cohort(cond_df, *, spark, cdr_dataset, billing_project, date_col, window_days=_WINDOW_DAYS, prior_obs_days=0)` — an eligible anchor now additionally requires `event_date ≥ observation_period_start_date + prior_obs_days`.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_cohorts.py`:

```python
def test_random_event_windows_enforces_prior_coverage(spark):
    """With prior_obs_days>0 an eligible anchor needs a fully-observed year
    BEFORE it, not just after — symmetric observability for the general arm."""
    import datetime as dt
    from charmpheno.omop.cohorts import _random_event_windows

    # One person, one event on 2015-06-01. Observation period 2015-01-01..2017-01-01:
    # forward year is observed (event+365 <= end), but only ~150d of prior coverage.
    events = spark.createDataFrame(
        [(1, dt.date(2015, 6, 1))], ["person_id", "condition_start_date"],
    )
    op = spark.createDataFrame(
        [(1, dt.date(2015, 1, 1), dt.date(2017, 1, 1))],
        ["person_id", "observation_period_start_date", "observation_period_end_date"],
    )
    # prior_obs_days=0 (current behavior): the anchor is eligible.
    kept = _random_event_windows(
        events, op, date_col="condition_start_date", prior_obs_days=0,
    )
    assert kept.count() == 1
    # prior_obs_days=365: insufficient prior coverage -> dropped.
    dropped = _random_event_windows(
        events, op, date_col="condition_start_date", prior_obs_days=365,
    )
    assert dropped.count() == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_random_event_windows_enforces_prior_coverage -v`
Expected: FAIL — `_random_event_windows() got an unexpected keyword argument 'prior_obs_days'`.

- [ ] **Step 3: Add the `prior_obs_days` predicate**

In `_random_event_windows`, add `prior_obs_days: int = 0` to the signature (after `window_days`) and a prior-coverage predicate in the `eligible` filter. The `eligible` block becomes:

```python
    eligible = (
        events.join(observation_period, on="person_id", how="inner")
        .where(F.col("event_date") >= F.date_add(
            F.col("observation_period_start_date"), prior_obs_days))
        .where(
            F.date_add(F.col("event_date"), window_days)
            <= F.col("observation_period_end_date")
        )
        .select("person_id", "event_date")
        .distinct()
    )
```

(The first `.where` replaces the existing `event_date >= observation_period_start_date` check; at `prior_obs_days=0` it is identical to before via `date_add(start, 0)`.)

In `_random_observed_year_cohort`, add `prior_obs_days: int = 0` to the signature and thread it into the `_random_event_windows` call:

```python
    windows = _random_event_windows(
        cond_df, op, date_col=date_col, window_days=window_days,
        prior_obs_days=prior_obs_days,
    )
```

- [ ] **Step 4: Run the test to verify it passes + no regressions**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -v`
Expected: the new test passes and all pre-existing tests stay green (the default `prior_obs_days=0` preserves existing behavior).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): optional prior-coverage predicate for the random general-arm anchor"
```

---

### Task 2: Ingredient resolver, first-drug-era helper, and drug registry

Add name-based RxNorm-ingredient resolution and the first-era-per-person helper — the two pure building blocks the drug cohort needs, plus the class→ingredient-names registry.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `_ingredient_concept_ids(concept_df, ingredient_names) -> DataFrame` — from a `concept`-shaped DataFrame (`concept_id`, `concept_name`, `vocabulary_id`, `concept_class_id`), returns the distinct `concept_id`s of RxNorm `Ingredient` concepts whose `concept_name` case-insensitively matches `ingredient_names`.
  - `_first_drug_era_dates(drug_era_df, ingredient_concept_ids) -> DataFrame` — from a `drug_era`-shaped DataFrame (`person_id`, `drug_concept_id`, `drug_era_start_date`) and a single-column `concept_id` DataFrame, returns `(person_id, index_date)` = each person's earliest era-start among those ingredients.
  - `_DRUG_REGISTRY: dict[str, dict]` with keys `glp1_ra`, `sglt2i`, `tirzepatide`, each `{"ingredient_names": tuple[str, ...]}`, and `_COMBO_MAX_GAP_DAYS = 90`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ingredient_concept_ids_matches_rxnorm_ingredients_case_insensitively(spark):
    from charmpheno.omop.cohorts import _ingredient_concept_ids
    concept = spark.createDataFrame(
        [
            (11, "semaglutide", "RxNorm", "Ingredient"),
            (12, "Empagliflozin", "RxNorm", "Ingredient"),   # case differs
            (13, "semaglutide", "RxNorm", "Brand Name"),      # wrong class
            (14, "metformin", "RxNorm", "Ingredient"),        # not requested
            (15, "semaglutide", "ATC", "Ingredient"),         # wrong vocab
        ],
        ["concept_id", "concept_name", "vocabulary_id", "concept_class_id"],
    )
    out = _ingredient_concept_ids(concept, ["semaglutide", "empagliflozin"])
    assert {r["concept_id"] for r in out.collect()} == {11, 12}


def test_first_drug_era_dates_picks_earliest_era_per_person(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _first_drug_era_dates
    drug_era = spark.createDataFrame(
        [
            (1, 11, dt.date(2020, 3, 1)),
            (1, 11, dt.date(2019, 5, 1)),   # earlier -> wins for person 1
            (2, 12, dt.date(2021, 1, 1)),
            (3, 99, dt.date(2020, 1, 1)),   # ingredient not in set -> person 3 absent
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date"],
    )
    ids = spark.createDataFrame([(11,), (12,)], ["concept_id"])
    out = {r["person_id"]: r["index_date"] for r in _first_drug_era_dates(drug_era, ids).collect()}
    assert out == {1: dt.date(2019, 5, 1), 2: dt.date(2021, 1, 1)}


def test_drug_registry_shape():
    from charmpheno.omop.cohorts import _DRUG_REGISTRY, _COMBO_MAX_GAP_DAYS
    assert _COMBO_MAX_GAP_DAYS == 90
    assert set(_DRUG_REGISTRY) == {"glp1_ra", "sglt2i", "tirzepatide"}
    assert "semaglutide" in _DRUG_REGISTRY["glp1_ra"]["ingredient_names"]
    assert _DRUG_REGISTRY["tirzepatide"]["ingredient_names"] == ("tirzepatide",)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k "ingredient_concept_ids or first_drug_era or drug_registry" -v`
Expected: FAIL with ImportError on `_ingredient_concept_ids`.

- [ ] **Step 3: Add the constants, registry, and helpers**

Add near the other module constants (after the disease registry block):

```python
# Co-initiation gap (days) below which a new-user of BOTH a GLP-1 RA and an
# SGLT2i is treated as combination therapy (glp1_sglt2_combo) rather than two
# separate single-drug years. v1 default; re-cut from the co-initiation gap
# histogram (_coinitiation_gap_histogram) emitted at build time. Not yet a
# frontmatter field.
_COMBO_MAX_GAP_DAYS = 90

# Drug classes for the population_glp1 gated cohort, resolved by RxNorm
# Ingredient NAME (not hard-coded concept_ids, so it is portable across CDR
# vocab versions). VERIFY ON FIRST RUN that each name set resolves to a
# non-empty ingredient set on the target CDR (see apply_population_drug_cohort's
# build-time diagnostic).
_DRUG_REGISTRY: dict[str, dict] = {
    "glp1_ra": {"ingredient_names": (
        "semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide",
    )},
    "sglt2i": {"ingredient_names": (
        "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    )},
    "tirzepatide": {"ingredient_names": ("tirzepatide",)},
}
```

Add the helpers (place them just above where the drug cohort functions will go, after the disease-track functions):

```python
def _ingredient_concept_ids(
    concept_df: DataFrame, ingredient_names: Sequence[str],
) -> DataFrame:
    """Resolve RxNorm Ingredient concept_ids by (case-insensitive) name.

    ``concept_df`` must have ``concept_id``, ``concept_name``, ``vocabulary_id``,
    ``concept_class_id``. Returns the distinct ``concept_id`` of standard RxNorm
    ingredients whose name matches ``ingredient_names``. drug_era is recorded at
    the ingredient level, so these ids join directly against
    ``drug_era.drug_concept_id`` — no ancestor expansion needed.
    """
    names_lower = [n.lower() for n in ingredient_names]
    return (
        concept_df
        .where(F.col("vocabulary_id") == "RxNorm")
        .where(F.col("concept_class_id") == "Ingredient")
        .where(F.lower(F.col("concept_name")).isin(names_lower))
        .select("concept_id")
        .distinct()
    )


def _first_drug_era_dates(
    drug_era_df: DataFrame, ingredient_concept_ids: DataFrame,
) -> DataFrame:
    """Earliest drug_era start per person among the given ingredient concept_ids.

    ``drug_era_df`` has ``person_id``, ``drug_concept_id``,
    ``drug_era_start_date``. Returns ``(person_id, index_date)`` — the person's
    first exposure to the class. Persons with no matching era are absent.
    """
    return (
        drug_era_df.join(
            F.broadcast(ingredient_concept_ids),
            drug_era_df.drug_concept_id == ingredient_concept_ids.concept_id,
            how="inner",
        )
        .groupBy("person_id")
        .agg(F.min("drug_era_start_date").alias("index_date"))
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k "ingredient_concept_ids or first_drug_era or drug_registry" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): RxNorm-ingredient resolver + first-drug-era helper + drug registry"
```

---

### Task 3: The five-way partition core (`_assign_drug_groups`)

The heart of the design: pure precedence + combo-gap + exclusion logic over per-class first-era dates. No BQ — fully unit-tested.

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: `_COMBO_MAX_GAP_DAYS` (Task 2).
- Produces: `_assign_drug_groups(g, s, t, *, combo_max_gap_days=_COMBO_MAX_GAP_DAYS) -> DataFrame` — inputs are three `(person_id, index_date)` DataFrames (first glp1_ra / sglt2i / tirzepatide era dates). Returns `(person_id, source_cohort, index_date)` for **drug-arm** persons only; non-combo both-users and drug-free persons are absent from the output.

- [ ] **Step 1: Write the failing test**

```python
def test_assign_drug_groups_precedence_combo_and_exclusion(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _assign_drug_groups
    d = dt.date

    def frame(rows):  # rows: list[(person_id, date)]
        return spark.createDataFrame(rows, ["person_id", "index_date"])

    # g = GLP-1 first era, s = SGLT2i first era, t = tirzepatide first era
    g = frame([(1, d(2021, 1, 1)),                     # glp1 only
               (4, d(2021, 1, 1)), (5, d(2021, 1, 1)), # combo / excluded
               (6, d(2021, 1, 1)), (7, d(2021, 1, 1))])# tirzepatide-precedence cases
    s = frame([(2, d(2021, 1, 1)),                     # sglt2i only
               (4, d(2021, 2, 1)),                     # +31d from g4 -> combo (<=90)
               (5, d(2021, 9, 1)),                     # +243d from g5 -> excluded (>90)
               (6, d(2021, 3, 1))])                    # p6 also has t -> tirzepatide
    t = frame([(3, d(2021, 1, 1)),                     # tirzepatide only
               (6, d(2021, 5, 1)), (7, d(2021, 6, 1))])# precedence over g/s

    out = {r["person_id"]: (r["source_cohort"], r["index_date"])
           for r in _assign_drug_groups(g, s, t, combo_max_gap_days=90).collect()}

    assert out[1] == ("glp1_ra", d(2021, 1, 1))
    assert out[2] == ("sglt2i", d(2021, 1, 1))
    assert out[3] == ("tirzepatide", d(2021, 1, 1))
    assert out[4] == ("glp1_sglt2_combo", d(2021, 1, 1))   # index = earlier of g,s
    assert 5 not in out                                    # both, gap>90 -> excluded
    assert out[6] == ("tirzepatide", d(2021, 5, 1))        # t wins over g+s
    assert out[7] == ("tirzepatide", d(2021, 6, 1))        # t wins over g
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_assign_drug_groups_precedence_combo_and_exclusion -v`
Expected: FAIL with ImportError on `_assign_drug_groups`.

- [ ] **Step 3: Implement `_assign_drug_groups`**

```python
def _assign_drug_groups(
    g: DataFrame, s: DataFrame, t: DataFrame,
    *, combo_max_gap_days: int = _COMBO_MAX_GAP_DAYS,
) -> DataFrame:
    """Assign each drug-exposed person to exactly one foreground group.

    Inputs are per-class first-era dates ``(person_id, index_date)`` for
    glp1_ra (``g``), sglt2i (``s``), tirzepatide (``t``). Precedence:

    - has tirzepatide            -> ``tirzepatide``       (index = first tirzepatide)
    - has glp1_ra AND sglt2i, no tirzepatide:
        - ``|g - s| <= combo_max_gap_days`` -> ``glp1_sglt2_combo`` (index = earlier)
        - otherwise                          -> EXCLUDED (row dropped)
    - has glp1_ra only           -> ``glp1_ra``           (index = g)
    - has sglt2i only            -> ``sglt2i``            (index = s)

    Non-combo both-users are dropped entirely (returned in neither a single arm
    nor the caller's ``general`` arm), so the single-drug arms are "only ever
    that class". Returns ``(person_id, source_cohort, index_date)``.
    """
    g2 = g.select("person_id", F.col("index_date").alias("g_date"))
    s2 = s.select("person_id", F.col("index_date").alias("s_date"))
    t2 = t.select("person_id", F.col("index_date").alias("t_date"))
    joined = (
        g2.join(s2, on="person_id", how="full_outer")
          .join(t2, on="person_id", how="full_outer")
    )
    has_g = F.col("g_date").isNotNull()
    has_s = F.col("s_date").isNotNull()
    has_t = F.col("t_date").isNotNull()
    gap = F.abs(F.datediff(F.col("g_date"), F.col("s_date")))
    combo_index = F.least(F.col("g_date"), F.col("s_date"))

    source = (
        F.when(has_t, F.lit("tirzepatide"))
        .when(has_g & has_s & (gap <= combo_max_gap_days), F.lit("glp1_sglt2_combo"))
        .when(has_g & has_s, F.lit(None))          # excluded
        .when(has_g, F.lit("glp1_ra"))
        .when(has_s, F.lit("sglt2i"))
        .otherwise(F.lit(None))
    )
    index_date = (
        F.when(has_t, F.col("t_date"))
        .when(has_g & has_s & (gap <= combo_max_gap_days), combo_index)
        .when(has_g & has_s, F.lit(None))
        .when(has_g, F.col("g_date"))
        .when(has_s, F.col("s_date"))
        .otherwise(F.lit(None))
    )
    return (
        joined.withColumn("source_cohort", source)
        .withColumn("index_date", index_date)
        .where(F.col("source_cohort").isNotNull())
        .select("person_id", "source_cohort", "index_date")
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_assign_drug_groups_precedence_combo_and_exclusion -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): _assign_drug_groups five-way precedence + combo-gap + exclusion core"
```

---

### Task 4: Co-initiation gap histogram diagnostic

A pure diagnostic over both-class users' `|g − s|` gaps, so `combo_max_gap_days` is set from data. Logged at build time by the cohort (Task 5).

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Produces: `_coinitiation_gap_histogram(g, s) -> DataFrame` — from the glp1_ra (`g`) and sglt2i (`s`) first-era `(person_id, index_date)` frames, returns `(bucket, n)` counts of `|g − s|` (days) for persons in both, bucketed `0-7 / 8-30 / 31-90 / 91-180 / 181-365 / 366+`.

- [ ] **Step 1: Write the failing test**

```python
def test_coinitiation_gap_histogram_buckets(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import _coinitiation_gap_histogram
    d = dt.date
    g = spark.createDataFrame(
        [(1, d(2021, 1, 1)), (2, d(2021, 1, 1)), (3, d(2021, 1, 1)),
         (4, d(2021, 1, 1)), (9, d(2021, 1, 1))],   # p9 has no s -> excluded from hist
        ["person_id", "index_date"])
    s = spark.createDataFrame(
        [(1, d(2021, 1, 4)),    # gap 3   -> 0-7
         (2, d(2021, 1, 21)),   # gap 20  -> 8-30
         (3, d(2021, 7, 20)),   # gap 200 -> 181-365
         (4, d(2022, 6, 1))],   # gap 516 -> 366+
        ["person_id", "index_date"])
    hist = {r["bucket"]: r["n"] for r in _coinitiation_gap_histogram(g, s).collect()}
    assert hist == {"0-7": 1, "8-30": 1, "181-365": 1, "366+": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_coinitiation_gap_histogram_buckets -v`
Expected: FAIL with ImportError on `_coinitiation_gap_histogram`.

- [ ] **Step 3: Implement `_coinitiation_gap_histogram`**

```python
def _coinitiation_gap_histogram(g: DataFrame, s: DataFrame) -> DataFrame:
    """Bucketed |g - s| gap counts for persons who are new-users of BOTH
    glp1_ra and sglt2i. A no-fit diagnostic: eyeball where the co-initiation
    cluster ends to set ``_COMBO_MAX_GAP_DAYS``. Returns ``(bucket, n)``.
    """
    both = (
        g.select("person_id", F.col("index_date").alias("g_date"))
        .join(s.select("person_id", F.col("index_date").alias("s_date")),
              on="person_id", how="inner")
    )
    gap = F.abs(F.datediff(F.col("g_date"), F.col("s_date")))
    bucket = (
        F.when(gap <= 7, "0-7")
        .when(gap <= 30, "8-30")
        .when(gap <= 90, "31-90")
        .when(gap <= 180, "91-180")
        .when(gap <= 365, "181-365")
        .otherwise("366+")
    )
    return (
        both.withColumn("bucket", bucket)
        .groupBy("bucket")
        .agg(F.count(F.lit(1)).alias("n"))
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py::test_coinitiation_gap_histogram_buckets -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): co-initiation gap-histogram diagnostic for combo threshold"
```

---

### Task 5: `apply_population_drug_cohort` orchestration + register `population_glp1`

Wire the pieces into the population cohort, register it, and add its defaults file. BQ-reading orchestration → validation-surface + registration tests (the logic pieces are already unit-tested in Tasks 1–4).

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py` (function + `SUPPORTED_COHORTS` + `COHORT_METADATA` + `apply_cohort`)
- Create: `experiments/defaults/population_glp1.yaml`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: `_ingredient_concept_ids`, `_first_drug_era_dates`, `_DRUG_REGISTRY`, `_COMBO_MAX_GAP_DAYS` (T2); `_assign_drug_groups` (T3); `_coinitiation_gap_histogram` (T4); `_window_observed_cohort`, `_random_observed_year_cohort` (existing / T1).
- Produces: `apply_population_drug_cohort(cond_df, *, window_days=_WINDOW_DAYS, prior_obs_days=_WINDOW_DAYS, combo_max_gap_days=_COMBO_MAX_GAP_DAYS, spark, cdr_dataset, billing_project, date_col) -> DataFrame`, and `"population_glp1"` registered in all three sites.

- [ ] **Step 1: Write the failing tests**

```python
def test_supported_cohorts_includes_population_glp1():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "population_glp1" in SUPPORTED_COHORTS


def test_cohort_metadata_has_population_glp1():
    from charmpheno.omop.cohorts import COHORT_METADATA
    m = COHORT_METADATA["population_glp1"]
    assert m["id"] == "population_glp1"
    assert m["label"] and m["description"]


def test_apply_population_drug_cohort_importable_signature():
    import inspect
    from charmpheno.omop.cohorts import apply_population_drug_cohort
    p = inspect.signature(apply_population_drug_cohort).parameters
    assert {"window_days", "prior_obs_days", "combo_max_gap_days", "date_col"} <= set(p)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -k population_glp1 -v`
Expected: FAIL (`population_glp1` not in `SUPPORTED_COHORTS`).

- [ ] **Step 3: Implement `apply_population_drug_cohort`**

Place it after the disease/sparse population cohorts:

```python
def apply_population_drug_cohort(
    cond_df: DataFrame,
    *,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = _WINDOW_DAYS,
    combo_max_gap_days: int = _COMBO_MAX_GAP_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
) -> DataFrame:
    """Whole-population background + four drug foreground arms, disjoint.

    Anchors on ``drug_era``: each person's first new-user era of glp1_ra /
    sglt2i / tirzepatide (ingredients resolved by name via
    :func:`_ingredient_concept_ids`) gives per-class index dates, partitioned by
    :func:`_assign_drug_groups` (tirzepatide → glp1_sglt2_combo → single-class,
    non-combo both-users excluded). Chosen index dates are new-user-bracketed
    (:func:`_window_observed_cohort`: ``prior_obs_days`` prior coverage + observed
    ``window_days`` follow-up). The ``general`` arm is every person with NO
    tracked drug exposure, windowed to a random observed year with the SAME
    prior+forward bracket (:func:`_random_observed_year_cohort`). Documents are
    the person's condition rows in ``[index_date, index_date + window_days)``.
    Returns ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    concept = _read("concept").select(
        "concept_id", "concept_name", "vocabulary_id", "concept_class_id",
    )
    drug_era = _read("drug_era").select(
        "person_id", "drug_concept_id", "drug_era_start_date",
    )
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )

    def _first_dates(class_key: str) -> DataFrame:
        ids = _ingredient_concept_ids(
            concept, _DRUG_REGISTRY[class_key]["ingredient_names"],
        )
        return _first_drug_era_dates(drug_era, ids)

    g = _first_dates("glp1_ra")
    s = _first_dates("sglt2i")
    t = _first_dates("tirzepatide")

    # Build-time diagnostic: co-initiation gap distribution sets combo_max_gap_days.
    print("[cohort population_glp1] GLP-1/SGLT2i co-initiation |g-s| gap histogram "
          f"(combo_max_gap_days={combo_max_gap_days}):", flush=True)
    for row in _coinitiation_gap_histogram(g, s).orderBy("bucket").collect():
        print(f"[cohort population_glp1]   {row['bucket']}: {row['n']}", flush=True)

    assigned = _assign_drug_groups(g, s, t, combo_max_gap_days=combo_max_gap_days)

    # New-user observation bracket on the chosen index; rejoin to recover the tag.
    bracketed = _window_observed_cohort(
        assigned.select("person_id", "index_date"), op,
        prior_obs_days=prior_obs_days, window_days=window_days,
    )
    drug_docs = assigned.join(bracketed, on=["person_id", "index_date"], how="inner")
    drug_windows = (
        cond_df.join(drug_docs, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )

    # general = persons with NO tracked drug exposure at all (excluded both-users
    # and inadequately-observed initiators are NOT background).
    drug_persons = (
        g.select("person_id")
        .unionByName(s.select("person_id"))
        .unionByName(t.select("person_id"))
        .distinct()
    )
    non_drug = cond_df.join(drug_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_drug, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days, prior_obs_days=prior_obs_days,
    ).withColumn("source_cohort", F.lit("general"))

    return drug_windows.unionByName(general)
```

- [ ] **Step 4: Register `population_glp1`**

Add `"population_glp1"` to `SUPPORTED_COHORTS`. Add to `COHORT_METADATA`:

```python
    "population_glp1": {
        "id": "population_glp1",
        "label": "Population + GLP-1 & comparators (gated)",
        "description": (
            "The whole population as a shared background, with four drug "
            "foreground arms anchored on the first year after starting a "
            "medication (incident new-user: a year of prior coverage, a "
            "fully-observed follow-up year). Arms: glp1_ra (GLP-1 receptor "
            "agonists), sglt2i (SGLT2 inhibitors, the active comparator), "
            "tirzepatide (dual GIP/GLP-1 agonist), and glp1_sglt2_combo "
            "(new-users of both a GLP-1 RA and an SGLT2i within a short "
            "co-initiation window). Documents are the conditions in that year; "
            "drugs are the anchor only. The general background carries the same "
            "1-year-prior + 1-year-follow-up observability bracket. A gated "
            "block-wise correlated STM then shows what is distinctive to each "
            "arm and its (anti-)correlations with the background comorbidity "
            "topics."
        ),
    },
```

Add the `apply_cohort` dispatch branch (before the final `raise`):

```python
    if cohort == "population_glp1":
        return apply_population_drug_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
```

(`combo_max_gap_days` and `window_days` use their defaults — v1 scoping.)

- [ ] **Step 5: Create the defaults file**

Create `experiments/defaults/population_glp1.yaml`:

```yaml
# Whole-population background + GLP-1/SGLT2i/tirzepatide/combo drug foreground
# gated cohort. The experiment frontmatter supplies the STM covariate + gating
# fields (K, background_k, foreground, group_var, covariate_formula); this file
# exists so load_defaults finds a <cohort>.yaml to merge over _base.yaml.
#
# `cohort` is the display/lookup id (matches filename); `cohort_def` is the
# value passed to the driver's --cohort.
cohort: population_glp1
cohort_def: population_glp1
```

- [ ] **Step 6: Run to verify pass + full suite green**

Run: `cd charmpheno && python -m pytest tests/test_cohorts.py -v`
Expected: all pass (new population_glp1 registration/signature tests + every pre-existing test).

- [ ] **Step 7: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py experiments/defaults/population_glp1.yaml
git commit -m "feat(cohorts): apply_population_drug_cohort + register population_glp1"
```

---

### Task 6: Experiment record 0044

Add the experiment definition so `run_experiment.py --id 44` resolves and dispatches.

**Files:**
- Create: `docs/experiments/0044-stm-population-glp1-comparator.md`

**Interfaces:**
- Consumes: `run_experiment.read_frontmatter`/`validate_frontmatter`/`load_defaults` (`EXPERIMENTS_DIR = docs/experiments`, `DEFAULTS_DIR = experiments/defaults`). Required frontmatter: `id`, `slug`, `cohort`, `model_class`; for STM also `covariate_formula`, `categorical_cols`, `continuous_cols`.

- [ ] **Step 1: Create the experiment record**

Create `docs/experiments/0044-stm-population-glp1-comparator.md`:

```markdown
---
id: 44
slug: stm-population-glp1-comparator
status: pending
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
K: 140
background_k: 80
foreground: "glp1_ra:15,sglt2i:15,tirzepatide:15,glp1_sglt2_combo:15"
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
one document per person, `source_cohort ∈ {glp1_ra, sglt2i, tirzepatide,
glp1_sglt2_combo, general}`:

- **glp1_ra / sglt2i** — incident new-users of that class only (never the other
  tracked classes); 365d prior coverage, 365d observed follow-up.
- **tirzepatide** — new-users of tirzepatide (dual GIP/GLP-1; precedence over
  the other arms).
- **glp1_sglt2_combo** — new-users of both a GLP-1 RA and an SGLT2i co-initiated
  within `combo_max_gap_days` (code default 90; set from the build-time gap
  histogram). Non-combo both-users are excluded from the cohort.
- **general** — no tracked drug exposure; random observed year with the SAME
  1yr-prior + 1yr-follow-up bracket.

Documents are the person's conditions in the post-index year; drugs are the
anchor only.

## Configuration

Full population (`person_mod: 1`) — the thin tirzepatide/combo arms need it.
K=140 = 80 background + 15 × 4 foreground. Incident new-user (`prior_obs_days:
365`), 1-year windows. Otherwise the exp 0043 hardened + slowed stack (subsample
0.1, tau0 256, kappa 0.7, max_iter 300, reference + dense spectral, sigma_init 1,
min_pair_support 10, block-wise unit-diagonal Σ / ADR 0034), `~ C(sex) + age`,
`known_sex_only`.

## Success criteria

- Build diagnostics report non-empty per-arm document counts (watch tirzepatide +
  combo); the co-initiation gap histogram is logged so `combo_max_gap_days` can
  be re-cut.
- Drug foreground arms recover recognizable, distinctive structure (e.g. GI /
  weight / appetite signal for GLP-1; genitourinary / volume for SGLT2i) and
  interpretable GLP-1↔background anti-correlations.
- Σ variance bounded (no runaway); honest correlation report.

## Related

First cohort on the drug-anchor track (parallel to the disease track). Follows
exp 0043 (population_eds) on the same hardened + slowed stack.
```

- [ ] **Step 2: Validate frontmatter + defaults resolve**

Run:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -c "
import sys; sys.path.insert(0, 'scripts')
import run_experiment as r
p = r.find_by_id(r.EXPERIMENTS_DIR, 44)
fm = r.read_frontmatter(p); r.validate_frontmatter(fm)
r.load_defaults(fm['cohort'], r.DEFAULTS_DIR)
print(44, p.name, 'cohort=', fm['cohort'], 'K=', fm['K'], 'foreground=', fm['foreground'], 'OK')
"
```

Expected: prints `44 0044-stm-population-glp1-comparator.md cohort= population_glp1 K= 140 foreground= glp1_ra:15,sglt2i:15,tirzepatide:15,glp1_sglt2_combo:15 OK`, no exception.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/0044-stm-population-glp1-comparator.md
git commit -m "feat(experiments): add exp 0044 population_glp1 drug-anchored gated STM"
```

---

### Task 7: Cluster fit + export + dashboard (operational, user-driven)

Not local TDD — runs on the Dataproc master from `analysis/cloud/`, after Tasks 1–6 are on the branch. Mirrors the population_eds (0043) flow.

- [ ] **Step 1: Verify the drug ingredient sets resolve on the CDR**

Before fitting, confirm each class resolves to non-empty RxNorm ingredients and workable person counts, e.g.:

```sql
SELECT LOWER(concept_name) AS ingredient, COUNT(*) n
FROM concept
WHERE vocabulary_id='RxNorm' AND concept_class_id='Ingredient'
  AND LOWER(concept_name) IN ('semaglutide','liraglutide','dulaglutide','exenatide',
    'lixisenatide','empagliflozin','dapagliflozin','canagliflozin','ertugliflozin','tirzepatide')
GROUP BY ingredient ORDER BY ingredient;
```

Expected: all ten ingredients present. If any is missing/renamed in this vocab version, adjust `_DRUG_REGISTRY` names accordingly.

- [ ] **Step 2: Build covariates + fit exp 0044**

Run: `cd analysis/cloud && make build-covariates EXP=44 && make exp ID=44`
Expected: the build logs the co-initiation gap histogram and per-arm doc counts; `make summary ID=44` shows all five `source_cohort` groups present, bounded Σ, and convergence. If tirzepatide/combo arms are too thin to carry 15 topics, note it and consult (merge tirzepatide→glp1_ra or drop, per the spec's split-now-merge-later plan); if the gap histogram shows 90 is the wrong combo cut, that motivates threading `combo_max_gap_days` to frontmatter (deferred fast-follow).

- [ ] **Step 3: Export, annotate, and add to the dashboard**

Run `make build-dashboard-exp ID=44`; download the bundle; ingest into `dashboard/public/data/population_glp1/`; run `scripts/label_phenotypes.py --bundle-dir dashboard/public/data/population_glp1`; add `population_glp1` to `dashboard/public/data/manifest.json` as an additional cohort (population_cancer stays default) — the same flow used for population_eds.

- [ ] **Step 4: Record the outcome**

Update `docs/experiments/0044-...md` `status:` and a `## Result` section (per-arm counts, gap histogram takeaway, distinctive arm phenotypes, Σ behavior); add a numbered `docs/insights/` entry if a non-obvious finding surfaces. Commit.

---

## Self-Review

**Spec coverage:**
- Drug-anchor track / drug registry / ingredient-name resolution → Tasks 2, 5. ✓
- `apply_first_drug_year_cohort` primitive — folded into `_first_drug_era_dates` + `apply_population_drug_cohort` (the single-class public function is unused by the spec's cohort; YAGNI — not built). ✓ (deviation noted below)
- Five-way partition + precedence + combo-gap + exclusion (single arms "only ever that class") → Task 3. ✓
- Combo = co-initiation within `combo_max_gap_days` (default 90, code-level) → Tasks 2, 3. ✓
- Gap-histogram diagnostic emitted at build → Tasks 4, 5. ✓
- Symmetric general-arm observability (prior-coverage predicate) → Task 1. ✓
- Register population_glp1 (three sites) + defaults + experiment record → Tasks 5, 6. ✓
- Fit/export/annotate/dashboard + ingredient-resolution gate → Task 7. ✓
- Disease track untouched; drug content anchor-only → honored throughout. ✓

**Intentional deviation from the spec:** the spec names a standalone `apply_first_drug_year_cohort` primitive; the plan instead factors the reusable core as `_first_drug_era_dates` (index dates) because the partition needs per-class dates *before* windowing, and a single-class public drug cohort has no consumer (YAGNI). Same capability, cleaner seam. Flag to the final reviewer.

**Placeholder scan:** No TBD/TODO. The drug ingredient sets are resolved by NAME in code (no unresolved concept_ids); the cluster ingredient-count check (Task 7 Step 1) is a data-verification gate, not a code gap. Cluster commands are exact Makefile targets; their outputs are environment-dependent and verified via `summary` / the build diagnostic.

**Type consistency:** `_ingredient_concept_ids` → `_first_drug_era_dates` (concept_id frame) → `_assign_drug_groups` (g/s/t `(person_id, index_date)` frames) → `apply_population_drug_cohort` (joins on `person_id`,`index_date`, tags `source_cohort`). `_coinitiation_gap_histogram(g, s)` takes the same `(person_id, index_date)` frames. `source_cohort` values (`glp1_ra`/`sglt2i`/`tirzepatide`/`glp1_sglt2_combo`/`general`) match the experiment `foreground` string and `group_var`. `_random_event_windows`/`_random_observed_year_cohort` gain `prior_obs_days=0` default (Task 1), consumed with `prior_obs_days=prior_obs_days` in Task 5.
