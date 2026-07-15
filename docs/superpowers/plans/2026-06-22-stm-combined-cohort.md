# Combined cancer/dementia cohort + source covariate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a combined cancer∪dementia STM corpus with a per-document `source_cohort` covariate (`~ C(source_cohort) + C(sex) + age`), as an end-to-end STM validation experiment.

**Architecture:** All changes are charmpheno-side, upstream of the corpus↔covariate join. A comorbid patient contributes two documents (one per cohort), kept distinct by encoding `source_cohort` into `doc_id` via a new DocSpec; the covariate builder is generalized to a composite `[person_id, source_cohort]` key; the fit driver joins on that composite key. The spark-vi engine and shim are untouched; the dashboard bundle stays aggregate.

**Tech Stack:** PySpark (DataFrame + RDD), formulaic (covariate formula), pytest with a session-scoped local Spark fixture.

## Global Constraints

- Spec: [docs/superpowers/specs/2026-06-22-stm-combined-cohort-covariate-design.md](../specs/2026-06-22-stm-combined-cohort-covariate-design.md). Every task implicitly includes these.
- spark-vi (engine + MLlib shim) must NOT be modified and must never reference `person_id`/`doc_id`. All work is in `charmpheno/` and `analysis/cloud/`.
- No LaTeX in code/docs (Unicode Greek if needed). No emojis in committed files.
- Markdown-linkable code references in any prose: `[name](path#Lstart-Lend)`.
- Tests run under the repo's `.venv`: from a package dir, `../.venv/bin/python -m pytest ...`. The `spark` fixture is session-scoped local Spark (already pins `PYSPARK_PYTHON`).
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Keep subjects tight.
- Comorbid = patient qualifying for BOTH cohorts. They are kept as two documents (not dropped, not merged).

---

## File Structure

- `charmpheno/charmpheno/omop/doc_spec.py` — add `PatientCohortDocSpec` (doc_id encodes source_cohort).
- `charmpheno/charmpheno/omop/cohorts.py` — add `apply_cancer_or_dementia_cohort` + `_combine_cohorts` helper; register `cancer_or_dementia`.
- `charmpheno/charmpheno/omop/covariates.py` — generalize `build_patient_covariate_df` with a `key_cols` parameter.
- `analysis/cloud/_covariates_load.py` — thread `key_cols` through `load_or_build_covariates`.
- `analysis/cloud/stm_bigquery_cloud.py` — combined-cohort branch: decode `source_cohort` from `doc_id`, expand person table, composite join.
- `docs/experiments/0002-stm-cancer-dementia.md` — the experiment definition.
- Tests: `charmpheno/tests/test_doc_spec.py`, `test_cohorts.py`, `test_covariates_build.py`, `test_stm_smoke.py`.

---

### Task 1: `PatientCohortDocSpec`

**Files:**
- Modify: `charmpheno/charmpheno/omop/doc_spec.py`
- Test: `charmpheno/tests/test_doc_spec.py`

**Interfaces:**
- Produces: `PatientCohortDocSpec(min_doc_length=0)`, registry name `"patient_cohort"`, `required_columns=("person_id","source_cohort")`, `derive_docs` adds `doc_id = "{source_cohort}:{person_id}"`. Round-trips via `manifest()` / `from_manifest()`.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_doc_spec.py`:

```python
def test_patient_cohort_doc_id_encodes_source_cohort(spark):
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    df = spark.createDataFrame(
        [(1, "cancer", 100), (1, "dementia", 100), (2, "cancer", 200)],
        ["person_id", "source_cohort", "concept_id"],
    )
    out = PatientCohortDocSpec().derive_docs(df)
    ids = {r["doc_id"] for r in out.select("doc_id").collect()}
    # Same person, two cohorts -> two distinct doc_ids (no merge).
    assert ids == {"cancer:1", "dementia:1", "cancer:2"}


def test_patient_cohort_manifest_roundtrips():
    from charmpheno.omop.doc_spec import PatientCohortDocSpec, DocSpec
    spec = PatientCohortDocSpec(min_doc_length=5)
    back = DocSpec.from_manifest(spec.manifest())
    assert isinstance(back, PatientCohortDocSpec)
    assert back.min_doc_length == 5
    assert back.name == "patient_cohort"


def test_patient_cohort_requires_source_cohort_column(spark):
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    df = spark.createDataFrame([(1, 100)], ["person_id", "concept_id"])
    import pytest
    with pytest.raises(ValueError):
        PatientCohortDocSpec().derive_docs(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_doc_spec.py -k patient_cohort -q`
Expected: FAIL with `ImportError: cannot import name 'PatientCohortDocSpec'`.

- [ ] **Step 3: Write minimal implementation**

In `charmpheno/charmpheno/omop/doc_spec.py`, after `PatientDocSpec` (around line 118), add:

```python
@_register
@dataclass(frozen=True)
class PatientCohortDocSpec(DocSpec):
    """One document per (patient, source_cohort).

    doc_id = "{source_cohort}:{person_id}". Used by the combined
    cancer_or_dementia corpus so a comorbid patient's two documents (one per
    cohort) stay distinct through the groupBy(doc_id) in to_bow_dataframe.
    The source_cohort is recovered downstream by splitting doc_id on ':'.
    """
    name: str = field(default="patient_cohort", init=False)
    required_columns: tuple[str, ...] = field(
        default=("person_id", "source_cohort"), init=False)
    min_doc_length: int = 0

    def derive_docs(self, events_df: DataFrame) -> DataFrame:
        self.validate(events_df)
        return events_df.withColumn(
            "doc_id",
            F.concat_ws(":",
                        F.col("source_cohort").cast("string"),
                        F.col("person_id").cast("string")),
        )

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "min_doc_length": self.min_doc_length}

    @classmethod
    def _from_manifest(cls, d: dict[str, Any]) -> "PatientCohortDocSpec":
        return cls(min_doc_length=int(d.get("min_doc_length", 0)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_doc_spec.py -k patient_cohort -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/doc_spec.py charmpheno/tests/test_doc_spec.py
git commit -m "feat(charmpheno/omop): PatientCohortDocSpec (doc_id encodes source_cohort)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Combined `cancer_or_dementia` cohort

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Consumes: existing `apply_first_cancer_year_cohort`, `apply_first_dementia_year_cohort`.
- Produces: `_combine_cohorts(cancer_events, dementia_events) -> DataFrame` (adds `source_cohort` literal, `unionByName`, no dedup); `apply_cancer_or_dementia_cohort(cond_df, *, spark, cdr_dataset, billing_project, date_col)`; `"cancer_or_dementia"` in `SUPPORTED_COHORTS`, dispatched by `apply_cohort`, present in `COHORT_METADATA`.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_cohorts.py`:

```python
def test_supported_cohorts_includes_cancer_or_dementia():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "cancer_or_dementia" in SUPPORTED_COHORTS


def test_cohort_metadata_has_cancer_or_dementia():
    from charmpheno.omop.cohorts import COHORT_METADATA
    assert "cancer_or_dementia" in COHORT_METADATA


def test_combine_cohorts_tags_and_unions_keeping_comorbid(spark):
    from charmpheno.omop.cohorts import _combine_cohorts
    cancer = spark.createDataFrame([(1, 10), (2, 20)], ["person_id", "concept_id"])
    dementia = spark.createDataFrame([(2, 30), (3, 40)], ["person_id", "concept_id"])
    out = _combine_cohorts(cancer, dementia)
    rows = {(r["person_id"], r["source_cohort"]) for r in out.collect()}
    # person 2 is comorbid -> appears under BOTH labels (no dedup).
    assert rows == {(1, "cancer"), (2, "cancer"), (2, "dementia"), (3, "dementia")}
    assert out.count() == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_cohorts.py -k "cancer_or_dementia or combine_cohorts" -q`
Expected: FAIL (ImportError on `_combine_cohorts` / missing registry entries).

- [ ] **Step 3: Write minimal implementation**

In `charmpheno/charmpheno/omop/cohorts.py`:

(a) Add to `SUPPORTED_COHORTS` (the tuple around line 84):

```python
SUPPORTED_COHORTS: tuple[str, ...] = (
    "first_cancer_year",
    "first_dementia_year",
    "cancer_or_dementia",
)
```

(b) Add a `COHORT_METADATA["cancer_or_dementia"]` entry (after the dementia entry, before the closing `}` near line 134):

```python
    "cancer_or_dementia": {
        "id": "cancer_or_dementia",
        "label": "Cancer or Dementia (combined, source-labeled)",
        "description": (
            "Union of the first-cancer-year and first-dementia-year cohorts, "
            "each document labeled by its source cohort. A patient qualifying "
            "for both contributes two documents (one per cohort). Used as an "
            "STM validation: a source_cohort covariate should produce strongly "
            "separable cancer vs dementia topic structure."
        ),
    },
```

(c) Add the dispatch branch inside `apply_cohort` (after the dementia branch, before the `raise`):

```python
    if cohort == "cancer_or_dementia":
        return apply_cancer_or_dementia_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
        )
```

(d) Add the helper + function (after `apply_first_dementia_year_cohort`, end of file):

```python
def _combine_cohorts(
    cancer_events: DataFrame, dementia_events: DataFrame,
) -> DataFrame:
    """Tag each cohort's events with source_cohort and union (no dedup).

    A comorbid patient's cancer-window events (tagged "cancer") and
    dementia-window events (tagged "dementia") both survive, so they become
    two distinct documents downstream via PatientCohortDocSpec.
    """
    c = cancer_events.withColumn("source_cohort", F.lit("cancer"))
    d = dementia_events.withColumn("source_cohort", F.lit("dementia"))
    return c.unionByName(d)


def apply_cancer_or_dementia_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
) -> DataFrame:
    """Combined cancer-or-dementia cohort with a source_cohort label column.

    Composes the two single-disease cohorts and unions their tagged events.
    Returns cond_df's schema plus a `source_cohort` string column.
    """
    cancer = apply_first_cancer_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
    )
    dementia = apply_first_dementia_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
    )
    return _combine_cohorts(cancer, dementia)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_cohorts.py -k "cancer_or_dementia or combine_cohorts" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(charmpheno/omop): cancer_or_dementia combined cohort with source label

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Generalize `build_patient_covariate_df` with `key_cols`

**Files:**
- Modify: `charmpheno/charmpheno/omop/covariates.py:43-118`
- Test: `charmpheno/tests/test_covariates_build.py`

**Interfaces:**
- Produces: `build_patient_covariate_df(person_df, *, covariate_formula, categorical_cols, continuous_cols, key_cols=("person_id",), max_levels=10_000) -> (cov_df, model_spec, names)`. `cov_df` has columns `[*key_cols, "covariates"]`, one row per distinct key tuple. Default `key_cols=("person_id",)` preserves existing `(person_id, covariates)` output.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_covariates_build.py` inside a new class:

```python
class TestCompositeKeyCovariates:
    def test_composite_key_one_row_per_person_cohort(self, spark):
        import pandas as pd
        from charmpheno.omop.covariates import build_patient_covariate_df
        # person 1 comorbid (two cohorts), person 2 cancer only.
        pdf = pd.DataFrame({
            "person_id":     [1, 1, 2],
            "source_cohort": ["cancer", "dementia", "cancer"],
            "sex":           ["M", "M", "F"],
            "age":           [60.0, 60.0, 70.0],
        })
        person_df = spark.createDataFrame(pdf)
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula="~ C(source_cohort) + C(sex) + age",
            categorical_cols=["source_cohort", "sex"],
            continuous_cols=["age"],
            key_cols=["person_id", "source_cohort"],
        )
        assert set(cov_df.columns) == {"person_id", "source_cohort", "covariates"}
        keys = {(r["person_id"], r["source_cohort"]) for r in cov_df.collect()}
        assert keys == {(1, "cancer"), (1, "dementia"), (2, "cancer")}

    def test_default_key_cols_unchanged(self, spark):
        import pandas as pd
        from charmpheno.omop.covariates import build_patient_covariate_df
        pdf = pd.DataFrame({"person_id": [1, 2], "age": [60.0, 70.0]})
        cov_df, _, _ = build_patient_covariate_df(
            spark.createDataFrame(pdf),
            covariate_formula="~ age",
            categorical_cols=[], continuous_cols=["age"],
        )
        assert set(cov_df.columns) == {"person_id", "covariates"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_covariates_build.py::TestCompositeKeyCovariates -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'key_cols'`).

- [ ] **Step 3: Write minimal implementation**

Modify `build_patient_covariate_df` in `charmpheno/charmpheno/omop/covariates.py`. Change the signature to add `key_cols`, and replace the projection / schema / `_flush` so they use the key columns:

```python
def build_patient_covariate_df(
    person_df: DataFrame,
    *,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    key_cols: tuple[str, ...] | list[str] = ("person_id",),
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
```

In the body, replace the `needed_cols` / `projected` lines (currently 70-72) with:

```python
    key_cols = list(key_cols)
    # De-dup the column list (a key col may also be a formula categorical,
    # e.g. source_cohort) while preserving order.
    needed_cols = list(dict.fromkeys([*key_cols, *categorical_cols, *continuous_cols]))
    projected = person_df.select(*needed_cols).dropDuplicates(key_cols)
```

Replace the output schema (currently 113-116) with one built from the key columns' actual types plus the covariates vector:

```python
    key_fields = [projected.schema[c] for c in key_cols]
    schema = StructType([*key_fields, StructField("covariates", VectorUDT(), False)])
```

Replace `_flush` (currently 104-111) to yield the key tuple followed by the vector:

```python
    def _flush(buf, spec, cat_set, cont_set):
        import pandas as pd
        from spark_vi.mllib.topic._formula import apply_model_spec
        from pyspark.ml.linalg import DenseVector
        pdf = pd.DataFrame(buf)
        X = apply_model_spec(spec, pdf)   # (chunk_size, P)
        for i, row in enumerate(X):
            key_vals = tuple(pdf[c].iloc[i] for c in key_cols)
            yield (*key_vals, DenseVector(list(row)))
```

Note: `LongType` key columns require Python `int`. Pandas yields `numpy.int64`; coerce in `_flush` by casting integer-typed keys. Replace the `key_vals` line with:

```python
            key_vals = tuple(
                int(pdf[c].iloc[i]) if str(projected.schema[c].dataType) in ("LongType()", "IntegerType()")
                else pdf[c].iloc[i]
                for c in key_cols
            )
```

(The existing `_apply_partition` / broadcast / `validate_formula` lines are unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_covariates_build.py -q`
Expected: PASS (new class + existing build tests + the composite-key tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/covariates.py charmpheno/tests/test_covariates_build.py
git commit -m "feat(charmpheno/omop): build_patient_covariate_df key_cols (composite covariate key)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Thread `key_cols` through `load_or_build_covariates`

**Files:**
- Modify: `analysis/cloud/_covariates_load.py`
- Test: covered by Task 7 smoke (no isolated unit test — this is thin pass-through over Task 3 + the cache).

**Interfaces:**
- Consumes: Task 3 `build_patient_covariate_df(..., key_cols=...)`.
- Produces: `load_or_build_covariates(..., key_cols=("person_id",))` forwarding `key_cols` to `build_patient_covariate_df`. Cache layer (`save`/`try_load`) is schema-generic, so no cache change.

- [ ] **Step 1: Add the parameter and forward it**

In `analysis/cloud/_covariates_load.py`, add `key_cols: tuple[str, ...] | list[str] = ("person_id",)` to the `load_or_build_covariates` signature (alongside `max_levels`), and pass it into the `build_patient_covariate_df(...)` call:

```python
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula=covariate_formula,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            key_cols=key_cols,
            max_levels=max_levels,
        )
```

- [ ] **Step 2: Syntax check**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m py_compile analysis/cloud/_covariates_load.py`
Expected: no output (compiles).

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/_covariates_load.py
git commit -m "feat(cloud): thread key_cols through load_or_build_covariates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: STM fit driver — combined-cohort branch

**Files:**
- Modify: `analysis/cloud/stm_bigquery_cloud.py:85-160`
- Test: cluster-validated end-to-end; the decode/expand logic is exercised by the Task 7 smoke.

**Interfaces:**
- Consumes: corpus `bow_df(person_id, doc_id, features)` whose `doc_id = "{source_cohort}:{person_id}"` when `cohort == "cancer_or_dementia"`; Task 3/4 `key_cols`.
- Produces: a composite `[person_id, source_cohort]` join when `source_cohort` is a declared categorical; otherwise the existing `person_id` path is unchanged.

- [ ] **Step 1: Add the combined-cohort branch**

In `analysis/cloud/stm_bigquery_cloud.py` `main()`, after the corpus load (around line 106) and before/within the covariates section, branch on whether `source_cohort` is a declared categorical (the experiment for the combined cohort declares it):

```python
        from pyspark.sql import functions as F
        composite = "source_cohort" in cat_cols

        if composite:
            # doc_id == "{source_cohort}:{person_id}"; recover the label.
            bow_df = bow_df.withColumn(
                "source_cohort",
                F.split(F.col("doc_id"), ":").getItem(0),
            )
            labels = bow_df.select("person_id", "source_cohort").distinct()
            cov_key_cols = ["person_id", "source_cohort"]
            join_on = ["person_id", "source_cohort"]
        else:
            labels = None
            cov_key_cols = ["person_id"]
            join_on = "person_id"
```

- [ ] **Step 2: Expand the person table with labels (composite only)**

Replace the person-table load block (lines 111-118) so the combined cohort joins the corpus-derived labels onto the person table (one row per (person, cohort)):

```python
        with _phase("person table load"):
            person_df = load_person_table(
                spark=spark, cdr_dataset=args.cdr,
                billing_project=args.billing,
                person_sample_mod=args.person_mod, cohort=args.cohort,
            )
            if composite:
                person_df = person_df.join(labels, on="person_id", how="inner")
```

- [ ] **Step 3: Pass `key_cols` to covariates and use the composite join**

In the covariates call (lines 122-133) add `key_cols=cov_key_cols`, and change the join (line 137) to `on=join_on`:

```python
            cov_df, model_spec, covariate_names = load_or_build_covariates(
                spark, person_df=person_df,
                covariate_formula=args.covariate_formula,
                categorical_cols=cat_cols, continuous_cols=cont_cols,
                cdr=args.cdr, source_table=args.source_table,
                cohort=args.cohort, person_mod=args.person_mod,
                cache_uri=args.cache_uri, key_cols=cov_key_cols,
            )
        ...
            joined = bow_df.join(F.broadcast(cov_df), on=join_on, how="inner")
```

- [ ] **Step 4: Syntax check**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m py_compile analysis/cloud/stm_bigquery_cloud.py`
Expected: compiles cleanly.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/stm_bigquery_cloud.py
git commit -m "feat(cloud/stm): combined-cohort composite join on [person_id, source_cohort]

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Experiment definition + cache_uri

**Files:**
- Create: `docs/experiments/0002-stm-cancer-dementia.md`
- Test: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -c "from scripts.run_experiment import read_frontmatter, validate_frontmatter; ..."` (frontmatter validity check below).

**Interfaces:**
- Consumes: `validate_frontmatter` (requires for STM: `covariate_formula`, `categorical_cols`, `continuous_cols`); `doc_unit: patient_cohort` resolves to `PatientCohortDocSpec` via `doc_spec_from_cli` (registered in Task 1).

- [ ] **Step 1: Write the experiment file**

Create `docs/experiments/0002-stm-cancer-dementia.md`:

```markdown
---
id: 2
slug: stm-cancer-dementia
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
doc_unit: patient_cohort
covariate_formula: "~ C(source_cohort) + C(sex) + age"
categorical_cols: [source_cohort, sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
max_iter: 100
---

# STM validation: combined cancer/dementia cohort with source covariate

Validates STM prevalence covariates end to end. The combined corpus unions the
first-cancer-year and first-dementia-year cohorts; each document is labeled by
its `source_cohort`. A comorbid patient contributes two documents (one per
cohort). Success criterion: the fitted Gamma shows strong, opposite-sign
`source_cohort` loadings on cancer-leaning vs dementia-leaning topics, and the
dashboard's faithful corpus_prevalence reflects the cohort mix.
```

- [ ] **Step 2: Validate the frontmatter**

Run:
```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
from run_experiment import read_frontmatter, validate_frontmatter
from pathlib import Path
fm = read_frontmatter(Path('docs/experiments/0002-stm-cancer-dementia.md'))
validate_frontmatter(fm)
print('frontmatter OK:', fm['model_class'], fm['cohort'], fm['covariate_formula'])
"
```
Expected: `frontmatter OK: stm cancer_or_dementia ~ C(source_cohort) + C(sex) + age` (no SystemExit).

- [ ] **Step 3: Verify cohort_def is accepted by the corpus loader path**

Confirm `cancer_or_dementia` is a valid `--cohort` value end-to-end: it must be in `SUPPORTED_COHORTS` (Task 2) so `load_omop_bigquery`/`apply_cohort` accept it. Run:
```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -c "
from charmpheno.omop.cohorts import SUPPORTED_COHORTS
assert 'cancer_or_dementia' in SUPPORTED_COHORTS; print('cohort registered OK')
"
```
Expected: `cohort registered OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/0002-stm-cancer-dementia.md
git commit -m "exp(0002): STM combined cancer/dementia cohort + source covariate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Smoke test — combined cohort end-to-end with a comorbid patient

**Files:**
- Modify: `charmpheno/tests/test_stm_smoke.py`
- Test: itself.

**Interfaces:**
- Consumes: `PatientCohortDocSpec` (Task 1), `_combine_cohorts` (Task 2), `build_patient_covariate_df(key_cols=...)` (Task 3), `StreamingSTM`.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_stm_smoke.py`:

```python
@pytest.mark.slow
def test_combined_cohort_comorbid_two_documents(spark, tmp_path):
    """A comorbid patient yields two source-labeled documents that fit and
    join on the composite key; covariates are per-(person, cohort)."""
    import numpy as np
    import pandas as pd
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from charmpheno.omop.covariates import build_patient_covariate_df
    from pyspark.sql import functions as F

    # Events already tagged with source_cohort (as _combine_cohorts would).
    # person 1 is comorbid (cancer + dementia), person 2 cancer, person 3 dementia.
    rng = np.random.default_rng(0)
    rows = []
    def emit(pid, cohort, codes):
        for c in codes:
            rows.append((pid, cohort, str(c)))
    emit(1, "cancer", rng.integers(0, 10, 40))
    emit(1, "dementia", rng.integers(10, 20, 40))
    emit(2, "cancer", rng.integers(0, 10, 40))
    emit(3, "dementia", rng.integers(10, 20, 40))
    events = spark.createDataFrame(rows, ["person_id", "source_cohort", "concept_id"])

    bow_df, vocab_map = to_bow_dataframe(
        events, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        token_col="concept_id",
    )
    bow_df = bow_df.withColumn(
        "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))
    # Comorbid person 1 -> two docs.
    docs = {(r["person_id"], r["source_cohort"]) for r in bow_df.collect()}
    assert (1, "cancer") in docs and (1, "dementia") in docs

    person_pdf = pd.DataFrame({
        "person_id":     [1, 1, 2, 3],
        "source_cohort": ["cancer", "dementia", "cancer", "dementia"],
        "sex":           ["M", "M", "F", "F"],
        "age":           [60.0, 60.0, 70.0, 80.0],
    })
    cov_df, _, names = build_patient_covariate_df(
        spark.createDataFrame(person_pdf),
        covariate_formula="~ C(source_cohort) + C(sex) + age",
        categorical_cols=["source_cohort", "sex"], continuous_cols=["age"],
        key_cols=["person_id", "source_cohort"],
    )
    joined = bow_df.join(cov_df, on=["person_id", "source_cohort"], how="inner")
    # Each document joins to exactly one covariate row.
    assert joined.count() == bow_df.count()
```

- [ ] **Step 2: Run to verify it passes (uses already-built components)**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_stm_smoke.py::test_combined_cohort_comorbid_two_documents -q -m ""`
Expected: PASS. (If Tasks 1/3 are incomplete it fails on import/`key_cols` — that is the red signal.)

- [ ] **Step 3: Commit**

```bash
git add charmpheno/tests/test_stm_smoke.py
git commit -m "test(charmpheno): combined-cohort comorbid two-document smoke

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
( cd charmpheno && ../.venv/bin/python -m pytest -q -m "" 2>&1 | tail -3 )
( cd spark-vi  && ../.venv/bin/python -m pytest -q -m "not slow" 2>&1 | tail -3 )
.venv/bin/python -m pytest scripts/tests/ -q 2>&1 | tail -3
```
Expected: all green; spark-vi unchanged (still passing). Then the cluster run via `make next-exp` (picks up experiment 0002) validates the full BigQuery → combined corpus → composite-join fit → eval → dashboard path.

## Self-Review

- **Spec coverage:** Component 1 (combined cohort) → Task 2; Component 2 (doc_id encodes source_cohort) → Task 1 + Task 5 decode; Component 3 (per-(person,cohort) covariates) → Tasks 3/4 + Task 5 person-table expansion; Component 4 (composite join) → Task 5; Component 5 (shared labels) → Task 5 derives labels from the corpus bow_df (single source); Component 6 (experiment + cache_uri) → Task 6. Testing section → Tasks 1,2,3,7. All covered.
- **Placeholder scan:** none (`NNNN` replaced by `0002`; all code shown).
- **Type consistency:** `key_cols` is `("person_id",)` default across Tasks 3/4/5; `source_cohort` is StringType throughout; `doc_id` format `"{source_cohort}:{person_id}"` is produced in Task 1 and decoded by `F.split(...).getItem(0)` in Task 5 and Task 7.
