# SP3b — Drug domain + multi-domain cloud driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the multi-domain gated model a real second domain (OMOP `drug_era`) and a way to run it: extend OMOP loading to drugs, assemble a two-domain case-finding corpus that emits SP3a's `featuresCols` shape over one shared window, and stand up a cloud driver that fits it multi-domain and writes a loadable artifact.

**Architecture:** Three layers, bottom-up. (1) `charmpheno/omop/bigquery.py` learns to read `drug_era` and normalize it to the same event shape conditions already use. (2) A new domain-agnostic two-domain BOW builder calls the EXISTING `to_bow_dataframe` once per domain (it already takes a `token_col` + a frame → one bow column + vocab map — no refactor) and joins the two feature columns per document; the case-finding assembly gains a two-domain path that produces a bundle carrying both columns and both vocab maps, with the leakage strip applied per domain. (3) A cloud driver mirrors `dag_placement_cloud.py`, loads both domains, assembles, fits via `GatedLDAEstimator(featuresCols=[...])`, surfaces a dead-node init-quality read, and writes the dict-λ artifact through SP3a's writer. Labels/gate stay condition-only (gate ⟂ domain).

**Tech Stack:** Python, PySpark (Spark SQL + MLlib shim), pytest with a local Spark fixture, BigQuery (cluster-only body), Dataproc.

**Spec:** `docs/superpowers/specs/2026-07-25-sp3b-drug-domain-and-cloud-driver-design.md`

## Global Constraints

- **This layer is `charmpheno/**` and `analysis/cloud/**` — clinical vocabulary is PERMITTED and expected here** (concept ids, drug eras, disease anchors). The opposite of the `spark_vi/**` domain-neutral rule. The `spark_vi` engine and shim are NOT modified by this plan — SP3b consumes SP3a's shipped `featuresCols` contract, it does not change it.
- **Drug domain = `drug_era`, vocabulary built EMPIRICALLY** from the `concept_id`s observed in-window. NO ingredient-class filter, NO assumed rollup (`drug_era` is not ingredient-only in practice — user decision 2026-07-25).
- **One window, both domains.** Drug events go through the SAME index-date + lookback split as conditions. **Labels and the gate stay condition-only** — a drug domain needs a vocabulary, not a DAG.
- **Per-domain vocabulary controls** (`vocab_size`, `min_df`, `min_patient_count`) are independent per domain — one shared threshold would starve the drug vocab or bloat the condition one.
- **The driver MUST set the spectral/projection seed explicitly** and never rely on the shim's `seed=(seed or 0)` default, and MUST surface a dead-node init-quality read after the fit (insight 0070: the scalable init is seed-fragile; a real corpus may expose a draw EM does not fully rescue).
- **Acceptance asserts STRUCTURE + concrete sanity reads, not a single green recovery number** (this arc's plant/metric lesson — insights 0067/0068/0070). Assembly unit tests assert shapes / id-ranges / alignment / leakage; the cluster smoke asserts no dead node topics + plausible per-domain vocab sizes + the two-column contract holds end to end.
- The BigQuery body and the Dataproc fit are **cluster-covered** (user-run); unit tests cover the pure-Spark assembly core and the drivers' arg surfaces, matching the existing convention (`assemble_from_events` is unit-tested; the BQ path's arg surface is).
- Run charmpheno tests from `/Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno` as `python -m pytest charmpheno/tests/...`. The **Bash `timeout` parameter is in MILLISECONDS, max 600000, default 120000** — local-Spark tests can take tens of seconds; pass `timeout` explicitly and never bundle so many suites that a run exceeds the cap and is auto-backgrounded.
- Commit trailer EXACTLY as the last line of every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- `charmpheno/charmpheno/omop/bigquery.py` — MODIFY. Add `"drug"` to `_SUPPORTED_CONCEPT_TYPES`, `"drug_era"` to `_SUPPORTED_SOURCE_TABLES`, and a `drug_era` read branch normalizing to `(person_id, concept_id, drug_era_start_date, drug_era_end_date)`.
- `charmpheno/charmpheno/omop/two_domain.py` — CREATE. The domain-agnostic two-domain BOW builder (`two_domain_bow`) + the two-domain bundle assembler (`assemble_two_domain_from_events`). Kept in its own module so the single-domain assembly is untouched and the two-domain logic is one focused unit.
- `charmpheno/charmpheno/omop/case_finding_assembly.py` — MODIFY MINIMALLY. Only if a helper (e.g. `strip_features`, `split_train_test`, `attach_frontiers`) needs to be imported by `two_domain.py`; do NOT change the single-domain path.
- `analysis/cloud/multidomain_cloud.py` — CREATE. The cloud driver, mirroring `dag_placement_cloud.py`.
- `analysis/cloud/Makefile` — MODIFY. Add a `multidomain-bq-smoke` target following the `lda-bq-smoke` pattern.
- Tests: `charmpheno/tests/test_bigquery.py` (MODIFY — drug arg surface), `charmpheno/tests/test_two_domain.py` (CREATE — the core), `analysis/cloud/tests/test_multidomain_cloud.py` (CREATE — arg surface + the dead-node read helper).

## Out of scope (do NOT build)

- The ω sweep / specificity green light — SP4.
- `drug_exposure` at clinical-drug level; dose/route/frequency.
- A drug DAG.
- Mid-fit checkpoint/resume (SP3a deferred it). If the planned fit sizes make Dataproc preemption likely, flag it — do not build it here.
- Any change to `spark_vi/**` (the engine/shim). If you believe one is needed, STOP and report — SP3a's contract is frozen for this plan.

---

### Task 1: Drug-domain OMOP loading (`drug_era`)

**Files:**
- Modify: `charmpheno/charmpheno/omop/bigquery.py`
- Test: `charmpheno/tests/test_bigquery.py`

**Interfaces:**
- Consumes: nothing from later tasks.
- Produces: `load_omop_bigquery(..., concept_types=("drug",), source_table="drug_era")` is accepted (no longer raises) and, on a cluster, returns a DataFrame with columns `person_id, concept_id, concept_name, drug_era_start_date, drug_era_end_date` (concept_id 0 dropped), the same normalized shape the condition path produces. Later tasks call this once per domain.

**Facts you need:** `_SUPPORTED_CONCEPT_TYPES = ("condition",)` and `_SUPPORTED_SOURCE_TABLES = ("condition_occurrence", "condition_era")` are module-level tuples near the top of `bigquery.py`. The load body validates `concept_types` and `source_table` separately, then branches on `source_table` to select+alias columns from the read. The `drug_era` OMOP table has columns `person_id`, `drug_concept_id`, `drug_era_start_date`, `drug_era_end_date`. The BQ read itself (`_read(table)`) is cluster-only — your unit tests exercise the VALIDATION and the branch SELECTION LOGIC, not a live read.

- [ ] **Step 1: Write the failing tests**

Add to `charmpheno/tests/test_bigquery.py`:

```python
def test_drug_concept_type_and_drug_era_source_are_supported():
    """drug/drug_era must pass validation (they raised NotImplementedError/ValueError
    before). We can't hit BigQuery in a unit test, so we assert the validation gate
    opens -- the read failure is a DIFFERENT, later error (no live spark.read)."""
    import pytest
    from charmpheno.omop import bigquery as bq
    assert "drug" in bq._SUPPORTED_CONCEPT_TYPES
    assert "drug_era" in bq._SUPPORTED_SOURCE_TABLES
    # A rejected concept type still raises NotImplementedError, unchanged:
    with pytest.raises(NotImplementedError, match="procedure"):
        bq.load_omop_bigquery(spark=object(), cdr_dataset="p.d", billing_project="b",
                              concept_types=("procedure",))


def test_drug_era_column_normalization_is_declared():
    """The drug_era branch must normalize to (person_id, concept_id, dates) -- the
    same event shape conditions use -- so the downstream window/doc-spec machinery
    is unchanged. We assert the branch's declared output columns via a small pure
    helper `_drug_era_select_cols` (extracted so it is testable without a read)."""
    from charmpheno.omop.bigquery import _drug_era_select_cols
    cols, extra = _drug_era_select_cols()
    # concept_id is the aliased drug_concept_id; dates carried through:
    assert "person_id" in cols and "concept_id" in cols
    assert extra == ("drug_era_start_date", "drug_era_end_date")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest charmpheno/tests/test_bigquery.py -k "drug" -v`
Expected: FAIL — `"drug"` not in the supported tuples; `_drug_era_select_cols` undefined.

- [ ] **Step 3: Implement**

In `bigquery.py`: extend the tuples, add the pure select-cols helper, and add the `drug_era` read branch.

```python
_SUPPORTED_CONCEPT_TYPES: tuple[str, ...] = ("condition", "drug")
_SUPPORTED_SOURCE_TABLES: tuple[str, ...] = (
    "condition_occurrence", "condition_era", "drug_era",
)


def _drug_era_select_cols():
    """Column projection for the drug_era read, normalized to the canonical event
    shape (person_id, concept_id, + span dates) conditions already use. Extracted
    as a pure function so the projection is unit-testable without a BigQuery read.
    `drug_era` is span-shaped like `condition_era`; the drug vocabulary is built
    EMPIRICALLY downstream from whatever concept classes the CDR populates here
    (no ingredient rollup -- SP3b design)."""
    cols = ("person_id",
            "drug_concept_id AS concept_id",
            "drug_era_start_date",
            "drug_era_end_date")
    extra = ("drug_era_start_date", "drug_era_end_date")
    return cols, extra
```

In the load body, add the `drug_era` branch alongside the condition branches (mirror the `condition_era` branch's `.select(...).alias(...)` shape, using `F.col("drug_concept_id").alias("concept_id")` and the two date columns; set `extra_cols = ("drug_era_start_date", "drug_era_end_date")`). Update the docstring's `concept_types`/`source_table`/`Returns` paragraphs to document drug support.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest charmpheno/tests/test_bigquery.py -q`
Expected: PASS (existing condition tests unchanged, new drug tests green).

- [ ] **Step 5: Mutation check (required)**

Revert `"drug"` from `_SUPPORTED_CONCEPT_TYPES` only, re-run `-k drug`, confirm `test_drug_concept_type_and_drug_era_source_are_supported` FAILS. Revert the mutation. Report the command + output; never commit the mutation.

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/bigquery.py charmpheno/tests/test_bigquery.py
git commit -m "feat(omop): load drug_era as a second domain, normalized to the condition event shape

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Domain-agnostic two-domain BOW builder

**Files:**
- Create: `charmpheno/charmpheno/omop/two_domain.py`
- Test: `charmpheno/tests/test_two_domain.py`

**Interfaces:**
- Consumes: `charmpheno.omop.topic_prep.to_bow_dataframe(df, *, doc_spec, token_col="concept_id", vocab_size, min_df, min_patient_count, vocab=None) -> (bow_df, vocab_map)` — the EXISTING, domain-agnostic BOW primitive. `bow_df` has `[doc_id, person_id, features(SparseVector), ...]`; `vocab_map` is `{concept_id: idx}`.
- Produces: `two_domain_bow(events_a, events_b, *, doc_spec, vocab_a, vocab_b) -> (df, vocab_map_a, vocab_map_b)` where each `vocab_*` is a `DomainVocabSpec(vocab_size, min_df, min_patient_count, vocab=None)`, and `df` has one row per doc_id with columns `[doc_id, person_id, features_a(SparseVector over |vocab_a|), features_b(SparseVector over |vocab_b|)]`. A doc present in one domain but not the other gets a zero vector (size = that domain's vocab) in the absent domain, NOT a dropped row. `features_a`/`features_b` are the exact per-domain columns SP3a's `GatedLDAEstimator(featuresCols=["features_a","features_b"])` consumes.

**Design notes:**
- `to_bow_dataframe` is already the right primitive — call it once per domain. Do NOT reimplement BOW/vocab fitting.
- The join is a FULL OUTER join on `doc_id` (a doc may have condition events but no drug events, or vice versa). Fill the missing side with an empty `SparseVector(V_domain, [], [])` so every doc has both columns and the per-domain vector size is CONSTANT across the corpus (SP3a's shim derives `domainBounds` from the first row and validates every row — a variable size would raise).
- `DomainVocabSpec` is a small dataclass so per-domain controls travel as one object.

- [ ] **Step 1: Write the failing tests**

Create `charmpheno/tests/test_two_domain.py`:

```python
import pytest

pyspark = pytest.importorskip("pyspark")


def _spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[2]").appName("two-domain-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


@pytest.fixture(scope="module")
def spark():
    s = _spark()
    yield s
    s.stop()


def _events(spark, rows, date_col):
    # rows: list of (person_id, concept_id). date is a constant in-window day.
    from pyspark.sql import Row
    return spark.createDataFrame(
        [Row(person_id=p, concept_id=c, **{date_col: "2020-01-01"}) for p, c in rows])


def test_two_domain_bow_emits_two_aligned_per_domain_columns(spark):
    from charmpheno.omop.two_domain import two_domain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    # person 1 has both domains; person 2 has only conditions; person 3 only drugs.
    cond = _events(spark, [(1, 201), (1, 202), (2, 201)], "condition_era_start_date")
    drug = _events(spark, [(1, 900), (3, 901)], "drug_era_start_date")
    df, va, vb = two_domain_bow(
        cond, drug, doc_spec=PatientDocSpec(),
        vocab_a=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1))
    rows = {r["person_id"]: r for r in df.collect()}
    # every doc has BOTH columns, each a SparseVector over its own vocab size:
    for r in rows.values():
        assert r["features_a"].size == len(va)
        assert r["features_b"].size == len(vb)
    # person 2 (no drugs) has an empty drug vector, not a dropped row:
    assert rows[2]["features_b"].numNonzeros() == 0
    assert 2 in rows and 3 in rows
    # person 3 (no conditions) has an empty condition vector:
    assert rows[3]["features_a"].numNonzeros() == 0
    # ids are within each domain's own [0, V) range:
    for r in rows.values():
        assert all(0 <= i < len(va) for i in r["features_a"].indices)
        assert all(0 <= i < len(vb) for i in r["features_b"].indices)


def test_two_domain_bow_vocab_sizes_are_independent(spark):
    """Per-domain vocab_size caps each domain separately."""
    from charmpheno.omop.two_domain import two_domain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201), (1, 202), (1, 203), (2, 204)],
                   "condition_era_start_date")
    drug = _events(spark, [(1, 900), (2, 901)], "drug_era_start_date")
    df, va, vb = two_domain_bow(
        cond, drug, doc_spec=PatientDocSpec(),
        vocab_a=DomainVocabSpec(vocab_size=2, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1))
    assert len(va) == 2 and len(vb) == 2   # condition capped at 2; drug has 2 tokens
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest charmpheno/tests/test_two_domain.py -q` (Bash `timeout` 300000)
Expected: FAIL — `two_domain` module / `two_domain_bow` / `DomainVocabSpec` undefined.

- [ ] **Step 3: Implement**

Create `charmpheno/charmpheno/omop/two_domain.py`:

```python
"""Two-domain (MixEHR-style) corpus assembly for the multi-domain gated model.

A document carries TWO bag-of-words feature columns -- one per domain (e.g.
conditions and drugs) -- over two INDEPENDENT vocabularies, plus a single
condition-derived frontier label (the gate is condition-only; gate is orthogonal
to domain -- arc design). This module is a thin two-domain layer over the
domain-agnostic `topic_prep.to_bow_dataframe`: it fits each domain's vocabulary
and BOW separately and joins the two feature columns per document. It does NOT
reimplement BOW/vocab fitting, and it does NOT touch the single-domain
`case_finding_assembly` path.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


@dataclass(frozen=True)
class DomainVocabSpec:
    """Per-domain vocabulary controls. Independent per domain because the two
    domains have very different natural sizes (SP3b design). `vocab` pins a
    pre-built vocabulary (concept-ids in assignment order) for eval/reproduce;
    None fits from the data."""
    vocab_size: int | None
    min_df: int | float = 1
    min_patient_count: int = 1
    vocab: list[int] | None = None


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def two_domain_bow(events_a: DataFrame, events_b: DataFrame, *, doc_spec,
                   vocab_a: DomainVocabSpec, vocab_b: DomainVocabSpec):
    """Two aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_a, features_b], vocab_map_a, vocab_map_b).

    A doc present in only one domain gets an EMPTY vector (of that domain's vocab
    size) on the absent side, never a dropped row -- so every doc carries both
    columns and each per-domain vector size is CONSTANT across the corpus (SP3a's
    shim derives domainBounds from the first row and validates every row against
    it; a variable size would raise).
    """
    from charmpheno.omop.topic_prep import to_bow_dataframe
    bow_a, vm_a = to_bow_dataframe(
        events_a, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_a.vocab_size, min_df=vocab_a.min_df,
        min_patient_count=vocab_a.min_patient_count, vocab=vocab_a.vocab)
    bow_b, vm_b = to_bow_dataframe(
        events_b, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_b.vocab_size, min_df=vocab_b.min_df,
        min_patient_count=vocab_b.min_patient_count, vocab=vocab_b.vocab)
    va, vb = len(vm_a), len(vm_b)
    a = bow_a.select("doc_id", "person_id", F.col("features").alias("features_a"))
    b = bow_b.select("doc_id", F.col("features").alias("features_b"),
                     F.col("person_id").alias("person_id_b"))
    joined = a.join(b, on="doc_id", how="fullouter")
    # coalesce person_id across the outer join; fill absent per-domain vectors.
    joined = (joined
              .withColumn("person_id",
                          F.coalesce(F.col("person_id"), F.col("person_id_b")))
              .drop("person_id_b")
              .withColumn("features_a",
                          F.coalesce(F.col("features_a"), _empty_vec_udf(va)()))
              .withColumn("features_b",
                          F.coalesce(F.col("features_b"), _empty_vec_udf(vb)())))
    return joined.select("doc_id", "person_id", "features_a", "features_b"), vm_a, vm_b
```

(Note: if `to_bow_dataframe`'s output column for the doc key is not literally `doc_id`, adjust the `.select`/join key to its real name — read `to_bow_dataframe`'s return columns first and use the actual name; do not guess.)

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m pytest charmpheno/tests/test_two_domain.py charmpheno/tests/test_topic_prep.py -q` (Bash `timeout` 300000)
Expected: PASS.

- [ ] **Step 5: Mutation check (required)**

Change the full-outer join to an `inner` join, re-run `-k emits_two_aligned`, and confirm the test FAILS (persons 2 and 3 drop out — the very rows the outer join + zero-fill exists to keep). Revert.

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/two_domain.py charmpheno/tests/test_two_domain.py
git commit -m "feat(omop): two-domain BOW builder -- two aligned per-domain feature columns

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Two-domain bundle + per-domain leakage strip

**Files:**
- Modify: `charmpheno/charmpheno/omop/two_domain.py`
- Test: `charmpheno/tests/test_two_domain.py`

**Interfaces:**
- Consumes: Task 2's `two_domain_bow`; from `case_finding_assembly`: `split_train_test(df, *, holdout_frac, split_salt)`, `strip_features(vec, drop_idxs)`, and the frontier machinery (`doc_attested_nodes`/`attach_frontiers`) — import them, do not reimplement.
- Produces: `assemble_two_domain_from_events(cond_events, drug_events, before_dag, *, doc_spec, min_n, vocab_a, vocab_b, holdout_frac=0.2, n_bg=2, tpn=1, strip_mode="test_only", label_events=None) -> TwoDomainBundle`. `TwoDomainBundle` mirrors `CaseFindingBundle` but its `train_df`/`test_df` carry `[person_id, doc_id, features_a, features_b, frontier, source_cohort]` and it holds `vocab_map_a` + `vocab_map_b`. The frontier/label side is CONDITION-ONLY (built from `cond_events`, or `label_events` in lookback mode) — unchanged from the single-domain assembly.

**Design notes:**
- The leakage strip (node-marker concept ids) applies PER DOMAIN and only to the domain whose vocabulary contains that marker — a condition marker strips from `features_a`, a drug that is a marker strips from `features_b`. Reuse `strip_features` on each column.
- The patient train/test split must land a person's condition rows and drug rows on the SAME side — apply the same `split_train_test(holdout_frac, split_salt)` keyed on `person_id` before the BOW join, exactly as the single-domain lookback path does for its two frames.
- Keep this a THIN assembler: frontier construction, DAG pruning, and the split are the single-domain helpers; only the two-column BOW + per-domain strip is new.

- [ ] **Step 1: Write the failing test**

Use the SAME fixture idiom as `charmpheno/tests/test_case_finding_assembly.py` (verified against it): `build_condition_dag(edges, anchor, node_ids, names)` for the DAG, a condition frame with columns `[person_id, concept_id, source_cohort, condition_era_start_date]`, a drug frame with `[person_id, concept_id, source_cohort, drug_era_start_date]`, and `PatientCohortDocSpec(min_doc_length=0)`.

```python
import datetime as dt

from charmpheno.omop.condition_dag import build_condition_dag


def test_assemble_two_domain_bundle_shape_and_per_domain_strip(spark):
    """The bundle exposes two aligned feature columns + a CONDITION-ONLY frontier;
    the per-domain leakage strip removes the condition node-marker ids from
    features_a and leaves features_b (drug) untouched (node markers are condition
    concept-ids -- they define the DAG -- so only the condition domain holds them)."""
    from charmpheno.omop.two_domain import (
        assemble_two_domain_from_events, DomainVocabSpec)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    # anchor 100 -> node 200, node 300 (a 2-node DAG). 200/300 ARE the node markers.
    before = build_condition_dag(
        [(100, 200), (100, 300)], anchor=100, node_ids=[200, 300],
        names={100: "root", 200: "A", 300: "B"})
    cond_rows, drug_rows = [], []
    for pid in range(20):                       # foreground: attest a node + a drug
        node = 200 if pid % 2 == 0 else 300
        cond_rows.append((pid, node, "dz", dt.date(2015, 1, 1)))
        cond_rows.append((pid, 999, "dz", dt.date(2015, 2, 1)))    # rides-along non-node
        drug_rows.append((pid, 900 + (pid % 3), "dz", dt.date(2015, 1, 5)))
    for pid in range(100, 115):                 # background
        cond_rows.append((pid, 888, "bg", dt.date(2016, 1, 1)))
        drug_rows.append((pid, 950, "bg", dt.date(2016, 1, 5)))
    cond = spark.createDataFrame(
        cond_rows, ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    drug = spark.createDataFrame(
        drug_rows, ["person_id", "concept_id", "source_cohort", "drug_era_start_date"])
    bundle = assemble_two_domain_from_events(
        cond, drug, before, doc_spec=PatientCohortDocSpec(min_doc_length=0), min_n=1,
        vocab_a=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        holdout_frac=0.3, strip_mode="both")
    cols = set(bundle.train_df.columns)
    assert {"person_id", "doc_id", "features_a", "features_b",
            "frontier", "source_cohort"} <= cols
    # frontier ids are engine node-ids from the CONDITION DAG only (2 nodes -> ids in a
    # small range); a drug never appears in a frontier:
    fr = [f for r in bundle.train_df.collect() for f in (r["frontier"] or [])]
    assert fr and max(fr) < len(bundle.parent_int) + 2
    # per-domain strip: the condition marker 200's vocab index is zeroed in features_a
    # for every doc; features_b is untouched (its drug tokens remain).
    a200 = bundle.vocab_map_a.get(200)
    assert a200 is not None
    for r in bundle.train_df.collect() + bundle.test_df.collect():
        assert a200 not in set(r["features_a"].indices)     # stripped from conditions
    assert any(r["features_b"].numNonzeros() > 0
               for r in bundle.train_df.collect())          # drugs intact
```

- [ ] **Step 2: Run to verify failure.** `assemble_two_domain_from_events` undefined.

- [ ] **Step 3: Implement** `assemble_two_domain_from_events` in `two_domain.py`: split patients once; build the condition-only frontier via the existing helpers; build the two-domain BOW via `two_domain_bow`; join frontier + both feature columns per doc; apply the per-domain strip. Return `TwoDomainBundle` (a dataclass alongside the function).

- [ ] **Step 4: Run to verify pass + regressions.** `charmpheno/tests/test_two_domain.py charmpheno/tests/test_case_finding_assembly.py` (Bash `timeout` 300000).

- [ ] **Step 5: Mutation check (required).** Apply the strip to `features_a` only (drop the per-domain loop), re-run the strip test, confirm the "drug marker strips from features_b" half FAILS. Revert.

- [ ] **Step 6: Commit** (`feat(omop): two-domain case-finding bundle with per-domain leakage strip`, trailer).

---

### Task 4: End-to-end shim-contract integration test (local, no CDR)

**Files:**
- Test only: `charmpheno/tests/test_two_domain.py`

**Interfaces:**
- Consumes: Task 3's `assemble_two_domain_from_events`; SP3a's `GatedLDAEstimator(featuresCols=[...])` and `save_result`/`load_result`.
- Produces: the proof that the assembly's output actually connects to SP3a's shim — the seam SP3b exists to build.

**Why this task:** Tasks 1-3 are charmpheno-side; SP3a is spark_vi-side. Nothing yet proves a two-domain bundle FITS through the shim and produces a per-domain dict-λ model. This is that test, and it is unit-runnable (local Spark, tiny corpus, no CDR).

- [ ] **Step 1: Write the test**

Extract the Task 3 fixture into a module-level helper `_two_domain_bundle(spark)` (the body of Task 3's test, returning the `bundle`) so both tests share it. Then:

```python
def test_two_domain_bundle_fits_through_the_gated_shim_and_round_trips(spark, tmp_path):
    """The SP3b<->SP3a seam: a two-domain bundle fits via GatedLDAEstimator with
    featuresCols=[features_a, features_b], yields a per-domain dict lambda, and the
    saved VIResult round-trips. Structural (shape + round-trip), not a recovery gate."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.io.export import save_result, load_result
    bundle = _two_domain_bundle(spark)
    est = GatedLDAEstimator(
        featuresCols=["features_a", "features_b"], labelCol="frontier",
        parent=bundle.parent_int, nBg=2, tpn=1, maxIter=2, seed=0)
    model = est.fit(bundle.train_df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and set(lam) == {0, 1}
    assert model.result.metadata["domains"] == [len(bundle.vocab_map_a),
                                                 len(bundle.vocab_map_b)]
    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert loaded.metadata["domains"] == [len(bundle.vocab_map_a),
                                          len(bundle.vocab_map_b)]
```

(This test IMPORTS spark_vi but changes nothing in it — the shim's `featuresCols` contract is frozen. `bundle.parent_int` is the engine-id parent map the assembly already produces.)

- [ ] **Step 2: Run.** It should pass once Tasks 1-3 are in (the contract is already built on both sides). Report honestly if it passes first run; if it FAILS, the seam is genuinely broken — diagnose which side, and fix in the charmpheno assembly (NOT in spark_vi).
- [ ] **Step 3: Commit** (`test(omop): two-domain bundle fits through the gated shim and round-trips`, trailer).

---

### Task 5: Multi-domain cloud driver + dead-node init-quality read

**Files:**
- Create: `analysis/cloud/multidomain_cloud.py`
- Modify: `analysis/cloud/Makefile`
- Test: `analysis/cloud/tests/test_multidomain_cloud.py`

**Interfaces:**
- Consumes: Tasks 1-3; `dag_placement_cloud.py`'s conventions (argparse, `_driver_common`, corpus/cache modules).
- Produces: a runnable driver. The BODY (BQ load + Dataproc fit) is cluster-covered; UNIT tests cover `parse_args` and a pure `dead_node_report(model, lay)` helper.

**Design notes (binding):**
- **Set the spectral seed explicitly** from a `--seed` arg into `GatedLDAEstimator(seed=...)`; never rely on `seed=(seed or 0)` (insight 0070).
- **`dead_node_report`** is a pure function: given the fitted per-domain dict λ and the layout, return the list of nodes whose max per-domain topic mass never rose off the prior (a dead-node flag). The driver logs it after the fit; the cluster smoke asserts it is empty. This is the concrete sanity read, not "it ran".
- Config surface: `--cdr --billing --person-mod --disease`, `--cond-vocab-size --cond-min-df --cond-min-patient-count`, `--drug-vocab-size --drug-min-df --drug-min-patient-count`, `--omega` (comma list, optional), `--eta-per-domain` (comma list, optional), `--seed` (required), plus the window/cohort knobs `dag_placement_cloud.py` already has.
- Writes the `VIResult` through SP3a's `save_result` (dict-λ aware).

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_args_requires_seed_and_per_domain_vocab_controls():
    from analysis.cloud.multidomain_cloud import parse_args
    import pytest
    with pytest.raises(SystemExit):            # --seed required
        parse_args(["--cdr", "p.d", "--billing", "b"])
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--seed", "7",
                    "--drug-vocab-size", "500"])
    assert a.seed == 7 and a.drug_vocab_size == 500


def test_dead_node_report_flags_a_node_stuck_at_the_prior():
    """A node whose per-domain topic never rose off the ~uniform prior is dead;
    a node with concentrated mass is not."""
    import numpy as np
    from analysis.cloud.multidomain_cloud import dead_node_report
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    V = 20
    # node 1's topic concentrated on 3 ids; node 2's flat (dead).
    lam = {0: np.full((lay.K, V), 1.0)}
    for k in lay.block[1]:
        lam[0][k] = 0.01; lam[0][k, :3] = 100.0
    dead = dead_node_report({0: lam[0]}, lay, min_peak_ratio=5.0)
    assert 2 in dead and 1 not in dead
```

- [ ] **Step 2: Run to verify failure.** module/functions undefined.
- [ ] **Step 3: Implement** `parse_args`, `dead_node_report` (pure), and a `main(argv)` that loads condition + drug via `load_omop_bigquery` (once each), assembles via `assemble_two_domain_from_events`, fits `GatedLDAEstimator(featuresCols=["features_a","features_b"], seed=args.seed, ...)`, logs `dead_node_report`, and `save_result`s the artifact. Mirror `dag_placement_cloud.py`'s structure and `_driver_common` usage. Add a `multidomain-bq-smoke` Makefile target after `lda-bq-smoke`.
- [ ] **Step 4: Run to verify pass + regressions.** `analysis/cloud/tests/test_multidomain_cloud.py` (arg-surface + dead-node unit only; the BQ/fit body is cluster-covered).
- [ ] **Step 5: Mutation check (required).** In `dead_node_report`, drop the ratio comparison (flag nothing), re-run, confirm `test_dead_node_report_flags...` FAILS. Revert.
- [ ] **Step 6: Commit** (`feat(cloud): multi-domain driver with explicit spectral seed + dead-node init-quality read`, trailer).

---

## Post-plan wrap-up (controller, after Task 5)

- [ ] Whole-branch review over the SP3b commit range on the most capable model, with attention to: the two-column contract matching SP3a exactly; the condition-only frontier being genuinely unaffected by the drug domain; and whether any assembly gate can pass by accident.
- [ ] The cluster smoke (`make multidomain-bq-smoke`) is USER-RUN on Dataproc — it is the first end-to-end multi-domain fit on real data. Report to the user what to run; do NOT claim it passed without their run.
- [ ] Add an insights entry IF the real-data run surfaces something non-obvious (drug-era concept-class heterogeneity, a dead-node flag firing, a per-domain volume ratio that pushes ω off 1). Do not pre-write it.
- [ ] Do NOT merge or push. Next: SP4 (ω-swept FDR-delta specificity green light).
```
