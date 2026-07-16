# Case-Finding Cohort + Frontier-Label Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble, from OMOP, the labeled hierarchical case-finding corpus — one document per patient tagged with its set-valued DAG frontier (the clinical truth), the pruned label `DagLayout`, and a held-out split with the leakage strip applied — as a pure, unit-testable transformation over OMOP frames that feeds the gated-SVI MLlib shim.

**Architecture:** A new module `charmpheno/charmpheno/omop/case_finding_assembly.py` *composes* existing pieces (piece-1 `condition_dag.py`, the `apply_population_disease_cohort` cohort machinery, `to_bow_dataframe`, and the engine's frontier helpers in `spark_vi.models.topic.dag_placement`) into a `CaseFindingBundle`. The substance is the disciplined translation across three integer id spaces (concept-id → engine-id → vocab-index). The testable core `assemble_from_events` takes already-windowed events + the DAG and runs the whole pipeline on synthetic Spark frames; a thin BQ wrapper `assemble_case_finding_corpus` supplies the real loaders.

**Tech Stack:** Python 3.12, PySpark (DataFrame + SparseVector), numpy-free domain layer. Consumes `spark_vi.models.topic.dag_placement` (`DagLayout`, `frontier_from_coded`). Tests: pytest with the repo's session-scoped local Spark fixture (`charmpheno/tests/conftest.py`).

## Global Constraints

- **Branch:** `case-finding`. Do NOT merge to `main` (highly experimental). Verify the remote after committing — this branch did not auto-push in the prior session; push explicitly (user pre-authorized).
- **Commit trailer, EXACT:** every commit ends with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **No LaTeX** anywhere (docstrings, comments, reports). Plain text + Unicode Greek (α, β, η, λ, Σ) only.
- **Cite literature** for any method/default/constant drawn from a paper, in the docstring.
- **Hash IDs in row-level log output:** any `.show()`/`print()` of doc- or patient-level rows SHA-256-truncates id columns first. Aggregates, counts, and probabilities may print raw. (This module logs no row-level ids; keep it that way.)
- **Domain layer:** this module is the concept-id domain bridge — concept-ids are *expected* here. The engine (`spark_vi`) stays integer-id agnostic; do not push concept-id knowledge into it.
- **Test honesty:** never loosen a threshold to make a test pass. If an assertion cannot hold, `xfail` with a reason string pointing at the cause; do not weaken it silently.
- **Determinism:** the train/test split uses a fixed salt constant (like `cohorts._RANDOM_WINDOW_SALT`), never `F.rand()` — resume-stable.

## Design correction (carry into every task)

The spec (`docs/superpowers/specs/2026-07-15-case-finding-assembly-design.md`) says "patient-year, one year per patient (PatientYearDocSpec)". **Use `PatientCohortDocSpec` instead**, matching every other gated population cohort (`population_eds`/`population_cancer`/`population_glp1`) and required by `pg_stm_bigquery_cloud.py:172`. Rationale: `apply_population_disease_cohort` already windows each patient to ONE 365-day span; `PatientCohortDocSpec` (doc_id = `"{source_cohort}:{person_id}"`) collapses that whole window into ONE document per patient. `PatientYearDocSpec` with era-replication would split a window that crosses a calendar-year boundary into up to two docs — reintroducing the history-length bias the user explicitly wanted gone. `PatientCohortDocSpec` delivers the spec's *stated goal* (one representative window per patient); the "patient-year" wording was imprecise.

## The three id spaces (the correctness spine)

Every integer in this module lives in exactly one space; the tests pin the translations:

1. **concept-id** — raw OMOP `concept_id`. The `ConditionDag` is built, counted, pruned, and rolled-up here.
2. **engine-id** — contiguous `0..N` from `ConditionDag.to_engine()` (anchor → 0). `DagLayout`, `frontier_from_coded`, and the emitted `frontier` column operate here.
3. **vocab-index** — `[0, V)` from `vocab_map: {concept_id: idx}`. The `features` SparseVector and the leakage strip live here.

Pipeline ordering (concept-id unless noted):
```
apply_population_disease_cohort(disease="diabetes")   -> windowed events + source_cohort
load_condition_dag(anchor=201820)                     -> before_dag (full type taxonomy)
doc_attested_nodes(events, before_dag.nodes())        -> per-doc attested_cids (array<int>)
node_patient_counts(attested)                         -> {concept-id: n_distinct_patients}
prune_by_attestation(before_dag, counts, min_n)       -> after_dag
after_dag.to_engine()                                 -> parent_int, int2cid, cid2int (ENGINE)
DagLayout(parent_int, n_bg, tpn)                       -> lay (ENGINE)
attach_frontiers: per doc, roll dropped attestations up to nearest surviving ancestor
   (concept-id, pre-prune DAG) -> map via cid2int -> frontier_from_coded(lay) -> array<engine-id>
to_bow_dataframe(events, PatientCohortDocSpec)         -> bow_df + vocab_map (VOCAB-INDEX)
join frontier onto bow_df by doc_id; split by person (salted hash)
strip {vocab_map[cid] for cid in before_dag.nodes()} from TEST features only  (VOCAB-INDEX)
```

## File Structure

- **Create:** `charmpheno/charmpheno/omop/case_finding_assembly.py` — the whole assembly module (pure helpers + Spark transforms + orchestrators + `CaseFindingBundle`).
- **Create:** `charmpheno/tests/test_case_finding_assembly.py` — all tests for the module.
- **Modify:** `charmpheno/charmpheno/omop/cohorts.py` — add ONE `"diabetes"` entry to `_DISEASE_REGISTRY`. No other edit to existing code.
- **Reference (read-only):** `charmpheno/charmpheno/omop/condition_dag.py` (piece 1: `build_condition_dag`, `prune_by_attestation`, `pruning_ledger`, `_nearest_surviving_ancestors`, `ConditionDag`), `spark_vi/spark_vi/models/topic/dag_placement.py` (`DagLayout`, `frontier_from_coded`), `charmpheno/charmpheno/omop/topic_prep.py` (`to_bow_dataframe`), `charmpheno/charmpheno/omop/doc_spec.py` (`PatientCohortDocSpec`).

Test harness (run from repo root): `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -v`

---

### Task 1: Add the "diabetes" entry to the disease registry

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py` (the `_DISEASE_REGISTRY` dict, ~line 116)
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cohorts._DISEASE_REGISTRY["diabetes"] == {"inclusion_ancestors": (201820,), "exclusion_ancestors": ()}`, enabling `apply_population_disease_cohort(disease="diabetes", ...)`.

Only the registry entry is added — NOT a `SUPPORTED_COHORTS` / `apply_cohort` / `COHORT_METADATA` entry. The assembly calls `apply_population_disease_cohort(disease="diabetes")` directly; it never routes through the CLI `--cohort` dispatch, so those surfaces stay untouched (YAGNI).

- [ ] **Step 1: Write the failing test**

```python
# charmpheno/tests/test_case_finding_assembly.py
"""Tests for charmpheno.omop.case_finding_assembly (piece 2 of the case-finding
cluster driver) + the diabetes disease-registry entry it depends on."""


def test_disease_registry_has_diabetes_anchor_201820():
    from charmpheno.omop.cohorts import _DISEASE_REGISTRY
    assert _DISEASE_REGISTRY["diabetes"] == {
        "inclusion_ancestors": (201820,),
        "exclusion_ancestors": (),
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py::test_disease_registry_has_diabetes_anchor_201820 -v`
Expected: FAIL with `KeyError: 'diabetes'`.

- [ ] **Step 3: Add the registry entry**

In `cohorts.py`, add a module-level anchor constant near `_EDS_ANCESTOR` (~line 111) with a citing comment, then the registry entry:

```python
# Top-level SNOMED concept for diabetes mellitus. concept_ancestor(201820)
# returns the diabetes TYPE/etiology/status taxonomy (T1/T2/MODY/neonatal/
# gestational/secondary x remission/control/pregnancy) — 127 standard-condition
# descendants on the AoU vocab (offline-verified 2026-07-15). Classic
# complications (nephropathy/retinopathy/neuropathy/CKD) are NOT is-a
# descendants of 201820 (they live under 442793); by design they ride along as
# learned vocabulary in each type node's topic rather than as DAG nodes. VERIFY
# ON FIRST RUN: SELECT COUNT(*) FROM concept_ancestor WHERE
# ancestor_concept_id = 201820 (expect ~hundreds; 0 means wrong id for this
# vocab version).
_DIABETES_ANCESTOR = 201820
```

Then in `_DISEASE_REGISTRY` add:

```python
    "diabetes": {
        "inclusion_ancestors": (_DIABETES_ANCESTOR,),
        "exclusion_ancestors": (),
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py::test_disease_registry_has_diabetes_anchor_201820 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): add diabetes (anchor 201820) to disease registry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Pure frontier helpers (concept-id → engine-id)

**Files:**
- Create: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: `condition_dag.ConditionDag` (`.children()`, `.parents`), `condition_dag._nearest_surviving_ancestors(dag, node, keep)`, `dag_placement.frontier_from_coded(coded_engine_ids, lay)`, `dag_placement.DagLayout`.
- Produces:
  - `_descendants(children_map: dict[int, list[int]], root: int) -> set[int]` — proper descendants of `root`.
  - `most_specific_cids(attested_cids, before_dag) -> set[int]` — attested concept-ids with no attested proper descendant (concept-id-space frontier; for the ledger).
  - `roll_up_to_survivors(attested_cids, before_dag, keep) -> set[int]` — each attested cid mapped to itself (if kept) or its nearest surviving ancestors.
  - `doc_frontier_engine_ids(attested_cids, before_dag, keep, cid2int, lay) -> list[int]` — the set-valued frontier in ENGINE-id space (sorted); empty attestation → `[]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
from charmpheno.omop.condition_dag import build_condition_dag
from spark_vi.models.topic.dag_placement import DagLayout


def _diamond_dag():
    # concept-id DAG rooted at anchor 100:
    #   100 -> 200, 300 ; 200 -> 400 ; 300 -> 400 (diamond) ; 200 -> 500
    edges = [(100, 200), (100, 300), (200, 400), (300, 400), (200, 500)]
    node_ids = [200, 300, 400, 500]
    return build_condition_dag(edges, anchor=100, node_ids=node_ids)


def test_descendants_walks_transitively():
    from charmpheno.omop.case_finding_assembly import _descendants
    dag = _diamond_dag()
    ch = dag.children()
    assert _descendants(ch, 200) == {400, 500}
    assert _descendants(ch, 400) == set()


def test_most_specific_cids_drops_attested_ancestors_keeps_incomparable():
    from charmpheno.omop.case_finding_assembly import most_specific_cids
    dag = _diamond_dag()
    # attest 200 and its descendant 400 -> only 400 is most-specific.
    assert most_specific_cids({200, 400}, dag) == {400}
    # attest incomparable 400 and 500 -> both kept.
    assert most_specific_cids({400, 500}, dag) == {400, 500}
    # single node -> itself.
    assert most_specific_cids({300}, dag) == {300}


def test_roll_up_to_survivors_reattaches_dropped_to_nearest_ancestor():
    from charmpheno.omop.case_finding_assembly import roll_up_to_survivors
    dag = _diamond_dag()
    keep = {100, 200, 300}          # 400 and 500 pruned
    # 400 (pruned) rolls up to BOTH surviving parents 200 and 300.
    assert roll_up_to_survivors({400}, dag, keep) == {200, 300}
    # a kept node stays itself; a dropped one rolls up, in one call.
    assert roll_up_to_survivors({200, 500}, dag, keep) == {200}


def test_doc_frontier_engine_ids_maps_and_reduces_to_most_specific():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    keep = dag.nodes()                        # nothing pruned
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    # attest 200 + 400: 400 is a descendant of 200 -> frontier = {engine(400)}.
    fr = doc_frontier_engine_ids({200, 400}, dag, keep, cid2int, lay)
    assert fr == [cid2int[400]]
    # empty attestation (background doc) -> [].
    assert doc_frontier_engine_ids(set(), dag, keep, cid2int, lay) == []
    # incomparable 400 + 500 -> both, in engine space, sorted.
    fr2 = doc_frontier_engine_ids({400, 500}, dag, keep, cid2int, lay)
    assert fr2 == sorted([cid2int[400], cid2int[500]])


def test_doc_frontier_engine_ids_rolls_pruned_attestation_up():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    # prune 400: a patient attesting only 400 rolls up to 200 and 300, which are
    # incomparable survivors -> frontier = {engine(200), engine(300)}.
    keep = {100, 200, 300, 500}
    from charmpheno.omop.condition_dag import prune_by_attestation
    counts = {200: 99, 300: 99, 500: 99, 400: 0}
    after = prune_by_attestation(dag, counts, min_n=1)
    parent_int, int2cid, cid2int = after.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    fr = doc_frontier_engine_ids({400}, dag, after.nodes(), cid2int, lay)
    assert fr == sorted([cid2int[200], cid2int[300]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "descendants or most_specific or roll_up or frontier_engine" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'charmpheno.omop.case_finding_assembly'`.

- [ ] **Step 3: Create the module with the pure helpers**

```python
# charmpheno/charmpheno/omop/case_finding_assembly.py
"""Assemble the labeled hierarchical case-finding corpus from OMOP.

One document per patient tagged with its set-valued DAG frontier (the clinical
truth), the pruned label DagLayout, and a held-out split with the leakage strip
applied. Piece 2 of the cluster driver (piece 1 = condition_dag.py; piece 3 =
the cloud driver). Composes the piece-1 DAG builder, the population+disease
cohort machinery (cohorts.apply_population_disease_cohort), to_bow_dataframe,
and the engine's frontier helpers (spark_vi.models.topic.dag_placement).

This is the concept-id DOMAIN bridge: concept-ids are expected here. The engine
stays integer-id agnostic. Three id spaces are threaded with care (the tests pin
the translations):

  concept-id  raw OMOP concept_id; DAG build/prune/counts/roll-up.
  engine-id   contiguous 0..N from ConditionDag.to_engine() (anchor->0);
              DagLayout, frontier_from_coded, the emitted frontier column.
  vocab-index [0,V) from vocab_map {concept_id: idx}; features SparseVector,
              leakage strip.

See docs/superpowers/specs/2026-07-15-case-finding-assembly-design.md.
"""
from __future__ import annotations

from charmpheno.omop.condition_dag import _nearest_surviving_ancestors
from spark_vi.models.topic.dag_placement import frontier_from_coded


def _descendants(children_map: dict[int, list[int]], root: int) -> set[int]:
    """Proper descendants of `root` in a {parent: [children]} concept-id map."""
    out: set[int] = set()
    stack = list(children_map.get(root, []))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(children_map.get(x, []))
    return out


def most_specific_cids(attested_cids, before_dag) -> set[int]:
    """The most-specific attested concept-ids: attested nodes with no attested
    proper descendant. Concept-id-space analogue of frontier_from_coded, used for
    the pruning ledger's coarsening accounting (which measures depths in the
    pre-prune ontology)."""
    C = set(attested_cids)
    ch = before_dag.children()
    return {c for c in C if not (_descendants(ch, c) & (C - {c}))}


def roll_up_to_survivors(attested_cids, before_dag, keep) -> set[int]:
    """Map each attested concept-id to a surviving node: itself if kept, else its
    nearest surviving ancestors (transitive walk up the PRE-PRUNE DAG). Mirrors
    prune_by_attestation's rewire (same _nearest_surviving_ancestors walk), so a
    rolled-up patient lands exactly where the pruned DAG reattaches its node."""
    surv: set[int] = set()
    for c in attested_cids:
        if c in keep:
            surv.add(c)
        else:
            surv |= _nearest_surviving_ancestors(before_dag, c, keep)
    return surv


def doc_frontier_engine_ids(attested_cids, before_dag, keep, cid2int, lay) -> list[int]:
    """The set-valued frontier in ENGINE-id space (sorted). Roll pruned
    attestations up to survivors (concept-id), map via cid2int, then
    frontier_from_coded over the pruned DagLayout (most-specific engine nodes;
    incomparable survivors kept as a set). Empty attestation (background doc) ->
    []."""
    if not attested_cids:
        return []
    survivors = roll_up_to_survivors(attested_cids, before_dag, keep)
    engine_ids = [cid2int[c] for c in survivors if c in cid2int]
    return sorted(frontier_from_coded(engine_ids, lay))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "descendants or most_specific or roll_up or frontier_engine" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): pure frontier helpers (concept-id to engine-id)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Pure leakage strip (SparseVector, vocab-index)

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: `pyspark.ml.linalg.SparseVector`.
- Produces: `strip_features(vec: SparseVector, drop_idxs: set[int]) -> SparseVector` — `vec` with the vocab dims in `drop_idxs` removed; `vec.size` preserved (only the dropped entries vanish from the sparse representation). The Spark-side numpy analogue is the engine's `strip_dag_node_codes` (token-array space); this is the SparseVector (vocab-index) version the corpus needs.

- [ ] **Step 1: Write the failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
def test_strip_features_drops_only_named_dims_preserving_size():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(10, [1, 3, 5, 7], [2.0, 1.0, 4.0, 1.0])
    out = strip_features(v, {3, 7})
    assert out.size == 10
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {1: 2.0, 5: 4.0}


def test_strip_features_empty_drop_is_identity():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(5, [0, 2], [1.0, 3.0])
    out = strip_features(v, set())
    assert out.size == 5
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {0: 1.0, 2: 3.0}


def test_strip_features_all_dropped_yields_empty_vector():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(4, [1, 2], [5.0, 6.0])
    out = strip_features(v, {1, 2})
    assert out.size == 4
    assert out.indices.tolist() == [] and out.values.tolist() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k strip_features -v`
Expected: FAIL with `ImportError: cannot import name 'strip_features'`.

- [ ] **Step 3: Implement `strip_features`**

Add to `case_finding_assembly.py`:

```python
def strip_features(vec, drop_idxs):
    """Return a SparseVector equal to `vec` with the vocab dims in `drop_idxs`
    removed (leakage strip; held-out docs only). `vec.size` is preserved so the
    vector still matches the model vocabulary; the dropped indices simply become
    zero (absent from the sparse representation). This is the case-finding test:
    a held-out patient must not read its own DAG-node type code off its features."""
    from pyspark.ml.linalg import SparseVector
    if not drop_idxs:
        return vec
    drop = {int(i) for i in drop_idxs}
    kept = [(int(i), float(v)) for i, v in zip(vec.indices, vec.values)
            if int(i) not in drop]
    if not kept:
        return SparseVector(vec.size, [], [])
    idxs, vals = zip(*kept)
    return SparseVector(vec.size, list(idxs), list(vals))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k strip_features -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): pure SparseVector leakage strip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Spark transforms — per-doc attested nodes + node patient counts

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: `doc_spec` (a `DocSpec` with `.derive_docs(events_df) -> DataFrame[..., doc_id]`), the `charmpheno/tests/conftest.py` `spark` fixture.
- Produces:
  - `doc_attested_nodes(events_df, node_cids, *, doc_spec) -> DataFrame[doc_id, person_id, source_cohort, attested_cids: array<bigint>]` — one row per doc; `attested_cids` = distinct in-window `concept_id` values that are DAG nodes (`∩ node_cids`); background docs (no node code) survive with `[]`.
  - `node_patient_counts(attested_df) -> dict[int, int]` — distinct `person_id` per attested node concept-id (patient count, NOT patient-year count).

Key correctness point: `doc_attested_nodes` must keep ALL docs (build a roster then LEFT-join the node-filtered attestations), or background docs vanish and never get a `[]` frontier.

- [ ] **Step 1: Write the failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
import datetime as dt
from charmpheno.omop.doc_spec import PatientCohortDocSpec


def _events(spark, rows):
    # rows: (person_id, concept_id, source_cohort, start_date)
    return spark.createDataFrame(
        rows,
        ["person_id", "concept_id", "source_cohort", "condition_era_start_date"],
    )


def test_doc_attested_nodes_keeps_only_dag_nodes_and_background_empty(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    node_cids = {200, 300, 400}
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),   # node
        (1, 999, "diabetes", dt.date(2015, 2, 1)),   # non-node (rides along)
        (1, 400, "diabetes", dt.date(2015, 3, 1)),   # node
        (2, 888, "general",  dt.date(2016, 1, 1)),   # background, no node code
    ])
    out = {
        r["doc_id"]: (r["person_id"], r["source_cohort"], sorted(r["attested_cids"]))
        for r in doc_attested_nodes(
            ev, node_cids, doc_spec=PatientCohortDocSpec()).collect()
    }
    assert out["diabetes:1"] == (1, "diabetes", [200, 400])
    assert out["general:2"] == (2, "general", [])       # background survives, empty


def test_doc_attested_nodes_distinct_within_doc(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (1, 200, "diabetes", dt.date(2015, 6, 1)),   # same node twice in the window
    ])
    row = doc_attested_nodes(ev, {200}, doc_spec=PatientCohortDocSpec()).collect()[0]
    assert sorted(row["attested_cids"]) == [200]


def test_node_patient_counts_counts_distinct_patients_not_docs(spark):
    from charmpheno.omop.case_finding_assembly import (
        doc_attested_nodes, node_patient_counts,
    )
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (2, 200, "diabetes", dt.date(2016, 1, 1)),   # node 200: 2 distinct patients
        (2, 300, "diabetes", dt.date(2016, 2, 1)),   # node 300: 1 patient
        (3, 999, "general",  dt.date(2017, 1, 1)),   # no node -> contributes nothing
    ])
    att = doc_attested_nodes(ev, {200, 300}, doc_spec=PatientCohortDocSpec())
    assert node_patient_counts(att) == {200: 2, 300: 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "attested or patient_counts" -v`
Expected: FAIL with `ImportError: cannot import name 'doc_attested_nodes'`.

- [ ] **Step 3: Implement the two transforms**

Add to `case_finding_assembly.py`:

```python
def doc_attested_nodes(events_df, node_cids, *, doc_spec):
    """Per document, the distinct in-window condition concept-ids that are DAG
    nodes. Derives doc_id via `doc_spec`, then LEFT-joins a full doc roster
    against the node-filtered attestations so background docs (no DAG-node code)
    survive with an empty `attested_cids` (they get a `[]` frontier downstream).

    Returns [doc_id, person_id, source_cohort, attested_cids: array<bigint>].
    person_id and source_cohort are constant within a doc_id (the cohort arms are
    disjoint by person and doc_id encodes source_cohort), so F.first is
    well-defined."""
    from pyspark.sql import functions as F

    ev = doc_spec.derive_docs(events_df)
    roster = ev.groupBy("doc_id").agg(
        F.first("person_id").alias("person_id"),
        F.first("source_cohort").alias("source_cohort"),
    )
    attested = (
        ev.where(F.col("concept_id").isin(list(node_cids)))
          .groupBy("doc_id")
          .agg(F.collect_set(F.col("concept_id").cast("long")).alias("attested_cids"))
    )
    return (
        roster.join(attested, on="doc_id", how="left")
        .withColumn(
            "attested_cids",
            F.coalesce(F.col("attested_cids"),
                       F.array().cast("array<bigint>")),
        )
    )


def node_patient_counts(attested_df) -> dict[int, int]:
    """Distinct `person_id` per attested node concept-id (patient count, the
    learnability measure the prune uses — NOT patient-year count). Collected to a
    small driver dict (one entry per DAG node)."""
    from pyspark.sql import functions as F

    exploded = attested_df.select(
        "person_id", F.explode("attested_cids").alias("node_cid"),
    ).distinct()
    rows = (
        exploded.groupBy("node_cid")
        .agg(F.countDistinct("person_id").alias("n"))
        .collect()
    )
    return {int(r["node_cid"]): int(r["n"]) for r in rows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "attested or patient_counts" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): doc_attested_nodes + node_patient_counts Spark transforms

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Spark transforms — frontier UDF, patient split, test-feature strip

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: Task-2 `doc_frontier_engine_ids`, Task-3 `strip_features`, `dag_placement.DagLayout`, `condition_dag` (`build_condition_dag`, `prune_by_attestation`).
- Produces:
  - `attach_frontiers(attested_df, before_dag, keep, cid2int, lay) -> DataFrame` — `attested_df` + a `frontier: array<bigint>` column (engine-ids), via a UDF wrapping `doc_frontier_engine_ids`. `before_dag`, `keep`, `cid2int`, `lay` are small picklable structures captured in the closure.
  - `split_train_test(df, *, holdout_frac, split_salt) -> (train_df, test_df)` — deterministic salted-hash split on `person_id`; a person's docs never straddle the split (resume-stable; `F.hash`, not `F.rand`).
  - `strip_test_features(test_df, drop_idxs, *, features_col="features") -> DataFrame` — UDF wrapping `strip_features` over the features column.
  - Module constant `_SPLIT_SALT` (a fixed int, e.g. `20260716`).

- [ ] **Step 1: Write the failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
from pyspark.ml.linalg import SparseVector


def test_attach_frontiers_emits_engine_ids_and_empty_for_background(spark):
    from charmpheno.omop.case_finding_assembly import (
        doc_attested_nodes, attach_frontiers,
    )
    from charmpheno.omop.condition_dag import build_condition_dag
    from spark_vi.models.topic.dag_placement import DagLayout
    edges = [(100, 200), (200, 400)]
    dag = build_condition_dag(edges, anchor=100, node_ids=[200, 400])
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)

    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (1, 400, "diabetes", dt.date(2015, 2, 1)),   # descendant -> frontier {400}
        (2, 777, "general",  dt.date(2016, 1, 1)),   # background -> []
    ])
    att = doc_attested_nodes(ev, dag.nodes(), doc_spec=PatientCohortDocSpec())
    out = {
        r["doc_id"]: sorted(r["frontier"])
        for r in attach_frontiers(att, dag, dag.nodes(), cid2int, lay).collect()
    }
    assert out["diabetes:1"] == [cid2int[400]]
    assert out["general:2"] == []


def test_split_train_test_is_deterministic_and_person_disjoint(spark):
    from charmpheno.omop.case_finding_assembly import split_train_test
    df = spark.createDataFrame(
        [(pid, f"diabetes:{pid}") for pid in range(200)],
        ["person_id", "doc_id"],
    )
    tr1, te1 = split_train_test(df, holdout_frac=0.25, split_salt=20260716)
    tr2, te2 = split_train_test(df, holdout_frac=0.25, split_salt=20260716)
    test_ids_1 = {r["person_id"] for r in te1.collect()}
    test_ids_2 = {r["person_id"] for r in te2.collect()}
    train_ids_1 = {r["person_id"] for r in tr1.collect()}
    assert test_ids_1 == test_ids_2                       # deterministic
    assert test_ids_1 & train_ids_1 == set()              # disjoint
    assert test_ids_1 | train_ids_1 == set(range(200))    # a partition
    assert 0.15 < len(test_ids_1) / 200 < 0.35            # roughly holdout_frac


def test_strip_test_features_removes_named_vocab_dims(spark):
    from charmpheno.omop.case_finding_assembly import strip_test_features
    df = spark.createDataFrame(
        [(1, SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0]))],
        ["person_id", "features"],
    )
    out = strip_test_features(df, {2}).collect()[0]["features"]
    assert out.size == 6
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {0: 1.0, 4: 3.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "attach_frontiers or split_train or strip_test" -v`
Expected: FAIL with `ImportError: cannot import name 'attach_frontiers'`.

- [ ] **Step 3: Implement the three transforms + salt constant**

Add to `case_finding_assembly.py` (put `_SPLIT_SALT` near the top imports):

```python
# Fixed salt for the deterministic train/test split. Hashing person_id with a
# constant salt makes the split reproducible + resume-stable across runs (Spark's
# F.rand() is not), while spreading patients pseudo-uniformly. Mirrors
# cohorts._RANDOM_WINDOW_SALT.
_SPLIT_SALT = 20260716


def attach_frontiers(attested_df, before_dag, keep, cid2int, lay):
    """Add a `frontier: array<bigint>` column (ENGINE-id space) to `attested_df`
    by applying `doc_frontier_engine_ids` per row. The DAG/keep/cid2int/lay
    structures are small and picklable; they are captured in the UDF closure and
    broadcast with the task."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, LongType

    def _fr(cids):
        return [int(x) for x in doc_frontier_engine_ids(
            [int(c) for c in (cids or [])], before_dag, keep, cid2int, lay)]

    udf = F.udf(_fr, ArrayType(LongType()))
    return attested_df.withColumn("frontier", udf(F.col("attested_cids")))


def split_train_test(df, *, holdout_frac, split_salt=_SPLIT_SALT):
    """Deterministic salted-hash split on person_id (resume-stable; F.hash, not
    F.rand). A person's docs never straddle the split — the bucket is a pure
    function of person_id + salt — so a patient-keyed holdout stays correct even
    if the doc unit ever becomes many-per-patient. Returns (train_df, test_df)."""
    from pyspark.sql import functions as F

    bucket = F.pmod(F.hash(F.col("person_id"), F.lit(split_salt)), F.lit(10000))
    thresh = int(round(holdout_frac * 10000))
    tagged = df.withColumn("_split_bucket", bucket)
    test = tagged.where(F.col("_split_bucket") < thresh).drop("_split_bucket")
    train = tagged.where(F.col("_split_bucket") >= thresh).drop("_split_bucket")
    return train, test


def strip_test_features(test_df, drop_idxs, *, features_col="features"):
    """Apply the SparseVector leakage strip to `features_col`, removing the vocab
    dims in `drop_idxs` (the DAG-node type codes). Held-out docs only — the caller
    passes only the test split here."""
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import VectorUDT

    drop = {int(i) for i in drop_idxs}

    def _strip(v):
        return strip_features(v, drop)

    udf = F.udf(_strip, VectorUDT())
    return test_df.withColumn(features_col, udf(F.col(features_col)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "attach_frontiers or split_train or strip_test" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): frontier UDF, deterministic patient split, test-feature strip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `CaseFindingBundle` + `assemble_from_events` (testable orchestration core)

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: everything above; `condition_dag.prune_by_attestation`, `condition_dag.pruning_ledger`, `topic_prep.to_bow_dataframe`, `dag_placement.DagLayout`.
- Produces:
  - `CaseFindingBundle` (dataclass): `train_df`, `test_df`, `parent_int: dict`, `int2cid: dict`, `cid2int: dict`, `vocab_map: dict[int,int]`, `name_by_id: dict[int,str]`, `ledger: dict`.
  - `assemble_from_events(events_df, before_dag, *, doc_spec, min_n, holdout_frac=0.2, split_salt=_SPLIT_SALT, vocab_size, min_df, min_patient_count, n_bg=2, tpn=1) -> CaseFindingBundle` — the whole assembly on already-windowed events (with `source_cohort`) + the pre-prune DAG. No BQ; fully synthetic-testable end to end.

`train_df`/`test_df` schema: `[person_id, doc_id, features, frontier: array<bigint>, source_cohort]` — exactly what `GatedLDAEstimator(featuresCol="features", labelCol="frontier").fit` consumes and `evaluate` scores.

Steps inside `assemble_from_events` (ordering is the id-space spine above):
1. `attested = doc_attested_nodes(events_df, before_dag.nodes(), doc_spec=doc_spec).cache()`
2. `counts = node_patient_counts(attested)`
3. `after_dag = prune_by_attestation(before_dag, counts, min_n)`; `parent_int, int2cid, cid2int = after_dag.to_engine()`; `lay = DagLayout(parent_int, n_bg, tpn)`; `keep = after_dag.nodes()`
4. Ledger with coarsening: collect the FOREGROUND attested sets only (`F.size(attested_cids) > 0`), reduce each to `most_specific_cids`, pass as `cohort_frontiers` to `pruning_ledger`. (Foreground-only: background docs have empty attestations and contribute nothing to coarsening. Collected to the driver — foreground scale; document the assumption.)
5. `fr = attach_frontiers(attested, before_dag, keep, cid2int, lay)`
6. `bow_df, vocab_map = to_bow_dataframe(events_df, doc_spec=doc_spec, vocab_size=..., min_df=..., min_patient_count=...)`
7. `labeled = bow_df.join(fr.select("doc_id","frontier","source_cohort"), on="doc_id", how="left")` then coalesce a null `frontier` to `[]` (a doc dropped by `min_doc_length` in the BOW simply won't appear; a doc kept in the BOW always has a frontier row, but coalesce defends the join).
8. `train_df, test_df = split_train_test(labeled, holdout_frac=holdout_frac, split_salt=split_salt)`
9. `drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}`; `test_df = strip_test_features(test_df, drop_idxs)`
10. `name_by_id = dict(before_dag.names)`; `attested.unpersist()`; return the bundle.

Leakage strip uses `before_dag.nodes()` (the FULL type taxonomy, pre-prune) — the conservative set: a held-out patient must not read ANY type-taxonomy code, including a pruned finer subtype that would leak more than its rolled-up label. Complication codes (not DAG nodes) stay intact by construction.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
def test_assemble_from_events_end_to_end(spark):
    """Full assembly on synthetic events + a tiny DAG: schema, frontier engine-ids,
    leakage strip on TEST only, and a DagLayout-loadable bundle."""
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from spark_vi.models.topic.dag_placement import DagLayout

    # DAG: anchor 100 -> 200 (T2), 300 (T1); 200 -> 400 (T2-with-complication node).
    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(
        edges, anchor=100, node_ids=[200, 300, 400],
        names={100: "diabetes", 200: "T2", 300: "T1", 400: "T2-renal"},
    )

    # 30 diabetes patients attest 200 (+ some 400) + a rides-along non-node 999;
    # 30 background patients attest only non-node codes. One 365-day window each,
    # collapsed to one doc by PatientCohortDocSpec.
    rows = []
    for pid in range(30):                       # diabetes / foreground
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
        if pid % 2 == 0:
            rows.append((pid, 400, "diabetes", dt.date(2015, 3, 1)))
    for pid in range(100, 130):                 # background / general
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])

    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    bundle = assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1, n_bg=2, tpn=1)

    # bundle plumbing
    assert set(bundle.parent_int) and 0 not in bundle.parent_int      # anchor has no parent
    lay = DagLayout(bundle.parent_int, n_bg=2, tpn=1)
    assert bundle.ledger["K_nodes"] == len(lay.nodes) + 0             # nodes exclude root
    assert bundle.name_by_id[200] == "T2"

    # schema the shim consumes
    for df in (bundle.train_df, bundle.test_df):
        assert set(["person_id", "doc_id", "features", "frontier",
                    "source_cohort"]) <= set(df.columns)

    # a foreground TEST doc that attested 400 has frontier == {engine(400)};
    # a background doc has frontier == [].
    test_rows = {r["doc_id"]: r for r in bundle.test_df.collect()}
    cid2int = bundle.cid2int
    fg = [r for did, r in test_rows.items() if did.startswith("diabetes:")]
    bg = [r for did, r in test_rows.items() if did.startswith("general:")]
    assert fg and bg
    for r in fg:
        assert set(r["frontier"]) in ({cid2int[200]}, {cid2int[400]})
    for r in bg:
        assert list(r["frontier"]) == []

    # leakage strip: TEST foreground features must NOT contain the node-200 vocab
    # dim, but MUST retain the rides-along non-node 999.
    vm = bundle.vocab_map
    node200_idx = vm[200]
    non_node_idx = vm[999]
    for r in fg:
        assert node200_idx not in set(r["features"].indices.tolist())
        assert non_node_idx in set(r["features"].indices.tolist())

    # train features are NOT stripped: a train foreground doc keeps node 200.
    train_fg = [r for r in bundle.train_df.collect()
                if r["doc_id"].startswith("diabetes:")]
    assert train_fg
    assert any(node200_idx in set(r["features"].indices.tolist()) for r in train_fg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py::test_assemble_from_events_end_to_end -v`
Expected: FAIL with `ImportError: cannot import name 'assemble_from_events'`.

- [ ] **Step 3: Implement `CaseFindingBundle` + `assemble_from_events`**

Add to `case_finding_assembly.py` (dataclass import at top):

```python
from dataclasses import dataclass


@dataclass
class CaseFindingBundle:
    """The assembled case-finding corpus. `train_df`/`test_df` carry
    [person_id, doc_id, features, frontier(engine-ids), source_cohort] — the exact
    shape GatedLDAEstimator(labelCol="frontier").fit consumes and dag_placement's
    evaluate scores. `parent_int`/`int2cid`/`cid2int` bridge engine <-> concept-id;
    `vocab_map` is {concept_id: vocab_idx}; `name_by_id` is {concept_id:
    concept_name} for interpretation (render_profile); `ledger` is the pruning
    receipt (kept/dropped/K_nodes + coarsening)."""
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_map: dict
    name_by_id: dict
    ledger: dict


def assemble_from_events(events_df, before_dag, *, doc_spec, min_n,
                         holdout_frac=0.2, split_salt=_SPLIT_SALT,
                         vocab_size, min_df, min_patient_count,
                         n_bg=2, tpn=1) -> CaseFindingBundle:
    """Assemble the case-finding bundle from already-windowed events (with a
    `source_cohort` column) + the pre-prune concept-id DAG. This is the testable
    core: no BigQuery, pure Spark + domain logic. See the module docstring for the
    id-space ordering.

    `cohort_frontiers` for the ledger's coarsening rate is computed from the
    FOREGROUND docs only (background attestations are empty) and collected to the
    driver — foreground scale, run once at prep time."""
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import prune_by_attestation, pruning_ledger
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from spark_vi.models.topic.dag_placement import DagLayout

    attested = doc_attested_nodes(
        events_df, before_dag.nodes(), doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(attested)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        fg_sets = [
            {int(c) for c in r["attested_cids"]}
            for r in attested.where(F.size("attested_cids") > 0)
                             .select("attested_cids").collect()
        ]
        cohort_frontiers = [most_specific_cids(s, before_dag) for s in fg_sets]
        ledger = pruning_ledger(before_dag, after_dag, counts,
                                cohort_frontiers=cohort_frontiers)

        fr = attach_frontiers(attested, before_dag, keep, cid2int, lay)

        bow_df, vocab_map = to_bow_dataframe(
            events_df, doc_spec=doc_spec, vocab_size=vocab_size,
            min_df=min_df, min_patient_count=min_patient_count)

        labeled = (
            bow_df.join(fr.select("doc_id", "frontier", "source_cohort"),
                        on="doc_id", how="left")
            .withColumn("frontier",
                        F.coalesce(F.col("frontier"),
                                   F.array().cast("array<bigint>")))
        )
        train_df, test_df = split_train_test(
            labeled, holdout_frac=holdout_frac, split_salt=split_salt)

        drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)

        return CaseFindingBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_map=vocab_map,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        attested.unpersist()
```

Note on the `finally`: `train_df`/`test_df` are lazy and reference the cached `attested` only transitively through `fr`; but `attach_frontiers`' UDF reads `attested_cids`, not the cache handle, and `fr` is re-derived from `attested` which is uncached at return. If a later reviewer finds the returned DataFrames must re-scan, that is acceptable (the bundle is materialized once by the driver); do NOT keep `attested` cached past return (leak). Confirm the end-to-end test still passes after `unpersist` — it collects the DataFrames, forcing recomputation.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py::test_assemble_from_events_end_to_end -v`
Expected: PASS. If `attested.unpersist()` in `finally` breaks the lazy DataFrames, drop the `try/finally` and unpersist is omitted (the driver stops the session); note the choice in the commit.

- [ ] **Step 5: Run the whole module suite**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -v`
Expected: all green (Tasks 1-6).

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): CaseFindingBundle + assemble_from_events core

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: BQ loaders — `_condition_dag_from_frames`, `load_condition_dag`, `assemble_case_finding_corpus`

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py`
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Consumes: `condition_dag.build_condition_dag`; `charmpheno.omop.load_omop_bigquery`; `cohorts.apply_population_disease_cohort`; `doc_spec.PatientCohortDocSpec`.
- Produces:
  - `_condition_dag_from_frames(concept_df, ca_df, anchor) -> ConditionDag` — pure-ish (Spark, no BQ): `node_ids` = standard-condition descendants of `anchor`; `edges` = min-sep-1 `concept_ancestor` pairs among the nodes (pushed down to ~DAG size, not the full table); names from `concept`. Synthetic-frame testable.
  - `load_condition_dag(spark, *, anchor, cdr, billing) -> ConditionDag` — the BQ wrapper (reads `concept` + `concept_ancestor`).
  - `assemble_case_finding_corpus(spark, *, anchor=201820, cdr, billing, source_table="condition_era", person_mod, min_n, holdout_frac=0.2, split_salt=_SPLIT_SALT, vocab_size, min_df, min_patient_count, n_bg=2, tpn=1, doc_min_length=0, prior_obs_days=365, window_days=365) -> CaseFindingBundle` — the thin BQ wrapper: load OMOP, apply the diabetes cohort, load the DAG, `assemble_from_events`.

- [ ] **Step 1: Write the failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
def test_condition_dag_from_frames_builds_taxonomy_from_omop_frames(spark):
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
    # concept: anchor 100 + standard conditions 200,300,400; 999 non-standard,
    # 555 wrong-domain -> excluded as nodes.
    concept = spark.createDataFrame(
        [
            (100, "diabetes", "S", "Condition"),
            (200, "T2",       "S", "Condition"),
            (300, "T1",       "S", "Condition"),
            (400, "T2-renal", "S", "Condition"),
            (999, "non-std",  None, "Condition"),   # not standard
            (555, "a drug",   "S", "Drug"),          # wrong domain
        ],
        ["concept_id", "concept_name", "standard_concept", "domain_id"],
    )
    # concept_ancestor: descendants of 100 + min-sep-1 edges. Include a sep-2 row
    # (100->400) that must NOT become a direct edge.
    ca = spark.createDataFrame(
        [
            (100, 200, 1), (100, 300, 1), (100, 400, 2),
            (200, 400, 1),
            (100, 999, 1), (100, 555, 1),   # candidates filtered out by concept join
        ],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
    )
    dag = _condition_dag_from_frames(concept, ca, anchor=100)
    assert dag.nodes() == {100, 200, 300, 400}
    assert dag.parents[400] == [200]            # sep-1 edge only, not 100->400
    assert set(dag.parents[200]) == {100}
    assert dag.names[200] == "T2"


def test_assemble_case_finding_corpus_importable_signature():
    import inspect
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    p = inspect.signature(assemble_case_finding_corpus).parameters
    assert {"anchor", "cdr", "billing", "person_mod", "min_n", "vocab_size",
            "holdout_frac", "n_bg", "tpn"} <= set(p)
    assert p["anchor"].default == 201820
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "from_frames or corpus_importable" -v`
Expected: FAIL with `ImportError: cannot import name '_condition_dag_from_frames'`.

- [ ] **Step 3: Implement the loaders**

Add to `case_finding_assembly.py`:

```python
def _condition_dag_from_frames(concept_df, ca_df, anchor):
    """Build the concept-id ConditionDag from `concept` + `concept_ancestor`
    frames. Nodes = standard-condition (standard_concept='S', domain_id=
    'Condition') descendants of `anchor` (+ the anchor); edges = min-sep-1
    concept_ancestor pairs among the nodes (the node membership is pushed into the
    edge scan so only ~DAG-size rows collect, not the full concept_ancestor
    table); names from `concept`. Delegates assembly to piece-1
    build_condition_dag."""
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import build_condition_dag

    desc = (
        ca_df.where(F.col("ancestor_concept_id") == anchor)
             .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    std_cond = (
        concept_df.where((F.col("standard_concept") == "S")
                         & (F.col("domain_id") == "Condition"))
                  .select("concept_id", "concept_name")
    )
    node_rows = desc.join(std_cond, on="concept_id", how="inner").collect()
    node_ids = [int(r["concept_id"]) for r in node_rows]
    names = {int(r["concept_id"]): r["concept_name"] for r in node_rows}
    nodeset = list(set(node_ids) | {anchor})

    edges = [
        (int(r["ancestor_concept_id"]), int(r["descendant_concept_id"]))
        for r in ca_df.where(F.col("min_levels_of_separation") == 1)
                      .where(F.col("ancestor_concept_id").isin(nodeset))
                      .where(F.col("descendant_concept_id").isin(nodeset))
                      .select("ancestor_concept_id", "descendant_concept_id")
                      .collect()
    ]
    anchor_row = (concept_df.where(F.col("concept_id") == anchor)
                  .select("concept_name").head(1))
    if anchor_row:
        names[anchor] = anchor_row[0][0]
    return build_condition_dag(edges, anchor, node_ids, names)


def load_condition_dag(spark, *, anchor, cdr, billing):
    """Read `concept` + `concept_ancestor` from BigQuery and build the anchor's
    condition DAG (concept-id space). BQ wrapper around
    _condition_dag_from_frames."""
    def _read(table):
        return (spark.read.format("bigquery")
                .option("table", f"{cdr}.{table}")
                .option("parentProject", billing).load())

    concept = _read("concept").select(
        "concept_id", "concept_name", "standard_concept", "domain_id")
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation")
    return _condition_dag_from_frames(concept, ca, anchor)


def assemble_case_finding_corpus(spark, *, anchor=201820, cdr, billing,
                                 source_table="condition_era", person_mod,
                                 min_n, holdout_frac=0.2, split_salt=_SPLIT_SALT,
                                 vocab_size, min_df, min_patient_count,
                                 n_bg=2, tpn=1, doc_min_length=0,
                                 prior_obs_days=365, window_days=365):
    """End-to-end BQ assembly: load OMOP (person_mod sample), apply the
    diabetes+background cohort (one 365-day window per patient), load the anchor
    DAG, and assemble the bundle. Thin wrapper over assemble_from_events; the
    per-doc unit is PatientCohortDocSpec (doc_id = source_cohort:person_id) so each
    patient's single window is exactly one document (see the plan's design
    correction). Requires a live CDR; unit tests cover assemble_from_events and
    _condition_dag_from_frames directly."""
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.cohorts import apply_population_disease_cohort
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    omop = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        person_sample_mod=person_mod, source_table=source_table)
    date_col = "condition_era_start_date"
    events = apply_population_disease_cohort(
        omop, disease="diabetes", window_days=window_days,
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        date_col=date_col, prior_obs_days=prior_obs_days)
    before_dag = load_condition_dag(spark, anchor=anchor, cdr=cdr, billing=billing)
    doc_spec = PatientCohortDocSpec(min_doc_length=doc_min_length)
    return assemble_from_events(
        events, before_dag, doc_spec=doc_spec, min_n=min_n,
        holdout_frac=holdout_frac, split_salt=split_salt, vocab_size=vocab_size,
        min_df=min_df, min_patient_count=min_patient_count, n_bg=n_bg, tpn=tpn)
```

`load_omop_bigquery`'s signature is confirmed (`bigquery.py:39`): keyword-only `spark, cdr_dataset, billing_project, concept_types=("condition",), person_sample_mod=None, source_table="condition_occurrence", cohort=None, prior_obs_days=None`. Call it WITHOUT `cohort` (leave the `None` default) — we apply the diabetes cohort separately via `apply_population_disease_cohort`, since "diabetes" is not a `SUPPORTED_COHORTS` name and would raise inside the loader.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "from_frames or corpus_importable" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full module suite**

Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -v`
Expected: all green (Tasks 1-7). Also run the existing cohort suite to confirm the registry edit broke nothing: `.venv/bin/python -m pytest charmpheno/tests/test_cohorts.py -q`.

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): BQ loaders + assemble_case_finding_corpus orchestrator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed against the spec)

**Spec coverage:** doc unit (Task 6/7, corrected to PatientCohortDocSpec) · cohort = diabetes+background (Task 1 registry + Task 7 orchestrator) · set-valued in-window frontier (Task 2/5) · prune counts = distinct patients (Task 4) · pruned attestation rolls up (Task 2 `roll_up_to_survivors`) · deterministic patient split (Task 5) · leakage strip at EVAL only (Task 5/6) · test set fg+bg (Task 6, `source_cohort` carried) · the three id spaces (module docstring + Task 2/5/6 translations) · ledger with coarsening (Task 6) · `CaseFindingBundle` shape matching the shim (Task 6). **Deferred to piece 3 (documented):** cache read/write (`_corpus_cache` discipline) — the assembly is deterministic in its args, so the driver keys the cache; real-data smoke wiring; the `init="random"` vs `"spectral"` A/B.

**Deviation from spec (surfaced):** `PatientCohortDocSpec`, not `PatientYearDocSpec` — see the Design correction section; it is required to meet the spec's own "one representative window per patient" goal.

**Placeholder scan:** none — every code step is complete and runnable.

**Type consistency:** `frontier` is `array<bigint>` (engine-ids) throughout (Task 5 UDF `ArrayType(LongType())`, Task 6 coalesce `array<bigint>`); `attested_cids` is `array<bigint>` (Task 4); `parent_int`/`cid2int`/`int2cid` are the exact `ConditionDag.to_engine()` triple; `strip_features`/`strip_test_features` share the same `drop_idxs` (vocab-index) type.
