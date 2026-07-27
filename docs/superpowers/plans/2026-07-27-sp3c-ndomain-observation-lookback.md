# SP3c — N-domain + observation + lookback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the two-domain gated model to N domains, add `observation` as a third OMOP data domain, give the multidomain cloud driver lookback windowing, and mint exp 0071 (rare6, condition + drug_era + observation).

**Architecture:** The engine + shim are already N-domain (dict-λ, `featuresCols` list). This plan generalizes the charmpheno assembly layer (`two_domain.py` → `multi_domain.py`, N feature columns), adds `observation` loading, adds a multi-domain lookback windower that reuses the already-domain-neutral `cohorts.lookback_feature_label_events` against one shared condition index, generalizes the driver over an ordered domain list, and migrates exp 0070 onto the same path as N=2.

**Tech Stack:** Python 3, PySpark (Spark ML), charmpheno OMOP layer, spark_vi engine, pytest.

## Global Constraints

- **Domain 0 is always conditions** — it is the DAG/frontier/gate source; domains 1…N−1 are feature-only. Verbatim from spec.
- **The gate/frontier is condition-only** (gate ⟂ domain): no drug/observation event ever defines a frontier; the label frame is condition-derived.
- **Feature columns are `features_0 … features_{N-1}`** (integer-indexed, domain 0 = conditions). The shim consumes `featuresCols=["features_0", …]`.
- **The leakage strip loops over ALL N domain vocabularies** — map DAG node-marker concept-ids through each domain's `vocab_map`, strip from that domain's column. Defensive (no-op on conforming data).
- **`observation` is a point event** — normalize to `(person_id, concept_id, observation_date)`; no era span; `date_col = observation_date`.
- **`--seed` stays required** in the driver (insight 0070: the scalable spectral init is seed-fragile).
- **Manifest per-domain vocab keyed by domain NAME** — `vocab_<name>` / `vocab_names_<name>` (e.g. `vocab_condition`), superseding SP3b's `vocab_a`/`vocab_b`.
- **charmpheno Spark tests run from the charmpheno dir**: `cd charmpheno && poetry run pytest tests/...` (repo-root `python -m pytest` breaks Python-UDF worker imports).
- **Bash `timeout` is in MILLISECONDS** (max 600000). Spark-session tests are slow; give them ≥300000.
- Reuse existing helpers verbatim; do NOT reimplement split/frontier/prune/strip/BOW/vocab or the cohort windowers.

---

## File Structure

- `charmpheno/charmpheno/omop/bigquery.py` — add `observation` source table + concept type (Task 1).
- `charmpheno/charmpheno/omop/multi_domain.py` — NEW (renamed from `two_domain.py`): N-domain BOW + bundle + assembler + lookback windower (Tasks 2, 3).
- `charmpheno/tests/test_multi_domain.py` — NEW (renamed from `test_two_domain.py`): generalized tests (Tasks 2, 3).
- `analysis/cloud/multidomain_cloud.py` — generalize driver over N domains + window modes (Task 4).
- `analysis/cloud/tests/test_multidomain_cloud.py` — driver arg-surface tests (Task 4).
- `scripts/run_experiment.py` — `build_multidomain_args` new flags (Task 5).
- `scripts/tests/test_run_experiment_multidomain.py` — wiring tests (Task 5).
- `docs/experiments/0071-multidomain-rare6-cond-drug-obs.md` — NEW; `0070-...md` migrated (Task 6).

---

### Task 1: `observation` OMOP loading

**Files:**
- Modify: `charmpheno/charmpheno/omop/bigquery.py` (`_SUPPORTED_CONCEPT_TYPES`, `_SUPPORTED_SOURCE_TABLES`, add `_observation_select_cols`, add a read branch)
- Test: `charmpheno/tests/test_bigquery_observation.py` (NEW)

**Interfaces:**
- Produces: `_observation_select_cols() -> (cols: tuple, extra: tuple)` where `cols = ("person_id", "concept_id", "observation_date")` and `extra = ("observation_date",)`. `"observation"` added to both `_SUPPORTED_CONCEPT_TYPES` and `_SUPPORTED_SOURCE_TABLES`.

- [ ] **Step 1: Write the failing test**

Create `charmpheno/tests/test_bigquery_observation.py`:

```python
def test_observation_select_cols_declares_point_event_shape():
    from charmpheno.omop.bigquery import _observation_select_cols
    cols, extra = _observation_select_cols()
    assert cols == ("person_id", "concept_id", "observation_date")
    assert extra == ("observation_date",)   # point event: a single date, no end/span


def test_observation_is_a_supported_source_table_and_concept_type():
    from charmpheno.omop import bigquery as bq
    assert "observation" in bq._SUPPORTED_SOURCE_TABLES
    assert "observation" in bq._SUPPORTED_CONCEPT_TYPES


def test_observation_rejects_cohort_filtering():
    # The existing fast-fail: cohort filtering needs a condition source_table (the
    # index date is condition-derived). observation is a feature-only domain.
    import pytest
    from charmpheno.omop.bigquery import load_omop_bigquery
    with pytest.raises(ValueError, match="cohort filtering requires a condition"):
        load_omop_bigquery(spark=None, cdr_dataset="p.d", billing_project="b",
                           source_table="observation", cohort="population_diabetes")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && poetry run pytest tests/test_bigquery_observation.py -v`
Expected: FAIL (`_observation_select_cols` undefined / `"observation"` not in the tuples).

- [ ] **Step 3: Implement**

In `bigquery.py`, extend the supported tuples:

```python
_SUPPORTED_CONCEPT_TYPES: tuple[str, ...] = ("condition", "drug", "observation")
_SUPPORTED_SOURCE_TABLES: tuple[str, ...] = (
    "condition_occurrence", "condition_era", "drug_era", "observation",
)
```

Add the pure select-cols helper next to `_drug_era_select_cols`:

```python
def _observation_select_cols():
    """Declares the observation read's output column names, normalized to a
    POINT-event shape (person_id, concept_id, observation_date) -- observation
    has a single `observation_date`, NOT a span (unlike condition_era/drug_era).
    Extracted as a pure function so the projection is unit-testable without a
    BigQuery read. The observation vocabulary is built EMPIRICALLY downstream from
    whatever concept classes the CDR populates (no rollup -- SP3c design;
    observation is the most heterogeneous OMOP domain)."""
    cols = ("person_id", "concept_id", "observation_date")
    extra = ("observation_date",)
    return cols, extra
```

In `load_omop_bigquery`, add the read branch. Find the `else:  # drug_era` block and change it to an explicit `elif`, then add the observation branch:

```python
    elif source_table == "drug_era":
        cond = _read("drug_era").select(
            "person_id",
            F.col("drug_concept_id").alias("concept_id"),
            "drug_era_start_date",
            "drug_era_end_date",
        )
        extra_cols = ("drug_era_start_date", "drug_era_end_date")
    else:  # observation
        cond = _read("observation").select(
            "person_id",
            F.col("observation_concept_id").alias("concept_id"),
            "observation_date",
        )
        extra_cols = ("observation_date",)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd charmpheno && poetry run pytest tests/test_bigquery_observation.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/bigquery.py charmpheno/tests/test_bigquery_observation.py
git commit -m "feat(omop): add observation source table (point-event) loading"
```

---

### Task 2: N-domain assembly (`multi_domain.py`)

**Files:**
- Create: `charmpheno/charmpheno/omop/multi_domain.py` (from `two_domain.py`, generalized)
- Delete: `charmpheno/charmpheno/omop/two_domain.py`
- Create: `charmpheno/tests/test_multi_domain.py` (from `test_two_domain.py`, generalized)
- Delete: `charmpheno/tests/test_two_domain.py`

**Interfaces:**
- Consumes: `DomainVocabSpec(vocab_size, min_df, min_patient_count, vocab)` (unchanged); `to_bow_dataframe`, `assemble_from_events` helpers (`split_train_test`, `doc_attested_nodes`, `node_patient_counts`, `attach_frontiers`, `most_specific_cids`, `strip_test_features`, `_SPLIT_SALT`), `prune_by_attestation`, `pruning_ledger`, `DagLayout`.
- Produces:
  - `multidomain_bow(domain_events: list[DataFrame], vocab_specs: list[DomainVocabSpec], *, doc_spec) -> (df, vocab_maps: list[dict])`. `df` has `doc_id, person_id, features_0 … features_{N-1}`.
  - `MultiDomainBundle` dataclass: `train_df, test_df, parent_int, int2cid, cid2int, vocab_maps: list[dict], name_by_id, ledger`. Feature columns `features_0 … features_{N-1}` (domain 0 = conditions). **No `domain_names`** — the driver owns clinical names.
  - `assemble_multidomain_from_events(cond_events, extra_events: list[DataFrame], before_dag, *, doc_spec, min_n, vocab_specs: list[DomainVocabSpec], holdout_frac=0.2, split_salt=None, n_bg=2, tpn=1, strip_mode="test_only", label_events=None) -> MultiDomainBundle`. `len(vocab_specs) == 1 + len(extra_events)`.

- [ ] **Step 1: Create `multi_domain.py` with the generalized module**

Create `charmpheno/charmpheno/omop/multi_domain.py`. Keep the module docstring's MixEHR framing (from `two_domain.py`), then:

```python
from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class DomainVocabSpec:
    """Per-domain vocabulary controls. Independent per domain because domains have
    very different natural sizes (SP3b/SP3c). `vocab` pins a pre-built vocabulary
    (concept-ids in assignment order) for eval/reproduce; None fits from data."""
    vocab_size: int | None
    min_df: int | float = 1
    min_patient_count: int = 1
    vocab: list[int] | None = None


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def multidomain_bow(domain_events, vocab_specs, *, doc_spec):
    """N aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_0 .. features_{N-1}], vocab_maps).

    domain_events[i] is bag-of-worded against vocab_specs[i]; domain 0 is
    conditions by convention. A doc present in only some domains gets an EMPTY
    vector (of the absent domain's vocab size) on each absent side -- never a
    dropped row -- so every doc carries all N columns and each per-domain vector
    size is CONSTANT across the corpus (SP3a's shim derives domainBounds from the
    first row and validates every row against it).
    """
    from charmpheno.omop.topic_prep import to_bow_dataframe

    if len(domain_events) != len(vocab_specs):
        raise ValueError(
            f"domain_events ({len(domain_events)}) and vocab_specs "
            f"({len(vocab_specs)}) must have the same length")

    bows, vms = [], []
    for ev, spec in zip(domain_events, vocab_specs):
        bow, vm = to_bow_dataframe(
            ev, doc_spec=doc_spec, token_col="concept_id",
            vocab_size=spec.vocab_size, min_df=spec.min_df,
            min_patient_count=spec.min_patient_count, vocab=spec.vocab)
        bows.append(bow)
        vms.append(vm)

    joined = bows[0].select(
        "doc_id", "person_id", F.col("features").alias("features_0"))
    for i in range(1, len(bows)):
        side = bows[i].select(
            "doc_id",
            F.col("features").alias(f"features_{i}"),
            F.col("person_id").alias(f"person_id_{i}"))
        joined = (joined.join(side, on="doc_id", how="full_outer")
                  .withColumn("person_id",
                              F.coalesce(F.col("person_id"), F.col(f"person_id_{i}")))
                  .drop(f"person_id_{i}"))

    for i, vm in enumerate(vms):
        col = f"features_{i}"
        joined = joined.withColumn(
            col, F.coalesce(F.col(col), _empty_vec_udf(len(vm))()))

    feat_cols = [f"features_{i}" for i in range(len(vms))]
    return joined.select("doc_id", "person_id", *feat_cols), vms


@dataclass
class MultiDomainBundle:
    """The assembled N-domain case-finding corpus. `train_df`/`test_df` carry
    feature columns features_0 .. features_{N-1} (domain 0 = conditions) plus
    `frontier` (engine-ids) and `source_cohort`. `vocab_maps` is the list of
    per-domain {concept_id: vocab_idx} maps in domain order.

    The frontier is CONDITION-ONLY (gate ⟂ domain): the bridge/receipt fields are
    identical to the single-domain bundle -- `parent_int`/`int2cid`/`cid2int`
    bridge engine <-> concept-id; `name_by_id` is {concept_id: concept_name};
    `ledger` is the pruning receipt. Clinical domain names live in the DRIVER, not
    here -- this bundle is index-based and domain-agnostic."""
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_maps: list
    name_by_id: dict
    ledger: dict


def _frozen_vocab_spec(spec: DomainVocabSpec, vocab_map: dict) -> DomainVocabSpec:
    """A DomainVocabSpec that pins `vocab_map` as a frozen concept-id list (index
    order), so a TEST BOW is bag-of-worded against the TRAIN-fit vocabulary. The
    fit knobs reset to defaults because to_bow_dataframe rejects a frozen `vocab`
    combined with non-default vocab_size/min_df/min_patient_count."""
    from dataclasses import replace

    vocab_list = [None] * len(vocab_map)
    for cid, idx in vocab_map.items():
        vocab_list[idx] = cid
    return replace(spec, vocab_size=None, min_df=1, min_patient_count=1,
                   vocab=vocab_list)


def assemble_multidomain_from_events(cond_events, extra_events, before_dag, *,
                                     doc_spec, min_n, vocab_specs,
                                     holdout_frac=0.2, split_salt=None, n_bg=2,
                                     tpn=1, strip_mode="test_only",
                                     label_events=None) -> MultiDomainBundle:
    """Assemble the N-domain case-finding bundle from already-windowed events.

    A thin N-domain layer over the single-domain `assemble_from_events`
    orchestration: it reuses that module's split, frontier, prune, ledger, and
    strip helpers verbatim, swaps the single BOW for the N-column `multidomain_bow`,
    and applies the leakage strip PER DOMAIN over all N vocabularies.

    Domain 0 is conditions (`cond_events`); domains 1..N-1 are `extra_events` in
    order. `vocab_specs` has one spec per domain (len == 1 + len(extra_events)).
    The frontier/label side is CONDITION-ONLY: the attestation frame is
    `cond_events`, or `label_events` when given (lookback mode). A non-condition
    event never enters a frontier.

    Split-first, leakage-free: the SAME salted split (keyed on person_id) is
    applied to every domain frame AND the label frame, so a person's rows across
    all domains and labels land on the same side. The DAG is pruned on TRAIN
    condition-node counts; each domain's vocabulary is fit on TRAIN, frozen for
    TEST. Per-domain strip: node-marker concept-ids are mapped through EACH
    domain's vocab_map and stripped from that column (defensive; a condition marker
    is expected only in vocab 0, but the strip is symmetric across domains).
    `strip_mode="test_only"` (default) strips TEST only; `"both"` also strips TRAIN.
    """
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import prune_by_attestation, pruning_ledger
    from charmpheno.omop.case_finding_assembly import (
        _SPLIT_SALT, split_train_test, doc_attested_nodes, node_patient_counts,
        attach_frontiers, most_specific_cids, strip_test_features)
    from spark_vi.models.topic.dag_placement import DagLayout

    if split_salt is None:
        split_salt = _SPLIT_SALT
    if strip_mode not in ("test_only", "both"):
        raise ValueError(
            f"strip_mode must be 'test_only' or 'both', got {strip_mode!r}")

    domain_events = [cond_events, *extra_events]
    if len(vocab_specs) != len(domain_events):
        raise ValueError(
            f"vocab_specs ({len(vocab_specs)}) must equal number of domains "
            f"({len(domain_events)} = 1 condition + {len(extra_events)} extra)")

    node_cids = before_dag.nodes()

    # 1) split PATIENTS first, keyed on person_id with the SAME salted hash, so a
    #    person's rows across all domains + labels land on the same side.
    train_doms, test_doms = [], []
    for ev in domain_events:
        tr, te = split_train_test(ev, holdout_frac=holdout_frac, split_salt=split_salt)
        train_doms.append(tr)
        test_doms.append(te)
    if label_events is None:
        train_lab, test_lab = train_doms[0], test_doms[0]
    else:
        train_lab, test_lab = split_train_test(
            label_events, holdout_frac=holdout_frac, split_salt=split_salt)

    # 2) condition-only frontier: prune the DAG on TRAIN attestation counts.
    train_att = doc_attested_nodes(train_lab, node_cids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_lab, node_cids, doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(train_att)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        # 3) ledger: TRAIN + TEST coarsening.
        def _fg_ms(att_df):
            return [most_specific_cids({int(c) for c in r["attested_cids"]}, before_dag)
                    for r in att_df.where(F.size("attested_cids") > 0)
                                   .select("attested_cids").collect()]
        train_fg = _fg_ms(train_att)
        ledger = pruning_ledger(before_dag, after_dag, counts,
                                cohort_frontiers=train_fg)
        test_fg = _fg_ms(test_att)
        test_coarsened = sum(1 for ms in test_fg if any(c not in keep for c in ms))
        ledger["test_fg_docs"] = len(test_fg)
        ledger["test_coarsening_rate"] = (
            test_coarsened / len(test_fg) if test_fg else 0.0)

        # 4) frontiers for both sides via the TRAIN-pruned DAG.
        train_fr = attach_frontiers(train_att, before_dag, keep, cid2int, lay)
        test_fr = attach_frontiers(test_att, before_dag, keep, cid2int, lay)

        # 5) N-domain BOW: fit each domain's vocab on TRAIN, freeze for TEST.
        train_bow, vms = multidomain_bow(train_doms, vocab_specs, doc_spec=doc_spec)
        test_bow, _ = multidomain_bow(
            test_doms, [_frozen_vocab_spec(s, vm) for s, vm in zip(vocab_specs, vms)],
            doc_spec=doc_spec)

        def _label(bow, fr):
            return (bow.join(fr.select("doc_id", "frontier", "source_cohort"),
                             on="doc_id", how="left")
                    .withColumn("frontier",
                                F.coalesce(F.col("frontier"),
                                           F.array().cast("array<bigint>"))))
        train_df = _label(train_bow, train_fr)
        test_df = _label(test_bow, test_fr)

        # 6) per-domain leakage strip over ALL N vocabularies (defensive).
        per_domain = [(f"features_{i}", vms[i]) for i in range(len(vms))]

        def _strip(df):
            for col, vm in per_domain:
                drop_idxs = {vm[c] for c in node_cids if c in vm}
                df = strip_test_features(df, drop_idxs, features_col=col)
            return df

        test_df = _strip(test_df)
        if strip_mode == "both":
            train_df = _strip(train_df)

        return MultiDomainBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_maps=vms,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()
```

- [ ] **Step 2: Delete the old module**

```bash
git rm charmpheno/charmpheno/omop/two_domain.py
```

- [ ] **Step 3: Create the generalized test file**

Create `charmpheno/tests/test_multi_domain.py`. Port the three `two_domain` tests to the N-domain API plus a 3-domain strip test with a marker injected into a NON-condition domain (the spec's guarantee):

```python
import datetime as dt

import pytest

pyspark = pytest.importorskip("pyspark")

from charmpheno.omop.condition_dag import build_condition_dag


def _events(spark, rows, date_col):
    from pyspark.sql import Row
    return spark.createDataFrame(
        [Row(person_id=p, concept_id=c, **{date_col: "2020-01-01"}) for p, c in rows])


def test_multidomain_bow_emits_n_aligned_per_domain_columns(spark):
    from charmpheno.omop.multi_domain import multidomain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201), (1, 202), (2, 201)], "condition_era_start_date")
    drug = _events(spark, [(1, 900), (3, 901)], "drug_era_start_date")
    obs = _events(spark, [(1, 700), (2, 701)], "observation_date")
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    df, vms = multidomain_bow([cond, drug, obs], [spec, spec, spec],
                              doc_spec=PatientDocSpec())
    assert len(vms) == 3
    rows = {r["person_id"]: r for r in df.collect()}
    for r in rows.values():                       # every doc has all 3 columns
        assert r["features_0"].size == len(vms[0])
        assert r["features_1"].size == len(vms[1])
        assert r["features_2"].size == len(vms[2])
    assert rows[3]["features_0"].numNonzeros() == 0   # person 3: no conditions
    assert rows[3]["features_1"].numNonzeros() == 1   #           has a drug
    assert rows[2]["features_1"].numNonzeros() == 0   # person 2: no drugs
    for r in rows.values():                       # ids within each domain's [0, V)
        for i in range(3):
            assert all(0 <= j < len(vms[i]) for j in r[f"features_{i}"].indices)


def test_multidomain_bow_length_mismatch_raises(spark):
    from charmpheno.omop.multi_domain import multidomain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201)], "condition_era_start_date")
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    with pytest.raises(ValueError, match="same length"):
        multidomain_bow([cond], [spec, spec], doc_spec=PatientDocSpec())


def _three_domain_bundle(spark, *, marker_in_obs=False, strip_mode="both"):
    """A CLEAN 3-domain bundle: conditions attest DAG nodes (+ a rides-along
    non-node code); drugs + observations are ordinary tokens in their own
    namespaces. When marker_in_obs, the condition node-marker id 200 is ALSO
    emitted as an OBSERVATION token, to pin the per-domain strip over a
    non-condition domain (the spec's defensive guarantee)."""
    from charmpheno.omop.multi_domain import (
        assemble_multidomain_from_events, DomainVocabSpec)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    before = build_condition_dag(
        [(100, 200), (100, 300)], anchor=100, node_ids=[200, 300],
        names={100: "root", 200: "A", 300: "B"})
    cond_rows, drug_rows, obs_rows = [], [], []
    for pid in range(20):
        node = 200 if pid % 2 == 0 else 300
        cond_rows.append((pid, node, "dz", dt.date(2015, 1, 1)))
        cond_rows.append((pid, 999, "dz", dt.date(2015, 2, 1)))     # rides-along
        drug_rows.append((pid, 900 + (pid % 3), "dz", dt.date(2015, 1, 5)))
        obs_rows.append((pid, 700 + (pid % 2), "dz", dt.date(2015, 1, 7)))
        if marker_in_obs:
            obs_rows.append((pid, 200, "dz", dt.date(2015, 1, 8)))  # marker in OBS
    for pid in range(100, 115):
        cond_rows.append((pid, 888, "bg", dt.date(2016, 1, 1)))
        drug_rows.append((pid, 950, "bg", dt.date(2016, 1, 5)))
        obs_rows.append((pid, 750, "bg", dt.date(2016, 1, 7)))
    cond = spark.createDataFrame(
        cond_rows, ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    drug = spark.createDataFrame(
        drug_rows, ["person_id", "concept_id", "source_cohort", "drug_era_start_date"])
    obs = spark.createDataFrame(
        obs_rows, ["person_id", "concept_id", "source_cohort", "observation_date"])
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    return assemble_multidomain_from_events(
        cond, [drug, obs], before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, vocab_specs=[spec, spec, spec], holdout_frac=0.3,
        strip_mode=strip_mode)


def test_assemble_multidomain_shape_and_condition_only_frontier(spark):
    bundle = _three_domain_bundle(spark)
    cols = set(bundle.train_df.columns)
    assert {"person_id", "doc_id", "features_0", "features_1", "features_2",
            "frontier", "source_cohort"} <= cols
    assert len(bundle.vocab_maps) == 3
    fr = [f for r in bundle.train_df.collect() for f in (r["frontier"] or [])]
    assert fr and max(fr) < len(bundle.parent_int) + 2   # engine node-ids only


def test_multidomain_strip_removes_marker_from_a_noncondition_domain(spark):
    """The defensive guarantee: a node-marker concept-id that (wrongly, per OMOP
    convention) lands in the OBSERVATION vocabulary is still stripped from
    features_2 -- the strip loops over all N vocabs, not just conditions."""
    bundle = _three_domain_bundle(spark, marker_in_obs=True, strip_mode="both")
    # domain 0 = conditions: marker 200 stripped from features_0
    m0 = bundle.vocab_maps[0].get(200)
    assert m0 is not None
    # domain 2 = observation: marker 200 ALSO present here (synthetic) and stripped
    m2 = bundle.vocab_maps[2].get(200)
    assert m2 is not None
    for r in bundle.train_df.collect() + bundle.test_df.collect():
        assert m0 not in set(r["features_0"].indices)
        assert m2 not in set(r["features_2"].indices)   # stripped from observation too
    assert any(r["features_2"].numNonzeros() > 0
               for r in bundle.train_df.collect())       # other obs tokens intact


def test_multidomain_bundle_fits_through_the_gated_shim_and_round_trips(spark, tmp_path):
    """The SP3c<->SP3a seam: a 3-domain bundle fits via GatedLDAEstimator with
    featuresCols=[features_0, features_1, features_2], yields a per-domain dict
    lambda over 3 domains, and the saved VIResult round-trips."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.io.export import save_result, load_result
    bundle = _three_domain_bundle(spark)
    est = GatedLDAEstimator(
        featuresCols=["features_0", "features_1", "features_2"], labelCol="frontier",
        parent=bundle.parent_int, nBg=2, tpn=1, maxIter=2, seed=0)
    model = est.fit(bundle.train_df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and set(lam) == {0, 1, 2}
    assert model.result.metadata["domains"] == [len(vm) for vm in bundle.vocab_maps]
    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert loaded.metadata["domains"] == [len(vm) for vm in bundle.vocab_maps]
```

Then delete the old test:

```bash
git rm charmpheno/tests/test_two_domain.py
```

- [ ] **Step 4: Run the tests**

Run: `cd charmpheno && poetry run pytest tests/test_multi_domain.py -v`
Expected: PASS (6 tests). If the seam test is slow, that is the Spark fit; allow up to 300000 ms.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/multi_domain.py charmpheno/tests/test_multi_domain.py
git commit -m "feat(omop): generalize two-domain assembly to N domains (multi_domain.py)"
```

---

### Task 3: Multi-domain lookback windower

**Files:**
- Modify: `charmpheno/charmpheno/omop/multi_domain.py` (add `lookback_feature_frames`)
- Modify: `charmpheno/tests/test_multi_domain.py` (add the windower test)

**Interfaces:**
- Consumes: `cohorts.lookback_feature_label_events(events_df, index_df, *, date_col, lookback_days, label_window_days) -> (feature, label)` (already domain-neutral).
- Produces: `lookback_feature_frames(domain_raws: list[DataFrame], index_df, date_cols: list[str], *, lookback_days, label_window_days) -> (feature_frames: list[DataFrame], cond_label: DataFrame)`. `domain_raws[0]` MUST be conditions; `feature_frames[i]` is domain i's pre-index window; `cond_label` is domain 0's forward window (the frontier source). Other domains' label frames are discarded (gate is condition-only).

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_multi_domain.py`:

```python
def test_lookback_feature_frames_splits_all_domains_against_one_index(spark):
    """One shared (condition-derived) index splits every domain into a pre-index
    FEATURE frame; the condition (domain 0) forward window is the LABEL frame. A
    post-index drug/observation event never enters any feature frame."""
    from charmpheno.omop.multi_domain import lookback_feature_frames
    from pyspark.sql import Row
    # index: person 1 indexed 2020-06-01.
    index_df = spark.createDataFrame(
        [Row(person_id=1, index_date=dt.date(2020, 6, 1), source_cohort="dz")])
    cond = spark.createDataFrame([
        Row(person_id=1, concept_id=201, condition_era_start_date=dt.date(2020, 1, 1)),  # pre  -> feature
        Row(person_id=1, concept_id=202, condition_era_start_date=dt.date(2020, 7, 1)),  # post -> label
    ])
    drug = spark.createDataFrame([
        Row(person_id=1, concept_id=900, drug_era_start_date=dt.date(2020, 2, 1)),       # pre  -> feature
        Row(person_id=1, concept_id=901, drug_era_start_date=dt.date(2020, 8, 1)),       # post -> dropped
    ])
    obs = spark.createDataFrame([
        Row(person_id=1, concept_id=700, observation_date=dt.date(2020, 3, 1)),          # pre  -> feature
    ])
    feats, cond_label = lookback_feature_frames(
        [cond, drug, obs], index_df,
        ["condition_era_start_date", "drug_era_start_date", "observation_date"],
        lookback_days=365, label_window_days=365)
    assert len(feats) == 3
    cond_feat_cids = {r["concept_id"] for r in feats[0].collect()}
    drug_feat_cids = {r["concept_id"] for r in feats[1].collect()}
    obs_feat_cids = {r["concept_id"] for r in feats[2].collect()}
    label_cids = {r["concept_id"] for r in cond_label.collect()}
    assert cond_feat_cids == {201}          # pre-index condition only
    assert drug_feat_cids == {900}          # pre-index drug; post-index 901 dropped
    assert obs_feat_cids == {700}
    assert label_cids == {202}              # forward-window condition only
    # every feature frame carries source_cohort (from the index join)
    assert "source_cohort" in feats[1].columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && poetry run pytest tests/test_multi_domain.py::test_lookback_feature_frames_splits_all_domains_against_one_index -v`
Expected: FAIL (`lookback_feature_frames` undefined).

- [ ] **Step 3: Implement**

Add to `multi_domain.py`:

```python
def lookback_feature_frames(domain_raws, index_df, date_cols, *,
                            lookback_days, label_window_days):
    """Split every domain's raw events against ONE shared index into pre-index
    FEATURE frames, and return the condition (domain 0) forward-window LABEL frame.

    domain_raws[0] MUST be conditions (the label/gate source). For each domain i,
    `cohorts.lookback_feature_label_events` splits domain_raws[i] on date_cols[i]:
    the pre-index [index - lookback_days, index) window is kept as that domain's
    feature frame. Only domain 0's forward [index, index + label_window_days)
    window is kept as the label frame -- the gate is condition-only, so a
    drug/observation event never defines a frontier. index_df carries
    source_cohort, which the join propagates onto every feature/label frame.
    """
    from charmpheno.omop.cohorts import lookback_feature_label_events

    if len(domain_raws) != len(date_cols):
        raise ValueError(
            f"domain_raws ({len(domain_raws)}) and date_cols ({len(date_cols)}) "
            f"must have the same length")

    feature_frames, cond_label = [], None
    for i, (raw, dc) in enumerate(zip(domain_raws, date_cols)):
        feat, lab = lookback_feature_label_events(
            raw, index_df, date_col=dc,
            lookback_days=lookback_days, label_window_days=label_window_days)
        feature_frames.append(feat)
        if i == 0:
            cond_label = lab
    return feature_frames, cond_label
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd charmpheno && poetry run pytest tests/test_multi_domain.py::test_lookback_feature_frames_splits_all_domains_against_one_index -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/multi_domain.py charmpheno/tests/test_multi_domain.py
git commit -m "feat(omop): multi-domain lookback windower (one shared condition index)"
```

---

### Task 4: Driver generalization (N domains + window modes)

**Files:**
- Modify: `analysis/cloud/multidomain_cloud.py`
- Modify: `analysis/cloud/tests/test_multidomain_cloud.py`

**Interfaces:**
- Consumes: `multi_domain.assemble_multidomain_from_events`, `multi_domain.lookback_feature_frames`, `multi_domain.DomainVocabSpec`; `cohorts.case_finding_index_table`, `cohorts.apply_population_disease_cohort`, `cohorts.disease_anchors`; the existing pure helpers `dead_node_report`, `_topic_block_labels`, `_log_topics`, `_vocab_concept_names`, `_idx_to_name`.
- Produces: `parse_args` gains `--domains`, `--window-mode`, `--lookback-days`, `--label-window-days`, `--obs-vocab-size`, `--obs-min-df`, `--obs-min-patient-count`; module-level `DOMAIN_REGISTRY` and `_domain_vocab_spec(args, source_table)`; `_window_events_to_cohort` (renamed/generalized from `_window_drug_events_to_cohort`).

- [ ] **Step 1: Write the failing arg-surface tests**

Add to `analysis/cloud/tests/test_multidomain_cloud.py`:

```python
def test_parse_args_domains_defaults_to_drug_era_and_splits_a_list():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).domains == ["drug_era"]                     # default
    a = parse_args(base + ["--domains", "drug_era,observation"])
    assert a.domains == ["drug_era", "observation"]


def test_parse_args_window_mode_and_lookback_knobs():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).window_mode == "forward"                   # default
    a = parse_args(base + ["--window-mode", "lookback",
                           "--lookback-days", "1825", "--label-window-days", "365"])
    assert a.window_mode == "lookback"
    assert a.lookback_days == 1825 and a.label_window_days == 365


def test_domain_vocab_spec_selects_the_right_arg_group():
    from multidomain_cloud import parse_args, _domain_vocab_spec
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0",
                    "--cond-vocab-size", "5000", "--drug-vocab-size", "2000",
                    "--obs-vocab-size", "1500"])
    assert _domain_vocab_spec(a, "condition_era").vocab_size == 5000
    assert _domain_vocab_spec(a, "drug_era").vocab_size == 2000
    assert _domain_vocab_spec(a, "observation").vocab_size == 1500


def test_domain_registry_maps_source_tables_to_date_cols_and_names():
    from multidomain_cloud import DOMAIN_REGISTRY
    assert DOMAIN_REGISTRY["condition_era"]["date_col"] == "condition_era_start_date"
    assert DOMAIN_REGISTRY["drug_era"]["date_col"] == "drug_era_start_date"
    assert DOMAIN_REGISTRY["observation"]["date_col"] == "observation_date"
    assert DOMAIN_REGISTRY["observation"]["name"] == "observation"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -k "domains or window_mode or domain_vocab or registry" -v`
Expected: FAIL (`--domains` unknown / `_domain_vocab_spec`, `DOMAIN_REGISTRY` undefined).

- [ ] **Step 3: Add the registry + vocab-spec selector + generalize parse_args**

At module scope in `multidomain_cloud.py` (near the top, after imports), add:

```python
# The clinical-semantics layer: which OMOP source tables can be domains, their
# per-domain event date column, and a short display/persistence name. Condition is
# always domain 0; the others are selected by --domains. The engine + assembler
# stay index-based and never see these names (SP3c design).
DOMAIN_REGISTRY = {
    "condition_era": {"date_col": "condition_era_start_date", "name": "condition",
                      "arg": "cond"},
    "drug_era":      {"date_col": "drug_era_start_date",      "name": "drug",
                      "arg": "drug"},
    "observation":   {"date_col": "observation_date",         "name": "observation",
                      "arg": "obs"},
}


def _domain_vocab_spec(args, source_table):
    """DomainVocabSpec for a source table, reading that domain's --<arg>-* controls
    (cond/drug/obs) off the parsed args."""
    from charmpheno.omop.multi_domain import DomainVocabSpec
    a = DOMAIN_REGISTRY[source_table]["arg"]
    return DomainVocabSpec(
        vocab_size=getattr(args, f"{a}_vocab_size"),
        min_df=getattr(args, f"{a}_min_df"),
        min_patient_count=getattr(args, f"{a}_min_patient_count"))
```

Note: the `from charmpheno.omop.multi_domain import DomainVocabSpec` import inside `_domain_vocab_spec` is deliberate — the module top must import cleanly on non-Spark unit-test hosts (matching the existing driver, whose charmpheno imports are all function-local).

In `parse_args`, add the new flags. After the existing `--drug-min-patient-count` line add the observation group:

```python
    p.add_argument("--obs-vocab-size", type=int, default=1500)
    p.add_argument("--obs-min-df", type=int, default=20)
    p.add_argument("--obs-min-patient-count", type=int, default=20)
```

After `--source-table-drug` add the domain selector + window-mode knobs:

```python
    p.add_argument("--domains", default="drug_era",
                   help="Comma list of EXTRA domains beyond conditions (subset of "
                        "{drug_era, observation}); condition is always domain 0. "
                        "Default 'drug_era' = the two-domain exp-0070 shape.")
    p.add_argument("--window-mode", choices=["forward", "lookback"], default="forward",
                   help="forward = one shared window (exp 0070). lookback = pre-index "
                        "feature window (all domains) + forward condition label window "
                        "(leakage-free; parity with the single-domain rare6 exps).")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--label-window-days", type=int, default=365)
```

At the end of `parse_args`, after the omega/eta parsing, normalize `--domains` to a list:

```python
    args.domains = [d for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in args.domains if d not in DOMAIN_REGISTRY or d == "condition_era"]
    if unknown:
        p.error(f"--domains entries must be extra domains in "
                f"{sorted(k for k in DOMAIN_REGISTRY if k != 'condition_era')}; "
                f"got {unknown}")
    return args
```

- [ ] **Step 4: Run the arg tests**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -k "domains or window_mode or domain_vocab or registry" -v`
Expected: PASS.

- [ ] **Step 5: Generalize `main()` over the domain list + window modes**

Rename `_window_drug_events_to_cohort` to `_window_events_to_cohort` and generalize its parameter names (`drug_df`→`dom_df`, `drug_date_col`→`dom_date_col`); the body is unchanged except those names:

```python
def _window_events_to_cohort(cond_windowed, dom_df, *,
                             cond_date_col, dom_date_col, window_days):
    """Window a secondary-domain event frame to the SAME per-patient cohort window
    as the (already-windowed) condition frame, carrying source_cohort across.
    Domain-neutral (was _window_drug_events_to_cohort; SP3c). Cluster-covered."""
    from pyspark.sql import functions as F
    bounds = (cond_windowed.groupBy("person_id", "source_cohort")
              .agg(F.min(cond_date_col).alias("_win_start")))
    return (
        dom_df.join(bounds, on="person_id", how="inner")
        .where(F.col(dom_date_col) >= F.col("_win_start"))
        .where(F.col(dom_date_col) < F.date_add(F.col("_win_start"), window_days))
        .drop("_win_start")
    )
```

Replace `main()`'s imports + load/window/assemble block. The imports (function-local) become:

```python
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.case_finding_assembly import (
        _FOREST_ROOT_CID, load_condition_dag)
    from charmpheno.omop.cohorts import (
        apply_population_disease_cohort, case_finding_index_table, disease_anchors)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    from charmpheno.omop.multi_domain import (
        assemble_multidomain_from_events, lookback_feature_frames)
    from spark_vi.io.export import save_result
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout
```

Build the ordered domain list once (condition always first), then load raws:

```python
        cond_table = args.source_table_cond
        domain_tables = [cond_table, *args.domains]     # source tables, domain order
        domain_names = [DOMAIN_REGISTRY[t]["name"] for t in domain_tables]
        date_cols = [DOMAIN_REGISTRY[t]["date_col"] for t in domain_tables]
        vocab_specs = [_domain_vocab_spec(args, t) for t in domain_tables]

        with _phase(f"load {len(domain_tables)} domains: {domain_names}"):
            raws = [load_omop_bigquery(
                        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                        person_sample_mod=args.person_mod, source_table=t)
                    for t in domain_tables]
```

Then branch on window mode to produce `(cond_events_for_dag, feature_events_list, label_arg)` and assemble:

```python
        with _phase(f"window ({args.window_mode}) + assemble"):
            cond_date_col = date_cols[0]
            if args.window_mode == "lookback":
                index_df = case_finding_index_table(
                    raws[0], disease=args.disease, spark=spark,
                    cdr_dataset=args.cdr, billing_project=args.billing,
                    date_col=cond_date_col, prior_obs_days=args.prior_obs_days,
                    label_window_days=args.label_window_days)
                feats, cond_label = lookback_feature_frames(
                    raws, index_df, date_cols,
                    lookback_days=args.lookback_days,
                    label_window_days=args.label_window_days)
                cond_feature, extra_features, label_arg = feats[0], feats[1:], cond_label
            else:  # forward
                cond_feature = apply_population_disease_cohort(
                    raws[0], disease=args.disease, window_days=args.window_days,
                    spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                    date_col=cond_date_col, prior_obs_days=args.prior_obs_days)
                extra_features = [
                    _window_events_to_cohort(
                        cond_feature, raw, cond_date_col=cond_date_col,
                        dom_date_col=dc, window_days=args.window_days)
                    for raw, dc in zip(raws[1:], date_cols[1:])]
                label_arg = None

            anchors = disease_anchors(args.disease)
            root = _FOREST_ROOT_CID if len(anchors) > 1 else None
            before_dag = load_condition_dag(
                spark, anchors=anchors, root=root, cdr=args.cdr, billing=args.billing)

            doc_spec = PatientCohortDocSpec(min_doc_length=args.doc_min_length)
            bundle = assemble_multidomain_from_events(
                cond_feature, extra_features, before_dag, doc_spec=doc_spec,
                min_n=args.min_n, vocab_specs=vocab_specs,
                holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                strip_mode=args.strip_mode, label_events=label_arg)
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)
```

Update `DagLayout`, the fit (`featuresCols`), the dead-node/topic-dump/persist to use `bundle.vocab_maps` (a list) and `domain_names`. The fit's `featuresCols`:

```python
        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        feature_cols = [f"features_{i}" for i in range(len(domain_tables))]
        ...
            est = GatedLDAEstimator(
                featuresCols=feature_cols, labelCol="frontier",
                parent=bundle.parent_int, nBg=args.n_bg, tpn=args.tpn,
                maxIter=args.max_iter, seed=args.seed,
                caviMaxIter=args.cavi_max_iter, caviTol=args.cavi_tol,
                init=args.init, spectralMaxVocab=args.spectral_max_vocab,
                spectralMethod=args.spectral_method, anchorScope=args.anchor_scope,
                spectralTopoOrder=args.spectral_topo_order,
                omega=args.omega, etaPerDomain=args.eta_per_domain)
```

Retain the existing **dead-node phase** unchanged in structure — it binds
`lam_dict = model.result.global_params["lambda"]` and
`names = {i: bundle.name_by_id.get(c, str(c)) for i, c in bundle.int2cid.items()}`
(the engine-id→name map). `dead_node_report(lam_dict, lay, ...)` is already
N-domain (it iterates `lam_dict.items()`), so it needs no change.

The per-domain vocab-name resolution + topic dump generalize from the hardcoded
a/b to a loop over `bundle.vocab_maps`, reusing `lam_dict`/`names` from the
dead-node phase:

```python
        with _phase("resolve per-domain vocab names"):
            names_bycid = [_vocab_concept_names(spark, args.cdr, args.billing, vm)
                           for vm in bundle.vocab_maps]
            idx2name = {i: _idx_to_name(vm, names_bycid[i])
                        for i, vm in enumerate(bundle.vocab_maps)}

        if args.top_n_tokens > 0:
            with _phase("final topic dump (top terms per domain)"):
                labels = _topic_block_labels(lay, names, args.n_bg)
                _log_topics(lam_dict, idx2name, labels, args.top_n_tokens,
                            domain_tags={i: n for i, n in enumerate(domain_names)})
```

The manifest's per-domain vocab persistence becomes name-keyed. **Delete** the old
`vocab_a`/`vocab_b`/`vocab_names_a`/`vocab_names_b` keys from `corpus_manifest`
(spec: superseded, no consumers) and replace with the name-keyed spreads:

```python
            **{f"vocab_{domain_names[i]}": {str(c): j for c, j in vm.items()}
               for i, vm in enumerate(bundle.vocab_maps)},
            **{f"vocab_names_{domain_names[i]}": {str(c): n for c, n in names_bycid[i].items()}
               for i in range(len(bundle.vocab_maps))},
```

Add `"domains": domain_names`, `"window_mode": args.window_mode` to the manifest's
top-level dict, and keep `dead_nodes`, `corpus_stats`, `ledger`. Update
`_log_corpus_stats` to report each domain's vocab size from `bundle.vocab_maps`
(loop over `enumerate(zip(domain_names, bundle.vocab_maps))`, not `vocab_map_a`/`_b`),
and take `domain_names` as a parameter so its printout is labeled.

- [ ] **Step 6: Run the full driver test file**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -v`
Expected: PASS (existing dead-node/topic-dump tests + the new arg tests). `main()` is cluster-covered, not run here.

- [ ] **Step 7: Commit**

```bash
git add analysis/cloud/multidomain_cloud.py analysis/cloud/tests/test_multidomain_cloud.py
git commit -m "feat(multidomain): generalize driver over N domains + lookback window mode"
```

---

### Task 5: `run_experiment` wiring

**Files:**
- Modify: `scripts/run_experiment.py` (`build_multidomain_args`)
- Modify: `scripts/tests/test_run_experiment_multidomain.py`

**Interfaces:**
- Consumes: the effective config dict (defaults ⊕ frontmatter).
- Produces: `build_multidomain_args` additionally emits `--domains`, `--window-mode`, `--lookback-days`, `--label-window-days`, and the `--obs-*` vocab trio, from the effective config.

- [ ] **Step 1: Write the failing tests**

Add to `scripts/tests/test_run_experiment_multidomain.py`:

```python
def test_build_multidomain_args_emits_domains_and_window_mode(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {**_min_eff(), "domains": "drug_era,observation",
           "window_mode": "lookback", "lookback_days": 365, "label_window_days": 365,
           "obs_vocab_size": 1500, "obs_min_df": 20, "obs_min_patient_count": 20}
    args = mod.build_multidomain_args(eff, "/out")
    assert args[args.index("--domains") + 1] == "drug_era,observation"
    assert args[args.index("--window-mode") + 1] == "lookback"
    assert args[args.index("--lookback-days") + 1] == "365"
    assert args[args.index("--label-window-days") + 1] == "365"
    assert args[args.index("--obs-vocab-size") + 1] == "1500"


def test_build_multidomain_args_defaults_domains_and_forward(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_multidomain_args(_min_eff(), "/out")
    assert args[args.index("--domains") + 1] == "drug_era"          # exp-0070 shape
    assert args[args.index("--window-mode") + 1] == "forward"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest scripts/tests/test_run_experiment_multidomain.py -k "domains_and_window or defaults_domains" -v`
Expected: FAIL (`--domains` not emitted).

- [ ] **Step 3: Implement**

In `build_multidomain_args`, after the `--top-n-tokens` line (inside the args list), add:

```python
        "--domains", str(effective.get("domains", "drug_era")),
        "--window-mode", str(effective.get("window_mode", "forward")),
        "--lookback-days", str(effective.get("lookback_days", 365)),
        "--label-window-days", str(effective.get("label_window_days", 365)),
        "--obs-vocab-size", str(effective.get("obs_vocab_size", 1500)),
        "--obs-min-df", str(effective.get("obs_min_df", 20)),
        "--obs-min-patient-count", str(effective.get("obs_min_patient_count", 20)),
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest scripts/tests/test_run_experiment_multidomain.py -v`
Expected: PASS (all, including the pre-existing multidomain wiring tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment_multidomain.py
git commit -m "feat(exp): wire --domains/--window-mode/--obs-* through build_multidomain_args"
```

---

### Task 6: Exp 0071 + migrate exp 0070

**Files:**
- Create: `docs/experiments/0071-multidomain-rare6-cond-drug-obs.md`
- Modify: `docs/experiments/0070-multidomain-diabetes-drug-condition.md` (add `domains`, `window_mode`)
- Test: `scripts/tests/test_experiment_defs_multidomain.py` (NEW)

**Interfaces:**
- Consumes: `run_experiment.read_frontmatter`, `run_experiment.validate_frontmatter`.
- Produces: two valid `model_class: multidomain` experiment definitions; 0071 is lookback 3-domain, 0070 is forward 2-domain (explicit `domains: drug_era`).

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_experiment_defs_multidomain.py`:

```python
import importlib
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent.parent / "docs" / "experiments"


def _fm(name):
    mod = importlib.import_module("run_experiment")
    return mod, mod.read_frontmatter(EXP / name)


def test_exp_0071_is_valid_lookback_three_domain_multidomain():
    mod, fm = _fm("0071-multidomain-rare6-cond-drug-obs.md")
    mod.validate_frontmatter(fm)                       # must not sys.exit
    assert fm["model_class"] == "multidomain"
    assert fm["disease"] == "rare6"
    assert fm["window_mode"] == "lookback"
    assert fm["domains"] == "drug_era,observation"


def test_exp_0070_migrated_to_explicit_forward_two_domain():
    mod, fm = _fm("0070-multidomain-diabetes-drug-condition.md")
    mod.validate_frontmatter(fm)
    assert fm["domains"] == "drug_era"
    assert fm["window_mode"] == "forward"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest scripts/tests/test_experiment_defs_multidomain.py -v`
Expected: FAIL (0071 missing; 0070 lacks `domains`/`window_mode`).

- [ ] **Step 3: Create exp 0071**

Create `docs/experiments/0071-multidomain-rare6-cond-drug-obs.md`:

```markdown
---
id: 71
slug: multidomain-rare6-cond-drug-obs
status: pending
model_class: multidomain
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
domains: drug_era,observation
window_mode: lookback
lookback_days: 365
label_window_days: 365
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 5
source_table_cond: condition_era
cond_vocab_size: 5000
cond_min_df: 20
cond_min_patient_count: 20
drug_vocab_size: 2000
drug_min_df: 20
drug_min_patient_count: 20
obs_vocab_size: 1500
obs_min_df: 20
obs_min_patient_count: 20
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
spectral_topo_order: forward
strip_mode: both
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0071 — Multi-domain (condition + drug + observation) gated fit, rare6, lookback

First **three-domain** gated case-finding fit (SP3c): conditions (domain 0,
`condition_era`) + drugs (domain 1, `drug_era`) + observations (domain 2,
`observation`), over three independent vocabularies, sharing one DAG-gated theta
with a **condition-only gate** (gate ⟂ domain). The rare6 six-disease forest is
the label DAG; drugs and observations are features, never labels.

**Lookback windowing** (parity with the single-domain rare6 exps 0061–0065): one
shared, condition-derived index date (`case_finding_index_table`) splits every
domain into a pre-index feature window (`lookback_days` back) while the frontier
labels come from the forward `label_window_days` condition window — leakage-free
by construction. `strip_mode`/`prior_obs_days` are moot in lookback (disjoint by
construction; the ≥1yr gate is intrinsic to the index table).

Runs via `make exp ID=71`. K is emergent (`n_bg` + surviving-DAG-nodes × `tpn`);
resume unsupported (v1). No NPMI eval (npz + manifest artifact).

## What to read (manifest.json + fit log)

- `dead_nodes`: MUST be empty (insight 0070 init-fragility signature; re-seed if not).
- `corpus_stats`: three per-domain vocab sizes (cond / drug / observation) in
  plausible bands; observation is the most heterogeneous domain (social history,
  surveys, findings) — expect a mixed-granularity vocabulary (drug_era finding
  precedent), tamed by `obs_min_df`/`obs_min_patient_count`.
- The final topic dump: do the rare6 disease nodes carry coherent conditions +
  corroborating drugs + observations? Does the observation domain add signal or
  erode to prior (the SP4 ω question)?
- `ledger`: the multi-domain assembly provenance.

## Knobs

- `person_mod: 1` = full population (rare diseases need the counts).
- `omega`/`eta_per_domain` unset = faithful MixEHR baseline (SP4 sweeps ω).
```

- [ ] **Step 4: Migrate exp 0070**

In `docs/experiments/0070-multidomain-diabetes-drug-condition.md`, add two lines to the frontmatter (after `disease: diabetes`), making the domain set + window mode explicit so it routes through the generalized path as N=2:

```yaml
domains: drug_era
window_mode: forward
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest scripts/tests/test_experiment_defs_multidomain.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add docs/experiments/0071-multidomain-rare6-cond-drug-obs.md docs/experiments/0070-multidomain-diabetes-drug-condition.md scripts/tests/test_experiment_defs_multidomain.py
git commit -m "feat(exp): mint exp 0071 (rare6 3-domain lookback); migrate 0070 to explicit N=2"
```

---

## Final verification (after all tasks)

- [ ] `cd charmpheno && poetry run pytest tests/test_multi_domain.py tests/test_bigquery_observation.py -v` — N-domain assembly + observation loading green.
- [ ] `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -v` — driver arg surface + pure helpers green.
- [ ] `python -m pytest scripts/tests/test_run_experiment_multidomain.py scripts/tests/test_experiment_defs_multidomain.py -v` — harness wiring + experiment defs green.
- [ ] `cd spark-vi && poetry run pytest tests/test_gated_lda_shim.py -q` — engine/shim regression unaffected.
- [ ] **Cluster smoke (user-run):** `make -C analysis/cloud exp ID=71` → `manifest.dead_nodes` empty; 3 per-domain vocab sizes reported; topic dump shows all 3 domains; artifact round-trips through SP3a's loader. And `make -C analysis/cloud exp ID=70` still fits (N=2 regression).
```
