# Pre-diagnosis (lookback) Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document each patient by their PRE-diagnosis history (a lookback window) with the subtype/comorbid label read from a forward label window, so the model learns to detect UNDIAGNOSED cases from what precedes the diagnosis.

**Architecture:** The forward path is untouched. Lookback is a parallel path: a case-finding INDEX table (foreground first-dx + background random-event, gated by the symmetric `[index−1yr, index+label_window)` bracket), a PURE windowing helper that splits raw events into a pre-index feature frame and a forward label frame, and a two-frame `assemble_from_events` that derives features from the feature frame and the frontier from the label frame. Leakage-free by construction (no disease code exists before index).

**Tech Stack:** Python, PySpark SQL, pytest (synthetic Spark frames via the charmpheno `spark` fixture).

## Global Constraints

- Domain layer (`charmpheno/omop`) owns concept-ids/OMOP; the engine (`spark_vi`) stays id-agnostic — this plan touches only `charmpheno` + the cloud driver + configs.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- No LaTeX; Unicode Greek only. Cite literature where a method/constant warrants it.
- Branch `case-finding` does NOT auto-push; push only when the user asks.
- Exploratory research code (no prod). Structural tests only; do not gold-plate.
- **Forward mode is the default and must be byte-for-byte unchanged** (existing exps 0052–0060 keep working). Lookback is opt-in via `window_mode`.
- Reuse the existing gate helpers `_window_observed_cohort` and `_random_event_windows` (with `prior_obs_days=365, window_days=label_window_days`) — do NOT reimplement the observation bracketing.
- Test harness (repo root): `.venv/bin/python -m pytest charmpheno/tests/<f>.py -q` ; config: `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -q`.

---

### Task 1: Two-frame `assemble_from_events` (frontier from a separate label frame)

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py` (`assemble_from_events`, ~line 232)
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:**
- Produces: `assemble_from_events(events_df, before_dag, *, doc_spec, min_n, holdout_frac, split_salt=_SPLIT_SALT, vocab_size, min_df, min_patient_count, n_bg, tpn, strip_mode="test_only", label_events=None)`. When `label_events is None` behavior is unchanged (frontier + features both from `events_df`). When given, the frontier (`doc_attested_nodes`/`attach_frontiers`) is derived from `label_events`, features (`to_bow_dataframe`) from `events_df`; the patient split is applied to BOTH frames with the SAME `split_train_test(holdout_frac, split_salt)` so a person's feature and label rows land on the same side.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_case_finding_assembly.py` (reuse the module's `_events`/`spark` conventions):

```python
def test_assemble_from_events_label_events_decouples_features_from_frontier(spark):
    # Feature frame carries ONLY non-node tokens (pre-index phenotype); the label
    # frame carries the DAG node code. The frontier must come from the label frame
    # and the BOW features from the feature frame — never mixed.
    import datetime as dt
    from charmpheno.omop.case_finding_assembly import assemble_from_events, _condition_dag_from_frames
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    # Minimal DAG: root anchor 100, one child node 200 (single-disease).
    concept = spark.createDataFrame(
        [(100, "anchor", "S", "Condition"), (200, "sub", "S", "Condition")],
        ["concept_id", "concept_name", "standard_concept", "domain_id"])
    ca = spark.createDataFrame(
        [(100, 100), (100, 200), (200, 200)],
        ["ancestor_concept_id", "descendant_concept_id"])
    dag = _condition_dag_from_frames(concept, ca, anchors=100, root=None)

    def ev(rows):   # (person, concept, source_cohort, date)
        return spark.createDataFrame(
            [(p, c, s, d) for (p, c, s, d) in rows],
            ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    feats, labels = [], []
    for p in range(1, 13):
        feats += [(p, 900, "dis", dt.date(2014, 1, 1)), (p, 901, "dis", dt.date(2014, 2, 1))]
        labels += [(p, 200, "dis", dt.date(2015, 1, 1))]        # node code, post-index
    for p in range(100, 130):
        feats += [(p, 900, "general", dt.date(2014, 1, 1))]     # background feature only
    feature_events, label_events = ev(feats), ev(labels)

    doc_spec = PatientCohortDocSpec(min_doc_length=0)
    bundle = assemble_from_events(
        feature_events, dag, doc_spec=doc_spec, min_n=2, holdout_frac=0.25,
        vocab_size=50, min_df=1, min_patient_count=1, n_bg=2, tpn=1,
        label_events=label_events)
    # Node code 200 defines the DAG; it must NOT be in the feature vocab (features
    # are the 900/901 tokens only) — proving features came from the feature frame.
    assert 200 not in bundle.vocab_map
    assert 900 in bundle.vocab_map
    # Foreground docs got a non-empty frontier (from the label frame's node code).
    fr = {r["doc_id"]: r["frontier"]
          for r in bundle.train_df.select("doc_id", "frontier").collect()}
    assert any(len(v) > 0 for k, v in fr.items() if k.startswith("dis:"))
    assert all(len(v) == 0 for k, v in fr.items() if k.startswith("general:"))
```

- [ ] **Step 2: Run it — expect FAIL** (`assemble_from_events` has no `label_events` kwarg → TypeError).
Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k label_events_decouples -q`

- [ ] **Step 3: Implement** — in `assemble_from_events`, add `label_events=None` to the signature, and derive attestations from the label frame when provided. Replace the attestation derivation (the two `doc_attested_nodes(...)` calls and the split) so BOTH frames split consistently:

```python
    # Split PATIENTS first (deterministic hash). When a separate label frame is
    # given (lookback mode: features are the pre-index window, the frontier is
    # read from the forward label window), split it on the SAME person hash so a
    # person's feature and label rows stay on the same side. label_events is None
    # in forward mode -> frontier and features both come from events_df (unchanged).
    train_events, test_events = split_train_test(
        events_df, holdout_frac=holdout_frac, split_salt=split_salt)
    if label_events is None:
        train_lab, test_lab = train_events, test_events
    else:
        train_lab, test_lab = split_train_test(
            label_events, holdout_frac=holdout_frac, split_salt=split_salt)

    train_att = doc_attested_nodes(train_lab, node_cids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_lab, node_cids, doc_spec=doc_spec).cache()
```

Everything else (prune, frontiers, vocab on `train_events`, `_label`, strip) stays as-is — it already reads features from `train_events`/`test_events` and the frontier from `train_att`/`test_att`. Note in a comment that in lookback mode the strip is a no-op (node codes are absent from the feature frame by construction).

- [ ] **Step 4: Run it — expect PASS.** Also run the whole file to confirm forward mode is unchanged:
Run: `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -q`

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): assemble_from_events reads frontier from a separate label frame

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Lookback windowing helper + case-finding index table (cohorts.py)

**Files:**
- Modify: `charmpheno/charmpheno/omop/cohorts.py`
- Test: `charmpheno/tests/test_cohorts.py`

**Interfaces:**
- Produces (pure, testable): `lookback_feature_label_events(events_df, index_df, *, date_col, lookback_days, label_window_days) -> (feature_events, label_events)`. `index_df` = `(person_id, index_date, source_cohort)`. `feature_events` = rows with `index_date − lookback_days ≤ date < index_date` (carrying `source_cohort`, `index_date` dropped); `label_events` = rows with `index_date ≤ date < index_date + label_window_days`. (No explicit op_start clip needed — events only exist within observation periods, so the lookback naturally captures available history; the ≥1yr-prior GATE lives in the index table.)
- Produces (BQ): `case_finding_index_table(cond_df, *, disease, spark, cdr_dataset, billing_project, date_col, prior_obs_days=365, label_window_days=365) -> DataFrame` = `(person_id, index_date, source_cohort)`. Foreground = first-dx index over the disease's inclusion/exclusion ancestors, gated by `_window_observed_cohort(prior_obs_days, window_days=label_window_days)`; background = `_random_event_windows` over the left-anti of the foreground persons, same gate; unioned with `source_cohort` = disease / "general".

- [ ] **Step 1: Write the failing test** (pure windowing — no BQ)

Add to `charmpheno/tests/test_cohorts.py`:

```python
def test_lookback_feature_label_events_splits_pre_and_post_index(spark):
    import datetime as dt
    from charmpheno.omop.cohorts import lookback_feature_label_events
    events = spark.createDataFrame(
        [   # person, concept, date
            (1, 900, dt.date(2013, 6, 1)),   # 1.5y pre-index -> feature (within 5y, before)
            (1, 901, dt.date(2014, 6, 1)),   # 0.5y pre-index -> feature
            (1, 200, dt.date(2015, 1, 1)),   # index day -> label
            (1, 201, dt.date(2015, 6, 1)),   # 0.5y post -> label
            (1, 999, dt.date(2011, 1, 1)),   # 4y pre -> feature only if lookback>=~4y
        ],
        ["person_id", "concept_id", "condition_era_start_date"])
    index_df = spark.createDataFrame(
        [(1, dt.date(2015, 1, 1), "dis")], ["person_id", "index_date", "source_cohort"])
    feat, lab = lookback_feature_label_events(
        events, index_df, date_col="condition_era_start_date",
        lookback_days=365, label_window_days=365)
    fc = {r["concept_id"] for r in feat.collect()}
    lc = {r["concept_id"] for r in lab.collect()}
    assert fc == {901}                    # only within [index-1y, index)
    assert lc == {200, 201}               # only within [index, index+1y)
    assert "index_date" not in feat.columns and "source_cohort" in feat.columns
    # 5-year lookback pulls the older feature events too
    feat5, _ = lookback_feature_label_events(
        events, index_df, date_col="condition_era_start_date",
        lookback_days=1825, label_window_days=365)
    assert {r["concept_id"] for r in feat5.collect()} == {900, 901, 999}
```

- [ ] **Step 2: Run it — expect FAIL** (import error).
Run: `.venv/bin/python -m pytest charmpheno/tests/test_cohorts.py -k lookback_feature_label -q`

- [ ] **Step 3: Implement** in `cohorts.py`:

```python
def lookback_feature_label_events(events_df, index_df, *, date_col,
                                  lookback_days, label_window_days):
    """Split raw events into a pre-index feature frame and a forward label frame.

    `index_df` = (person_id, index_date, source_cohort). Feature frame = events in
    [index_date - lookback_days, index_date); label frame = events in
    [index_date, index_date + label_window_days). Each frame carries source_cohort
    (index_date dropped). Events only occur within observation periods, so the
    lookback naturally yields the available history (up to lookback_days); the
    >=1yr-prior observation requirement is enforced upstream in the index table.
    """
    joined = events_df.join(F.broadcast(index_df), on="person_id", how="inner")
    feature = (joined
               .where(F.col(date_col) < F.col("index_date"))
               .where(F.col(date_col) >= F.date_sub(F.col("index_date"), lookback_days))
               .drop("index_date"))
    label = (joined
             .where(F.col(date_col) >= F.col("index_date"))
             .where(F.col(date_col) < F.date_add(F.col("index_date"), label_window_days))
             .drop("index_date"))
    return feature, label


def case_finding_index_table(cond_df, *, disease, spark, cdr_dataset,
                             billing_project, date_col, prior_obs_days=365,
                             label_window_days=_WINDOW_DAYS):
    """(person_id, index_date, source_cohort) for the disease + general arms.

    Foreground: first qualifying dx (min over the disease's inclusion-minus-
    exclusion descendants), gated by _window_observed_cohort so the symmetric
    bracket [index - prior_obs_days, index + label_window_days) is observed.
    Background: _random_event_windows over everyone else, same gate. No windowing
    of events here — just the gated index per person; lookback_feature_label_events
    does the windowing. Reuses the same helpers as the forward cohorts."""
    spec = _DISEASE_REGISTRY[disease]

    def _read(table):
        return (spark.read.format("bigquery")
                .option("table", f"{cdr_dataset}.{table}")
                .option("parentProject", billing_project).load())

    ca = _read("concept_ancestor").select("ancestor_concept_id", "descendant_concept_id")
    concepts = _concept_set_from_ancestors(
        ca, inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"])
    first_dx = (cond_df.join(F.broadcast(concepts), on="concept_id", how="inner")
                .groupBy("person_id").agg(F.min(date_col).alias("index_date")))
    op = _read("observation_period").select(
        "person_id", "observation_period_start_date", "observation_period_end_date")

    fg = (_window_observed_cohort(first_dx, op, prior_obs_days=prior_obs_days,
                                  window_days=label_window_days)
          .withColumn("source_cohort", F.lit(disease)))
    non = cond_df.join(fg.select("person_id").distinct(), on="person_id", how="left_anti")
    bg = (_random_event_windows(non, op, date_col=date_col,
                                window_days=label_window_days, prior_obs_days=prior_obs_days)
          .withColumn("source_cohort", F.lit("general")))
    return fg.unionByName(bg)
```

- [ ] **Step 4: Run it — expect PASS.** Full file: `.venv/bin/python -m pytest charmpheno/tests/test_cohorts.py -q`
(The BQ `case_finding_index_table` is thin reuse of already-tested gates; it is exercised end-to-end on the cluster, not unit-tested here — note this in the report.)

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/cohorts.py charmpheno/tests/test_cohorts.py
git commit -m "feat(cohorts): lookback windowing helper + case-finding index table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `assemble_case_finding_corpus` lookback path

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py` (`assemble_case_finding_corpus`, ~line 420)
- Test: `charmpheno/tests/test_case_finding_assembly.py` (signature/params only — the BQ body is cluster-covered)

**Interfaces:**
- Consumes: `lookback_feature_label_events`, `case_finding_index_table` (Task 2); `assemble_from_events(label_events=...)` (Task 1).
- Produces: `assemble_case_finding_corpus(..., window_mode="forward", lookback_days=365, label_window_days=365)`. `window_mode="forward"` = current path (unchanged). `window_mode="lookback"` = index table → two windows → `assemble_from_events(feature_events, ..., label_events=label_events)`.

- [ ] **Step 1: Add params + branch.** Add `window_mode="forward", lookback_days=365, label_window_days=365` to the signature. Replace the cohort/assembly tail:

```python
    from charmpheno.omop.cohorts import (
        apply_population_disease_cohort, disease_anchors,
        case_finding_index_table, lookback_feature_label_events,
    )
    ...
    if window_mode == "lookback":
        index_df = case_finding_index_table(
            omop, disease=disease, spark=spark, cdr_dataset=cdr,
            billing_project=billing, date_col=date_col,
            prior_obs_days=365, label_window_days=label_window_days)
        feature_events, label_events = lookback_feature_label_events(
            omop, index_df, date_col=date_col,
            lookback_days=lookback_days, label_window_days=label_window_days)
        events, label_arg = feature_events, label_events
    elif window_mode == "forward":
        events = apply_population_disease_cohort(
            omop, disease=disease, window_days=window_days, spark=spark,
            cdr_dataset=cdr, billing_project=billing, date_col=date_col,
            prior_obs_days=prior_obs_days)
        label_arg = None
    else:
        raise ValueError(f"window_mode must be 'forward' or 'lookback', got {window_mode!r}")
    ...
    return assemble_from_events(
        events, before_dag, doc_spec=doc_spec, min_n=min_n,
        holdout_frac=holdout_frac, split_salt=split_salt, vocab_size=vocab_size,
        min_df=min_df, min_patient_count=min_patient_count, n_bg=n_bg, tpn=tpn,
        strip_mode=strip_mode, label_events=label_arg)
```

- [ ] **Step 2: Test the arg surface** (no BQ) — assert the function accepts the new params and rejects a bad `window_mode`:

```python
def test_assemble_case_finding_corpus_accepts_window_mode_params():
    import inspect
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    sig = inspect.signature(assemble_case_finding_corpus)
    assert sig.parameters["window_mode"].default == "forward"
    assert sig.parameters["lookback_days"].default == 365
    assert sig.parameters["label_window_days"].default == 365
```

- [ ] **Step 3: Run** `.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -q` — expect PASS (new test + all prior).

- [ ] **Step 4: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(case-finding): assemble_case_finding_corpus window_mode lookback path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Driver + cache + run_experiment + config + exps 0061/0062

**Files:**
- Modify: `analysis/cloud/dag_placement_cloud.py`, `analysis/cloud/_case_finding_cache.py`, `scripts/run_experiment.py`, `experiments/defaults/_base.yaml`
- Create: `docs/experiments/0061-dag-placement-rare6-lookback-1yr.md`, `docs/experiments/0062-dag-placement-rare6-lookback-5yr.md`
- Test: `scripts/tests/test_dag_placement_config.py`

**Interfaces:**
- Consumes: `assemble_case_finding_corpus(window_mode, lookback_days, label_window_days)` (Task 3).
- Produces: `--window-mode`/`--lookback-days`/`--label-window-days` on the driver (passed to the cached assembly + recorded in the manifest); cache key includes the three params (version bump); `build_dag_placement_args` emits them; `_base.yaml` defaults `window_mode: forward`, `lookback_days: 365`, `label_window_days: 365`; exps 0061 (lookback 365) and 0062 (lookback 1825).

- [ ] **Step 1: Write the failing config test** in `scripts/tests/test_dag_placement_config.py`:

```python
def test_lookback_window_config_and_argv(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    for slug, lb in [("0061-dag-placement-rare6-lookback-1yr", 365),
                     ("0062-dag-placement-rare6-lookback-5yr", 1825)]:
        eff = _load_effective(mod, f"docs/experiments/{slug}.md")
        assert eff["window_mode"] == "lookback"
        assert eff["lookback_days"] == lb
        assert eff["label_window_days"] == 365
        args = mod.build_dag_placement_args(eff, "/out")
        assert args[args.index("--window-mode") + 1] == "lookback"
        assert args[args.index("--lookback-days") + 1] == str(lb)
    eff52 = _load_effective(mod, "docs/experiments/0052-dag-placement-diabetes-random.md")
    assert eff52["window_mode"] == "forward"     # _base default preserves forward
```

- [ ] **Step 2: Run — expect FAIL.** `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -k lookback_window -q`

- [ ] **Step 3: Wire it.**
1. `analysis/cloud/dag_placement_cloud.py`: add args `--window-mode` (choices `forward|lookback`, default `forward`), `--lookback-days` (int, default 365), `--label-window-days` (int, default 365). The driver builds the bundle via `load_or_build_case_finding_bundle(spark, cache_uri=..., window_days=..., disease=..., ...)` (~line 204) — add `window_mode=args.window_mode, lookback_days=args.lookback_days, label_window_days=args.label_window_days` to THAT call. Add all three to the `manifest` dict (~line 306).
2. `analysis/cloud/_case_finding_cache.py`: (a) `compute_bundle_cache_key` (~line 36) — add `window_mode="forward", lookback_days=365, label_window_days=365` to the signature + payload, bump `"v"` 3 → 4. (b) `load_or_build_case_finding_bundle` (~line 130) — it both computes the key AND forwards a fixed param-name list (~lines 145-147) to `compute_bundle_cache_key` and `assemble_case_finding_corpus`; add the three new names to that list and to the function's accepted kwargs so they thread to both the key and the assembly. Defaults (`forward`/365/365) keep existing cached keys valid for forward runs (modulo the v-bump, which forces a one-time rebuild — acceptable).
3. `scripts/run_experiment.py` `build_dag_placement_args`: emit `--window-mode`, `--lookback-days`, `--label-window-days` from `effective` (defaults forward/365/365).
4. `experiments/defaults/_base.yaml`: add `window_mode: forward`, `lookback_days: 365`, `label_window_days: 365` in the dag_placement block.
5. Create `docs/experiments/0061-…md` and `0062-…md`: copy 0060's frontmatter (rare6, `init: spectral`, `spectral_method: scalable`, `anchor_scope: frontier`, `node_alpha_scale: 1.0`, `strip_mode: both`, `n_bg: 40`, `tpn: 5`, `min_n: 20`, `person_mod: 1`, `max_iter: 200`, `seed: 42`), set `window_mode: lookback`, `label_window_days: 365`, and `lookback_days: 365` (0061) / `lookback_days: 1825` (0062). Give each a body noting the A/B (0061 vs 0060; 0062 vs 0061) and that `strip_mode`/`prior_obs_days` are moot in lookback (leakage-free by construction; the ≥1yr-prior gate is intrinsic).

- [ ] **Step 4: Run — expect PASS.** `.venv/bin/python -m pytest scripts/tests/test_dag_placement_config.py -q` (all, incl. the earlier rare6/anchor-scope tests). Syntax-check the driver: `.venv/bin/python -c "import ast; ast.parse(open('analysis/cloud/dag_placement_cloud.py').read()); ast.parse(open('analysis/cloud/_case_finding_cache.py').read())"`.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/dag_placement_cloud.py analysis/cloud/_case_finding_cache.py scripts/run_experiment.py experiments/defaults/_base.yaml docs/experiments/0061-dag-placement-rare6-lookback-1yr.md docs/experiments/0062-dag-placement-rare6-lookback-5yr.md scripts/tests/test_dag_placement_config.py
git commit -m "feat(dag-placement-cloud): --window-mode lookback + exps 0061/0062

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Task 1 = two-frame assembly (spec component 2); Task 2 = lookback windowing + index table (component 1); Task 3 = `assemble_case_finding_corpus` lookback path (component 2); Task 4 = driver/cache/config + exps 0061/0062 (components 3–4). Gate (symmetric bracket) is delivered by reusing `_window_observed_cohort(prior_obs_days=365, window_days=label_window_days)` in Task 2's index table.

**Placeholder scan:** every code step carries concrete code; the one BQ function (`case_finding_index_table`) is explicitly cluster-covered, not stubbed.

**Type consistency:** `lookback_feature_label_events` returns `(feature_events, label_events)` consumed in Task 3; `case_finding_index_table` returns `(person_id, index_date, source_cohort)` consumed by the windowing helper; `assemble_from_events(label_events=...)` (Task 1) consumed by Task 3; config keys `window_mode`/`lookback_days`/`label_window_days` consistent across Tasks 3–4.

**Forward-mode safety:** Tasks 1 and 3 both keep `label_events=None` / `window_mode="forward"` as the exact prior path; the whole-file assembly test in Task 1 Step 4 guards it.

**Open item for the driver task:** confirm whether `assemble_case_finding_corpus` is called directly in `dag_placement_cloud.py` or via a `_case_finding_cache` wrapper; thread the three params through whichever call path builds the bundle, and into the cache key.
