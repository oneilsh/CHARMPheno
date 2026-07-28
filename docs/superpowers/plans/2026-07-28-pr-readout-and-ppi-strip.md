# PR readout + PPI vocabulary strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a precision/recall readout (PR-AUC + precision@recall per rare disease per domain subset) to the multi-domain LR readout, and strip the All of Us survey vocabulary (`vocabulary_id='PPI'`) from the observation domain, so one re-fit of exp 0071/0072 answers both deployability and whether curated observation stops being a drag.

**Architecture:** Task 1 is readout-only (pure numpy over the already-persisted test set, reusing the existing subtree/positive-set logic and the already-computed per-subset scores). Task 2 is fit-side: a general `exclude_vocabularies` filter on the existing `concept` left-join in `load_omop_bigquery`, surfaced as an observation-only driver knob and switched on in exp 0071/0072 frontmatter.

**Tech Stack:** Python 3, NumPy, PySpark (BigQuery read), pytest.

## Global Constraints

- **Task 1 is POST-HOC:** no re-fit, no BigQuery, no Spark session. It reads the driver-local test set already persisted in the run dir.
- **PR positive set MUST be identical to the AUC table's:** positive = doc's frontier ∩ `subtree(anchor)`; per-disease score = max over `subtree(anchor) ∩ lay.nodes` columns. The PR numbers must describe the same detection problem as the AUC numbers.
- **PR-AUC = step-wise average precision** `Σ_i (recall_i − recall_{i−1}) · precision_i` (Davis & Goadrich 2006) — NOT trapezoidal interpolation (optimistically biased for PR curves).
- **precision@recall r** = precision at the smallest threshold achieving recall ≥ r; `nan` if r is unreachable.
- **One-class input** (`n_pos == 0`, or every doc positive) → `pr_auc = nan` and `nan` precisions, mirroring `_auc`'s one-class convention. No crash.
- **`prev` column** = `n_pos / n_docs`, printed beside PR-AUC: prevalence IS the random-classifier PR-AUC baseline (unlike ROC, PR's baseline moves with the base rate).
- **`exclude_vocabularies` defaults to `()`** = today's behavior byte-identical (no extra column selected, no filter applied).
- **The vocabulary filter is NULL-SAFE:** a concept absent from the `concept` table has a null `vocabulary_id` from the left join and is KEPT (an unmapped code is not evidence of being a survey item).
- **`vocabulary_id` is NOT in the output projection** — the canonical output schema (`person_id`, `concept_id`, `concept_name`, *extra_cols) is unchanged.
- **`--obs-exclude-vocab` applies ONLY to the observation domain's load.** Condition and drug loads are unaffected. Default empty.
- **Do NOT touch exp 0070** (two-domain, no observation).
- **Deferred, do NOT implement:** `max_df` / document-frequency cap; the measurement (labs) domain; the ω sweep.
- charmpheno tests: `cd charmpheno && poetry run pytest tests/...`. driver/analysis + scripts tests: `./.venv/bin/python -m pytest <path> -q` from the repo root (the `poetry run` variant in `analysis/cloud` hits a known stale-venv `charmpheno.omop` import miss). Bash `timeout` is MILLISECONDS.

---

## File Structure

- `analysis/cloud/multidomain_lr_readout.py` — add `per_disease_pr` + two prints in `main()` (Task 1).
- `analysis/cloud/tests/test_multidomain_lr_readout.py` — PR helper tests (Task 1).
- `charmpheno/charmpheno/omop/bigquery.py` — `exclude_vocabularies` param + NULL-safe filter (Task 2).
- `charmpheno/tests/test_bigquery_exclude_vocab.py` — NEW: filter predicate test (Task 2).
- `analysis/cloud/multidomain_cloud.py` — `--obs-exclude-vocab` knob, observation-only application, manifest record (Task 2).
- `analysis/cloud/tests/test_multidomain_cloud.py` — knob parse test (Task 2).
- `scripts/run_experiment.py` — emit `--obs-exclude-vocab` (Task 2).
- `scripts/tests/test_run_experiment_multidomain.py` — emission test (Task 2).
- `docs/experiments/0071-*.md`, `docs/experiments/0072-*.md` — `obs_exclude_vocab: PPI` (Task 2).
- `scripts/tests/test_experiment_defs_multidomain.py` — frontmatter test (Task 2).

---

### Task 1: Precision/recall readout

**Files:**
- Modify: `analysis/cloud/multidomain_lr_readout.py` (add `per_disease_pr` after `per_disease_auc_row`; add two prints at the end of `main()`)
- Test: `analysis/cloud/tests/test_multidomain_lr_readout.py`

**Interfaces:**
- Consumes (already in the module): `subtree_nodes(parent_int, root) -> set[int]`; `per_disease_auc_row(scores, frontiers, anchor, lay, parent_int) -> (auc, n_pos)`; and in `main()` the bound locals `subset_scores` (`{subset_name: [n_docs x n_nodes] ndarray}`), `subsets` (`{name: [domain indices]}`), `frontiers`, `aff`, `aff_frontiers`, `anchors` (engine ids), `name_by_engine`, `int2cid`, `lay`, `parent_int`, `n_docs`, `domain_names`.
- Produces: `per_disease_pr(scores, frontiers, anchor, lay, parent_int, recalls=(0.5, 0.8)) -> (pr_auc: float, prec_at: dict[float, float], n_pos: int)`.

- [ ] **Step 1: Write the failing tests**

Add to `analysis/cloud/tests/test_multidomain_lr_readout.py`:

```python
def _pr_fixture():
    """A 1-node layout (node 1, no children) + scores/frontiers helper: node 1 is
    the anchor, so subtree(1) == {1} and the per-disease score is that column."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    parent_int = {1: [0]}
    return lay, parent_int


def test_per_disease_pr_perfect_ranker():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    # 2 positives ranked strictly above 6 negatives -> perfect PR.
    scores = np.zeros((8, len(lay.nodes)))
    scores[:, col] = [9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    frontiers = [[1], [1], [], [], [], [], [], []]
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int,
                                           recalls=(0.5, 0.8))
    assert n_pos == 2
    assert pr_auc == 1.0
    assert prec_at[0.5] == 1.0 and prec_at[0.8] == 1.0


def test_per_disease_pr_uninformative_ranker_is_near_prevalence():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    # Constant score -> every doc tied; AP of a random ranker ~= prevalence.
    n, n_pos_want = 100, 20
    scores = np.zeros((n, len(lay.nodes)))
    scores[:, col] = 1.0
    frontiers = [[1] if i % 5 == 0 else [] for i in range(n)]   # 20/100 = 0.2
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int)
    assert n_pos == n_pos_want
    assert abs(pr_auc - 0.2) < 0.05            # ~= prevalence baseline
    assert abs(prec_at[0.5] - 0.2) < 0.05


def test_per_disease_pr_reads_the_right_operating_point():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    # Ranked order: P N P N  (2 positives). Cumulative precision/recall:
    #   rank1 P: tp=1 -> prec 1/1=1.00, rec 0.5
    #   rank2 N: tp=1 -> prec 1/2=0.50, rec 0.5
    #   rank3 P: tp=2 -> prec 2/3=0.667, rec 1.0
    # precision@0.5 = 1.00 (first index reaching rec>=0.5)
    # precision@1.0 = 0.667 ; precision@1.5 unreachable -> nan
    scores = np.zeros((4, len(lay.nodes)))
    scores[:, col] = [4.0, 3.0, 2.0, 1.0]
    frontiers = [[1], [], [1], []]
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int,
                                           recalls=(0.5, 1.0, 1.5))
    assert n_pos == 2
    assert prec_at[0.5] == 1.0
    assert abs(prec_at[1.0] - 2.0 / 3.0) < 1e-9
    assert np.isnan(prec_at[1.5])              # unreachable recall
    # AP = sum over positives of precision at that positive = (1.0 + 0.667)/2
    assert abs(pr_auc - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_per_disease_pr_one_class_is_nan():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    scores = np.zeros((4, len(lay.nodes)))
    pr_auc, prec_at, n_pos = per_disease_pr(scores, [[], [], [], []], 1, lay,
                                           parent_int)
    assert n_pos == 0
    assert np.isnan(pr_auc) and np.isnan(prec_at[0.5])
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/python -m pytest analysis/cloud/tests/test_multidomain_lr_readout.py -q`
Expected: FAIL (`per_disease_pr` undefined).

- [ ] **Step 3: Implement `per_disease_pr`**

In `analysis/cloud/multidomain_lr_readout.py`, immediately AFTER `per_disease_auc_row`:

```python
def per_disease_pr(scores, frontiers, anchor, lay, parent_int, recalls=(0.5, 0.8)):
    """(pr_auc, {recall: precision}, n_pos) for detecting disease `anchor`.

    Same detection problem as `per_disease_auc_row` -- positive = the doc's
    frontier intersects subtree(anchor); per-disease score = max over that
    subtree's columns -- so PR and ROC numbers are directly comparable.

    pr_auc = step-wise AVERAGE PRECISION, Sum_i (rec_i - rec_{i-1}) * prec_i
    (Davis & Goadrich 2006, "The relationship between Precision-Recall and ROC
    curves", ICML): the trapezoidal rule is optimistically biased on PR curves
    because precision is not linear between operating points. At rare-disease
    base rates the random-classifier PR-AUC is the PREVALENCE (not 0.5), so the
    caller prints n_pos/n_docs beside it.

    prec_at[r] = precision at the smallest threshold reaching recall >= r
    (nan if r is unreachable). One-class input -> nan, matching `_auc`.
    """
    sub = subtree_nodes(parent_int, anchor) & set(lay.nodes)
    nan_out = (float("nan"), {float(r): float("nan") for r in recalls}, 0)
    if not sub:
        return nan_out
    cols = [lay.nodes.index(u) for u in sub]
    node_score = np.asarray(scores)[:, cols].max(axis=1)
    y = np.array([1 if (set(fr) & sub) else 0 for fr in frontiers], dtype=int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return (float("nan"), {float(r): float("nan") for r in recalls}, n_pos)

    order = np.argsort(-node_score, kind="mergesort")   # stable: ties keep row order
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # Average precision: precision summed at each POSITIVE (where recall steps).
    ap = float(np.sum(precision * y_sorted) / n_pos)

    prec_at = {}
    for r in recalls:
        hit = np.nonzero(recall >= float(r))[0]
        prec_at[float(r)] = float(precision[hit[0]]) if hit.size else float("nan")
    return ap, prec_at, n_pos
```

- [ ] **Step 4: Run to verify they pass**

Run: `./.venv/bin/python -m pytest analysis/cloud/tests/test_multidomain_lr_readout.py -q`
Expected: PASS (all, including the 4 new PR tests).

- [ ] **Step 5: Add the two prints to `main()`**

In `analysis/cloud/multidomain_lr_readout.py`, replace the final summary line

```python
    print(f"[lr] scored {n_docs} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0
```

with (the PR tables reuse `subset_scores` -- no re-scoring):

```python
    # --- PR-AUC table (same subsets as the AUC table). `prev` = n_pos/n_docs is
    # the random-classifier PR-AUC, the baseline that makes PR-AUC readable at
    # rare-disease base rates. ---
    print(f"[lr] === per-disease x domain-subset PR-AUC (avg precision, "
          f"alpha={a_head}) ===", flush=True)
    header = "disease".ljust(26) + "n+".rjust(5) + "   prev"
    for name in subsets:
        header += "  " + name[:12].rjust(12)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        _, _, n_pos = per_disease_pr(subset_scores["all"], frontiers, u, lay,
                                     parent_int)
        prev = (n_pos / n_docs) if n_docs else float("nan")
        line = dname.ljust(26) + str(n_pos).rjust(5) + f"  {prev:5.4f}"
        for name in subsets:
            pr_auc, _, _ = per_disease_pr(subset_scores[name], frontiers, u, lay,
                                          parent_int)
            line += "  " + f"{pr_auc:12.3f}"
        print("[lr] " + line, flush=True)

    # --- Precision@recall: the deployability read ("flag enough patients to catch
    # 80% of true cases -- what fraction of the flagged list is real?") for the
    # three headline subsets, incl. the cond vs cond+drug operational comparison.
    # drop:<last domain> is cond+drug when observation is the last domain; fall
    # back to whatever exists so this never KeyErrors on a 1- or 2-domain run. ---
    headline = [n for n in ("all", "only:condition",
                            f"drop:{domain_names[-1]}") if n in subsets]
    print("[lr] === precision @ recall (headline subsets) ===", flush=True)
    header = "disease".ljust(26) + "n+".rjust(5)
    for name in headline:
        header += "  " + f"{name[:10]}@50%".rjust(16) + f"{name[:10]}@80%".rjust(16)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        cells, n_pos_seen = "", 0
        for name in headline:
            _, prec_at, n_pos = per_disease_pr(subset_scores[name], frontiers, u,
                                               lay, parent_int, recalls=(0.5, 0.8))
            n_pos_seen = n_pos          # same positive set for every subset
            cells += "  " + f"{prec_at[0.5]:16.3f}" + f"{prec_at[0.8]:16.3f}"
        print("[lr] " + f"{dname:<26}{n_pos_seen:>5}" + cells, flush=True)

    print(f"[lr] scored {n_docs} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0
```

- [ ] **Step 6: Verify the module still imports and tests pass**

Run:
```bash
./.venv/bin/python -c "import py_compile; py_compile.compile('analysis/cloud/multidomain_lr_readout.py', doraise=True); print('compile OK')"
./.venv/bin/python -m pytest analysis/cloud/tests/test_multidomain_lr_readout.py -q
```
Expected: compile OK; all tests PASS. (`main()`'s printing is cluster-run.)

- [ ] **Step 7: Commit**

```bash
git add analysis/cloud/multidomain_lr_readout.py analysis/cloud/tests/test_multidomain_lr_readout.py
git commit -m "feat(multidomain): PR-AUC + precision@recall readout per disease per domain subset"
```

---

### Task 2: PPI vocabulary strip (loader + driver + experiment defs)

**Files:**
- Modify: `charmpheno/charmpheno/omop/bigquery.py` (`load_omop_bigquery` signature + docstring + the `concept` join/filter)
- Create: `charmpheno/tests/test_bigquery_exclude_vocab.py`
- Modify: `analysis/cloud/multidomain_cloud.py` (`--obs-exclude-vocab` knob; observation-only application at the load site; manifest record)
- Modify: `analysis/cloud/tests/test_multidomain_cloud.py`
- Modify: `scripts/run_experiment.py` (`build_multidomain_args` emission)
- Modify: `scripts/tests/test_run_experiment_multidomain.py`
- Modify: `docs/experiments/0071-multidomain-rare6-cond-drug-obs.md`, `docs/experiments/0072-multidomain-rare6-cond-drug-obs-minibatch.md`
- Modify: `scripts/tests/test_experiment_defs_multidomain.py`

**Interfaces:**
- Consumes: `load_omop_bigquery(*, spark, cdr_dataset, billing_project, concept_types=("condition",), person_sample_mod=None, source_table="condition_occurrence", cohort=None, prior_obs_days=None) -> DataFrame` (current signature — the new param is added keyword-only at the end); `DOMAIN_REGISTRY` (maps source_table → `{date_col, name, arg}`); the driver's load site builds `raws = [load_omop_bigquery(spark=..., cdr_dataset=args.cdr, billing_project=args.billing, person_sample_mod=args.person_mod, source_table=t) for t in domain_tables]`.
- Produces: `load_omop_bigquery(..., exclude_vocabularies: tuple[str, ...] = ())`; driver arg `args.obs_exclude_vocab` (a tuple of strings, `()` when unset); `--obs-exclude-vocab` CLI flag on both the driver and `build_multidomain_args`.

- [ ] **Step 1: Write the failing loader test**

Create `charmpheno/tests/test_bigquery_exclude_vocab.py`:

```python
"""The exclude_vocabularies filter predicate: drops named vocabularies, keeps
NULL (unmapped concepts). The live BigQuery read is cluster-covered; this tests
the pure predicate against a local-Spark frame."""
import pytest


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[1]")
         .appName("test-exclude-vocab").getOrCreate())
    yield s
    s.stop()


def test_exclude_vocabularies_predicate_drops_named_keeps_null(spark):
    from charmpheno.omop.bigquery import _exclude_vocab_filter
    df = spark.createDataFrame(
        [(1, "PPI"), (2, "SNOMED"), (3, None), (4, "PPI")],
        "concept_id int, vocabulary_id string")
    kept = {r["concept_id"] for r in _exclude_vocab_filter(df, ("PPI",)).collect()}
    assert kept == {2, 3}          # PPI dropped; NULL (unmapped) KEPT


def test_exclude_vocabularies_empty_is_identity(spark):
    from charmpheno.omop.bigquery import _exclude_vocab_filter
    df = spark.createDataFrame(
        [(1, "PPI"), (2, None)], "concept_id int, vocabulary_id string")
    assert _exclude_vocab_filter(df, ()).count() == 2      # no filter applied
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && poetry run pytest tests/test_bigquery_exclude_vocab.py -q`
Expected: FAIL (`_exclude_vocab_filter` undefined).

- [ ] **Step 3: Implement the loader change**

In `charmpheno/charmpheno/omop/bigquery.py`, add the helper near the other module-level helpers (e.g. after `_observation_select_cols`):

```python
def _exclude_vocab_filter(df, exclude_vocabularies):
    """Drop rows whose `vocabulary_id` is in `exclude_vocabularies`; NULL-SAFE --
    a concept absent from the OMOP `concept` table has a null vocabulary_id from
    the left join and is KEPT (an unmapped code is not evidence of being a survey
    item). Empty tuple = identity (no filter).

    Motivation: All of Us stores its survey/SDOH observations under
    vocabulary_id 'PPI' (Participant-Provided Information), which dominates the
    observation domain's token volume with low disease specificity (insight
    0071). The mechanism is general -- any domain, any vocabulary."""
    if not exclude_vocabularies:
        return df
    keep = (~F.col("vocabulary_id").isin(list(exclude_vocabularies))
            ) | F.col("vocabulary_id").isNull()
    return df.where(keep)
```

Add the parameter to the signature (keyword-only, last):

```python
    prior_obs_days: int | None = None,
    exclude_vocabularies: tuple[str, ...] = (),
) -> DataFrame:
```

Document it in the Args block (after `prior_obs_days`):

```
        exclude_vocabularies: OMOP `vocabulary_id` values to drop (e.g.
            ("PPI",) to strip the All of Us survey/SDOH vocabulary from the
            observation domain -- insight 0071). Default () = no filtering,
            byte-identical to before. NULL-safe: concepts missing from the
            `concept` table (null vocabulary_id) are KEPT. `vocabulary_id` is
            not added to the output schema.
```

Change the `concept` read + join so `vocabulary_id` is available only when needed, filter, then project it away. Replace:

```python
    concept = _read("concept").select("concept_id", "concept_name")
```

with:

```python
    # vocabulary_id is selected ONLY when a filter needs it, so the default path
    # reads exactly the same columns as before.
    concept_cols = ["concept_id", "concept_name"]
    if exclude_vocabularies:
        concept_cols.append("vocabulary_id")
    concept = _read("concept").select(*concept_cols)
```

and after the join (`omop = cond.join(concept, on="concept_id", how="left")`), BEFORE the canonical projection, add:

```python
    # Vocabulary exclusion runs after the left join (it needs vocabulary_id) and
    # before the canonical projection (which drops it again).
    omop = _exclude_vocab_filter(omop, exclude_vocabularies)
```

The existing canonical projection line is unchanged and drops `vocabulary_id`:

```python
    omop = omop.select("person_id", "concept_id", "concept_name", *extra_cols)
```

- [ ] **Step 4: Run to verify the loader tests pass**

Run: `cd charmpheno && poetry run pytest tests/test_bigquery_exclude_vocab.py tests/test_multi_domain.py -q`
Expected: PASS (new filter tests + the multi-domain suite unaffected).

- [ ] **Step 5: Write the failing driver + harness + exp-def tests**

Add to `analysis/cloud/tests/test_multidomain_cloud.py`:

```python
def test_parse_args_obs_exclude_vocab_defaults_empty_and_parses_a_list():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).obs_exclude_vocab == ()          # default = no strip
    a = parse_args(base + ["--obs-exclude-vocab", "PPI"])
    assert a.obs_exclude_vocab == ("PPI",)
    b = parse_args(base + ["--obs-exclude-vocab", "PPI,SNOMED"])
    assert b.obs_exclude_vocab == ("PPI", "SNOMED")
```

Add to `scripts/tests/test_run_experiment_multidomain.py`:

```python
def test_build_multidomain_args_emits_obs_exclude_vocab(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_multidomain_args({**_min_eff(), "obs_exclude_vocab": "PPI"},
                                      "/out")
    assert args[args.index("--obs-exclude-vocab") + 1] == "PPI"
    # unset -> empty string (the driver parses that to ())
    d = mod.build_multidomain_args(_min_eff(), "/out")
    assert d[d.index("--obs-exclude-vocab") + 1] == ""
```

Add to `scripts/tests/test_experiment_defs_multidomain.py`:

```python
def test_exps_0071_0072_strip_the_ppi_vocabulary():
    # insight 0071: observation is net-negative; strip the AoU survey vocabulary.
    for name in ("0071-multidomain-rare6-cond-drug-obs.md",
                 "0072-multidomain-rare6-cond-drug-obs-minibatch.md"):
        mod, fm = _fm(name)
        mod.validate_frontmatter(fm)
        assert fm["obs_exclude_vocab"] == "PPI", name
```

- [ ] **Step 6: Run to verify they fail**

Run:
```bash
./.venv/bin/python -m pytest analysis/cloud/tests/test_multidomain_cloud.py scripts/tests/test_run_experiment_multidomain.py scripts/tests/test_experiment_defs_multidomain.py -q
```
Expected: FAIL (no `obs_exclude_vocab` attr / flag / frontmatter key).

- [ ] **Step 7: Implement the driver knob**

In `analysis/cloud/multidomain_cloud.py` `parse_args`, next to the other `--obs-*` knobs (near `--obs-vocab-size`):

```python
    p.add_argument("--obs-exclude-vocab", default="",
                   type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
                   help="Comma list of OMOP vocabulary_id values to drop from the "
                        "OBSERVATION domain (e.g. 'PPI' = the All of Us survey/SDOH "
                        "vocabulary; insight 0071 found observation net-negative). "
                        "Empty = no strip.")
```

Note: argparse applies `type` to the DEFAULT only when the default is a string, which it is (`""` → `()`), so an unset flag yields `()`.

At the load site, apply it to the observation domain ONLY. Replace:

```python
            raws = [load_omop_bigquery(
                        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                        person_sample_mod=args.person_mod, source_table=t)
                    for t in domain_tables]
```

with:

```python
            raws = [load_omop_bigquery(
                        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
                        person_sample_mod=args.person_mod, source_table=t,
                        # observation-only: strip the AoU survey/SDOH vocabulary
                        # (insight 0071). Other domains load unfiltered.
                        exclude_vocabularies=(args.obs_exclude_vocab
                                              if t == "observation" else ()))
                    for t in domain_tables]
```

Record it in the manifest's `corpus_manifest` (next to `person_mod`), so the artifact says what was stripped:

```python
                    "obs_exclude_vocab": list(args.obs_exclude_vocab),
```

- [ ] **Step 8: Implement the harness emission + experiment frontmatter**

In `scripts/run_experiment.py` `build_multidomain_args`, next to the other `--obs-*` flags:

```python
        "--obs-exclude-vocab", str(effective.get("obs_exclude_vocab", "")),
```

In BOTH `docs/experiments/0071-multidomain-rare6-cond-drug-obs.md` and
`docs/experiments/0072-multidomain-rare6-cond-drug-obs-minibatch.md`, add to the
frontmatter next to the other `obs_*` keys:

```yaml
# insight 0071: the observation domain was net-negative for all six rare
# diseases (drop:observation >= all everywhere). Strip the All of Us survey/SDOH
# vocabulary (vocabulary_id='PPI'), which dominates its token volume with low
# disease specificity. Re-fit required (this changes the observation vocabulary).
obs_exclude_vocab: PPI
```

Also append one paragraph to each experiment's markdown body under "What to read":

```markdown
- **Did the PPI strip help?** Compare the readout's `drop:observation` column to
  `all`: insight 0071 had `drop:observation >= all` for all six diseases (observation
  was pure drag). If the gap narrows or closes, the AoU survey vocabulary was the
  drag; if it persists, the remaining clinical junk ("History of event",
  "Long-term current use of...") is, and a max_df cap is the next lever.
- **The new PR tables:** PR-AUC beside `prev` (prevalence = the random baseline)
  and precision@50%/80% recall for all / only:condition / drop:observation --
  whether the ranking AUC translates into deployable precision at these base
  rates, and whether cond+drug beats cond-alone operationally.
```

- [ ] **Step 9: Run all the Task-2 tests**

Run:
```bash
./.venv/bin/python -m pytest analysis/cloud/tests/test_multidomain_cloud.py scripts/tests/test_run_experiment_multidomain.py scripts/tests/test_experiment_defs_multidomain.py -q
cd charmpheno && poetry run pytest tests/test_bigquery_exclude_vocab.py -q
```
Expected: PASS (all).

- [ ] **Step 10: Commit**

```bash
git add charmpheno/charmpheno/omop/bigquery.py charmpheno/tests/test_bigquery_exclude_vocab.py analysis/cloud/multidomain_cloud.py analysis/cloud/tests/test_multidomain_cloud.py scripts/run_experiment.py scripts/tests/test_run_experiment_multidomain.py scripts/tests/test_experiment_defs_multidomain.py docs/experiments/0071-multidomain-rare6-cond-drug-obs.md docs/experiments/0072-multidomain-rare6-cond-drug-obs-minibatch.md
git commit -m "feat(omop): exclude_vocabularies loader filter + PPI strip on exp 0071/0072 observation domain"
```

---

## Final verification (after both tasks)

- [ ] `./.venv/bin/python -m pytest analysis/cloud/tests/ scripts/tests/ -q` — driver + harness suites green.
- [ ] `cd charmpheno && poetry run pytest tests/ -q` — charmpheno suite green (loader change is additive/default-off).
- [ ] **Cluster (user-run):** re-fit `make exp ID=71` + `ID=72` (PPI-stripped observation vocab), then `make multidomain-lr-readout ID=71` + `ID=72`. Expect: the observation topic dump free of PMI / "DNA Quiz" / "Can't Afford Care" / "Are you still seeing…"; `manifest.dead_nodes` still empty; `manifest.corpus_manifest.obs_exclude_vocab == ["PPI"]`; the readout printing the AUC table PLUS the PR-AUC table and the precision@recall summary.
