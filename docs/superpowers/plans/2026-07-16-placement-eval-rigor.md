# Placement-Eval Rigor: tie-fix, PR metrics, leakage-free split, strip ablation — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the dag_placement evaluation trustworthy and leakage-free before drawing real-data conclusions from exp 0052/0053: fix the AUC tie bug, add PR-AUC + recall@k (+ light bootstrap CIs), fit DAG-pruning and vocabulary on TRAIN patients only (map test onto the train DAG), and expose a train/test leakage-strip ablation knob.

**Architecture:** Four tasks across two layers. Engine (`spark_vi.models.topic.dag_placement`): (1) `_auc` midranks; (2) PR-AUC/recall@k/CIs in `evaluate`. Domain (`charmpheno.omop.case_finding_assembly`): (3) split-first, train-only prune+vocab, test roll-up + coarsening report; (4) `strip_mode` ablation knob threaded through assembly → cache → driver → config.

**Tech Stack:** numpy + `scipy.stats.rankdata` (no sklearn — average precision implemented manually, tie-aware). PySpark for the assembly. Tests: repo pytest + local Spark fixtures.

## Global Constraints

- **Branch:** `case-finding`. Do NOT merge to main. Push to `origin/case-finding` after committing (no auto-push).
- **Commit trailer, EXACT** (blank line before): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **No LaTeX**; Unicode Greek only. **Cite literature** for methods (average precision, Mann-Whitney AUC midranks).
- **Hash IDs in row-level logs.** Metrics/rates print raw.
- **Test honesty:** no threshold-loosening; `xfail` with a reason if needed.
- **Engine stays integer-id agnostic** (Tasks 1-2 touch only ids/scores). Concept-ids live in the assembly (Tasks 3-4).
- **Backward-compatible defaults:** `evaluate`'s existing keys keep their meaning (new keys added); `strip_mode` defaults to `"test_only"` (current behavior). The leakage fix (Task 3) DOES change bundle contents — that is intended; the cache `v` is bumped so old entries miss.

## Reference (read-only)

- `spark-vi/spark_vi/models/topic/dag_placement.py` — `_auc` (line 228), `evaluate` (line 257), `_hops`, `DagLayout` (`.subtree`, `.nodes`, `.depth`).
- `charmpheno/charmpheno/omop/case_finding_assembly.py` — `assemble_from_events` (line 225), the helpers `doc_attested_nodes`, `node_patient_counts`, `attach_frontiers`, `split_train_test`, `strip_test_features`, `most_specific_cids`; `CaseFindingBundle`.
- `charmpheno/charmpheno/omop/topic_prep.py` — `to_bow_dataframe(df, *, doc_spec, vocab_size, min_df, min_patient_count, vocab=None)` (frozen-vocab path requires `vocab_size=None, min_df=1, min_patient_count=1`).
- `analysis/cloud/_case_finding_cache.py` — `compute_bundle_cache_key` (`v` field), `load_or_build_case_finding_bundle` key whitelist.
- `analysis/cloud/dag_placement_cloud.py` — `main` metric-print + manifest; `parse_args`.
- `scripts/run_experiment.py` — `build_dag_placement_args`.
- `experiments/defaults/_base.yaml` dag_placement block.

---

### Task 1: `_auc` midranks + explicit tie policy

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (`_auc` ~228; docstring notes on `evaluate`'s MRR/hops)
- Test: `spark-vi/tests/test_dag_placement.py` (extend; create if absent — check first)

**Interfaces:** `_auc(scores, y) -> float` unchanged signature; now returns the Mann-Whitney AUC with **average (mid)ranks** for ties (0.5 for all-ties).

- [ ] **Step 1: Write failing tests**

```python
# spark-vi/tests/test_dag_placement.py  (append; add imports at top if new file)
import numpy as np
from spark_vi.models.topic.dag_placement import _auc


def test_auc_all_ties_is_half():
    # identical scores -> AUC must be 0.5 regardless of label order.
    assert abs(_auc(np.zeros(6), [1, 0, 1, 0, 1, 0]) - 0.5) < 1e-9
    assert abs(_auc(np.zeros(6), [0, 0, 0, 1, 1, 1]) - 0.5) < 1e-9


def test_auc_partial_ties_midrank():
    # scores [1,1,0,0], labels [1,0,1,0]: the two positives tie a positive with a
    # negative at each score level -> AUC 0.5 (midranks), NOT order-dependent 0/1.
    assert abs(_auc(np.array([1.0, 1.0, 0.0, 0.0]), [1, 0, 1, 0]) - 0.5) < 1e-9


def test_auc_perfect_and_degenerate():
    assert _auc(np.array([3.0, 2.0, 1.0, 0.0]), [1, 1, 0, 0]) == 1.0
    assert np.isnan(_auc(np.array([1.0, 2.0]), [1, 1]))     # one class -> nan
```

- [ ] **Step 2: Run to verify fail**

`cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k auc -v` → `test_auc_all_ties_is_half`/`_midrank` FAIL (argsort gives order-dependent ranks).

- [ ] **Step 3: Fix `_auc` with midranks**

```python
def _auc(scores, y):
    """Mann-Whitney (rank-sum) AUC. Ties in `scores` get AVERAGE (mid)ranks
    (scipy.stats.rankdata method='average'), so tied score blocks contribute
    0.5 per positive-negative pair — the correct ROC-AUC. (argsort's distinct
    ranks made a tie block read as 0 or 1 depending on row order.) One-class
    input -> nan."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
```

- [ ] **Step 4: Document the MRR/hops tie policies**

In `evaluate`'s docstring append: "Tie policy: node AUC/PR use midranks (see `_auc`). MRR/top2 count only nodes with STRICTLY greater affinity than the true node (best-rank-among-ties — optimistic, appropriate for a set-valued truth). mean_hops uses `argmax` (ties broken by node id)."

- [ ] **Step 5: Run to verify pass + no regression**

`cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v` (all green). The existing slow equivalence gates live in `test_gated_lda.py`, not here.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "fix(dag-placement): AUC uses midranks so ties score 0.5 not order-dependent

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: PR-AUC, recall@k, and bootstrap CIs in `evaluate`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add `_average_precision`, `_bootstrap_ci`; extend `evaluate`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:** `evaluate(profiles, test_labels, lay)` keeps every current key and ADDS: `node_ap` (per-node average precision, {node: float}), `ap_macro`, `ap_micro`, `ap_prevalence_weighted`, `recall_at_k` ({1: float, 2: float, 3: float}, set-valued recall over the full frontier), and `ci` (percentile bootstrap over docs for `{ap_macro, mrr, top2, recall_at_1}`). Under heavy background imbalance, PR-AUC/recall are the clinically meaningful complements to ROC-AUC.

- [ ] **Step 1: Write failing tests**

```python
# append to spark-vi/tests/test_dag_placement.py
from spark_vi.models.topic.dag_placement import (
    DagLayout, evaluate, _average_precision,
)


def test_average_precision_perfect_and_constant():
    # perfect ranking -> AP 1.0
    assert abs(_average_precision([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]) - 1.0) < 1e-9
    # constant scores -> AP == prevalence (2/4), NOT 0/1 (the AUC-tie failure mode)
    assert abs(_average_precision([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]) - 0.5) < 1e-9
    import numpy as np
    assert np.isnan(_average_precision([1.0, 2.0], [0, 0]))   # no positives -> nan


def test_evaluate_adds_pr_recall_ci_keys():
    parent = {1: 0, 2: 0, 3: 1}
    lay = DagLayout(parent, n_bg=2, tpn=1)     # nodes [1,2,3], depth(3)=2
    # 4 docs: two truly under node 1 (via leaf 3), two under node 2.
    profiles = [
        {1: 0.6, 2: 0.1, 3: 0.5},   # true {3}
        {1: 0.4, 2: 0.2, 3: 0.3},   # true {3}
        {1: 0.1, 2: 0.7, 3: 0.0},   # true {2}
        {1: 0.2, 2: 0.6, 3: 0.1},   # true {2}
    ]
    labels = [{3}, {3}, {2}, {2}]
    ev = evaluate(profiles, labels, lay)
    assert set(ev["node_ap"]) == {1, 2, 3}
    assert 0.0 <= ev["ap_macro"] <= 1.0
    assert 0.0 <= ev["ap_micro"] <= 1.0
    assert 0.0 <= ev["ap_prevalence_weighted"] <= 1.0
    assert set(ev["recall_at_k"]) == {1, 2, 3}
    assert 0.0 <= ev["recall_at_k"][1] <= ev["recall_at_k"][3] <= 1.0
    # CI present for the headline metrics and brackets the point estimate.
    for key in ("ap_macro", "mrr", "top2", "recall_at_1"):
        lo, hi = ev["ci"][key]
        assert lo <= hi
    assert ev["ci"]["ap_macro"][0] <= ev["ap_macro"] <= ev["ci"]["ap_macro"][1] + 1e-9
```

- [ ] **Step 2: Run to verify fail**

`cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "average_precision or pr_recall_ci" -v` → FAIL (`_average_precision` missing; new keys absent).

- [ ] **Step 3: Add `_average_precision` (tie-aware, sklearn-consistent)**

```python
def _average_precision(scores, y):
    """Average precision (area under the precision-recall curve, the step
    definition used by sklearn.metrics.average_precision_score): AP = sum_i
    (R_i - R_{i-1}) * P_i over distinct score thresholds i (descending). Tied
    scores share a threshold, so AP is order-invariant and a constant scorer
    yields AP == prevalence. No positives -> nan."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    n1 = y.sum()
    if n1 == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")          # desc, stable
    s, yy = scores[order], y[order]
    tp_cum = np.cumsum(yy)
    pp_cum = np.arange(1, len(yy) + 1)
    group_end = np.concatenate((s[1:] != s[:-1], [True]))  # last index of each tie group
    ends = np.where(group_end)[0]
    recall = tp_cum[ends] / n1
    precision = tp_cum[ends] / pp_cum[ends]
    r_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - r_prev) * precision))
```

- [ ] **Step 4: Extend `evaluate` with PR, recall@k, CI**

Inside `evaluate`, after the existing `node_auc`/ranks/hops computation and before the return, add (reusing `P`, `fronts`, `lay`):

```python
    # --- PR-AUC (average precision) per node + summaries ---------------------
    node_pos = {u: [bool(f & lay.subtree(u)) for f in fronts] for u in lay.nodes}
    node_ap = {u: _average_precision(P[:, i], node_pos[u])
               for i, u in enumerate(lay.nodes)}
    valid_ap = {u: a for u, a in node_ap.items() if not np.isnan(a)}
    npos = {u: int(np.sum(node_pos[u])) for u in lay.nodes}
    ap_macro = float(np.mean(list(valid_ap.values()))) if valid_ap else float("nan")
    tot_pos = sum(npos[u] for u in valid_ap)
    ap_prevalence_weighted = (
        float(sum(valid_ap[u] * npos[u] for u in valid_ap) / tot_pos)
        if tot_pos else float("nan"))
    # micro AP: pool every (node, doc) pair into one ranking.
    flat_scores = P.reshape(-1)
    flat_labels = np.array([node_pos[u][d] for d in range(len(fronts))
                            for u in [lay.nodes[i] for i in range(len(lay.nodes))]])
    # NOTE order: reshape(-1) walks docs-major (row d, then node i); build labels to match.
    flat_labels = np.array([node_pos[lay.nodes[i]][d]
                            for d in range(P.shape[0]) for i in range(P.shape[1])])
    ap_micro = _average_precision(flat_scores, flat_labels)

    # --- recall@k over the FULL set-valued frontier -------------------------
    def _recall_at_k(k):
        rec = []
        for i, f in enumerate(fronts):
            true_idx = [lay.nodes.index(t) for t in f if t in lay.nodes]
            if not true_idx:
                continue
            topk = set(np.argsort(-P[i], kind="mergesort")[:k].tolist())
            rec.append(len(topk & set(true_idx)) / len(true_idx))
        return float(np.mean(rec)) if rec else float("nan")
    recall_at_k = {k: _recall_at_k(k) for k in (1, 2, 3)}

    # --- percentile bootstrap CIs (resample docs = patients; 1 doc/patient) --
    ci = _bootstrap_ci(P, fronts, lay, node_pos)
```

Add the return keys:

```python
    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": ..., "top2": ..., "mean_hops": ...,
            "frontier_size_mean": ..., "multi_frontier_rate": ...,
            "node_ap": node_ap, "ap_macro": ap_macro, "ap_micro": ap_micro,
            "ap_prevalence_weighted": ap_prevalence_weighted,
            "recall_at_k": recall_at_k, "ci": ci}
```

(Keep the existing computed values for the first seven keys verbatim; only append the new ones.)

- [ ] **Step 5: Add `_bootstrap_ci`**

```python
def _bootstrap_ci(P, fronts, lay, node_pos, *, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CIs for the headline metrics, resampling DOCS
    with replacement (docs are one-per-patient here, so this is a patient-
    clustered bootstrap). Returns {metric: (lo, hi)} for ap_macro, mrr, top2,
    recall_at_1. Fixed seed -> resume-stable."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    nodes = lay.nodes

    def _metrics(idx):
        Pb = P[idx]
        fb = [fronts[j] for j in idx]
        posb = {u: [node_pos[u][j] for j in idx] for u in nodes}
        aps = [_average_precision(Pb[:, i], posb[u]) for i, u in enumerate(nodes)]
        aps = [a for a in aps if not np.isnan(a)]
        apm = float(np.mean(aps)) if aps else np.nan
        r1, ranks, top2 = [], [], []
        for i, f in enumerate(fb):
            ti = [nodes.index(t) for t in f if t in nodes]
            if not ti:
                continue
            top1 = int(np.argmax(Pb[i]))
            r1.append(1.0 if top1 in ti else 0.0)
            rk = min(1 + int((Pb[i] > Pb[i][j]).sum()) for j in ti)
            ranks.append(1.0 / rk)
            top2.append(1.0 if rk <= 2 else 0.0)
        return (apm,
                float(np.mean(ranks)) if ranks else np.nan,
                float(np.mean(top2)) if top2 else np.nan,
                float(np.mean(r1)) if r1 else np.nan)

    draws = {"ap_macro": [], "mrr": [], "top2": [], "recall_at_1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        apm, mrr, top2, r1 = _metrics(idx)
        draws["ap_macro"].append(apm); draws["mrr"].append(mrr)
        draws["top2"].append(top2); draws["recall_at_1"].append(r1)
    out = {}
    for k, vals in draws.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
            if len(v) else (float("nan"), float("nan"))
    return out
```

- [ ] **Step 6: Run to verify pass + full engine file**

`cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v` (all green).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): PR-AUC + recall@k + bootstrap CIs in evaluate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Leakage-free split — train-only prune + vocab, test rolled onto the train DAG

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py` (`assemble_from_events`)
- Modify: `analysis/cloud/_case_finding_cache.py` (bump `v` to 2)
- Test: `charmpheno/tests/test_case_finding_assembly.py`

**Interfaces:** `assemble_from_events(...)` signature unchanged; behavior now **splits patients first**, prunes the DAG on TRAIN patient counts, fits the vocabulary on TRAIN docs (frozen for test), and rolls test attestations onto the train-derived (pruned) DAG. `ledger` gains `test_coarsening_rate` (fraction of test foreground docs whose most-specific attested node was pruned by train and rolled up) and `test_fg_docs`. Bundle shape unchanged.

The reason: pruning + vocabulary must not see held-out patients. Previously both were fit on train+test, so test labels influenced which nodes survived and test docs influenced vocab selection — mild transductive leakage.

- [ ] **Step 1: Write failing tests**

```python
# append to charmpheno/tests/test_case_finding_assembly.py
def test_assemble_prunes_and_fits_vocab_on_train_only(spark):
    """A node attested ONLY by a held-out (test) patient must be pruned (train
    count 0 < min_n) and that test doc's frontier rolled up; a token appearing
    ONLY in test docs must be absent from the (train-fit) vocab."""
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    import datetime as dt

    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300, 400],
                                 names={100: "dm", 200: "T2", 300: "T1", 400: "T2r"})
    # Choose person ids so the deterministic split puts some on each side; make
    # node 400 (T2r) attested by exactly one patient, and token 555 appear only
    # in that same patient's doc. Assert 400 pruned + 555 not in vocab, AND
    # verify which side that patient landed on drives the expectation.
    rows = []
    for pid in range(40):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    for pid in range(100, 140):
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])
    bundle = assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=2, holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1, n_bg=2, tpn=1)
    # train/test still person-disjoint; ledger reports test coarsening.
    tr = {r["person_id"] for r in bundle.train_df.collect()}
    te = {r["person_id"] for r in bundle.test_df.collect()}
    assert tr and te and (tr & te == set())
    assert "test_coarsening_rate" in bundle.ledger
    assert "test_fg_docs" in bundle.ledger
    # DagLayout loads; K emergent from TRAIN-surviving nodes.
    from spark_vi.models.topic.dag_placement import DagLayout
    DagLayout(bundle.parent_int, n_bg=2, tpn=1)
```

(The implementer refines the fixture so a specific node/token is provably test-only after the deterministic split — inspect `split_train_test` assignment with the fixed salt, then assert that node's absence from `cid2int`/that token's absence from `vocab_map`.)

- [ ] **Step 2: Run to verify fail**

`.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -k "train_only" -v` → FAIL (`test_coarsening_rate` absent; old flow prunes on all patients).

- [ ] **Step 3: Restructure `assemble_from_events`**

Replace the body (keep signature + imports; add `most_specific_cids` is already in-module). New flow:

```python
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import (
        prune_by_attestation, pruning_ledger, _nearest_surviving_ancestors,
    )
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from spark_vi.models.topic.dag_placement import DagLayout

    node_cids = before_dag.nodes()
    # 1) split PATIENTS first (events carry person_id); nothing downstream sees
    #    the other side. Same deterministic hash as the doc-level split.
    train_events, test_events = split_train_test(
        events_df, holdout_frac=holdout_frac, split_salt=split_salt)

    # 2) prune the DAG on TRAIN patient counts only.
    train_att = doc_attested_nodes(train_events, node_cids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_events, node_cids, doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(train_att)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        # 3) ledger: TRAIN coarsening (as before) + TEST coarsening (new).
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

        # 4) frontiers for both sides via the TRAIN DAG (test attestations to
        #    pruned nodes roll up to nearest surviving ancestor).
        train_fr = attach_frontiers(train_att, before_dag, keep, cid2int, lay)
        test_fr = attach_frontiers(test_att, before_dag, keep, cid2int, lay)

        # 5) vocab fit on TRAIN; frozen for TEST.
        train_bow, vocab_map = to_bow_dataframe(
            train_events, doc_spec=doc_spec, vocab_size=vocab_size,
            min_df=min_df, min_patient_count=min_patient_count)
        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid
        test_bow, _ = to_bow_dataframe(test_events, doc_spec=doc_spec, vocab=vocab_list)

        def _label(bow, fr):
            return (bow.join(fr.select("doc_id", "frontier", "source_cohort"),
                             on="doc_id", how="left")
                    .withColumn("frontier",
                                F.coalesce(F.col("frontier"),
                                           F.array().cast("array<bigint>"))))
        train_df = _label(train_bow, train_fr)
        test_df = _label(test_bow, test_fr)

        # 6) leakage strip (test_only): drop DAG-node type codes from TEST features.
        drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)

        return CaseFindingBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_map=vocab_map,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()
```

(Delete the old single-pass `attested`/`labeled`/`split_train_test(labeled…)` body. `_nearest_surviving_ancestors` import is unused here — drop it unless referenced; keep the import list minimal.)

- [ ] **Step 4: Bump the cache version**

In `analysis/cloud/_case_finding_cache.py` `compute_bundle_cache_key`, change `"v": 1` to `"v": 2` (the split-first semantics change the bundle; force a rebuild independent of the source-hash).

- [ ] **Step 5: Run to verify pass + full assembly suite**

`.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py -v` (all green; the pre-existing end-to-end test still passes with the new flow — it asserts schema + strip + frontier semantics that hold under split-first).

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/omop/case_finding_assembly.py analysis/cloud/_case_finding_cache.py charmpheno/tests/test_case_finding_assembly.py
git commit -m "fix(case-finding): split patients first; prune+vocab on train only; report test coarsening

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `strip_mode` ablation knob + driver/config wiring for the new metrics

**Files:**
- Modify: `charmpheno/charmpheno/omop/case_finding_assembly.py` (`assemble_from_events`, `assemble_case_finding_corpus` add `strip_mode`)
- Modify: `analysis/cloud/_case_finding_cache.py` (add `strip_mode` to key + whitelist)
- Modify: `analysis/cloud/dag_placement_cloud.py` (`--strip-mode` arg; pass through; print + manifest the new metrics + `test_coarsening_rate`)
- Modify: `scripts/run_experiment.py` (`build_dag_placement_args` emits `--strip-mode`)
- Modify: `experiments/defaults/_base.yaml` (`strip_mode: test_only`)
- Create: `docs/experiments/0054-dag-placement-diabetes-strip-both.md`
- Test: `charmpheno/tests/test_case_finding_assembly.py`, `analysis/cloud/tests/test_dag_placement_cloud.py`, `scripts/tests/test_run_experiment_dag_placement.py`

**Interfaces:** `assemble_from_events(..., strip_mode="test_only")` and `assemble_case_finding_corpus(..., strip_mode="test_only")`; `strip_mode="both"` also strips TRAIN features. `compute_bundle_cache_key(..., strip_mode="test_only")`. The `both` arm is the corpus-level ablation (does train keeping the type codes cause shortcut learning?). The third arm — codes supervise gating but are excluded from β sufficient statistics — is a deeper GatedOnlineLDA change and is DEFERRED (documented).

- [ ] **Step 1: Write failing tests**

```python
# charmpheno/tests/test_case_finding_assembly.py — append
def test_strip_mode_both_strips_train_features(spark):
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    import datetime as dt
    edges = [(100, 200), (100, 300)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300],
                                 names={100: "dm", 200: "T2", 300: "T1"})
    rows = []
    for pid in range(40):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    ev = spark.createDataFrame(rows, ["person_id", "concept_id", "source_cohort",
                                      "condition_era_start_date"])
    kw = dict(doc_spec=PatientCohortDocSpec(min_doc_length=0), min_n=2,
              holdout_frac=0.3, split_salt=20260716, vocab_size=100, min_df=1,
              min_patient_count=1, n_bg=2, tpn=1)
    b_test = assemble_from_events(ev, before, strip_mode="test_only", **kw)
    b_both = assemble_from_events(ev, before, strip_mode="both", **kw)
    node200 = b_both.vocab_map.get(200)
    if node200 is not None:
        train_has = any(node200 in set(r["features"].indices.tolist())
                        for r in b_test.train_df.collect())
        train_stripped = all(node200 not in set(r["features"].indices.tolist())
                             for r in b_both.train_df.collect())
        assert train_has and train_stripped
```

```python
# scripts/tests/test_run_experiment_dag_placement.py — append to the existing shape test
def test_build_dag_placement_args_includes_strip_mode(monkeypatch):
    import importlib
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "min_n": 50,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "strip_mode": "both"}
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--strip-mode") + 1] == "both"
```

(Also extend `analysis/cloud/tests/test_dag_placement_cloud.py::test_parse_args_surface` to assert `--strip-mode both` parses to `a.strip_mode == "both"`.)

- [ ] **Step 2: Run to verify fail**

Run the three `-k` selections; all FAIL (no `strip_mode`).

- [ ] **Step 3: Add `strip_mode` to the assembly**

In `assemble_from_events`, add `strip_mode="test_only"` kwarg. Replace step 6 with:

```python
        drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)
        if strip_mode == "both":
            train_df = strip_test_features(train_df, drop_idxs)
        elif strip_mode != "test_only":
            raise ValueError(
                f"strip_mode must be 'test_only' or 'both', got {strip_mode!r}")
```

In `assemble_case_finding_corpus`, add `strip_mode="test_only"` kwarg and thread it into the `assemble_from_events(...)` call.

- [ ] **Step 4: Thread `strip_mode` through the cache key**

In `_case_finding_cache.compute_bundle_cache_key`, add `strip_mode="test_only"` param and include `"strip_mode": strip_mode` in the payload. In `load_or_build_case_finding_bundle`, add `"strip_mode"` to the `key_params` whitelist.

- [ ] **Step 5: Driver — `--strip-mode` + print/manifest the new metrics**

In `dag_placement_cloud.py`: add `p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")`; pass `strip_mode=args.strip_mode` into `load_or_build_case_finding_bundle(...)`. Extend the metrics print line to include `ap_macro`, `ap_prevalence_weighted`, `recall_at_k`, and `bundle.ledger.get("test_coarsening_rate")`; the manifest already serializes the whole `metrics` dict, so `node_ap`/`ci`/etc. are captured automatically — also add `"strip_mode": args.strip_mode` to the manifest.

- [ ] **Step 6: `build_dag_placement_args` — `--strip-mode`**

Append to the args list: `"--strip-mode", str(effective.get("strip_mode", "test_only"))`.

- [ ] **Step 7: Config — `_base.yaml` + exp 0054**

Add `strip_mode: test_only` to the `_base.yaml` dag_placement block. Create `docs/experiments/0054-dag-placement-diabetes-strip-both.md` — frontmatter copied from 0053 (spectral) with `id: 54`, `slug: dag-placement-diabetes-strip-both`, `init: spectral`, plus `strip_mode: both`; body explains it is the leakage-strip ablation arm against 0053 (same init, train codes also stripped) testing whether keeping type codes in training inflates placement via shortcut learning.

- [ ] **Step 8: Run all affected suites + commit**

```bash
.venv/bin/python -m pytest charmpheno/tests/test_case_finding_assembly.py analysis/cloud/tests/test_dag_placement_cloud.py scripts/tests/test_run_experiment_dag_placement.py scripts/tests/test_dag_placement_config.py -v
git add charmpheno/charmpheno/omop/case_finding_assembly.py analysis/cloud/_case_finding_cache.py analysis/cloud/dag_placement_cloud.py analysis/cloud/tests/test_dag_placement_cloud.py scripts/run_experiment.py scripts/tests/test_run_experiment_dag_placement.py experiments/defaults/_base.yaml docs/experiments/0054-dag-placement-diabetes-strip-both.md charmpheno/tests/test_case_finding_assembly.py
git commit -m "feat(dag-placement): strip_mode ablation knob + surface PR/recall/coarsening in driver

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Coverage:** tie fix = T1; PR-AUC/recall@k/CIs = T2; leakage-free split (train-only prune+vocab, test roll-up, coarsening report) = T3; strip ablation knob + driver/config surfacing = T4. Baselines are explicitly out (user). The β-exclusion (two-channel) ablation arm is deferred with a note.

**Placeholder scan:** none — every step has runnable code. The two "implementer refines the fixture" notes (T3 Step 1, T4 Step 1) are about pinning which node/token lands test-only under the deterministic split; the assertions are concrete, only the fixture-id choice is left to inspection.

**Type consistency:** `evaluate` return keeps its 7 existing keys and adds `node_ap`/`ap_*`/`recall_at_k`/`ci`; `ci` is `{metric: (lo, hi)}`. `strip_mode` is a `str` in {"test_only","both"} everywhere (assembly, cache key, driver choices, config, args). Cache `v=2` + the new `strip_mode` key both force correct invalidation. Bundle shape unchanged; `ledger` gains `test_coarsening_rate`/`test_fg_docs` (additive).
