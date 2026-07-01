# Gated CTM Correlation Reporting (Plan 1 of 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface an honest topic-correlation heatmap in the dashboard, sourced from the fitted logistic-normal Σ with a support-keyed identified mask, and retire the full-matrix conditioning diagnostics that optimized a reporting artifact.

**Architecture:** The domain-agnostic engine (`spark-vi`) persists the per-pair document-support matrix N and gains a correlation-with-mask helper; it drops the `sigma_cond` / `max_offdiag` conditioning diagnostics. The OMOP layer (`charmpheno`) exports `correlation.json`. The dashboard (`dashboard/`) renders one heatmap panel with NA cells greyed. The model fit mechanics are unchanged; `pd_complete` stays.

**Tech Stack:** Python / NumPy (`spark-vi`), PySpark + Python (`charmpheno`), TypeScript / Vite / Vitest (`dashboard/`).

**Spec:** [docs/superpowers/specs/2026-07-01-gated-ctm-correlation-reporting-design.md](../specs/2026-07-01-gated-ctm-correlation-reporting-design.md) (Components 2, 3, 4, the reporting parts of 6, and the 0032/0033/0025 amendments of 5). Component 1 (multi-membership representation) and the new ADR are Plan 2.

## Global Constraints

- `spark-vi` stays domain-agnostic: integer topic indices only; no OMOP/EHR vocabulary or cohort names in the library. N, R, and the mask are topic-index-keyed.
- No LaTeX in docs or docstrings; plain text and Unicode Greek (Σ, η, θ, μ, Γ, λ, ν).
- Any literature-derived method/default/constant cites its source in the docstring; otherwise label it a heuristic.
- No personal information in committed artifacts.
- Code references in prose use markdown links (`[name](path#Lstart-Lend)`).
- Row-level document output that prints identifier columns hashes them (SHA-256-truncated); aggregates and probabilities may print raw. (No such output in this plan.)
- TDD per task; commit after each green task.

## Under-test environment note

The engine test env is provisioned (`spark-vi/` importable, `pytest` runs). Run engine tests from `spark-vi/`. Run `charmpheno` tests from `charmpheno/`. Run dashboard tests from `dashboard/` with `npm test`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| [spark-vi/spark_vi/models/topic/stm.py](../../spark-vi/spark_vi/models/topic/stm.py) | persist `n_pairs`; drop cond diagnostics | 1, 3 |
| [spark-vi/spark_vi/models/topic/_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py) | `topic_correlation_identified` helper | 2 |
| spark-vi/tests/test_stm_full_sigma.py | helper + persistence tests | 1, 2 |
| spark-vi/tests/test_stm_pd_completion_conditioning.py (exists) | recovery-invariant-to-cond test | 4 |
| spark-vi/tests/test_mllib_stm_persistence.py | n_pairs round-trip | 1 |
| charmpheno/charmpheno/export/correlation.py (new) | `build_correlation_json` | 5 |
| charmpheno/tests/test_correlation_export.py (new) | export tests | 5 |
| [analysis/local/build_dashboard.py](../../analysis/local/build_dashboard.py) + cloud builder | write `correlation.json` | 6 |
| [dashboard/src/lib/bundle.ts](../../dashboard/src/lib/bundle.ts) + types | load `correlation.json` | 7 |
| dashboard CorrelationHeatmap component + panel | render heatmap, grey NA | 8 |
| docs/insights/0032-*, docs/decisions/0033-*, docs/experiments/0025-* | amendments (LAST) | 9 |

---

## Task 1: Persist per-pair document-support N in the model

**Files:**
- Modify: [spark-vi/spark_vi/models/topic/stm.py](../../spark-vi/spark_vi/models/topic/stm.py) — `initialize_global` (seed `n_pairs`), `update_global` (stash `n_pairs`).
- Test: spark-vi/tests/test_stm_full_sigma.py, spark-vi/tests/test_mllib_stm_persistence.py

**Interfaces:**
- Consumes: `update_global`'s `target_stats["n_pairs_stat"]` (K×K float, per-pair doc support, [stm.py:710](../../spark-vi/spark_vi/models/topic/stm.py#L710)).
- Produces: `global_params["n_pairs"]` — a K×K float matrix of per-pair document support in the final M-step, persisted as `params/n_pairs.npy` and round-tripped by `STMModel.load`. Consumed by Task 2's helper and Task 5's export.

Note: for a full-batch fit (the finalization config: `max_iter` passes over the full corpus, no minibatch) the final M-step's `n_pairs_stat` equals the whole-corpus per-pair support, which is the reporting quantity. (Minibatch fits would capture the last batch only; out of scope here, noted for Plan 2.)

- [ ] **Step 1: Write the failing test** (append to spark-vi/tests/test_stm_full_sigma.py)

```python
def test_update_global_stashes_n_pairs_support():
    """global_params carries the final M-step per-pair support N so reporting
    can build the identified mask without a re-pass."""
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))  # K=4
    m = OnlineSTM(K=4, vocab_size=6, P=1, reference_topic=False,
                  topic_blocks=part, min_pair_support=10)
    gp = m.initialize_global(None)
    assert "n_pairs" in gp and gp["n_pairs"].shape == (4, 4)  # seeded
    gp["Gamma"] = np.zeros((1, 4))
    rng = np.random.default_rng(7)
    docs = _gated_multigroup_docs(rng, n_comorbid=300)  # module-level helper in this file
    gp = m.update_global(gp, m.local_update(docs, gp), 1.0)
    N = gp["n_pairs"]
    a = part.block_indices("A")[0]; b = part.block_indices("B")[0]
    assert N[a, b] == 300          # cross-foreground support == comorbid docs
    assert N[0, 0] >= 800          # background seen by all docs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py::test_update_global_stashes_n_pairs_support -q`
Expected: FAIL with `KeyError: 'n_pairs'` (not yet stashed).

- [ ] **Step 3: Seed `n_pairs` in `initialize_global`**

In both return dicts of `initialize_global` ([stm.py:442-462](../../spark-vi/spark_vi/models/topic/stm.py#L442-L462)), add the key alongside `"Sigma"`:

```python
"n_pairs": np.zeros((self.K, self.K), dtype=np.float64),
```

- [ ] **Step 4: Stash `n_pairs` in `update_global`**

In `update_global`'s return dict ([stm.py:729-734](../../spark-vi/spark_vi/models/topic/stm.py#L729-L734)), add:

```python
"n_pairs": np.asarray(target_stats["n_pairs_stat"], dtype=np.float64),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py::test_update_global_stashes_n_pairs_support -q`
Expected: PASS.

- [ ] **Step 6: Write the persistence round-trip test** (append to spark-vi/tests/test_mllib_stm_persistence.py)

```python
def test_n_pairs_round_trips_through_save_load(tmp_path):
    """params/n_pairs.npy persists and STMModel.load restores it."""
    import numpy as np
    from spark_vi.mllib.topic.stm import STMModel
    gp = {"lambda": np.ones((3, 5)), "eta": np.array(0.01),
          "Gamma": np.zeros((1, 3)), "Sigma": np.eye(3),
          "n_pairs": np.array([[9, 4, 0], [4, 9, 2], [0, 2, 9]], dtype=float)}
    model = STMModel(global_params=gp, metadata={}, model_spec=None,
                     covariate_names=[], n_iterations=1)
    model.save(tmp_path)
    loaded = STMModel.load(tmp_path)
    assert np.array_equal(loaded.global_params["n_pairs"], gp["n_pairs"])
```

- [ ] **Step 7: Run it — verify it passes without shim changes**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm_persistence.py::test_n_pairs_round_trips_through_save_load -q`
Expected: PASS — `save_result` writes one `params/<key>.npy` per `global_params` key and `load_result` restores all keys ([mllib/topic/stm.py:441-503](../../spark-vi/spark_vi/mllib/topic/stm.py#L441-L503)), so `n_pairs` round-trips with no shim change. If it FAILS because `model_spec=None` can't pickle, pass a trivial object instead; the assertion under test is the `n_pairs` array.

- [ ] **Step 8: Run the STM suites to confirm no regression from the new key**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py tests/test_stm_contract.py tests/test_mllib_stm.py -q`
Expected: PASS (the new `global_params` key is read only where needed; contract math untouched).

- [ ] **Step 9: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py spark-vi/tests/test_mllib_stm_persistence.py
git commit -m "feat(spark-vi): persist per-pair support n_pairs in global_params for correlation reporting"
```

---

## Task 2: Correlation-with-identified-mask engine helper

**Files:**
- Modify: [spark-vi/spark_vi/models/topic/_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py) — add `topic_correlation_identified`.
- Test: spark-vi/tests/test_stm_full_sigma.py

**Interfaces:**
- Consumes: `Sigma` (K×K), `n_pairs` (K×K, from Task 1), `min_pair_support` (int, the model's floor).
- Produces: `topic_correlation_identified(Sigma, n_pairs, min_pair_support) -> tuple[np.ndarray, np.ndarray]` returning `(R, identified)`. `R` is `topic_correlation(Sigma)` with `np.nan` on unidentified off-diagonal cells; `identified` is a K×K bool mask (`n_pairs >= min_pair_support`, diagonal forced True). Consumed by Task 5's export.

- [ ] **Step 1: Write the failing test** (append to spark-vi/tests/test_stm_full_sigma.py)

```python
def test_topic_correlation_identified_masks_thin_pairs():
    from spark_vi.models.topic._linalg import (
        topic_correlation, topic_correlation_identified)
    Sigma = np.array([[4.0, 2.0, 1.2],
                      [2.0, 9.0, 0.6],
                      [1.2, 0.6, 1.0]])
    N = np.array([[100, 30, 2],     # pair (0,2) thin: N=2
                  [30, 100, 40],
                  [2, 40, 100]], dtype=float)
    R, ident = topic_correlation_identified(Sigma, N, min_pair_support=10)
    R_full = topic_correlation(Sigma)
    assert ident[0, 1] and ident[1, 2]              # supported
    assert not ident[0, 2] and not ident[2, 0]      # thin -> unidentified
    assert bool(ident[0, 0]) and bool(ident[1, 1])  # diagonal always identified
    assert np.isnan(R[0, 2]) and np.isnan(R[2, 0])  # thin -> NA
    assert np.isclose(R[0, 1], R_full[0, 1])        # supported cells == full R
    assert np.isclose(R[0, 0], 1.0)                 # unit diagonal preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py::test_topic_correlation_identified_masks_thin_pairs -q`
Expected: FAIL with `ImportError: cannot import name 'topic_correlation_identified'`.

- [ ] **Step 3: Implement the helper** (add to [_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py) after `topic_correlation`)

```python
def topic_correlation_identified(Sigma, n_pairs, min_pair_support):
    """Logistic-normal correlation R (topic_correlation) with an identified mask.

    A cell (i,j) is identified iff n_pairs[i,j] >= min_pair_support — the same
    document-support floor the M-step uses to decide estimated-vs-completed
    (stm.py). Unidentified OFF-diagonal cells are set to NaN in R (no joint data
    supports that correlation); the diagonal is always identified (unit value).
    Domain-agnostic: topic indices only.

    Returns (R, identified): R is (K,K) float with NaN on unidentified off-diag
    cells; identified is (K,K) bool.
    """
    R = topic_correlation(Sigma)
    identified = np.asarray(n_pairs) >= float(min_pair_support)
    identified = identified | identified.T          # symmetric support
    np.fill_diagonal(identified, True)              # diagonal always identified
    mask_na = ~identified
    np.fill_diagonal(mask_na, False)                # never NaN the unit diagonal
    R = R.copy()
    R[mask_na] = np.nan
    return R, identified
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py::test_topic_correlation_identified_masks_thin_pairs -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/_linalg.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(spark-vi): topic_correlation_identified — logistic-normal R with support-keyed NA mask"
```

---

## Task 3: Remove the full-matrix conditioning diagnostics

**Files:**
- Modify: [spark-vi/spark_vi/models/topic/stm.py](../../spark-vi/spark_vi/models/topic/stm.py) — `iteration_summary` ([804](../../spark-vi/spark_vi/models/topic/stm.py#L804)), `iteration_diagnostics` ([837](../../spark-vi/spark_vi/models/topic/stm.py#L837)).
- Modify: any eval-driver conditioning readout ([analysis/cloud/eval_coherence_cloud.py](../../analysis/cloud/eval_coherence_cloud.py), added in commit 52f5ba5).
- Test: spark-vi/tests/test_mllib_stm.py (or wherever `iteration_diagnostics` keys are asserted).

**Interfaces:**
- Produces: `iteration_diagnostics` returns a dict WITHOUT `sigma_cond` and `max_abs_offdiag_corr`. It KEEPS `sigma_eig_min`, `sigma_eig_max` (genuine within-fit variance-scale signals), `Gamma`, `Sigma`, and `topic_block_labels`. `iteration_summary` drops the `cond=` and `Σ_corr[max_offdiag=...]` fields.

Rationale (spec Component 2): `sigma_cond` and `max_abs_offdiag_corr` are computed over the FULL assembled Σ, whose cross-block entries never enter the fit (only within-allowed-set marginal sub-blocks do). They measured a reporting artifact; topic recovery is invariant to them (Task 4).

- [ ] **Step 1: Update the failing test first** (find the assertion in spark-vi/tests/test_mllib_stm.py)

Run: `cd spark-vi && grep -rn "sigma_cond\|max_abs_offdiag_corr" tests/`
For each test asserting these keys exist, change it to assert they are ABSENT and the kept keys remain:

```python
def test_iteration_diagnostics_drops_full_matrix_conditioning():
    from spark_vi.models.topic.stm import OnlineSTM
    m = OnlineSTM(K=3, vocab_size=6, P=1, reference_topic=False)
    gp = m.initialize_global(None)
    d = m.iteration_diagnostics(gp)
    assert "sigma_cond" not in d and "max_abs_offdiag_corr" not in d
    assert "sigma_eig_min" in d and "sigma_eig_max" in d   # kept
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm.py::test_iteration_diagnostics_drops_full_matrix_conditioning -q`
Expected: FAIL (keys still present).

- [ ] **Step 3: Strip the diagnostics from `iteration_diagnostics`**

In `iteration_diagnostics` ([stm.py:837-862](../../spark-vi/spark_vi/models/topic/stm.py#L837-L862)) remove the `sigma_cond` and `max_abs_offdiag_corr` computations and dict entries, and the now-unused `topic_correlation` import and `K` local. Keep `sigma_eig_min` / `sigma_eig_max`. Resulting dict core:

```python
Sigma = np.asarray(global_params["Sigma"])
eigs = np.linalg.eigvalsh(Sigma)
diag = {
    "Gamma": np.asarray(global_params["Gamma"]),
    "Sigma": Sigma,
    "sigma_eig_min": float(eigs.min()),
    "sigma_eig_max": float(eigs.max()),
}
```

- [ ] **Step 4: Strip the fields from `iteration_summary`**

In `iteration_summary` ([stm.py:804-835](../../spark-vi/spark_vi/models/topic/stm.py#L804-L835)) remove the `cond` computation, the `max_offdiag` computation, the unused `topic_correlation` import, and drop the `cond=...` and `Σ_corr[max_offdiag=...]` fragments from the `base` f-string. Keep `Σ_eig[min=... max=...]` and `Σ_var[...]`.

- [ ] **Step 5: Remove the eval-driver conditioning readout**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && grep -rn "sigma_cond\|conditioning\|max_abs_offdiag" analysis/cloud/eval_coherence_cloud.py analysis/local/`
Remove any print/emit of `sigma_cond` / conditioning from the eval readout (the commit-52f5ba5 addition). If a Σ readout is kept, keep only `sigma_eig_min`/`max`, labelled a reporting statistic.

- [ ] **Step 6: Run the affected suites**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm.py tests/test_stm_integration.py -q`
Expected: PASS (updated assertions green; nothing else referenced the removed keys).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_mllib_stm.py analysis/cloud/eval_coherence_cloud.py
git commit -m "refactor(spark-vi): drop full-matrix sigma_cond/max_offdiag diagnostics (reporting artifact, not fit health)"
```

---

## Task 4: "Recovery is invariant to full-Σ condition number" — the finding, on the record

**Files:**
- Modify: [spark-vi/tests/test_stm_pd_completion_conditioning.py](../../spark-vi/tests/test_stm_pd_completion_conditioning.py) (exists; add the test).
- Uses: [spark-vi/tests/_stm_synth.py](../../spark-vi/tests/_stm_synth.py) helpers `synthetic_gated_corpus_overlap`, `fit_stm`, `planted_recovery` (exist).

**Interfaces:** none produced; this is a characterization test that pins the central finding — the gated fit's topic recovery does not depend on the full assembled Σ condition number, because the fit uses only within-allowed-set marginal sub-blocks.

- [ ] **Step 1: Write the test**

```python
def test_recovery_invariant_to_full_sigma_condition_number():
    """The full-matrix Sigma condition number is a reporting artifact: topic
    recovery holds across corpora whose assembled-Sigma cond spans orders of
    magnitude, because the gated E-step only ever inverts within-allowed-set
    marginal sub-blocks (safe_inverse-repaired per doc), never the full matrix.
    Guards the Component-2 decision to drop the conditioning diagnostics."""
    import numpy as np
    from _stm_synth import (synthetic_gated_corpus_overlap, fit_stm,
                            planted_recovery)

    def cond(M):
        w = np.linalg.eigvalsh(0.5 * (M + M.T))
        return w.max() / w.min() if w.min() > 0 else float("inf")

    conds, recs = [], []
    for seed in range(4):
        docs, planted, part = synthetic_gated_corpus_overlap(
            groups=("A", "B"), fg_per_group=2, bg_k=4, V=80, D=150,
            doc_len=70, bg_frac=0.4, shared_frac=0.5, seed=seed)
        gp = fit_stm(docs, K=part.K, V=80, sigma_init=1.0, n_iter=22,
                     seed=42, partition=part, reference_topic=False)
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        conds.append(cond(gp["Sigma"]))
        recs.append(planted_recovery(beta, planted))

    # Some seeds land near-singular (cond huge), others well-conditioned...
    assert max(conds) / max(min(conds), 1.0) >= 1e3, conds   # cond spans >=3 orders
    # ...yet recovery is stable across all of them (does not track cond).
    assert min(recs) >= max(recs) - 1, recs                  # within 1 topic
```

- [ ] **Step 2: Run it**

Run: `cd spark-vi && python -m pytest tests/test_stm_pd_completion_conditioning.py::test_recovery_invariant_to_full_sigma_condition_number -q`
Expected: PASS. (If the cond spread across seeds is under 1e3 on this machine, widen to `seed in range(6)` — the bg_frac=0.4 regime straddles the completability boundary, so more seeds guarantee the spread; do not weaken the recovery assertion.)

- [ ] **Step 3: Run the whole conditioning file green**

Run: `cd spark-vi && python -m pytest tests/test_stm_pd_completion_conditioning.py -q`
Expected: PASS (5 tests: the 4 Layer-1 tests already present + this one).

- [ ] **Step 4: Commit**

```bash
git add spark-vi/tests/test_stm_pd_completion_conditioning.py
git commit -m "test(spark-vi): recovery is invariant to full-Sigma condition number (reporting-artifact finding on record)"
```

---

## Task 5: `correlation.json` export in charmpheno

**Files:**
- Create: charmpheno/charmpheno/export/correlation.py
- Test: charmpheno/tests/test_correlation_export.py
- Pattern reference: [charmpheno/charmpheno/export/gating.py](../../charmpheno/charmpheno/export/gating.py)

**Interfaces:**
- Consumes: `R` and `identified` (Task 2 helper output, K×K over ORIGINAL topic ids), `support` (`n_pairs`, Task 1), `partition` (TopicBlockPartition), `kept_topic_ids` (list[int], the dashboard's kept topics from gating).
- Produces: `build_correlation_json(R, identified, support, partition, kept_topic_ids) -> dict` with keys: `topic_order` (kept ids in block order, reference topic excluded), `block_labels` (per-kept-topic label), `R` (list-of-lists over kept order, `null` for NaN/unidentified), `identified` (bool list-of-lists), `support` (int list-of-lists).

- [ ] **Step 1: Write the failing test** (charmpheno/tests/test_correlation_export.py)

```python
import numpy as np
from charmpheno.export.correlation import build_correlation_json
from spark_vi.models.topic.partition import TopicBlockPartition


def test_build_correlation_json_orders_blocks_and_nulls_unidentified():
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))  # K=4, ids 0..3
    R = np.array([[1.0, 0.3, 0.2, np.nan],
                  [0.3, 1.0, 0.1, np.nan],
                  [0.2, 0.1, 1.0, np.nan],
                  [np.nan, np.nan, np.nan, 1.0]])
    identified = np.array([[1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [0, 0, 0, 1]], dtype=bool)
    support = np.array([[300, 300, 150, 0],
                        [300, 300, 150, 0],
                        [150, 150, 150, 0],
                        [0, 0, 0, 150]], dtype=float)
    kept = [0, 1, 2, 3]
    out = build_correlation_json(R, identified, support, part, kept)
    assert out["topic_order"] == [0, 1, 2, 3]
    assert out["block_labels"] == ["background", "background", "A", "B"]
    # cross-foreground (A id=2, B id=3) is unidentified -> null in R
    assert out["R"][2][3] is None and out["R"][3][2] is None
    assert out["identified"][2][3] is False
    assert out["support"][2][3] == 0
    assert out["R"][0][1] == 0.3            # identified cell preserved
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd charmpheno && python -m pytest tests/test_correlation_export.py -q`
Expected: FAIL with `ModuleNotFoundError: charmpheno.export.correlation`.

- [ ] **Step 3: Implement the export**

```python
"""Correlation bundle export: logistic-normal topic correlation R + identified
mask, over the dashboard's kept topics in block order. Unidentified cells (no
joint document support) serialize as null so the frontend can grey them.

The identifiability floor is the model's min_pair_support: a cell is identified
iff the two topics were co-realized in >= min_pair_support documents (Blei &
Lafferty 2007 logistic-normal correlation; identifiability by support).
"""
from __future__ import annotations

import math


def _cell(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def build_correlation_json(R, identified, support, partition, kept_topic_ids):
    """correlation.json over kept topics in block order; null for unidentified R."""
    labels = partition.topic_labels()                 # length K, by original id
    order = [i for i in kept_topic_ids]               # already block-ordered upstream
    block_labels = [labels[i] for i in order]
    R_out, id_out, sup_out = [], [], []
    for i in order:
        R_out.append([_cell(R[i][j]) for j in order])
        id_out.append([bool(identified[i][j]) for j in order])
        sup_out.append([int(support[i][j]) for j in order])
    return {
        "topic_order": [int(i) for i in order],
        "block_labels": block_labels,
        "R": R_out,
        "identified": id_out,
        "support": sup_out,
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd charmpheno && python -m pytest tests/test_correlation_export.py -q`
Expected: PASS.

- [ ] **Step 5: Add the split-representation NA test** (append to the same test file)

```python
def test_cross_foreground_all_null_under_split_representation():
    """Under today's split (single-group) corpus, no doc co-realizes an A and a
    B foreground topic, so the whole cross-foreground block is unidentified."""
    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A", 1), ("B", 1)))  # ids: bg=0,A=1,B=2
    R = np.array([[1.0, 0.4, 0.5], [0.4, 1.0, np.nan], [0.5, np.nan, 1.0]])
    identified = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool)
    support = np.array([[300, 200, 100], [200, 200, 0], [100, 0, 100]], dtype=float)
    out = build_correlation_json(R, identified, support, part, [0, 1, 2])
    assert out["R"][1][2] is None and out["identified"][1][2] is False
```

- [ ] **Step 6: Run both tests**

Run: `cd charmpheno && python -m pytest tests/test_correlation_export.py -q`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add charmpheno/charmpheno/export/correlation.py charmpheno/tests/test_correlation_export.py
git commit -m "feat(charmpheno): correlation.json export — logistic-normal R + support-keyed NA mask"
```

---

## Task 6: Wire `correlation.json` into the dashboard build

**Files:**
- Modify: [analysis/local/build_dashboard.py](../../analysis/local/build_dashboard.py) (write it next to `gating.json`, [line 201-203](../../analysis/local/build_dashboard.py#L201-L203)).
- Modify: analysis/cloud/build_dashboard_cloud.py (same wiring on the cloud path).
- Test: charmpheno/tests/test_correlation_export.py (integration-style, using a tiny saved model) or a build_dashboard test if one exists.

**Interfaces:**
- Consumes: the loaded `STMModel` (`global_params["Sigma"]`, `global_params["n_pairs"]`, `min_pair_support` from metadata), `partition`, `kept_ids` (already computed for `gating.json`).
- Produces: `<out_dir>/correlation.json`.

- [ ] **Step 1: Write the failing integration test** (append to charmpheno/tests/test_correlation_export.py)

```python
def test_build_dashboard_writes_correlation_json(tmp_path):
    """The dashboard build emits correlation.json from a saved STM model."""
    import json, numpy as np
    from spark_vi.models.topic._linalg import topic_correlation_identified
    from spark_vi.models.topic.partition import TopicBlockPartition
    # Minimal stand-in for the wiring: Sigma + n_pairs + partition -> json file.
    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A", 1), ("B", 1)))
    Sigma = np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 0.0], [1.0, 0.0, 4.0]])
    N = np.array([[300, 200, 0], [200, 200, 0], [0, 0, 100]], dtype=float)
    R, ident = topic_correlation_identified(Sigma, N, min_pair_support=10)
    from charmpheno.export.correlation import build_correlation_json
    out = build_correlation_json(R, ident, N, part, [0, 1, 2])
    (tmp_path / "correlation.json").write_text(json.dumps(out))
    loaded = json.loads((tmp_path / "correlation.json").read_text())
    assert loaded["R"][1][2] is None                # cross-foreground NA
    assert loaded["block_labels"] == ["background", "A", "B"]
```

- [ ] **Step 2: Run it to verify it fails / passes**

Run: `cd charmpheno && python -m pytest tests/test_correlation_export.py::test_build_dashboard_writes_correlation_json -q`
Expected: PASS immediately (it exercises the composed helpers). This test pins the wiring contract; Steps 3-4 make the real driver produce the same file.

- [ ] **Step 3: Add the write to `build_dashboard.py`**

In the STM block after `gating.json` is written ([build_dashboard.py:201-203](../../analysis/local/build_dashboard.py#L201-L203)), add:

```python
from spark_vi.models.topic._linalg import topic_correlation_identified
from charmpheno.export.correlation import build_correlation_json
Sigma = model.global_params["Sigma"]
n_pairs = model.global_params["n_pairs"]
mps = int(model.metadata.get("min_pair_support", 1))
R, ident = topic_correlation_identified(Sigma, n_pairs, mps)
corr = build_correlation_json(R, ident, n_pairs, partition, kept_ids)
(args.out_dir / "correlation.json").write_text(json.dumps(corr, indent=2))
log.info("STM: wrote correlation.json (topics=%d, min_pair_support=%d)",
         len(kept_ids), mps)
```

(Use the exact names for the loaded model / partition / kept_ids already in scope in that block — confirm by reading lines 172-205; `min_pair_support` is in `model.metadata` via `get_metadata`, [stm.py:464](../../spark-vi/spark_vi/models/topic/stm.py#L464).)

- [ ] **Step 4: Mirror the write in the cloud builder**

Add the same block in analysis/cloud/build_dashboard_cloud.py at its `gating.json` write site (grep `gating.json` there). If `n_pairs` is absent from an older saved model (pre-Task-1), guard: skip `correlation.json` with a warning rather than crash.

- [ ] **Step 5: Run the charmpheno suite**

Run: `cd charmpheno && python -m pytest tests/test_correlation_export.py -q && cd ../spark-vi && python -m pytest tests/test_mllib_stm_persistence.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add analysis/local/build_dashboard.py analysis/cloud/build_dashboard_cloud.py charmpheno/tests/test_correlation_export.py
git commit -m "feat(dashboard-build): emit correlation.json alongside gating.json"
```

---

## Task 7: Load `correlation.json` in the dashboard bundle

**Files:**
- Modify: [dashboard/src/lib/bundle.ts](../../dashboard/src/lib/bundle.ts) (add optional fetch), dashboard/src/lib/types.ts (add `Correlation` + field on `DashboardBundle`).
- Test: dashboard/src/lib/bundle.test.ts (exists).

**Interfaces:**
- Produces: `DashboardBundle.correlation?: Correlation` where `Correlation = { topic_order: number[]; block_labels: string[]; R: (number|null)[][]; identified: boolean[][]; support: number[][] }`.

- [ ] **Step 1: Add the type** (dashboard/src/lib/types.ts)

```typescript
export interface Correlation {
  topic_order: number[]
  block_labels: string[]
  R: (number | null)[][]
  identified: boolean[][]
  support: number[][]
}
```
And add `correlation?: Correlation` to the `DashboardBundle` interface.

- [ ] **Step 2: Write the failing test** (dashboard/src/lib/bundle.test.ts — follow the existing optional-sidecar test pattern for `gating`)

```typescript
it('loads correlation.json as an optional bundle', async () => {
  // mirror the existing gating optional-load test: mock fetch to serve
  // correlation.json, assert bundle.correlation.R and .identified are parsed,
  // and that an absent file leaves bundle.correlation undefined.
})
```
(Match the exact mock/fetch harness the neighboring `gating` test uses in this file.)

- [ ] **Step 3: Run it to verify it fails**

Run: `cd dashboard && npm test -- bundle`
Expected: FAIL (`correlation` not loaded).

- [ ] **Step 4: Add the optional fetch** (in `loadBundle`, [bundle.ts:44-51](../../dashboard/src/lib/bundle.ts#L44-L51))

Add `correlation` to the optional `Promise.all` group and the returned object:

```typescript
const [covariateSchema, covariateEffects, gating, correlation] = await Promise.all([
  fetchJsonOptional<CovariateSchema>(`${base}data/${cohortId}/covariate_schema.json`),
  fetchJsonOptional<CovariateEffects>(`${base}data/${cohortId}/covariate_effects.json`),
  fetchJsonOptional<GatingSpec>(`${base}data/${cohortId}/gating.json`),
  fetchJsonOptional<Correlation>(`${base}data/${cohortId}/correlation.json`),
])
return { model, phenotypes, vocab, corpusStats, covariateSchema, covariateEffects, gating, correlation }
```
(Import `Correlation` in the type import block at the top of the file.)

- [ ] **Step 5: Run it to verify it passes**

Run: `cd dashboard && npm test -- bundle`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/lib/bundle.ts dashboard/src/lib/types.ts dashboard/src/lib/bundle.test.ts
git commit -m "feat(dashboard): load optional correlation.json into the bundle"
```

---

## Task 8: Correlation heatmap panel

**Files:**
- Create: dashboard/src/components/CorrelationHeatmap.tsx (or the project's component idiom — match neighbors under dashboard/src/components/).
- Modify: the dashboard panel/tab registry to add the panel when `bundle.correlation` is present.
- Test: a component test mirroring an existing component test in dashboard/src/.

**Interfaces:**
- Consumes: `bundle.correlation` (Task 7's `Correlation`).
- Renders: a K×K heatmap, topics ordered by `topic_order`, block separators by `block_labels`; diverging color scale for R from −1 to 1; **cells with `R[i][j] === null` (or `identified[i][j] === false`) rendered grey** with a "no joint support" title/tooltip; identified cells show R and `support` on hover.

- [ ] **Step 1: Read two neighbor components** to match the project's rendering + test idiom.

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && ls dashboard/src/components/ && sed -n '1,60p' dashboard/src/components/$(ls dashboard/src/components/ | head -1)`

- [ ] **Step 2: Write the failing component test** (mirror a neighbor's test)

```typescript
it('greys unidentified cells and colors identified ones', () => {
  const correlation = {
    topic_order: [0, 1, 2],
    block_labels: ['background', 'A', 'B'],
    R: [[1, 0.4, 0.5], [0.4, 1, null], [0.5, null, 1]],
    identified: [[true, true, true], [true, true, false], [true, false, true]],
    support: [[300, 200, 100], [200, 200, 0], [100, 0, 100]],
  }
  // render <CorrelationHeatmap correlation={correlation} /> and assert:
  //  - the (1,2)/(2,1) cells carry the "no joint support" NA styling/title
  //  - an identified cell (0,1) carries a non-NA fill
})
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd dashboard && npm test -- CorrelationHeatmap`
Expected: FAIL (component missing).

- [ ] **Step 4: Implement `CorrelationHeatmap`** rendering the grid per the Interfaces block (diverging scale; NA cells grey with title `"no joint support: N < min_pair_support"`). Use the project's existing charting/color idiom (match a neighbor component — e.g. the prevalence or covariate-effects visual).

- [ ] **Step 5: Wire the panel** into the dashboard's tab/panel registry, shown only when `bundle.correlation` is defined.

- [ ] **Step 6: Run the test + the dashboard build**

Run: `cd dashboard && npm test -- CorrelationHeatmap && npm run build`
Expected: test PASS; build succeeds.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/components/CorrelationHeatmap.tsx dashboard/src/
git commit -m "feat(dashboard): correlation heatmap panel (NA cells greyed)"
```

---

## Task 9 (LAST — gated on Tasks 1-8 green): Documentation amendments

Do this task ONLY after Tasks 1-8 are all green, so the amendments cite passing tests, not intentions (spec: evidence before assertions).

**Files:**
- Modify: docs/insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md
- Modify: docs/decisions/0033-stm-full-covariance-sigma.md
- Modify: docs/experiments/0025-stm-comorbid-fullcov-gated-pdcompletion.md

**No code; prose only. Conventions: no LaTeX (Unicode Greek), impersonal, cite the passing tests by path, markdown-linkable refs.**

- [ ] **Step 1: Amend insight 0032** — add a "Resolution (superseding Findings 4-6): conditioning was a reporting artifact" section. State: the gated fit inverts only within-allowed-set marginal sub-blocks (`safe_inverse(Σ[allowed, allowed])`, [stm.py:777](../../spark-vi/spark_vi/models/topic/stm.py#L777)); the cross-foreground block never enters single-group inference; the full-matrix condition number is a reporting quantity; topic recovery is invariant to it, proven by [test_recovery_invariant_to_full_sigma_condition_number](../../spark-vi/tests/test_stm_pd_completion_conditioning.py). Note Findings 4-6 remain the empirical record of the lever hunt but their conditioning-as-pathology framing is superseded.

- [ ] **Step 2: Amend ADR 0033** — reframe `pd_complete`: it is the fit-time completion giving multi-group documents a coherent cross-foreground prior (Dempster 1972 zero-precision completion), NOT a conditioning cure. Record that the `sigma_cond`/`max_offdiag` diagnostics were removed (Task 3) as reporting artifacts, and that correlation reporting now uses the support-keyed identified mask ([topic_correlation_identified](../../spark-vi/spark_vi/models/topic/_linalg.py)).

- [ ] **Step 3: Amend exp 0025** — change its success criteria from a condition-number target to: topic recovery + an honest correlation report (identified within-group blocks; cross-foreground NA under the split representation, populated under multi-membership per Plan 2).

- [ ] **Step 4: Verify docs are LaTeX-clean and refs resolve**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && grep -nE '\$[^$]*\$|\\\(|\\begin' docs/insights/0032-*.md docs/decisions/0033-*.md docs/experiments/0025-*.md || echo CLEAN`
Expected: CLEAN.

- [ ] **Step 5: Commit**

```bash
git add docs/insights/0032-*.md docs/decisions/0033-*.md docs/experiments/0025-*.md
git commit -m "docs(stm): reframe conditioning as reporting artifact; correlation reporting via identified mask"
```

---

## Final verification (whole-plan)

- [ ] `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py tests/test_stm_pd_completion_conditioning.py tests/test_mllib_stm.py tests/test_mllib_stm_persistence.py tests/test_stm_contract.py tests/test_stm_integration.py -q` — all green.
- [ ] `cd charmpheno && python -m pytest tests/test_correlation_export.py -q` — green.
- [ ] `cd dashboard && npm test && npm run build` — green.
- [ ] Sanity: a saved model's `params/n_pairs.npy` exists; `correlation.json` has `null` across the cross-foreground block under a split-cohort fit; the heatmap greys those cells.
- [ ] Then dispatch the final whole-branch review (subagent-driven-development's final step).

## Notes for the executor

- Tasks 1-4 (engine) and 5-6 (export) are Python/TDD with exact code above. Tasks 7-8 (dashboard) are TypeScript; match the neighboring component/test idioms (the plan gives the bundle.ts diff exactly and the component contract; the visual details follow existing components).
- The Layer-1 conditioning tests and the term-sharing generator already exist in the working tree ([test_stm_pd_completion_conditioning.py](../../spark-vi/tests/test_stm_pd_completion_conditioning.py), [_stm_synth.py](../../spark-vi/tests/_stm_synth.py)); Task 4 adds one test to that file. Include the pre-existing tests/generator in the first commit that touches that file if they are not yet committed.
- Commit/push cadence: this branch (`stm`) auto-pushes to origin; commit per task as written, do not force-push or rebase.
