# Gated STM Dashboard (Plan 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. THIS PLAN IS ALSO A CONTINUATION/HANDOFF DOC — the "Continuation Context" section below carries everything needed to resume after a context compaction.

**Goal:** Render a gated background/foreground STM result in the dashboard with gating-aware behavior — masked prevalence on server and client, per-topic block labels, block-level k-anon, and a group selector that makes foreground bubbles appear/vanish by group — validated locally against the Plan-2a checkpoint.

**Architecture:** Shared `charmpheno.export` gains a masked corpus-mean prevalence helper, a `gating.json` builder with block-level k-anon (full suppression of sub-k groups), and per-topic block labels on `DashboardExport`. The local `build_dashboard.py` emits the STM bundle files (covariate + gating) from the locally-saved covariate matrix. The Svelte front-end adds a group selector and a gating-aware `covariatePrevalence`.

**Tech Stack:** Python, NumPy, pandas, PySpark (local), pytest; TypeScript, Svelte, Vite, vitest.

---

## Continuation Context (READ FIRST after any compaction)

**Where we are:** On branch `stm`. Plan 1 (gated STM model) is merged. **Plan 2a (local harness) is COMPLETE** — `git log --oneline pre-gated-stm-harness..HEAD`. Rollback tags: `pre-gated-stm` (before the whole gated feature), `pre-gated-stm-harness` (before Plan 2a). This plan (2b) is the dashboard. The SDD ledger is `.superpowers/sdd/progress.md`.

**Spec:** [docs/superpowers/specs/2026-06-23-gated-stm-local-dashboard-design.md](../specs/2026-06-23-gated-stm-local-dashboard-design.md).

**CRITICAL venv note:** Python `.fit()`/Spark tests and the local fitter MUST run under `poetry run` (the root `.venv` has a working `distutils` for Spark workers). The bare `python -m pytest` in `spark-vi/` uses a different poetry venv where Spark workers crash on `distutils` (Python 3.12 removed it). Pure-numpy / TS tests don't need `poetry run`.

**How to produce a gated checkpoint to test against (the Plan-2a chain):**
```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
poetry run python scripts/simulate_gated_omop.py --n-patients 5000 --seed 0 \
    --background-k 3 --foreground rare_dx:2 \
    --group-props common:0.99,rare_dx:0.01 --age-means common:55,rare_dx:72 \
    --n-background-concepts 40 --n-group-concepts 12
poetry run python analysis/local/fit_stm_local.py \
    --omop data/simulated/gated_omop_N5000_seed0.parquet \
    --person data/simulated/gated_person_N5000_seed0.parquet \
    --K 5 --background-k 3 --foreground rare_dx:2 \
    --covariate-formula "~ C(sex) + age" --max-iter 40 \
    --out-dir data/runs/gated_demo
```
Note: at N=5000, `rare_dx` ≈ 50 patients (> k=20) so it survives k-anon; to exercise suppression use a sub-k group (e.g. `--group-props common:0.99,rare_dx:0.006,ultrarare:0.004` with `--foreground rare_dx:2,ultrarare:1` and a small N so `ultrarare` < 20).

**Checkpoint contract (what `fit_stm_local` writes to `--out-dir`, the dashboard INPUT):**
- `manifest.json` → `metadata`:
  - top-level: `vocab` (list[int], index→concept_id), `name_by_id` (dict[int→str]), `model_class="stm"`, `concept_names` (dict[str(cid)→str]), `concept_domains` (dict[str(cid)→"condition"]).
  - `corpus_manifest`: `cdr="local"`, `source_table`, `cohort="gated_sim"`, `prior_obs_days`, `person_mod`, `doc_spec` (patient_cohort manifest), `vocab_size`, `vocab`, `name_by_id`, `min_patient_count` (the k-anon threshold, default 20), **`topic_block_spec`** (`{group_var, background_k, foreground:[[g,K],...]}`).
  - `covariate_manifest`: `covariate_formula`, `categorical_cols`, `continuous_cols`, `covariate_names` (length P).
- `params/lambda.npy` (K×V), `params/Gamma.npy` (P×K), `params/Sigma.npy` (K,), `params/eta.npy`.
- `model_spec.pkl`, `covariate_names.json` (STMModel sidecars).
- **`covariates.parquet`** — columns `person_id`, `source_cohort`, `covariates` (array of length P, the design vector). One row per joined doc. THIS is the offline source for masked prevalence + covariate_schema (no Spark cov_df / BQ cache needed locally).

**Key existing interfaces (verified during design):**
- `spark-vi/spark_vi/models/topic/partition.py`: `TopicBlockPartition` — `.K`, `.groups` (tuple), `.background_indices()`, `.block_indices(g)`, `.allowed_indices(frozenset)` (foreground-less group → background-only after Plan 2a), `.topic_labels()` (list[str] len K: "background"|group), `.to_dict()`/`.from_dict()`.
- `spark-vi/spark_vi/models/topic/stm.py`: pure-numpy `corpus_mean_topic_proportions(Gamma (P,K), X (D,P)) -> (K,)` = (1/D) Σ softmax(Γᵀx_d); `prior_topic_proportions(Gamma, x)`.
- `charmpheno/charmpheno/export/model_adapter.py`: `adapt(result, *, hdp_top_k=50, stm_corpus_prevalence=None) -> DashboardExport`. `DashboardExport(beta (K×V), alpha (K), corpus_prevalence (K), topic_indices (K, original ids), theta_histogram, theta_percentiles)`. `adapt_stm` computes α-equiv = softmax(Gamma[intercept]); corpus_prevalence = supplied or intercept stand-in.
- `charmpheno/charmpheno/export/dashboard.py`: `write_model_and_vocab_bundles(...)` (model.json, vocab.json), `write_phenotypes_bundle(...)` (phenotypes.json), `adapt_stm(*, out_dir, Gamma (P,K), covariate_names, K, P)` → writes `covariate_effects.json` = `[{covariate, per_topic:[...]}, ...]`.
- `charmpheno/charmpheno/export/covariate_schema.py`: `build_covariate_schema(*, covariate_names, continuous_cols, categorical_levels, level_counts, continuous_stats, k) -> dict` with keys `k, controls[], design_columns[], unsupported[]`.
- `analysis/local/build_dashboard.py`: args `--checkpoint --input <OMOP parquet> --out-dir --vocab-top-n --hdp-top-k --top-n-codes-for-npmi`. Reads `model_class` from metadata, top-level `metadata["vocab"]`, `concept_names`/`concept_domains` from metadata. Loads OMOP via `load_omop_parquet`. Today emits ONLY model/vocab/phenotypes/corpus_stats (NO covariate/gating files).
- `analysis/cloud/build_dashboard_cloud.py`: cloud sibling; `_stm_corpus_prevalence` (reads covariate cache via `try_load`, uses `corpus_mean_proportions_from_covariate_df`), `_write_covariate_schema`, `dashboard_adapt_stm`. Needs BQ — NOT runnable locally (parity only).
- Front-end `dashboard/`: run with `cd dashboard && npm install && npm run dev` (Vite, localhost:5173). Bundle served from `dashboard/public/data/<cohortId>/*.json`. `npm run test` = vitest.
  - `src/lib/bundle.ts`: `loadBundle(baseUrl, cohortId)` fetches model/phenotypes/vocab/corpusStats + optional `covariate_schema.json`/`covariate_effects.json` via `fetchJsonOptional`.
  - `src/lib/types.ts`: `DashboardBundle { model, phenotypes, vocab, corpusStats, covariateSchema?, covariateEffects? }`. `CovariateSchema { k, controls[], design_columns[], unsupported[] }`. `CovariateEffects = {covariate, per_topic:number[]}[]`. `Phenotype { id, original_topic_id, corpus_prevalence, ... }`.
  - `src/lib/covariate.ts`: `buildDesignVector(design_columns, values) -> number[]`; `covariatePrevalence(effects, x) -> number[]` = softmax(Γᵀx) over ALL K.
  - `src/lib/store.ts`: `covariateMode` (writable bool), `covariateValues` (writable Record), `prevalenceReader = derived([bundle, tauThreshold, covariateMode, covariateValues], ...)` → returns `(p:Phenotype)=>number`. In covariate mode + schema+effects + `unsupported.length===0`: `buildDesignVector` then `covariatePrevalence`, read `prev[p.id]`.
  - `src/lib/atlas/CovariatePanel.svelte`: renders `schema.controls`, toggles `covariateMode`, binds `covariateValues` (via `./covariate-panel` helpers `initialValues`, `canInteract`).
  - `src/lib/atlas/TopicMap.svelte`: bubble radius via `d3.scaleSqrt().domain([0, max(reader)]).range([5,26])`, `reader = $prevalenceReader`.

**Plan-2a known notes carried forward:** `min_patient_count` is metadata only (k-anon happens here in the dashboard, not the fit). `concept_names`/`concept_domains` are top-level string-keyed and match `build_dashboard.py`.

---

## Global Constraints

- Non-gated bundles (no `topic_block_spec` / no `gating.json`) behave EXACTLY as today (backward compatible) across builders and front-end.
- Masked prevalence masks BEFORE the softmax: η=Γᵀx, set disallowed foreground η=−∞, softmax over allowed — matches the model. Never softmax-over-all-then-zero.
- Block-level k-anon = FULL suppression: a group with patient count `< k` (`k = corpus_manifest.min_patient_count`) is dropped from `gating.json.groups` and its foreground topics are EXCLUDED from the exported bundle (via the `topic_indices` mask, same mechanism HDP top-K uses). One consistent threshold.
- Γ stays (P, K). Gating is one new optional bundle file (`gating.json`) + the masked-prevalence swap + a topic-suppression mask. No reshaping of existing arrays.
- `source_cohort` is the charmpheno group label; the engine/shim/front-end stay domain-agnostic ("group").
- No LaTeX in docstrings/prose (Unicode Greek OK); no emojis in committed files. Markdown-linkable code refs in docs.
- TDD throughout. Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- Modify: `spark-vi/spark_vi/models/topic/stm.py` — add `corpus_mean_topic_proportions_gated`.
- Create: `charmpheno/charmpheno/export/gating.py` — `build_gating_json` + `suppressed_topic_ids` (block-level k-anon).
- Modify: `charmpheno/charmpheno/export/model_adapter.py` — `DashboardExport.topic_blocks`; `adapt_stm` accepts a partition + suppressed-topic mask.
- Modify: `analysis/local/build_dashboard.py` — STM branch emits covariate_effects + covariate_schema + gating.json + masked corpus_prevalence from `covariates.parquet`.
- Modify: `analysis/cloud/build_dashboard_cloud.py` — gated masked prevalence + gating.json (parity, unvalidated).
- Modify: `dashboard/src/lib/types.ts`, `bundle.ts`, `covariate.ts`, `store.ts` — gating types, optional load, masked prevalence, `selectedGroup`.
- Modify: `dashboard/src/lib/atlas/CovariatePanel.svelte` (or new `GatingPanel.svelte`) + `TopicMap.svelte` — group selector + vanish behavior.
- Create: `scripts/gated_dashboard_demo.sh` (or a Make target) — one-command chain.
- Tests: `spark-vi/tests/test_stm_math.py` (masked helper), `charmpheno/tests/test_gating_export.py` (new), `tests/scripts/test_build_dashboard_gated.py` (new, integration), `dashboard/src/lib/*.test.ts`.

---

## Task 1: Masked corpus-mean prevalence helper (pure numpy)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (add function near `corpus_mean_topic_proportions`)
- Test: `spark-vi/tests/test_stm_math.py`

**Interfaces:**
- Produces: `corpus_mean_topic_proportions_gated(Gamma, X, groups_per_doc, partition) -> np.ndarray (K,)`. `Gamma` (P,K), `X` (D,P), `groups_per_doc` is a list of `frozenset[str]` length D, `partition` a `TopicBlockPartition`. For each doc: softmax of Γᵀx over `partition.allowed_indices(groups)` only (0 elsewhere); average over docs.

- [ ] **Step 1: Write the failing test**

```python
# append in spark-vi/tests/test_stm_math.py
def test_corpus_mean_topic_proportions_gated_zeros_out_of_group_foreground():
    import numpy as np
    from spark_vi.models.topic.stm import corpus_mean_topic_proportions_gated
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))  # K=3
    P = 2
    Gamma = np.zeros((P, 3))
    X = np.ones((4, P))
    # 3 'common' docs (no foreground block -> background only) + 1 'rare'
    groups = [frozenset({"common"})] * 3 + [frozenset({"rare"})]
    prev = corpus_mean_topic_proportions_gated(Gamma, X, groups, part)
    assert prev.shape == (3,)
    np.testing.assert_allclose(prev.sum(), 1.0, atol=1e-9)
    # foreground topic 2 only gets mass from the 1 rare doc (1/4 of corpus * its share)
    assert prev[2] > 0.0 and prev[2] < 0.3
    # with Gamma=0, each common doc is uniform over background {0,1}; rare doc
    # uniform over {0,1,2}. mean[2] = (1/4)*(1/3).
    np.testing.assert_allclose(prev[2], 0.25 / 3, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k gated -v`
Expected: FAIL — function not defined.

- [ ] **Step 3: Implement**

In `spark-vi/spark_vi/models/topic/stm.py`, add after `corpus_mean_topic_proportions`:

```python
def corpus_mean_topic_proportions_gated(Gamma, X, groups_per_doc, partition):
    """Gating-aware corpus-mean prior proportions: (1/D) Σ_d softmax_allowed(Γᵀ x_d).

    For each document, the softmax is taken over that document's ALLOWED topics
    only (background ∪ its group's foreground, per partition.allowed_indices);
    disallowed topics are exactly 0. So a foreground topic's corpus-mean
    prevalence reflects only its group's share. Γ is (P, K), X is (D, P),
    groups_per_doc is a length-D sequence of frozenset[str].
    """
    import numpy as np
    Gamma = np.asarray(Gamma, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    K = Gamma.shape[1]
    eta = X @ Gamma                                   # (D, K)
    acc = np.zeros(K, dtype=np.float64)
    for d in range(X.shape[0]):
        allowed = partition.allowed_indices(groups_per_doc[d])
        e = eta[d, allowed]
        e = e - e.max()
        p = np.exp(e)
        p = p / p.sum()
        acc[allowed] += p
    return acc / max(X.shape[0], 1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k gated -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_math.py
git commit -m "$(printf 'feat(stm): gating-aware masked corpus-mean topic proportions\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: gating.json builder + block-level k-anon + DashboardExport.topic_blocks

**Files:**
- Create: `charmpheno/charmpheno/export/gating.py`
- Modify: `charmpheno/charmpheno/export/model_adapter.py` (`DashboardExport` + `adapt_stm`)
- Test: `charmpheno/tests/test_gating_export.py`

**Interfaces:**
- Produces:
  - `suppressed_topic_ids(partition, group_counts, k) -> set[int]` — original topic ids of foreground blocks whose group count `< k` (background topics are never suppressed).
  - `build_gating_json(partition, group_counts, k, kept_topic_ids) -> dict` → `{group_var, groups (kept, count>=k), topic_blocks}` where `topic_blocks` is the block label per KEPT topic id, in `kept_topic_ids` order.
  - `DashboardExport` gains `topic_blocks: list[str] | None` (per displayed topic, aligned to `topic_indices`), default None.
  - `adapt_stm(result, *, corpus_prevalence=None, partition=None, suppressed=frozenset())` — when `partition` given, EXCLUDES `suppressed` topic ids from beta/alpha/corpus_prevalence/topic_indices (same masking HDP uses) and sets `topic_blocks` from `partition.topic_labels()` for the surviving topics.

- [ ] **Step 1: Write the failing tests**

```python
# charmpheno/tests/test_gating_export.py
import numpy as np
from spark_vi.models.topic.partition import TopicBlockPartition
from charmpheno.export.gating import suppressed_topic_ids, build_gating_json


def test_suppressed_topic_ids_drops_sub_k_group():
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))  # K=6
    # rare_dx has 50 patients (>=20), ultrarare has 8 (<20) -> suppress topic id 5
    counts = {"rare_dx": 50, "ultrarare": 8}
    assert suppressed_topic_ids(part, counts, k=20) == {5}


def test_build_gating_json_keeps_only_above_k_groups():
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))
    counts = {"rare_dx": 50, "ultrarare": 8}
    supp = suppressed_topic_ids(part, counts, k=20)              # {5}
    kept_ids = [i for i in range(part.K) if i not in supp]       # [0,1,2,3,4]
    gj = build_gating_json(part, counts, k=20, kept_topic_ids=kept_ids)
    assert gj["group_var"] == "source_cohort"
    assert gj["groups"] == ["rare_dx"]                          # ultrarare suppressed
    assert gj["topic_blocks"] == ["background", "background", "background",
                                  "rare_dx", "rare_dx"]


def test_adapt_stm_excludes_suppressed_topics():
    from charmpheno.export.model_adapter import adapt_stm
    # build a minimal STM-like result object
    class R:
        global_params = {"lambda": np.abs(np.random.default_rng(0).normal(
            size=(6, 8))) + 0.1, "Gamma": np.zeros((2, 6))}
        metadata = {"covariate_manifest": {"covariate_names": ["Intercept", "age"]}}
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))
    exp = adapt_stm(R(), partition=part, suppressed=frozenset({5}))
    assert exp.beta.shape[0] == 5                                # 6 - 1 suppressed
    assert list(exp.topic_indices) == [0, 1, 2, 3, 4]
    assert exp.topic_blocks == ["background", "background", "background",
                                "rare_dx", "rare_dx"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_gating_export.py -v`
Expected: FAIL — `charmpheno.export.gating` missing; `adapt_stm` has no `partition`/`suppressed` kwargs.

- [ ] **Step 3: Implement**

Create `charmpheno/charmpheno/export/gating.py`:

```python
"""Gating bundle export: per-topic block labels + block-level k-anon.

A foreground group whose patient count is below the small-cell threshold k is
fully suppressed: dropped from the gating groups list AND its foreground topics
are excluded from the bundle (the dashboard never reveals a sub-k group). This
is the honest information floor for the rare-disease case.
"""
from __future__ import annotations


def suppressed_topic_ids(partition, group_counts, k):
    """Original topic ids of foreground blocks whose group count < k.

    Background topics are never suppressed. group_counts maps group label ->
    patient count.
    """
    supp = set()
    for g in partition.groups:
        if int(group_counts.get(g, 0)) < int(k):
            supp.update(int(i) for i in partition.block_indices(g))
    return supp


def build_gating_json(partition, group_counts, k, kept_topic_ids):
    """gating.json: kept groups (count >= k) + per-kept-topic block label."""
    kept_groups = [g for g in partition.groups
                   if int(group_counts.get(g, 0)) >= int(k)]
    labels = partition.topic_labels()                 # length K, by original id
    topic_blocks = [labels[i] for i in kept_topic_ids]
    return {
        "group_var": partition.group_var,
        "groups": kept_groups,
        "topic_blocks": topic_blocks,
    }
```

In `charmpheno/charmpheno/export/model_adapter.py`, add `topic_blocks: list | None = None` to the `DashboardExport` dataclass (after `theta_percentiles`, with default None). Then change `adapt_stm` to accept `partition=None, suppressed=frozenset()` and, when `partition` is set, build a kept-index list and subset:

```python
def adapt_stm(result, *, corpus_prevalence=None, partition=None,
              suppressed=frozenset()):
    import numpy as np
    gp = _global_params(result)
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    K = lambda_.shape[0]
    kept = [i for i in range(K) if i not in suppressed]
    beta_full = lambda_ / lambda_.sum(axis=1, keepdims=True)
    Gamma = np.asarray(gp["Gamma"], dtype=np.float64)
    covariate_names = result.metadata["covariate_manifest"]["covariate_names"]
    intercept_idx = next(
        (i for i, n in enumerate(covariate_names) if "intercept" in str(n).lower()),
        None)
    if intercept_idx is not None:
        eta_bar = Gamma[intercept_idx]
        alpha_eq = np.exp(eta_bar - eta_bar.max()); alpha_eq /= alpha_eq.sum()
    else:
        alpha_eq = np.full(K, 1.0 / K)
    corpus_prev = (np.asarray(corpus_prevalence, dtype=np.float64)
                   if corpus_prevalence is not None else alpha_eq.copy())

    topic_blocks = None
    if partition is not None:
        labels = partition.topic_labels()
        topic_blocks = [labels[i] for i in kept]
    beta = beta_full[kept]
    alpha = alpha_eq[kept]
    corpus_prev = corpus_prev[kept]
    return DashboardExport(
        beta=beta, alpha=alpha, corpus_prevalence=corpus_prev,
        topic_indices=np.array(kept, dtype=np.int64),
        theta_histogram=None, theta_percentiles=None,
        topic_blocks=topic_blocks,
    )
```

(Keep the existing non-gated `adapt_stm` behavior for `partition=None`: `kept` is all topics, `topic_blocks=None`, identical output to before.)

- [ ] **Step 4: Run to verify they pass**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_gating_export.py -v`
Expected: PASS (3). Also run the existing adapter tests to confirm no regression: `poetry run python -m pytest charmpheno/tests/ -k adapt -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/gating.py charmpheno/charmpheno/export/model_adapter.py charmpheno/tests/test_gating_export.py
git commit -m "$(printf 'feat(export): gating.json + block-level k-anon + topic_blocks on DashboardExport\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Local build_dashboard.py emits the STM gating bundle

**Files:**
- Modify: `analysis/local/build_dashboard.py`
- Test: `tests/scripts/test_build_dashboard_gated.py` (new — runs the Plan-2a chain to a tiny checkpoint, then the dashboard build, asserts the JSON files)

**Interfaces:**
- Consumes: Task 1 (`corpus_mean_topic_proportions_gated`), Task 2 (`gating`, `adapt_stm(partition, suppressed)`), the checkpoint contract + `covariates.parquet`, `TopicBlockPartition.from_dict`, `build_covariate_schema`, `dashboard.adapt_stm` (covariate_effects writer).
- Produces: when `model_class=="stm"` AND `corpus_manifest.topic_block_spec` present, `build_dashboard.py` reads `covariates.parquet` (sibling of the checkpoint), reconstructs the partition + per-doc groups + design matrix X, computes per-group patient counts and `suppressed`, builds the masked `corpus_prevalence` via the gated helper, passes `partition`+`suppressed` to `adapt`/`adapt_stm`, and writes `covariate_effects.json`, `covariate_schema.json`, and `gating.json` (k-anon applied). Non-gated checkpoints are unchanged.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/scripts/test_build_dashboard_gated.py
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis" / "local"))


def test_local_dashboard_emits_gating_files(tmp_path):
    from simulate_gated_omop import main as sim_main
    from fit_stm_local import main as fit_main
    import build_dashboard

    sim_main(["--n-patients", "400", "--seed", "9",
              "--background-k", "3", "--foreground", "rare_dx:2",
              "--group-props", "common:0.7,rare_dx:0.3",
              "--age-means", "common:55,rare_dx:72",
              "--n-background-concepts", "12", "--n-group-concepts", "6",
              "--codes-per-visit-mean", "6", "--output-dir", str(tmp_path)])
    omop = tmp_path / "gated_omop_N400_seed9.parquet"
    person = tmp_path / "gated_person_N400_seed9.parquet"
    ckpt = tmp_path / "ckpt"
    fit_main(["--omop", str(omop), "--person", str(person),
              "--K", "5", "--background-k", "3", "--foreground", "rare_dx:2",
              "--covariate-formula", "~ C(sex) + age",
              "--max-iter", "10", "--out-dir", str(ckpt)])

    out = tmp_path / "bundle"
    rc = build_dashboard.main(["--checkpoint", str(ckpt), "--input", str(omop),
                               "--out-dir", str(out)])
    assert rc == 0
    gating = json.loads((out / "gating.json").read_text())
    assert gating["group_var"] == "source_cohort"
    assert "rare_dx" in gating["groups"]              # >=k at this size
    assert len(gating["topic_blocks"]) == 5
    assert (out / "covariate_effects.json").exists()
    assert (out / "covariate_schema.json").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_build_dashboard_gated.py -v`
Expected: FAIL — no `gating.json` written (the STM branch doesn't emit it yet).

- [ ] **Step 3: Implement**

In `analysis/local/build_dashboard.py`, after the existing model-class detection and `adapt(...)` call, add an STM-gating branch. Read the checkpoint's `corpus_manifest`; if `model_class=="stm"` and `corpus_manifest.get("topic_block_spec")`:

```python
    # --- STM gating: masked prevalence + covariate + gating.json (offline) ---
    corpus = result.metadata.get("corpus_manifest", {})
    tbs = corpus.get("topic_block_spec") if result.metadata.get("model_class") == "stm" else None
    if tbs:
        import numpy as np
        import pandas as pd
        from spark_vi.models.topic.partition import TopicBlockPartition
        from spark_vi.models.topic.stm import corpus_mean_topic_proportions_gated
        from charmpheno.export.gating import suppressed_topic_ids, build_gating_json
        from charmpheno.export.dashboard import adapt_stm as write_cov_effects
        from charmpheno.export.covariate_schema import build_covariate_schema

        partition = TopicBlockPartition.from_dict(tbs)
        cov_path = Path(args.checkpoint) / "covariates.parquet"
        cov = pd.read_parquet(cov_path)
        X = np.vstack(cov["covariates"].to_numpy())               # (D, P)
        groups_per_doc = [frozenset({g}) for g in cov["source_cohort"]]
        Gamma = np.asarray(result.global_params["Gamma"], dtype=np.float64)

        # per-group patient counts (distinct person_id) for k-anon
        gc = (cov.groupby("source_cohort")["person_id"].nunique().to_dict())
        k = int(corpus.get("min_patient_count", 20))
        suppressed = suppressed_topic_ids(partition, gc, k)

        masked_prev = corpus_mean_topic_proportions_gated(
            Gamma, X, groups_per_doc, partition)

        from charmpheno.export.model_adapter import adapt_stm as adapt_stm_export
        export = adapt_stm_export(result, corpus_prevalence=masked_prev,
                                  partition=partition, suppressed=suppressed)
        kept_ids = [int(i) for i in export.topic_indices]
        gating = build_gating_json(partition, gc, k, kept_ids)
        (out_dir / "gating.json").write_text(json.dumps(gating, indent=2))

        # covariate_effects.json for the KEPT topics (Gamma columns subset)
        write_cov_effects(out_dir=out_dir, Gamma=Gamma[:, kept_ids],
                          covariate_names=corpus_manifest_covariate_names(result),
                          K=len(kept_ids), P=Gamma.shape[0])

        # covariate_schema.json from the local covariate matrix
        _write_local_covariate_schema(out_dir, result, cov, X, k)
        # use `export` (subset beta/alpha/prevalence + topic_blocks) for the
        # model/phenotypes bundles below instead of the plain adapt() result.
```

Add two small helpers in `build_dashboard.py`:
- `corpus_manifest_covariate_names(result)` → `result.metadata["covariate_manifest"]["covariate_names"]`.
- `_write_local_covariate_schema(out_dir, result, cov_pdf, X, k)` — replicate `build_dashboard_cloud._write_covariate_schema`'s logic but from the local pandas covariate matrix: per-dummy-column sums = `X[:, idx].sum()` for `C(var)[T.level]` names, continuous p5/p50/p95 via `np.percentile`, categorical levels/reference from `result.model_spec` (loaded sidecar) — reuse `build_dashboard_cloud._categorical_levels_from_spec` (import it) — then `build_covariate_schema(...)`, write `covariate_schema.json`.

Then wire the gated `export` (with `topic_blocks` and suppressed topics removed) into the existing `write_model_and_vocab_bundles` / `write_phenotypes_bundle` calls (they already accept `beta`/`alpha`/`corpus_prevalence`/`topic_indices`). The phenotypes' `original_topic_id` must be `export.topic_indices` so the front-end can map `gating.topic_blocks` by original id.

NOTE for the implementer: read the current `analysis/local/build_dashboard.py` end-to-end first — the exact insertion point is right after the `export = adapt(...)` line and before `write_model_and_vocab_bundles`. The non-STM and non-gated STM paths must keep calling `adapt(...)` unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_build_dashboard_gated.py -v`
Expected: PASS. Inspect the emitted `gating.json` + `covariate_schema.json` for sanity.

- [ ] **Step 5: Commit**

```bash
git add analysis/local/build_dashboard.py tests/scripts/test_build_dashboard_gated.py
git commit -m "$(printf 'feat(dashboard): local builder emits gated STM bundle (masked prevalence + gating.json + covariate files)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Cloud builder parity (unvalidated locally)

**Files:**
- Modify: `analysis/cloud/build_dashboard_cloud.py`
- Test: none runnable locally (needs BigQuery). Verify by `python -m py_compile` + inspection against Task 3's logic.

**Interfaces:**
- Consumes: Task 1/2 helpers.
- Produces: `build_dashboard_cloud` derives each cov-row's group from `source_cohort` (split from the covariate cache's key or recomputed), computes per-group counts + `suppressed`, swaps `_stm_corpus_prevalence` to the gated helper, passes `partition`+`suppressed` to `adapt_stm`, and writes `gating.json`. Mirrors Task 3 but sourced from the Spark covariate cache.

- [ ] **Step 1: Implement (no local test — BQ-bound)**

Read `analysis/cloud/build_dashboard_cloud.py`. In `_stm_corpus_prevalence`, after loading `cov_df`, also select `source_cohort` (the cov_df is keyed `(person_id[, source_cohort])`; for the gated combined cohort `source_cohort` is present). Collect `(covariates, source_cohort)` to build X + groups_per_doc OR add a gated Spark reduce; for the cohort sizes here a `.toPandas()` collect on the driver is acceptable (document the assumption). Compute `suppressed` from per-group `countDistinct(person_id)`. In `main`, build the `TopicBlockPartition.from_dict(corpus["topic_block_spec"])`, pass `partition`/`suppressed` into `adapt`/`adapt_stm`, and write `gating.json` next to `covariate_schema.json`. Guard the whole block on `corpus.get("topic_block_spec")` so non-gated STM checkpoints are unchanged.

- [ ] **Step 2: Verify it compiles + inspect**

Run: `cd analysis/cloud && python -m py_compile build_dashboard_cloud.py` — Expected: clean.
Manually diff the gating logic against Task 3 to confirm parity (same `suppressed_topic_ids`, same `build_gating_json`, same masked prevalence semantics).

- [ ] **Step 3: Commit**

```bash
git add analysis/cloud/build_dashboard_cloud.py
git commit -m "$(printf 'feat(dashboard): cloud builder gated parity (masked prevalence + gating.json)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: Front-end types + optional gating.json load

**Files:**
- Modify: `dashboard/src/lib/types.ts`, `dashboard/src/lib/bundle.ts`
- Test: `dashboard/src/lib/bundle.test.ts` (or extend an existing loader test)

**Interfaces:**
- Produces: `GatingSpec { group_var: string; groups: string[]; topic_blocks: string[] }`; `DashboardBundle` gains `gating?: GatingSpec`. `loadBundle` fetches `gating.json` via `fetchJsonOptional` (null when absent → behaves as today).

- [ ] **Step 1: Write the failing test**

```typescript
// dashboard/src/lib/bundle.test.ts (add; mirror existing loader test mocking fetch)
import { describe, it, expect, vi } from 'vitest'
import { loadBundle } from './bundle'

describe('loadBundle gating', () => {
  it('attaches gating when gating.json is present', async () => {
    const files: Record<string, unknown> = {
      'data/c/model.json': { K: 1, V: 1, alpha: [1], beta: [[1]] },
      'data/c/phenotypes.json': { phenotypes: [] },
      'data/c/vocab.json': { codes: [] },
      'data/c/corpus_stats.json': { corpus_size_docs: 0, mean_codes_per_doc: 0, k: 20, v: 1, v_full: 1 },
      'data/c/gating.json': { group_var: 'source_cohort', groups: ['rare_dx'], topic_blocks: ['background'] },
    }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k))
      return key ? { ok: true, json: async () => files[key] } : { ok: false, status: 404 }
    }))
    const b = await loadBundle('', 'c')
    expect(b.gating?.groups).toEqual(['rare_dx'])
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm run test -- bundle`
Expected: FAIL — `gating` not on the bundle / not fetched.

- [ ] **Step 3: Implement**

In `types.ts` add `GatingSpec` and `gating?: GatingSpec` on `DashboardBundle`. In `bundle.ts`, add to the optional-fetch block:
```typescript
  const gating = await fetchJsonOptional<GatingSpec>(`${base}data/${cohortId}/gating.json`)
```
and include `gating` in the returned object.

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm run test -- bundle`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/types.ts dashboard/src/lib/bundle.ts dashboard/src/lib/bundle.test.ts
git commit -m "$(printf 'feat(dashboard-fe): GatingSpec type + optional gating.json load\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: Front-end masked prevalence + selectedGroup store

**Files:**
- Modify: `dashboard/src/lib/covariate.ts`, `dashboard/src/lib/store.ts`
- Test: `dashboard/src/lib/covariate.test.ts`

**Interfaces:**
- Produces:
  - `covariatePrevalenceGated(effects, x, allowedMask: boolean[]) -> number[]` — η=Γᵀx, set η[k]=−Infinity where `!allowedMask[k]`, softmax → 0 on masked topics, normalized over allowed.
  - `allowedMaskForGroup(topic_blocks: string[], selectedGroup: string | null) -> boolean[]` — true for `topic_blocks[k]==="background"` OR `=== selectedGroup`; "background only" (`selectedGroup===null`) → only background true.
  - `store.ts`: `selectedGroup = writable<string | null>(null)`; `prevalenceReader` derived gains `selectedGroup` + `bundle.gating`: in covariate mode with gating, map each phenotype by `original_topic_id` to its block label, build the mask, use `covariatePrevalenceGated`. Non-gated path unchanged.

- [ ] **Step 1: Write the failing tests**

```typescript
// dashboard/src/lib/covariate.test.ts (add)
import { describe, it, expect } from 'vitest'
import { covariatePrevalenceGated, allowedMaskForGroup } from './covariate'

describe('gated prevalence', () => {
  it('masks before softmax and zeros out-of-group foreground', () => {
    const effects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    const x = [1]
    const blocks = ['background', 'rare_dx', 'other']
    const mask = allowedMaskForGroup(blocks, 'rare_dx')   // [true,true,false]
    expect(mask).toEqual([true, true, false])
    const prev = covariatePrevalenceGated(effects, x, mask)
    expect(prev[2]).toBe(0)
    expect(Math.abs(prev[0] + prev[1] - 1)).toBeLessThan(1e-9)
  })
  it('background-only selection zeros all foreground', () => {
    const blocks = ['background', 'rare_dx']
    expect(allowedMaskForGroup(blocks, null)).toEqual([true, false])
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm run test -- covariate`
Expected: FAIL — functions not exported.

- [ ] **Step 3: Implement**

In `covariate.ts` add:
```typescript
export function allowedMaskForGroup(
  topicBlocks: string[], selectedGroup: string | null,
): boolean[] {
  return topicBlocks.map((b) => b === 'background' || b === selectedGroup)
}

export function covariatePrevalenceGated(
  effects: CovariateEffects, x: number[], allowedMask: boolean[],
): number[] {
  const K = effects[0]?.per_topic.length ?? 0
  const eta = new Array(K).fill(0)
  for (let p = 0; p < effects.length; p++) {
    const row = effects[p].per_topic
    for (let k = 0; k < K; k++) eta[k] += row[k] * x[p]
  }
  for (let k = 0; k < K; k++) if (!allowedMask[k]) eta[k] = -Infinity
  const finite = eta.filter((e) => e !== -Infinity)
  const m = finite.length ? Math.max(...finite) : 0
  const exp = eta.map((e) => (e === -Infinity ? 0 : Math.exp(e - m)))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}
```

In `store.ts` add `export const selectedGroup = writable<string | null>(null)` and extend `prevalenceReader`'s dependency array + body: when `$mode && schema && effects && schema.unsupported.length === 0 && $b?.gating`, build `topicBlocks` aligned to the effects' topic order (the bundle's phenotypes carry `original_topic_id`; `gating.topic_blocks` is aligned to the KEPT topic ids in the same order the bundle emits topics — i.e., index k of the displayed model corresponds to `gating.topic_blocks[k]`), compute `allowedMaskForGroup(gating.topic_blocks, $selectedGroup)`, and return `(p) => covariatePrevalenceGated(effects, x, mask)[<index of p>]`. Map a phenotype to its index via its position in the model (the existing reader uses `prev[p.id]`; keep that — `gating.topic_blocks[k]` is aligned to displayed topic index k = `p.id`). When no `$b?.gating`, fall through to the existing non-gated `covariatePrevalence`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm run test -- covariate`
Expected: PASS. Then full FE suite: `npm run test` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/covariate.ts dashboard/src/lib/store.ts dashboard/src/lib/covariate.test.ts
git commit -m "$(printf 'feat(dashboard-fe): gating-aware masked covariatePrevalence + selectedGroup store\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: Group selector UI + Atlas vanish behavior

**Files:**
- Modify: `dashboard/src/lib/atlas/CovariatePanel.svelte` (add a group selector when `bundle.gating` present)
- Test: build + a small component test if the project has Svelte component tests; otherwise `npm run build` + visual.

**Interfaces:**
- Consumes: `selectedGroup` store, `bundle.gating` (Task 5/6).
- Produces: a group selector (radio/select) listing `gating.groups` + a "Background only" option (maps to `selectedGroup=null`), bound to the `selectedGroup` store. It appears only when `bundle.gating` is present. Because `prevalenceReader` (Task 6) already consumes `selectedGroup`, the `TopicMap` bubbles update reactively — out-of-group foreground bubbles get prevalence 0 → radius ~0 (vanish). No `TopicMap.svelte` change is strictly required (it reads `prevalenceReader`), but verify the sqrt-scale `domain([0, max])` handles all-but-few-zero gracefully.

- [ ] **Step 1: Read the current component + wire the selector**

Read `dashboard/src/lib/atlas/CovariatePanel.svelte` (its structure: a `<header>` with the covariate-mode toggle, then `schema.controls`). Import `selectedGroup` from `../store` and the `gating` from the bundle store (or pass `gating` as a prop from the parent that renders `CovariatePanel`). Add, above or below the covariate controls, a "Group" control rendered only when `gating` is present:
```svelte
{#if gating}
  <div class="control-row">
    <span class="control-label">{gating.group_var}</span>
    <select bind:value={$selectedGroup} class="cat-select">
      <option value={null}>Background only</option>
      {#each gating.groups as g}
        <option value={g}>{g}</option>
      {/each}
    </select>
  </div>
{/if}
```
Find where `CovariatePanel` is instantiated (grep `CovariatePanel`) and thread the `gating` prop from `$bundle.gating`. Ensure `selectedGroup` is reset to null on bundle change.

- [ ] **Step 2: Build + manual check**

Run: `cd dashboard && npm run build`
Expected: build succeeds (no type errors). Then `npm run dev` and confirm the Group selector appears for a gated bundle and bubbles change when switching groups (covariate mode on).

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/lib/atlas/CovariatePanel.svelte <parent component>
git commit -m "$(printf 'feat(dashboard-fe): group selector wired to masked prevalence (foreground appears/vanishes by group)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: One-command demo chain + run + screenshot

**Files:**
- Create: `scripts/gated_dashboard_demo.sh` (or a Make target in the analysis Makefile)

**Interfaces:**
- Produces: a script that runs `simulate_gated_omop → fit_stm_local → build_dashboard`, copies the bundle into `dashboard/public/data/gated_demo/`, and prints the `npm run dev` instruction. Idempotent; uses `poetry run`.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# scripts/gated_dashboard_demo.sh — end-to-end local gated STM dashboard demo.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-0}; N=${N:-5000}
poetry run python scripts/simulate_gated_omop.py --n-patients "$N" --seed "$SEED" \
    --background-k 3 --foreground rare_dx:2 \
    --group-props common:0.99,rare_dx:0.01 --age-means common:55,rare_dx:72 \
    --n-background-concepts 40 --n-group-concepts 12
poetry run python analysis/local/fit_stm_local.py \
    --omop "data/simulated/gated_omop_N${N}_seed${SEED}.parquet" \
    --person "data/simulated/gated_person_N${N}_seed${SEED}.parquet" \
    --K 5 --background-k 3 --foreground rare_dx:2 \
    --covariate-formula "~ C(sex) + age" --max-iter 40 --out-dir data/runs/gated_demo
poetry run python analysis/local/build_dashboard.py \
    --checkpoint data/runs/gated_demo \
    --input "data/simulated/gated_omop_N${N}_seed${SEED}.parquet" \
    --out-dir data/runs/gated_demo/dashboard_bundle
mkdir -p dashboard/public/data/gated_demo
cp data/runs/gated_demo/dashboard_bundle/*.json dashboard/public/data/gated_demo/
echo "Bundle ready. Run: cd dashboard && npm install && npm run dev  (select cohort 'gated_demo')"
```

NOTE: the dashboard's cohort manifest (`dashboard/public/data/manifest.json`) may need a `gated_demo` entry — check how cohorts are listed (grep `manifest.json` under `dashboard/public/data/`) and add an entry if required.

- [ ] **Step 2: Run it end-to-end**

Run: `bash scripts/gated_dashboard_demo.sh`
Expected: bundle files (incl `gating.json`) in `dashboard/public/data/gated_demo/`. Then `cd dashboard && npm run dev`, open the Atlas, toggle covariate mode, switch the Group selector between "rare_dx" and "Background only", and confirm the rare_dx foreground bubbles appear/vanish.

- [ ] **Step 3: Commit + capture a screenshot**

```bash
git add scripts/gated_dashboard_demo.sh dashboard/public/data/manifest.json
git commit -m "$(printf 'feat(demo): one-command local gated STM dashboard chain\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```
Capture a screenshot of the Atlas with the rare_dx group selected (use the run/verify tooling) to confirm the foreground-by-group behavior end-to-end.

---

## Final verification

- [ ] Python: `cd spark-vi && python -m pytest tests/test_stm_math.py -k gated -v` and `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_gating_export.py tests/scripts/test_build_dashboard_gated.py -v` — all PASS.
- [ ] Front-end: `cd dashboard && npm run test && npm run build` — PASS.
- [ ] The demo chain produces a `gating.json` with `rare_dx` in `groups`, and the running dashboard shows foreground bubbles appearing/vanishing with the group selector.
- [ ] Non-gated regression: an LDA local bundle (no `gating.json`) renders unchanged.
- [ ] Whole-branch review (opus) before finishing; then `superpowers:finishing-a-development-branch` (likely "keep as-is on stm" pending the real cluster run).

## Debugging notes (for resuming)

- If `build_dashboard.py` raises `SystemExit("checkpoint metadata has no 'vocab'")`: the checkpoint predates the Plan-2a top-level-vocab fix (`8e2eae3`) — re-fit, or the fitter regressed.
- If the Spark fit crashes on `distutils`/`Py4JJavaError`: you're not in the `poetry run` venv. Use `poetry run`.
- If `corpus_mean_topic_proportions_gated` gives unexpected foreground mass: check `groups_per_doc` is `frozenset({source_cohort})` per doc and the partition's `foreground` groups match the `source_cohort` values in `covariates.parquet`.
- If foreground bubbles don't vanish in the UI: confirm `bundle.gating` loaded (Network tab → `gating.json` 200), `covariateMode` is ON, and `gating.topic_blocks` indexing aligns with displayed topic index = `p.id` (Task 6 assumption). If topics were suppressed, `topic_indices`/`original_topic_id` must be consulted for alignment.
- Rare group below k disappears entirely (no bubbles, not in selector) — that's the intended k-anon, not a bug; raise `--n-patients` or the group's proportion to clear k.
