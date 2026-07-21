# Likelihood-ratio placement readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A post-hoc, background-relative likelihood-ratio readout of a gated-LDA case-finding fit: read the learned topic-word counts (λ) as a per-node Naive-Bayes detector so coherent-but-mass-starved node topics can surface cases the θ-mass placement buries. Report LR-AUC vs θ-mass-AUC over a shrinkage-α sweep (the fork-settler), plus a per-code decomposition viewer and NPMI coherence — all on the EXISTING 0061/0062 runs, no re-fit.

**Architecture:** Pure-numpy engine functions (`spark_vi/models/topic/dag_placement.py`, id-agnostic — arrays + `DagLayout`, no concept-ids) compute the shrunk log-LR scores, the per-code decomposition, and the α-AUC sweep. A standalone in-enclave driver (`analysis/cloud/lr_readout.py`, the domain edge) loads a run's saved λ (`dag_placement_result.npz`) + the cached `CaseFindingBundle` (test BOW + concept names + node structure), computes the fork-settler + NPMI + writes a per-code viewer report, rendering engine-ids to concept names. A `make lr-readout ID=N` target runs it. Nothing re-fits; the fit already persisted everything needed.

**Tech Stack:** Python, numpy, scipy.sparse, PySpark (bundle load + NPMI reference RDD), pytest.

## Global Constraints

- Engine (`spark_vi`) stays id-agnostic: `lr_placement_scores` / `lr_decompose` / `lr_auc_sweep` take numpy/scipy arrays + `DagLayout` (engine node-ids) — never concept-ids. Concept-name rendering lives only in the `analysis/cloud` driver.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- No LaTeX; Unicode Greek only. Cite in docstrings: the score is a per-node Naive-Bayes log-likelihood-ratio; the shrinkage is Dirichlet / empirical-Bayes smoothing toward the background base rate.
- Branch `case-finding` does NOT auto-push; push only when the user asks.
- Exploratory research code (no prod). Structural tests only; do not gold-plate. The Spark / bundle-loading / name-rendering paths are cluster-covered (exercised in-enclave on the real runs), NOT unit-tested; the pure numpy helpers ARE unit-tested.
- **No re-fit / no model change:** this reads the saved λ. Do not touch `profile`, the model, the gate, windowing, or the fit driver's fit path.
- **α is swept, always including 0.** α=0 = MLE with an ε floor on log(0); rising α shrinks toward the base rate. No hardcoded default α is chosen — the α-AUC curve is the experiment.
- **In-enclave egress:** the driver writes row-level per-patient views to the run dir (stays in the enclave); only aggregates (LR-AUC(α), NPMI) are printed/summarized.
- Test harness — engine: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -q` (use `-k` for focus; the full file runs Gibbs, ~90s). Driver arg-surface: `.venv/bin/python -m pytest analysis/cloud/tests/test_lr_readout.py -q`.

---

### Task 1: Engine — shrunk log-LR scores + per-code decomposition

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add near the other stats helpers)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `lr_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9, count_mode="raw", length_normalize=False) -> np.ndarray [n_docs x n_nodes]` (columns = `lay.nodes` order). `lr_decompose(bow_row, lam, lay, u, *, alpha, background, epsilon=1e-9, count_mode="raw") -> list[(w, count, contribution)]`, sorted by |contribution| desc, summing to the raw score for (that doc, node u). `bow` is [n_docs x V] counts (numpy dense or scipy.sparse CSR); `lam` is the [K x V] fit λ.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from spark_vi.models.topic.dag_placement import (
    DagLayout, lr_placement_scores, lr_decompose)

def _lr_lay():
    return DagLayout({1: [0], 2: [0]}, n_bg=1, tpn=1)   # 2 nodes, blocks [1],[2]; K=3

def test_lr_scores_distinctive_code_separates_where_thetamass_would_not():
    lay = _lr_lay()
    V = 6
    lam = np.full((3, V), 1.0)          # bg topic 0 flat
    lam[1] = np.array([1, 1, 1, 40, 1, 1.0])   # node 1 signature = code 3
    lam[2] = np.array([1, 1, 1, 1, 40, 1.0])   # node 2 signature = code 4
    # background base rate: code 3 and 4 are globally rare, code 0 common
    bg = np.array([50, 10, 10, 1, 1, 1.0]); bg = bg / bg.sum()
    case = np.zeros(V); case[3] = 1; case[0] = 5     # has the node-1 signature + common noise
    ctrl = np.zeros(V); ctrl[0] = 6                  # only the common code
    S = lr_placement_scores(np.vstack([case, ctrl]), lam, lay, alpha=1.0, background=bg)
    # Raw LR scores can be negative (the shared common-code terms are penalised
    # under every node); what matters is RANKING. The case outranks the control on
    # node 1 (has its signature), and for the case node 1 (its signature) beats
    # node 2. That separation is exactly what the θ-mass readout misses.
    assert S[0, 0] > S[1, 0]                          # case > control on node 1
    assert S[0, 0] > S[0, 1]                          # case's node 1 (signature) > its node 2

def test_lr_scores_shrinkage_pulls_toward_zero():
    lay = _lr_lay()
    V = 5
    lam = np.full((3, V), 1.0); lam[1, 2] = 20.0       # node 1 likes code 2
    bg = np.array([10, 10, 1, 10, 10.0]); bg = bg / bg.sum()
    doc = np.zeros(V); doc[2] = 1
    s_small = lr_placement_scores(doc[None], lam, lay, alpha=0.0, background=bg)[0, 0]
    s_big = lr_placement_scores(doc[None], lam, lay, alpha=1e6, background=bg)[0, 0]
    assert s_small > s_big                              # strong shrinkage -> toward 0
    assert abs(s_big) < 1e-2                            # alpha huge -> ~neutral

def test_lr_scores_alpha_zero_unseen_code_is_finite():
    lay = _lr_lay()
    V = 4
    lam = np.full((3, V), 1.0); lam[1, 1] = 5.0
    lam[1, 3] = 0.0                                     # node 1 NEVER saw code 3
    bg = np.array([1, 1, 1, 1.0]) / 4
    doc = np.zeros(V); doc[3] = 1                       # patient has the unseen code
    s = lr_placement_scores(doc[None], lam, lay, alpha=0.0, background=bg)[0, 0]
    assert np.isfinite(s)                               # epsilon floor, not -inf

def test_lr_decompose_sums_to_score():
    lay = _lr_lay()
    V = 5
    lam = np.full((3, V), 1.0); lam[1] = np.array([1, 1, 20, 1, 5.0])
    bg = np.array([20, 10, 1, 5, 2.0]); bg = bg / bg.sum()
    doc = np.array([0, 1, 2, 0, 3.0])                   # counts
    parts = lr_decompose(doc, lam, lay, 1, alpha=1.0, background=bg)
    score = lr_placement_scores(doc[None], lam, lay, alpha=1.0, background=bg)[0, 0]
    assert abs(sum(c for _, _, c in parts) - score) < 1e-9
    assert all(cnt > 0 for _, cnt, _ in parts)          # only present codes listed
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "lr_scores or lr_decompose" -q`

- [ ] **Step 3: Implement** in `dag_placement.py`:

```python
def _lr_logratio_rows(lam, lay, *, alpha, bg, epsilon):
    """Per-node shrunk log-ratio row log[P(w|node u)/bg(w)], stacked [n_nodes x V].

    P(w|node u) = (Σ_{k in block(u)} λ[k,w] + α·bg(w)) / (Σλ(u) + α) — Dirichlet /
    empirical-Bayes smoothing toward the background base rate bg: large α pulls a
    mass-starved node toward bg (under-evidenced and unseen codes -> log-ratio ≈ 0),
    small α trusts the node's own counts. Floored at epsilon so α=0 never yields
    log(0)."""
    n_nodes = len(lay.nodes)
    logratio = np.zeros((n_nodes, lam.shape[1]))
    for i, u in enumerate(lay.nodes):
        nc = lam[lay.block[u]].sum(axis=0)
        p_u = (nc + alpha * bg) / (nc.sum() + alpha)
        logratio[i] = np.log(np.maximum(p_u, epsilon) / bg)
    return logratio


def _lr_base_rate(bow, background, epsilon):
    if background is None:
        col = np.asarray(bow.sum(axis=0)).ravel().astype(float)
        bg = col / max(col.sum(), 1.0)
    else:
        bg = np.asarray(background, dtype=float)
    return np.maximum(bg, epsilon)


def lr_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9,
                        count_mode="raw", length_normalize=False):
    """Per-node Naive-Bayes log-likelihood-ratio placement score.

    s(i,u) = Σ_w cnt(i,w)·log[P(w|node u)/bg(w)], reading the learned topic-word
    counts λ as class-conditional distributions (P(w|node u) = the node block's λ
    rows summed+normalized+shrunk toward bg; see `_lr_logratio_rows`). Unlike
    θ-mass this does not compete on the simplex, and the log-ratio down-weights
    common codes automatically (idf-for-free). `bow` [n_docs x V] counts (dense or
    scipy.sparse); `background` = base rate (None -> corpus code frequency from
    bow). count_mode 'raw'|'log1p' (saturate repeated codes); length_normalize
    divides by the per-doc token count. Returns [n_docs x n_nodes], columns in
    lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    bg = _lr_base_rate(bow, background, epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)
    X = bow
    if count_mode == "log1p":
        if hasattr(X, "data"):
            X = X.copy(); X.data = np.log1p(X.data)
        else:
            X = np.log1p(X)
    scores = np.asarray(X @ logratio.T, dtype=float)
    if length_normalize:
        tok = np.asarray(bow.sum(axis=1)).ravel().astype(float)
        scores = scores / np.maximum(tok, 1.0)[:, None]
    return scores


def lr_decompose(bow_row, lam, lay, u, *, alpha, background, epsilon=1e-9,
                 count_mode="raw"):
    """Itemized (w, count, contribution) for lr_placement_scores(...)[i, node u]
    (raw, no length-normalization). Σ contributions == that score. Only codes
    present in bow_row are returned, sorted by |contribution| desc."""
    lam = np.asarray(lam, dtype=float)
    bg = np.maximum(np.asarray(background, dtype=float), epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg,
                                 epsilon=epsilon)[lay.nodes.index(u)]
    row = np.asarray(bow_row).ravel().astype(float)
    cnt = np.log1p(row) if count_mode == "log1p" else row
    contrib = cnt * logratio
    out = [(int(w), float(row[w]), float(contrib[w])) for w in np.nonzero(row)[0]]
    out.sort(key=lambda t: -abs(t[2]))
    return out
```

- [ ] **Step 4: Run — expect PASS.**
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "lr_scores or lr_decompose" -q`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): shrunk log-LR placement scores + per-code decomposition

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Engine α-AUC sweep + standalone driver (fork-settler) + make target

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (`lr_auc_sweep`)
- Create: `analysis/cloud/lr_readout.py`
- Modify: `analysis/cloud/Makefile` (add `lr-readout` target)
- Test: `spark-vi/tests/test_dag_placement.py` (sweep); `analysis/cloud/tests/test_lr_readout.py` (arg surface)

**Interfaces:**
- Consumes: `lr_placement_scores` (Task 1), `_auc` (existing, dag_placement.py:229).
- Produces: `lr_auc_sweep(bow, lam, lay, is_fg, *, alpha_grid, background=None, count_mode="raw", length_normalize=False) -> dict{alpha: auc}` — case-vs-background ROC-AUC of the max-over-nodes LR score at each α. The driver `lr_readout.py`: loads a run's npz (λ) + manifest + cached bundle, runs the sweep, prints LR-AUC(α) beside the run's θ-mass detection AUC (`manifest["metrics"]["detection"]["auc"]`).

- [ ] **Step 1: Write the failing tests**

Engine sweep test (`test_dag_placement.py`):

```python
from spark_vi.models.topic.dag_placement import lr_auc_sweep

def test_lr_auc_sweep_separates_planted_cases():
    rng = np.random.default_rng(0)
    lay = _lr_lay()                                  # 2 nodes; BOTH informative so
    V = 8                                            # max-over-nodes has no flat node
    lam = np.full((3, V), 1.0)                       # to win at ~0 (a real-data hazard,
    lam[1, 5] = 60.0                                 # noted in the caveats)
    lam[2, 6] = 60.0                                 # node 1 -> code 5, node 2 -> code 6
    bg = np.full(V, 1.0) / V
    rows, is_fg = [], []
    for _ in range(20):                              # node-1 cases: code 5 + light noise
        d = np.zeros(V); d[5] = 1; d[0] = rng.integers(0, 2); rows.append(d); is_fg.append(True)
    for _ in range(20):                              # node-2 cases: code 6
        d = np.zeros(V); d[6] = 1; d[0] = rng.integers(0, 2); rows.append(d); is_fg.append(True)
    for _ in range(300):                             # controls: only the common code
        d = np.zeros(V); d[0] = rng.integers(1, 4); rows.append(d); is_fg.append(False)
    bow = np.array(rows)
    out = lr_auc_sweep(bow, lam, lay, np.array(is_fg),
                       alpha_grid=[0.0, 1.0, 10.0, 100.0], background=bg)
    assert set(out) == {0.0, 1.0, 10.0, 100.0}
    assert max(out.values()) > 0.9                   # SOME alpha cleanly separates the signal
```

Driver arg-surface test (`analysis/cloud/tests/test_lr_readout.py`) — no Spark:

```python
def test_lr_readout_arg_surface():
    import importlib
    mod = importlib.import_module("lr_readout")
    p = mod.build_parser()
    ns = p.parse_args(["--run-dir", "/runs/0061", "--alpha-grid", "0,1,10"])
    assert ns.run_dir == "/runs/0061"
    assert mod.parse_alpha_grid(ns.alpha_grid) == [0.0, 1.0, 10.0]
    # 0 is always included even if omitted
    assert 0.0 in mod.parse_alpha_grid("1,10")
```

- [ ] **Step 2: Run — expect FAIL.**
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k lr_auc_sweep -q` and `.venv/bin/python -m pytest analysis/cloud/tests/test_lr_readout.py -q`

- [ ] **Step 3: Implement.**

Engine (`dag_placement.py`):

```python
def lr_auc_sweep(bow, lam, lay, is_fg, *, alpha_grid, background=None,
                 count_mode="raw", length_normalize=False):
    """{alpha: case-vs-background ROC-AUC} of the max-over-nodes LR score, for each
    alpha in alpha_grid. The fork-settler vs the θ-mass detection AUC: LR-AUC ≫
    θ-AUC => signal present but buried (θ-mass was the wrong lens); LR-AUC ≈ θ-AUC
    => signal genuinely absent."""
    y = np.asarray(is_fg, dtype=int)
    out = {}
    for a in alpha_grid:
        s = lr_placement_scores(bow, lam, lay, alpha=float(a), background=background,
                                count_mode=count_mode, length_normalize=length_normalize)
        out[float(a)] = _auc(s.max(axis=1), y)
    return out
```

Driver (`analysis/cloud/lr_readout.py`) — reuse `make_spark_session` from `_driver_common`, `try_load` + `compute_bundle_cache_key` from `_case_finding_cache`, `DagLayout`/`lr_auc_sweep` from the engine. Structure:

1. `build_parser()` -> argparse with `--run-dir` (required), `--alpha-grid` (default "0,0.1,1,10 x medianΣλ" — see `parse_alpha_grid`), `--bundle-path` (override), `--count-mode`, `--length-normalize`, `--sample-cases` (for Task 3), `--person` (Task 3).
2. `parse_alpha_grid(s)` -> sorted list of floats, ALWAYS containing 0.0. When a token is the literal `median`, the caller scales it by median node Σλ after λ is loaded. (Concretely: parse the numeric multipliers; the driver multiplies non-zero entries by `median(node Σλ)` when `--alpha-grid` uses the `xmed` suffix; default grid = `[0, 0.1, 1, 10]` × median node Σλ, with 0 kept absolute.)
3. `load_run(run_dir)` -> `(lam, alpha_dirichlet, manifest)`: `np.load(run_dir/"dag_placement_result.npz")` for `lambda`/`alpha`; `json.load(run_dir/"manifest.json")` for config + `manifest["metrics"]["detection"]["auc"]`.
4. `locate_bundle(spark, manifest, bundle_path)` -> `CaseFindingBundle`: if `bundle_path`, load directly; else recompute the cache key via `compute_bundle_cache_key(**corpus_cfg_from_manifest)` and `try_load(spark, cache_uri, key)`; print a WARNING if the bundle is None or the manifest's assembly/DAG source-hash differs from the current one (the no-re-fit fragility), telling the user to pass `--bundle-path`.
5. Build the test BOW: `test_df.select("features","frontier")` -> scipy.sparse CSR `[n_docs x V]` (from the SparseVectors) + `is_fg = size(frontier) > 0`. Build `lay = DagLayout(bundle.parent_int, n_bg, tpn)` (n_bg/tpn from manifest).
6. Run `lr_auc_sweep(...)`; print a table: each α, LR-AUC, and the θ-mass detection AUC from the manifest; label the verdict (LR ≫ θ / LR ≈ θ). `main()` wires argparse -> the above; returns 0.

`analysis/cloud/Makefile` — add after `exp`:

```make
# Post-hoc likelihood-ratio readout on an EXISTING run (no re-fit). Reads the
# run's saved lambda + cached bundle; prints LR-AUC(alpha) vs theta-mass AUC.
lr-readout: zip cluster-overlay $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/analysis/cloud/lr_readout.py \
	    --run-dir $(RUNS_DIR)/$(shell printf '%04d' $(ID))-* $(LR_ARGS)
```

**Real-data note (for interpreting the cluster run, not the unit tests):** two
subtleties, both handled by the α sweep and worth surfacing in the printed
verdict. (1) The gated node β under-represents common codes (the gate sent them
to background during training), so at α=0 common codes are over-penalised
(log-ratio very negative) — α>0 shrinkage neutralises them toward 0, so LR-AUC is
EXPECTED to be low at α=0 and rise with α; that curve is informative, not a bug.
(2) Floor/junk node topics (Σλ≈55, near-flat) score ≈0 for everyone and can win
the max-over-nodes, masking real signal; if LR-AUC is surprisingly low, a cheap
follow-up is to mask nodes below a Σλ floor (or via the NPMI ranking) before the
max — out of scope for v1, noted so the reviewer/user reads a low number
correctly.

- [ ] **Step 4: Run — expect PASS**, plus syntax-check the driver:
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k lr_auc_sweep -q`
Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_lr_readout.py -q`
Run: `.venv/bin/python -c "import ast; ast.parse(open('analysis/cloud/lr_readout.py').read()); print('ast OK')"`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py analysis/cloud/lr_readout.py analysis/cloud/tests/test_lr_readout.py analysis/cloud/Makefile
git commit -m "feat(lr-readout): alpha-AUC sweep + standalone fork-settler driver + make target

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Per-code viewer + NPMI coherence (in the driver)

**Files:**
- Modify: `analysis/cloud/lr_readout.py`
- Test: `analysis/cloud/tests/test_lr_readout.py`

**Interfaces:**
- Consumes: `lr_decompose` (Task 1), `compute_npmi_coherence` (existing, `spark_vi.eval.topic`), `BOWDocument.from_spark_row` (existing, `spark_vi.models.topic.types`).
- Produces: in `lr_readout.py`, a per-code viewer that writes, for a sample of known foreground cases (or `--person` ids), each case's `lr_decompose` for its true frontier node(s) rendered with concept NAMES (via `bundle.name_by_id` / `int2cid` / `vocab_map`) to `{run_dir}/lr_readout/decompose.txt` (in-enclave, stays there); and an NPMI table printed for all topics (β = λ row-normalized), node topics labeled, sorted by coherence.

- [ ] **Step 1: Write the failing test** (`analysis/cloud/tests/test_lr_readout.py`) — pure rendering, no Spark:

```python
def test_render_decompose_rows_uses_concept_names():
    import importlib
    mod = importlib.import_module("lr_readout")
    # (w, count, contribution) engine tuples + a vocab-index -> concept-id -> name chain
    rows = [(0, 2.0, 5.9), (1, 3.0, -3.9)]
    idx_to_cid = {0: 111, 1: 222}
    name_by_id = {111: "Lupus erythematosus", 222: "Essential hypertension"}
    lines = mod.render_decompose_rows(rows, idx_to_cid, name_by_id)
    assert "Lupus erythematosus" in lines[0] and "+5.9" in lines[0]
    assert "Essential hypertension" in lines[1] and "-3.9" in lines[1]
```

- [ ] **Step 2: Run — expect FAIL.**
Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_lr_readout.py -k render_decompose -q`

- [ ] **Step 3: Implement** in `lr_readout.py`:

1. `render_decompose_rows(rows, idx_to_cid, name_by_id) -> list[str]` — pure: for each `(w, count, contribution)`, format `f"{sign}{contribution:+.1f}  x{count:g}  {name_by_id.get(idx_to_cid.get(w), idx_to_cid.get(w, w))}"`. Sorted input is preserved.
2. `write_case_viewer(run_dir, bundle, lam, lay, cases, *, alpha, background)` — for each selected case (person + its BOW row + its frontier node(s)), call `lr_decompose` for each true frontier node, render with names, and write to `{run_dir}/lr_readout/decompose.txt`. Hash person ids in any printed/log line (row-level hygiene); the written file stays in the run dir. Sample size from `--sample-cases` (default a small N), or specific `--person` ids.
3. `npmi_table(spark, lam, bundle, topic_labels, *, top_n=20)` — β = `lam / lam.sum(1, keepdims=True)`; `ref = bundle.test_df.select("features").rdd.map(BOWDocument.from_spark_row)` (cache it); `report = compute_npmi_coherence(beta, ref, top_n=top_n)`; return rows `(topic, label, npmi)` sorted desc, with node topics labeled from `lay`/`name_by_id`. Print it.
4. Wire both into `main()` after the fork-settler: always print the NPMI table; write the viewer when `--sample-cases`/`--person` is set.

- [ ] **Step 4: Run — expect PASS.**
Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_lr_readout.py -q`
Run: `.venv/bin/python -c "import ast; ast.parse(open('analysis/cloud/lr_readout.py').read()); print('ast OK')"`

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/lr_readout.py analysis/cloud/tests/test_lr_readout.py
git commit -m "feat(lr-readout): per-code decomposition viewer + NPMI coherence table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Task 1 = the LR score + per-code decomposition (spec "The likelihood-ratio readout", engine component 1); Task 2 = α-AUC sweep + the standalone driver + make target (spec "α is a swept diagnostic", the fork-settler, component 2 loading); Task 3 = per-code viewer + NPMI (spec components (b), (c)). "Load existing / no re-fit" is honored — every input is read from the run's npz + cached bundle; the `--bundle-path` override + source-hash warning covers the documented fragility. α always includes 0 (`parse_alpha_grid`). Engine stays id-agnostic; name rendering is driver-only.

**Placeholder scan:** Task 1 carries complete engine code + tests; Tasks 2–3 give the driver structure concretely with the reused primitives named (make_spark_session, try_load/compute_bundle_cache_key, compute_npmi_coherence, BOWDocument.from_spark_row, _auc) — the Spark/bundle body is cluster-covered by design, with pure helpers (`parse_alpha_grid`, `render_decompose_rows`, `lr_auc_sweep`) unit-tested.

**Type consistency:** `lr_placement_scores`/`lr_decompose` (Task 1) consumed by `lr_auc_sweep` (Task 2) and the viewer (Task 3); `bow` is [n_docs x V] counts throughout, columns of the score = `lay.nodes`; `lr_decompose` returns `(w, count, contribution)` consumed by `render_decompose_rows`; the α grid (list incl 0.0) is consistent between `parse_alpha_grid` and `lr_auc_sweep`.

**Note on the plan's code:** the Task-1 code is written but unexecuted; earlier plans in this project had genuine bugs in plan code (a degenerate test, an ECDF tie error, an f-string). Implementers should TDD and fix genuine plan-code bugs with a minimal, documented change, as before.
