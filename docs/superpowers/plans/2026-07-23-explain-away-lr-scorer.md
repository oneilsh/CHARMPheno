# Explain-away LR Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a responsibility-weighted ("explain-away") LR placement scorer so comorbid codes route to background and stop penalizing/supporting foreground nodes, and report it beside plain LR in the case-finding readout.

**Architecture:** Reuse the existing LR machinery (`_lr_logratio_rows`, `_lr_base_rate`) for the evidence term; add a routing matrix `Rnode[u,w]` from the normalized topic-word distributions; the score is `cnt(bow) @ (Rnode ⊙ logratio)ᵀ`. Post-hoc readout on saved λ — no engine re-fit.

**Tech Stack:** Python, NumPy, scipy.sparse, PySpark driver, pytest. Spec: `docs/superpowers/specs/2026-07-23-explain-away-lr-scorer-design.md`.

## Global Constraints

- Engine code (`spark_vi/**`) is integer-id agnostic: no OMOP/concept ids, no vocabulary. Integer topic/node/code space only.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- No LaTeX; plain ASCII + Unicode Greek (α, λ, Σ) only.
- Cite literature for methods (Pearl 1988 "explain-away"; the mixture-model E-step soft assignment; existing Naive-Bayes LR refs).
- Tests: never loosen a threshold to pass; fix the implementation.
- Soft routing only; uniform topic prior; routing on raw normalized λ, evidence keeps the α handling (incl. α=inf lift limit).
- No engine re-fit, no new experiment; the readout runs post-hoc on exp 0067's saved λ.

## Canonical definitions

- `lay.nodes`: sorted node ids; `lay.block[u]`: the `tpn` topic-row indices for node u; `lay.K`, `lay.n_bg`.
- Evidence `logratio[u,w]` from `_lr_logratio_rows(lam, lay, alpha=..., bg=..., epsilon=...)` — [n_nodes x V], α=inf gives the lift limit.
- Base rate `bg` from `_lr_base_rate(bow, background, epsilon)`.

---

### Task 1: `_routing_rows` — soft per-node responsibility

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add function near `_lr_logratio_rows`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `_routing_rows(lam, lay, *, epsilon=1e-9) -> np.ndarray` shape `[n_nodes x V]`, rows in `lay.nodes` order.

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_dag_placement.py`:

```python
def test_routing_rows_soft_responsibility_and_conservation():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout, _routing_rows
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)               # K=3: [bg, node1, node2]
    V = 3
    lam = np.zeros((3, V))
    lam[0] = [8.0, 0.0, 2.0]     # background topic: code0 (and some code2)
    lam[1] = [0.0, 5.0, 0.0]     # node1 topic: code1 only (distinctive)
    lam[2] = [0.0, 0.0, 6.0]     # node2 topic: code2 only
    r = _routing_rows(lam, lay)                                # [2 nodes x V]
    # code1 is emitted only by node1's topic -> fully node1
    assert np.isclose(r[0, 1], 1.0) and np.isclose(r[1, 1], 0.0)
    # code0 is emitted only by background -> neither node claims it
    assert np.isclose(r[0, 0], 0.0) and np.isclose(r[1, 0], 0.0)
    # code2 is shared by background (P=2/10=0.2) and node2 (P=6/6=1.0):
    # node2 responsibility = 1.0 / (0.2 + 1.0) = 0.8333...
    assert np.isclose(r[1, 2], 1.0 / 1.2, atol=1e-6)
    # conservation: node responsibilities + background responsibility = 1 per seen code
    #   (background resp = 1 - sum of node resp); must be in [0,1].
    node_sum = r.sum(axis=0)
    assert np.all(node_sum <= 1.0 + 1e-9) and np.all(node_sum >= -1e-9)
    assert np.isclose(node_sum[2], 1.0 / 1.2, atol=1e-6)       # only node2 (+bg) see code2
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_routing_rows_soft_responsibility_and_conservation -q`
Expected: FAIL (`_routing_rows` not defined).

- [ ] **Step 3: Implement `_routing_rows`**

Add to `spark-vi/spark_vi/models/topic/dag_placement.py` (near `_lr_logratio_rows`):

```python
def _routing_rows(lam, lay, *, epsilon=1e-9):
    """Per-node soft responsibility Rnode[u,w] = fraction of code w's total
    topic-probability that lands on node u's topic block. Codes compete across ALL
    topics (background + nodes) with a UNIFORM topic prior (responsibility ∝
    P(w|topic) = λ[k]/Σλ[k]); a code unseen in every topic -> 0 everywhere. This is
    the "explain-away" routing (Pearl 1988; the mixture E-step's soft assignment):
    a comorbid code claimed by a background topic gets ~0 node responsibility, so it
    neither penalizes nor spuriously supports a foreground node. Returns [n_nodes x V]
    in lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    ptopic = lam / np.maximum(lam.sum(axis=1, keepdims=True), epsilon)  # P(w|topic k)
    rtopic = ptopic / np.maximum(ptopic.sum(axis=0, keepdims=True), epsilon)  # responsibility
    rnode = np.zeros((len(lay.nodes), lam.shape[1]), dtype=float)
    for i, u in enumerate(lay.nodes):
        rnode[i] = rtopic[lay.block[u]].sum(axis=0)
    return rnode
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_routing_rows_soft_responsibility_and_conservation -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): _routing_rows soft explain-away responsibility

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `explain_away_placement_scores`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `_routing_rows` (Task 1); `_lr_logratio_rows`, `_lr_base_rate` (existing).
- Produces: `explain_away_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9, count_mode="raw", length_normalize=False) -> np.ndarray` [n_docs x n_nodes].

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_dag_placement.py`:

```python
def test_explain_away_suppresses_comorbid_negatives():
    # A doc = 1 distinctive rare code (d, emitted by node1) + several generic codes
    # (g*, emitted by background). Plain LR docks node1 for the generic codes (they
    # are below base rate under node1); explain-away routes them to background, so
    # they contribute ~0 -> explain-away score(node1) >= plain LR score(node1).
    import numpy as np
    from spark_vi.models.topic.dag_placement import (
        DagLayout, explain_away_placement_scores, lr_placement_scores)
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)               # K=3
    V = 4                                                      # d=0, g1=1, g2=2, g3=3
    lam = np.zeros((3, V))
    lam[0] = [0.0, 40.0, 40.0, 40.0]   # background: the generic codes
    lam[1] = [30.0, 0.0, 0.0, 0.0]     # node1: distinctive code d only
    lam[2] = [0.0, 1.0, 1.0, 1.0]      # node2: weak/uniform
    bow = np.zeros((1, V)); bow[0] = [1, 1, 1, 1]              # d + 3 generic codes
    bg = np.array([0.10, 0.30, 0.30, 0.30])                   # base rate (d rarer)
    i = lay.nodes.index(1)
    lr = lr_placement_scores(bow, lam, lay, alpha=float("inf"), background=bg)[0, i]
    ea = explain_away_placement_scores(bow, lam, lay, alpha=float("inf"),
                                       background=bg)[0, i]
    assert ea >= lr - 1e-9        # comorbid negatives suppressed
    assert ea > 0.0               # the distinctive code still carries positive signal
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_explain_away_suppresses_comorbid_negatives -q`
Expected: FAIL (`explain_away_placement_scores` not defined).

- [ ] **Step 3: Implement**

Add to `spark-vi/spark_vi/models/topic/dag_placement.py`:

```python
def explain_away_placement_scores(bow, lam, lay, *, alpha, background=None,
                                  epsilon=1e-9, count_mode="raw",
                                  length_normalize=False):
    """Explain-away (responsibility-weighted) LR placement score:
    s(i,u) = Σ_w cnt(i,w) · r(u|w) · log[P(w|u)/bg(w)], where r(u|w) is code w's soft
    responsibility on node u's block (_routing_rows). Codes competing to a background
    topic (comorbidities) get r(u|w) ~ 0, so their evidence -- crucially the SMALL
    NEGATIVE log-ratios that make the plain LR penalize comorbidity-heavy patients --
    is suppressed toward 0 instead of docking the node. Same signature/shape as
    lr_placement_scores; the α->∞ lift limit applies to the evidence term, routing is
    α-independent (raw normalized λ). Returns [n_docs x n_nodes], lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    bg = _lr_base_rate(bow, background, epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)
    weight = _routing_rows(lam, lay, epsilon=epsilon) * logratio   # Rnode ⊙ logratio
    X = bow
    if count_mode == "log1p":
        if hasattr(X, "data"):
            X = X.copy(); X.data = np.log1p(X.data)
        else:
            X = np.log1p(X)
    scores = np.asarray(X @ weight.T, dtype=float)
    if length_normalize:
        tok = np.asarray(bow.sum(axis=1)).ravel().astype(float)
        scores = scores / np.maximum(tok, 1.0)[:, None]
    return scores
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_explain_away_suppresses_comorbid_negatives -q`
Expected: PASS.

- [ ] **Step 5: Add a no-signal guard test**

```python
def test_explain_away_background_only_doc_scores_near_zero():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout, explain_away_placement_scores
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)
    V = 3
    lam = np.zeros((3, V)); lam[0] = [5.0, 5.0, 5.0]           # only background has mass
    bow = np.zeros((1, V)); bow[0] = [1, 1, 1]
    s = explain_away_placement_scores(bow, lam, lay, alpha=float("inf"),
                                      background=np.array([0.34, 0.33, 0.33]))
    assert np.allclose(s, 0.0, atol=1e-6)                      # nodes have no routing -> ~0
```

- [ ] **Step 6: Run both**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -q -k explain_away`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): explain_away_placement_scores (routing-weighted LR)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `explain_away_decompose`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `_routing_rows`, `_lr_logratio_rows`, `_lr_base_rate`.
- Produces: `explain_away_decompose(bow_row, lam, lay, u, *, alpha, background, epsilon=1e-9, count_mode="raw") -> list[(int w, float count, float r_u_w, float contribution)]`, sorted by |contribution| desc; Σ contribution == the score for node u.

- [ ] **Step 1: Write the failing test**

```python
def test_explain_away_decompose_shows_routing_and_sums_to_score():
    import numpy as np
    from spark_vi.models.topic.dag_placement import (
        DagLayout, explain_away_decompose, explain_away_placement_scores)
    lay = DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)
    V = 4
    lam = np.zeros((3, V))
    lam[0] = [0.0, 40.0, 40.0, 40.0]; lam[1] = [30.0, 0.0, 0.0, 0.0]; lam[2] = [0.0, 1.0, 1.0, 1.0]
    row = np.array([1.0, 1.0, 1.0, 1.0])
    bg = np.array([0.10, 0.30, 0.30, 0.30])
    rows = explain_away_decompose(row, lam, lay, 1, alpha=float("inf"), background=bg)
    by_w = {w: (cnt, r, c) for (w, cnt, r, c) in rows}
    # distinctive code d=0 routes to node1 (r ~ 1), positive contribution
    assert by_w[0][1] > 0.9 and by_w[0][2] > 0.0
    # generic codes route to background (r ~ 0) -> contribution ~ 0 (not negative)
    for g in (1, 2, 3):
        assert abs(by_w[g][1]) < 0.05 and abs(by_w[g][2]) < 1e-3
    # Σ contribution == the node score
    total = sum(c for (_w, _cnt, _r, c) in rows)
    score = explain_away_placement_scores(row[None], lam, lay, alpha=float("inf"),
                                          background=bg)[0, lay.nodes.index(1)]
    assert np.isclose(total, score, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_explain_away_decompose_shows_routing_and_sums_to_score -q`
Expected: FAIL (`explain_away_decompose` not defined).

- [ ] **Step 3: Implement**

```python
def explain_away_decompose(bow_row, lam, lay, u, *, alpha, background,
                           epsilon=1e-9, count_mode="raw"):
    """Itemized (w, count, r(u|w), contribution) for
    explain_away_placement_scores(...)[i, node u]. contribution = cnt · r(u|w) ·
    log[P(w|u)/bg(w)]; Σ contribution == that node's score. r(u|w) is the routing
    weight (0 = the code went to background/another node; ~1 = it belongs to u), so
    the viewer can show WHERE each code routed. Only codes present in bow_row are
    returned, sorted by |contribution| desc."""
    lam = np.asarray(lam, dtype=float)
    bg = np.maximum(np.asarray(background, dtype=float), epsilon)
    i = lay.nodes.index(u)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)[i]
    rnode = _routing_rows(lam, lay, epsilon=epsilon)[i]
    row = np.asarray(bow_row).ravel().astype(float)
    cnt = np.log1p(row) if count_mode == "log1p" else row
    contrib = cnt * rnode * logratio
    out = [(int(w), float(row[w]), float(rnode[w]), float(contrib[w]))
           for w in np.nonzero(row)[0]]
    out.sort(key=lambda t: -abs(t[3]))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_explain_away_decompose_shows_routing_and_sums_to_score -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): explain_away_decompose with per-code routing weight

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Readout wiring — side-by-side detection block + viewer score-mode

**Files:**
- Modify: `analysis/cloud/lr_readout.py` (`render_decompose_rows`, `detection_report`, `_render_case`, `write_case_viewer`/`write_case_viewer_by_class`, `build_parser`, `main`)
- Test: `analysis/cloud/tests/test_lr_readout.py`

**Interfaces:**
- Consumes: `explain_away_placement_scores`, `explain_away_decompose` (Tasks 2-3).
- Produces: `--viewer-score-mode {lr,explain_away}` (default `lr`); an explain-away detection block printed beside plain LR; `render_decompose_rows` accepting 3- or 4-tuples.

- [ ] **Step 1: Write the failing test (render_decompose_rows handles the 4-tuple)**

Add to `analysis/cloud/tests/test_lr_readout.py`:

```python
def test_render_decompose_rows_handles_routing_tuple():
    import importlib
    mod = importlib.import_module("lr_readout")
    idx_to_cid = {0: 100, 1: 200}
    name_by_id = {100: "Distinctive code", 200: "Generic code"}
    # 4-tuples: (w, count, r_u_w, contribution)
    rows = [(0, 1.0, 0.95, 3.2), (1, 4.0, 0.02, -0.01)]
    lines = mod.render_decompose_rows(rows, idx_to_cid, name_by_id)
    assert any("Distinctive code" in ln and "r=0.95" in ln for ln in lines)
    assert any("Generic code" in ln and "r=0.02" in ln for ln in lines)
    # backward compatible with the 3-tuple (no routing column)
    lines3 = mod.render_decompose_rows([(0, 1.0, 3.2)], idx_to_cid, name_by_id)
    assert any("Distinctive code" in ln for ln in lines3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd analysis/cloud/tests && python -m pytest test_lr_readout.py::test_render_decompose_rows_handles_routing_tuple -q`
Expected: FAIL (`render_decompose_rows` only handles 3-tuples / no `r=` in output).

- [ ] **Step 3: Update `render_decompose_rows` for the optional routing column**

Replace `render_decompose_rows` in `analysis/cloud/lr_readout.py`:

```python
def render_decompose_rows(rows, idx_to_cid, name_by_cid) -> list[str]:
    """[(w, count, contribution)] OR [(w, count, r_u_w, contribution)] (the
    explain-away form, with the routing weight r(u|w)) -> rendered lines with
    concept NAMES for the raw vocab indices. `idx_to_cid` = vocab-idx -> concept-id;
    `name_by_cid` = concept-id -> name (full vocabulary). Falls back to concept id
    then vocab index. Pure string formatting; order preserved."""
    lines = []
    for r in rows:
        if len(r) == 4:
            w, count, r_uw, contribution = r
            cid = idx_to_cid.get(w, w)
            name = name_by_cid.get(cid, cid)
            lines.append(f"{contribution:+.1f}  x{count:g}  r={r_uw:.2f}  {name}")
        else:
            w, count, contribution = r
            cid = idx_to_cid.get(w, w)
            name = name_by_cid.get(cid, cid)
            lines.append(f"{contribution:+.1f}  x{count:g}  {name}")
    return lines
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd analysis/cloud/tests && python -m pytest test_lr_readout.py::test_render_decompose_rows_handles_routing_tuple -q`
Expected: PASS.

- [ ] **Step 5: Add the explain-away detection block to `detection_report`**

In `analysis/cloud/lr_readout.py`, at the end of `detection_report` (after the plain-LR `_ops` prints), add an explain-away block computed the same way:

```python
    # Explain-away (responsibility-weighted) LR, at the alpha->inf lift limit, beside
    # plain LR: does routing comorbid codes to background lift detection?
    from spark_vi.models.topic.dag_placement import explain_away_placement_scores
    ea = explain_away_placement_scores(
        bow, lam, lay, alpha=float("inf"), background=background,
        count_mode=count_mode, length_normalize=length_normalize)
    ea_det = _detection_metrics(ea.max(axis=1), np.asarray(is_fg, dtype=bool))
    _ops(ea_det, "explain-away @alpha=inf")
```

(`_ops` is the existing inner printer; it takes the detection dict + a label. Ensure it is in scope — if `_ops` is a local closure, call it before `detection_report` returns.)

- [ ] **Step 6: Thread `score_mode` into the case viewer**

In `_render_case` (add param `score_mode="lr"`): when `score_mode == "explain_away"`, use `explain_away_placement_scores` for the ranking scores and `explain_away_decompose` for the per-code `_decomp` (its 4-tuples flow through the updated `render_decompose_rows`); also print the plain-LR max score on the header line for contrast. Thread `score_mode` through `write_case_viewer` and `write_case_viewer_by_class` (and use it for the classification `scores` in `main` when set). Add `--viewer-score-mode {lr,explain_away}` (default `lr`) in `build_parser`, and pass `args.viewer_score_mode` at both viewer call sites + the classification `lr_placement_scores`/`explain_away_placement_scores` branch in `main`.

- [ ] **Step 7: Run the readout test file + a compile check**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m py_compile analysis/cloud/lr_readout.py && cd analysis/cloud/tests && python -m pytest test_lr_readout.py -q`
Expected: all pass (render 3-/4-tuple, arg surface, classify, ranking).

- [ ] **Step 8: Commit**

```bash
git add analysis/cloud/lr_readout.py analysis/cloud/tests/test_lr_readout.py
git commit -m "feat(lr-readout): explain-away detection block + --viewer-score-mode

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** `_routing_rows` (T1), `explain_away_placement_scores` (T2), `explain_away_decompose` (T3), detection side-by-side + viewer score-mode + render 4-tuple (T4). Soft routing / uniform prior / raw-λ routing / α=inf evidence all in T1-T2. Validation: comorbid-suppression + routing-conservation + no-signal guard (T1-T3), cluster A/B via T4's detection block. ✓
- **Placeholder scan:** T4 Steps 5-6 describe threading in prose rather than full literals because they touch many existing call sites already in the file; the implementer has the exact function names and the pattern (mirror the existing `count_mode` threading committed in 4c93a86). All new-function steps have complete code. If the reviewer wants stricter literals for T4, expand from the current file.
- **Type consistency:** `explain_away_placement_scores`/`_decompose`/`_routing_rows` signatures identical across definition (T1-T3) and use (T4). `render_decompose_rows` accepts 3- and 4-tuples. ✓
