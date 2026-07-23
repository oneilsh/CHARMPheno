# LR-FDR readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the existing empirical-null per-node FDR to the LR and explain-away score matrices (not just theta-mass), reporting discoveries side-by-side, to test whether LR's detection edge yields FDR-controlled discoveries where theta-mass found zero.

**Architecture:** Extract `evaluate`'s inline FDR `by_q`/multimorbidity reporting into a reusable engine function `fdr_discovery_report`; `evaluate` (theta-mass) and `lr_readout` (LR + explain-away) both call it — one id-agnostic FDR path, guaranteed like-for-like. Driver builds truth/mm_rows/lengths from the held-out frontiers. Post-hoc on saved λ, no re-fit.

**Tech Stack:** Python, NumPy, scipy, PySpark driver, pytest. Spec: `docs/superpowers/specs/2026-07-23-lr-fdr-readout-design.md`.

## Global Constraints

- Engine code (`spark_vi/**`) is integer-id agnostic: no concept ids/vocabulary; integer node/doc/topic space only.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- No LaTeX; plain ASCII + Unicode Greek (α, λ, Σ) only.
- Cite literature already cited in-file (Efron two-groups empirical null; BH 1995 / BY 2001 — do not re-derive).
- Tests: never loosen a threshold/assertion to pass; fix the implementation.
- The refactor MUST leave `evaluate`'s output byte-identical (existing `evaluate` FDR tests stay green).

## Canonical definitions

- `evaluate` currently builds the FDR block inline (`spark-vi/spark_vi/models/topic/dag_placement.py`, the section from `# --- FDR readout` through the `fdr_block = {...}` dict, roughly lines 779-833). It computes `node_list=lay.nodes`, `lengths`, `nlb`, `q_grid`, `disc = per_node_discoveries(...)`, `truth` [n_docs x n_nodes] (`node_pos[u][i]` = `bool(fronts[i] & lay.subtree(u))`), the `by_q` loop, the `multimorbidity` block (using `mm_rows = [len(f & set(node_list)) >= 2 ...]`), `zib_gap`, and assembles `fdr_block`.
- `per_node_discoveries(P, is_fg, doc_lengths, *, q_grid, n_length_bins=4, method="bh")` and `_zib_empirical_gap` already exist (unchanged).
- Driver: `build_test_bow(bundle, vocab_size, lay) -> (bow CSR, is_fg bool[n], meta)` where `meta[i] = (person_id, sv, frontier_engine_ids list)`; `is_fg[i] = bool(set(frontier) & set(lay.nodes))`. `lr_placement_scores` / `explain_away_placement_scores` return `[n_docs x n_nodes]`.

---

### Task 1: Extract `fdr_discovery_report` (engine) + refactor `evaluate` to call it

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `fdr_discovery_report(P, is_fg, doc_lengths, truth, mm_rows, *, q_grid=(0.05,0.10,0.20), n_length_bins=4, method="bh") -> dict` with keys `by_q`, `multimorbidity`, `saturation_rate`, `zib_gap_mean`, `zib_gap_max`, `n_length_bins_effective` (the exact `fdr_block` `evaluate` returns today).
- Consumes: `per_node_discoveries`, `_zib_empirical_gap` (existing).

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_dag_placement.py`:

```python
def test_fdr_discovery_report_planted_and_null():
    import numpy as np
    from spark_vi.models.topic.dag_placement import fdr_discovery_report
    n_bg_docs, n_fg_docs, n_nodes = 200, 40, 3
    rng = np.random.default_rng(0)
    # background docs: low mass on every node; foreground docs (all true positives for
    # node 0) sit clearly above the background null on node 0 only.
    P = np.abs(rng.normal(0.0, 0.01, size=(n_bg_docs + n_fg_docs, n_nodes)))
    P[n_bg_docs:, 0] = 0.9 + np.abs(rng.normal(0, 0.01, size=n_fg_docs))   # planted signal
    is_fg = np.zeros(n_bg_docs + n_fg_docs, dtype=bool); is_fg[n_bg_docs:] = True
    truth = np.zeros((n_bg_docs + n_fg_docs, n_nodes), dtype=bool)
    truth[n_bg_docs:, 0] = True                                            # fg docs are node-0 positives
    mm_rows = np.zeros(n_bg_docs + n_fg_docs, dtype=bool)                  # none multimorbid
    lengths = np.ones(n_bg_docs + n_fg_docs)
    rep = fdr_discovery_report(P, is_fg, lengths, truth, mm_rows,
                               q_grid=(0.05, 0.10, 0.20), n_length_bins=1)
    # planted node-0 signal -> discoveries at q=0.20 with precision 1.0 (only true node-0 docs)
    assert rep["by_q"][0.20]["n_discoveries"] >= 1
    assert rep["by_q"][0.20]["precision"] == 1.0
    assert set(rep.keys()) == {"by_q", "multimorbidity", "saturation_rate",
                               "zib_gap_mean", "zib_gap_max", "n_length_bins_effective"}


def test_fdr_discovery_report_all_null_no_discoveries():
    import numpy as np
    from spark_vi.models.topic.dag_placement import fdr_discovery_report
    n, n_nodes = 120, 2
    rng = np.random.default_rng(1)
    P = np.abs(rng.normal(0.0, 0.01, size=(n, n_nodes)))     # no fg/bg separation
    is_fg = np.zeros(n, dtype=bool); is_fg[100:] = True
    truth = np.zeros((n, n_nodes), dtype=bool); truth[100:, 0] = True
    mm_rows = np.zeros(n, dtype=bool)
    rep = fdr_discovery_report(P, is_fg, np.ones(n), truth, mm_rows,
                               q_grid=(0.05, 0.10, 0.20), n_length_bins=1)
    assert all(rep["by_q"][q]["n_discoveries"] == 0 for q in (0.05, 0.10, 0.20))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -q -k fdr_discovery_report`
Expected: FAIL (`fdr_discovery_report` not defined).

- [ ] **Step 3: Extract the function; refactor `evaluate` to call it**

In `spark-vi/spark_vi/models/topic/dag_placement.py`, add the new function (place it just BEFORE `evaluate`):

```python
def fdr_discovery_report(P, is_fg, doc_lengths, truth, mm_rows, *,
                         q_grid=(0.05, 0.10, 0.20), n_length_bins=4, method="bh"):
    """Length-conditioned, background-relative per-node FDR discovery report for ANY
    [n_docs x n_nodes] score matrix P (theta-mass, LR, or explain-away). Reuses the
    Efron two-groups empirical null (per_node_discoveries): background docs are the
    per-node/per-length-bin null, BH (or BY) per node column. Returns the by_q
    precision/recall (vs `truth` = subtree-membership positives), the multimorbidity
    payoff (mean TRUE node-discoveries per truly-multimorbid patient `mm_rows` vs the
    argmax true-hit baseline), the p-floor saturation rate, and the zero-inflated-Beta
    KS diagnostic. `truth` [n_docs x n_nodes] bool and `mm_rows` [n_docs] bool are
    supplied by the caller (mm_rows is NOT derivable from truth: a single deep frontier
    node makes truth true for all its ancestors). This is the exact block evaluate used
    to build inline; both evaluate (theta-mass) and the LR readout (LR/explain-away)
    call it so the comparison is like-for-like."""
    P = np.asarray(P, dtype=float)
    is_fg = np.asarray(is_fg, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    mm_rows = np.asarray(mm_rows, dtype=bool)
    n_nodes = P.shape[1]
    q_grid = list(q_grid)
    disc = per_node_discoveries(P, is_fg, np.asarray(doc_lengths, dtype=float),
                                q_grid=q_grid, n_length_bins=n_length_bins,
                                method=method)
    by_q = {}
    for q in q_grid:
        m = disc["discoveries"][q]
        ndisc = int(m.sum())
        tp = int((m & truth).sum())
        total_pos = int(truth.sum())
        by_q[q] = {
            "n_discoveries": ndisc,
            "precision": float(tp / ndisc) if ndisc else float("nan"),
            "recall": float(tp / total_pos) if total_pos else float("nan")}
    q_mid = q_grid[len(q_grid) // 2]
    if mm_rows.any():
        m_mid = disc["discoveries"][q_mid]
        mean_true_disc = float((m_mid & truth)[mm_rows].sum(axis=1).mean())
        mean_total_disc = float(m_mid[mm_rows].sum(axis=1).mean())
        argmax_node = np.argmax(P[mm_rows], axis=1)
        argmax_tp = truth[mm_rows][np.arange(mm_rows.sum()), argmax_node]
        argmax_base = float(argmax_tp.mean())
    else:
        mean_true_disc = mean_total_disc = argmax_base = float("nan")
    gaps = [_zib_empirical_gap(P[~is_fg, u]) for u in range(n_nodes)]
    gaps = [g for g in gaps if not np.isnan(g)]
    return {
        "by_q": by_q,
        "multimorbidity": {
            "mean_true_discoveries_per_multimorbid": mean_true_disc,
            "argmax_true_baseline_per_multimorbid": argmax_base,
            "mean_total_discoveries_per_multimorbid": mean_total_disc},
        "saturation_rate": float(disc["floor"][disc["discoveries"][q_mid]].mean())
            if disc["discoveries"][q_mid].any() else float("nan"),
        "zib_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "zib_gap_max": float(np.max(gaps)) if gaps else float("nan"),
        "n_length_bins_effective": int(len(np.unique(disc["bins"]))),
    }
```

Then in `evaluate`, REPLACE the inline FDR block (from `# --- FDR readout` through the `fdr_block = {...}` assignment) with: keep the `node_list`, `lengths`, `nlb`, `q_grid`, `truth`, and `mm_rows` computations, then call the helper. Concretely, evaluate should end up with:

```python
    # --- FDR readout: background-relative, per-node, multiple-testing corrected -
    node_list = lay.nodes
    lengths = (np.asarray(doc_lengths, dtype=float) if doc_lengths is not None
               else np.ones(len(fronts)))
    nlb = n_length_bins if doc_lengths is not None else 1
    truth = np.array([[node_pos[u][i] for u in node_list]
                      for i in range(len(fronts))], dtype=bool)
    mm_rows = np.array([len(f & set(node_list)) >= 2 for f in fronts])
    fdr_block = fdr_discovery_report(P, is_fg, lengths, truth, mm_rows,
                                     q_grid=tuple(fdr_q_grid), n_length_bins=nlb)
```

(Delete the now-moved `disc = per_node_discoveries(...)`, the `by_q` loop, the `q_mid`/`mm`/multimorbidity block, the `gaps` lines, and the old `fdr_block = {...}` dict from `evaluate` — they now live in the helper. Leave everything else in `evaluate` unchanged; `P`, `node_pos`, `is_fg` are already computed above.)

- [ ] **Step 4: Run the new tests + the full evaluate suite (refactor guard)**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -q`
Expected: the 2 new tests pass AND every existing `evaluate` test stays green (byte-identical output).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "refactor(dag-placement): extract fdr_discovery_report; evaluate calls it

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: LR-FDR block in the readout (LR + explain-away, beside theta-mass)

**Files:**
- Modify: `analysis/cloud/lr_readout.py`
- Test: `analysis/cloud/tests/test_lr_readout.py`

**Interfaces:**
- Consumes: `fdr_discovery_report` (Task 1), `lr_placement_scores`, `explain_away_placement_scores`, `_background_from_bow`, `build_test_bow` (existing).
- Produces: a `fdr_report_lines(...)` helper returning the printed lines; an FDR block in `main` after `detection_report`.

- [ ] **Step 1: Write the failing test (truth/mm_rows construction + render)**

Add to `analysis/cloud/tests/test_lr_readout.py`:

```python
def test_fdr_truth_and_mm_rows_from_frontiers():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    import importlib
    mod = importlib.import_module("lr_readout")
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=1, tpn=1)      # 2,3 children of 1
    # frontiers per doc (engine ids): doc0 = {2} (deep single), doc1 = {2,3} (multimorbid),
    # doc2 = {} (background)
    frontiers = [[2], [2, 3], []]
    truth, mm_rows = mod.fdr_truth_mm_rows(frontiers, lay)
    node_idx = {u: i for i, u in enumerate(lay.nodes)}
    # doc0 frontier {2}: true for node 2 AND its ancestor node 1 (subtree membership)
    assert truth[0, node_idx[2]] and truth[0, node_idx[1]] and not truth[0, node_idx[3]]
    # doc0 is NOT multimorbid (single frontier node), despite truth having 2 trues
    assert not mm_rows[0]
    # doc1 frontier {2,3}: multimorbid (>=2 scoreable frontier nodes)
    assert mm_rows[1]
    # doc2 background: no truth, not multimorbid
    assert not truth[2].any() and not mm_rows[2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd analysis/cloud/tests && python -m pytest test_lr_readout.py::test_fdr_truth_and_mm_rows_from_frontiers -q`
Expected: FAIL (`fdr_truth_mm_rows` not defined).

- [ ] **Step 3: Add `fdr_truth_mm_rows` + `fdr_report_lines` + wire into `main`**

In `analysis/cloud/lr_readout.py`, add two helpers:

```python
def fdr_truth_mm_rows(frontiers, lay):
    """From per-doc frontier engine-id lists -> (truth [n_docs x n_nodes] bool subtree
    membership, mm_rows [n_docs] bool = >=2 scoreable frontier nodes). Mirrors evaluate's
    node_pos/mm_rows so the LR-FDR is scored on the SAME truth as the theta-mass FDR."""
    import numpy as np
    node_set = set(lay.nodes)
    truth = np.zeros((len(frontiers), len(lay.nodes)), dtype=bool)
    mm_rows = np.zeros(len(frontiers), dtype=bool)
    for i, fr in enumerate(frontiers):
        fset = set(int(x) for x in fr)
        for j, u in enumerate(lay.nodes):
            truth[i, j] = bool(fset & lay.subtree(u))
        mm_rows[i] = len(fset & node_set) >= 2
    return truth, mm_rows


def fdr_report_lines(P, is_fg, lengths, truth, mm_rows, label, *, q_grid=(0.05, 0.10, 0.20)):
    """Rendered by_q lines for one score matrix's FDR discovery report (n_disc, precision,
    recall per q), labeled. Uses the engine's fdr_discovery_report so LR/explain-away are
    scored identically to the theta-mass FDR."""
    from spark_vi.models.topic.dag_placement import fdr_discovery_report
    rep = fdr_discovery_report(P, is_fg, lengths, truth, mm_rows,
                               q_grid=q_grid, n_length_bins=4)
    parts = []
    for q in q_grid:
        v = rep["by_q"][q]
        parts.append(f"q={q}: (n={v['n_discoveries']}, "
                     f"prec={v['precision']:.3f}, rec={v['recall']:.3f})")
    return [f"[lr_readout]   fdr {label}: " + "  ".join(parts)]
```

Then in `main`, immediately AFTER the `detection_report(...)` call, add:

```python
        # FDR discovery: run the SAME empirical-null per-node FDR on LR + explain-away
        # scores (alpha->inf lift limit) beside the theta-mass FDR from the manifest,
        # which found zero discoveries (buried signal). Does LR surface discoveries?
        from spark_vi.models.topic.dag_placement import (
            lr_placement_scores, explain_away_placement_scores)
        frontiers = [fr for (_pid, _sv, fr) in meta]
        truth, mm_rows = fdr_truth_mm_rows(frontiers, lay)
        lengths = np.asarray(bow.sum(axis=1)).ravel().astype(float)
        bg_rate = _background_from_bow(bow)
        theta_fdr = manifest.get("metrics", {}).get("fdr") or manifest.get("fdr")
        if theta_fdr and "by_q" in theta_fdr:
            tparts = "  ".join(
                f"q={q}: (n={v['n_discoveries']}, prec={v.get('precision', float('nan'))}, "
                f"rec={v.get('recall', float('nan'))})"
                for q, v in theta_fdr["by_q"].items())
            print(f"[lr_readout]   fdr theta-mass (manifest): {tparts}", flush=True)
        for label, P in (
            ("LR @alpha=inf", lr_placement_scores(
                bow, lam, lay, alpha=float("inf"), background=bg_rate,
                count_mode=args.count_mode, length_normalize=args.length_normalize)),
            ("explain-away @alpha=inf", explain_away_placement_scores(
                bow, lam, lay, alpha=float("inf"), background=bg_rate,
                count_mode=args.count_mode, length_normalize=args.length_normalize))):
            for ln in fdr_report_lines(P, is_fg, lengths, truth, mm_rows, label):
                print(ln, flush=True)
```

(Note: the manifest's theta-mass `by_q` keys may be strings after JSON round-trip; print them as-is. LR/explain-away FDR are computed fresh here so their q keys are floats.)

- [ ] **Step 4: Run the driver test + compile check**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m py_compile analysis/cloud/lr_readout.py && cd analysis/cloud/tests && python -m pytest test_lr_readout.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/lr_readout.py analysis/cloud/tests/test_lr_readout.py
git commit -m "feat(lr-readout): FDR discovery block for LR + explain-away beside theta-mass

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** engine helper extraction + evaluate refactor (T1), driver truth/mm_rows + LR/EA FDR block beside theta-mass (T2). Like-for-like guaranteed by the shared helper. Validation: planted+null helper tests + evaluate refactor guard (T1), truth/mm_rows construction test (T2), cluster three-way table via `make lr-readout`. ✓
- **Placeholder scan:** T1 Step 3 gives the full helper body + the exact evaluate replacement; T2 gives full helper + wiring code. The one prose instruction (delete the moved lines from evaluate) is unavoidable for an extraction and names the exact block. ✓
- **Type consistency:** `fdr_discovery_report(P, is_fg, doc_lengths, truth, mm_rows, *, q_grid, n_length_bins, method)` identical in definition (T1) and both call sites (evaluate T1, `fdr_report_lines` T2). `truth`/`mm_rows` bool arrays; `fdr_truth_mm_rows` returns them in `lay.nodes` order matching `P`'s columns. ✓
