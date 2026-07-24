# Case-finding FDR readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a background-relative, FDR-corrected case-finding readout to the gated-LDA placement eval — per-node empirical p-values on node-block mass, Benjamini-Hochberg per node — so multimorbid cases are scored on each node independently and the operating point becomes a false-discovery rate, all post-hoc on the already-fitted profiles.

**Architecture:** Pure-numpy FDR primitives + a length-conditioned per-node discovery routine live in the engine (`spark-vi/spark_vi/models/topic/dag_placement.py`, domain-neutral — arrays and engine node-ids, no concept-ids). `evaluate` gains an optional `doc_lengths` parameter and returns a new `fdr` block assembled from those primitives plus a zero-inflated-Beta-vs-empirical diagnostic. The cloud driver threads per-doc token counts into `evaluate` and reports the block. No model, gate, window, or anchor change — this reads the fitted `profile` output.

**Tech Stack:** Python, numpy, scipy.stats (Beta fit/CDF for the diagnostic only), pytest.

## Global Constraints

- Engine (`spark_vi`) stays id-agnostic: the FDR helpers take numpy arrays + the engine `DagLayout` (engine node-ids), never concept-ids. Only the cloud driver (`analysis/cloud`) is the domain edge.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- No LaTeX; Unicode Greek only. Cite the methods in docstrings: Benjamini & Hochberg 1995 (JRSS-B 57:289) for BH, Benjamini & Yekutieli 2001 (Ann. Statist. 29:1165) for BY, Efron 2004/2008 (two-groups / local FDR).
- Branch `case-finding` does NOT auto-push; push only when the user asks.
- Exploratory research code (no prod). Structural tests only; do not gold-plate.
- **Backward compatibility:** `evaluate(profiles, test_labels, lay)` with no `doc_lengths` must return every prior key unchanged, plus a single-length-bin `fdr` block. Existing `evaluate` callers/tests keep passing.
- **Post-hoc only:** no change to `profile`, the model, the gate, or windowing. This consumes `P` (node-block affinity) as-is.
- Test harness — engine: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -q` (use `-k <name>` for focused iteration; the full file runs Gibbs and is ~90s). Driver: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -q`.

---

### Task 1: Pure-numpy FDR primitives (empirical p, BH, BY)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add module-level helpers near the other stats helpers, e.g. after `_average_precision`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `_empirical_right_tail_p(values, reference) -> np.ndarray` — right-tail empirical p with the +1/(n+1) plug (p in (0,1], never 0). `_fdr_reject(pvals, q, method="bh") -> np.ndarray[bool]` — step-up rejection mask; `method` in {"bh","by"}, BY applying the harmonic c(m)=Σ 1/i penalty.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_dag_placement.py`:

```python
import numpy as np
from spark_vi.models.topic.dag_placement import _empirical_right_tail_p, _fdr_reject

def test_empirical_right_tail_p_counts_and_floor():
    ref = np.array([0.0, 0.1, 0.2, 0.3])          # n=4
    # value above all -> ge=0 -> p=(0+1)/(4+1)=0.2 (floored, never 0)
    # value 0.2 -> ge counts {0.2,0.3}=2 -> p=(2+1)/5=0.6
    p = _empirical_right_tail_p(np.array([0.5, 0.2, -1.0]), ref)
    assert np.allclose(p, [0.2, 0.6, 1.0])
    assert (p > 0).all()

def test_fdr_reject_bh_uniform_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)                    # pure null
    rej = _fdr_reject(p, 0.1, "bh")
    assert rej.sum() <= 0.02 * len(p)             # few false rejections under the null

def test_fdr_reject_bh_planted_and_by_subset():
    p = np.concatenate([np.full(20, 1e-6), np.random.default_rng(1).uniform(size=980)])
    bh = _fdr_reject(p, 0.1, "bh")
    by = _fdr_reject(p, 0.1, "by")
    assert bh[:20].all()                          # the strong signals are found
    assert set(np.nonzero(by)[0]).issubset(set(np.nonzero(bh)[0]))   # BY ⊆ BH
    assert by.sum() <= bh.sum()
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "right_tail_p or fdr_reject" -q`

- [ ] **Step 3: Implement** in `dag_placement.py`:

```python
def _empirical_right_tail_p(values, reference):
    """Right-tail empirical p-value of each `values[i]` against an empirical
    `reference` sample: p = (#{reference >= value} + 1) / (n + 1). The +1/(n+1)
    plug (never 0) bounds the resolution at the reference size and keeps BH
    well-defined. Vectorised via searchsorted on the sorted reference."""
    ref = np.sort(np.asarray(reference, dtype=float))
    v = np.asarray(values, dtype=float)
    n = len(ref)
    if n == 0:
        return np.ones_like(v)
    ge = n - np.searchsorted(ref, v, side="left")     # count of ref >= v
    return (ge + 1.0) / (n + 1.0)


def _fdr_reject(pvals, q, method="bh"):
    """Step-up multiple-testing rejection at false-discovery-rate q.

    method='bh': Benjamini & Hochberg 1995 (JRSS-B 57:289) — reject the k largest
    ranks with p_(i) <= (i/m) q. method='by': Benjamini & Yekutieli 2001 (Ann.
    Statist. 29:1165) — the same with the harmonic penalty c(m)=sum_{i<=m} 1/i,
    valid under arbitrary dependence (conservative). Returns a boolean mask
    aligned to `pvals`."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    c = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
    thresh = (np.arange(1, m + 1) / m) * (q / c)
    below = ranked <= thresh
    kmax = int(np.max(np.nonzero(below)[0])) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax:
        reject[order[:kmax]] = True
    return reject
```

- [ ] **Step 4: Run — expect PASS.**
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "right_tail_p or fdr_reject" -q`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): empirical right-tail p + BH/BY FDR primitives

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Length-binning + per-node discovery routine

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `_empirical_right_tail_p`, `_fdr_reject` (Task 1).
- Produces: `_assign_length_bins(lengths, ref_lengths, n_bins) -> np.ndarray[int]` (bin id 0..n_bins-1 by quantiles of `ref_lengths`; n_bins<=1 -> all zeros). `per_node_discoveries(P, is_fg, doc_lengths, *, q_grid, n_length_bins=4, method="bh") -> dict` where `P` is the [n_docs x n_nodes] node-block affinity, returning `{"pmat": [n_docs x n_nodes] length-conditioned p-values, "floor": bool mask of p at the empirical floor, "discoveries": {q: bool [n_docs x n_nodes]}, "bins": [n_docs] bin ids}`. Per node u and length bin b, the null reference is `P[background & bin==b, u]`; BH/BY is applied per node column across all patients.

- [ ] **Step 1: Write the failing tests**

```python
from spark_vi.models.topic.dag_placement import _assign_length_bins, per_node_discoveries

def test_assign_length_bins_quantile_and_single():
    ref = np.arange(100.0)
    b = _assign_length_bins(np.array([1.0, 50.0, 99.0]), ref, 4)
    assert b[0] == 0 and b[2] == 3 and 0 <= b[1] <= 3
    assert (_assign_length_bins(np.array([1.0, 9.0]), ref, 1) == 0).all()

def test_per_node_discoveries_recovers_planted_signal():
    rng = np.random.default_rng(0)
    n_bg, n_case, n_nodes = 500, 40, 3
    P = rng.uniform(0, 0.05, size=(n_bg + n_case, n_nodes))
    is_fg = np.zeros(n_bg + n_case, bool); is_fg[n_bg:] = True
    P[n_bg:, 1] += 0.6                       # cases are elevated on node 1
    out = per_node_discoveries(P, is_fg, np.full(n_bg + n_case, 10.0),
                               q_grid=[0.1], n_length_bins=1)
    disc = out["discoveries"][0.1]
    assert disc[n_bg:, 1].mean() > 0.7       # most true cases discovered on node 1
    assert disc[:n_bg, 1].mean() < 0.1       # few background discovered

def test_per_node_discoveries_length_conditioning_controls_fdr():
    # Long records carry more mass on node 0 for EVERYONE (a length confound, not
    # signal). Unconditioned scoring falsely flags long background docs; the
    # length-conditioned null removes it.
    rng = np.random.default_rng(2)
    n, n_nodes = 1200, 2
    length = rng.choice([5.0, 50.0], size=n)
    is_fg = np.zeros(n, bool)                 # ALL background: any discovery is false
    P = rng.uniform(0, 0.02, size=(n, n_nodes))
    P[length == 50.0, 0] += 0.3              # confound: long docs, node 0
    uncond = per_node_discoveries(P, is_fg, length, q_grid=[0.1], n_length_bins=1)
    cond = per_node_discoveries(P, is_fg, length, q_grid=[0.1], n_length_bins=2)
    assert cond["discoveries"][0.1][:, 0].sum() < uncond["discoveries"][0.1][:, 0].sum()
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "length_bins or per_node_discoveries" -q`

- [ ] **Step 3: Implement** in `dag_placement.py`:

```python
def _assign_length_bins(lengths, ref_lengths, n_bins):
    """Assign each `lengths[i]` to a quantile bin (0..n_bins-1) of the reference
    length distribution `ref_lengths` (the background records). n_bins<=1 returns
    all zeros (the unconditioned null). Ties/degenerate quantiles collapse to
    fewer effective bins, which is harmless (a bin just holds more docs)."""
    lengths = np.asarray(lengths, dtype=float)
    if n_bins <= 1 or len(ref_lengths) == 0:
        return np.zeros(len(lengths), dtype=int)
    edges = np.quantile(np.asarray(ref_lengths, dtype=float),
                        np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(lengths, edges).astype(int)


def per_node_discoveries(P, is_fg, doc_lengths, *, q_grid,
                         n_length_bins=4, method="bh"):
    """Length-conditioned, background-relative per-node discovery.

    P is the [n_docs x n_nodes] node-block affinity (profile mass per node). For
    each node u and length bin b, the null reference is the background docs'
    node-u mass in bin b; the per-doc p-value is the right-tail empirical p
    against that reference (Efron two-groups empirical null; the background arm is
    the null sample). Benjamini-Hochberg (or BY) is then applied per node column
    across all docs, giving a discovery set at each q in q_grid. Returns pmat, the
    floor mask (p at the 1/(n_ref+1) resolution floor), the per-q discovery masks,
    and the bin ids."""
    P = np.asarray(P, dtype=float)
    is_fg = np.asarray(is_fg, dtype=bool)
    n, n_nodes = P.shape
    ref_lengths = doc_lengths[~is_fg] if (~is_fg).any() else doc_lengths
    bins = _assign_length_bins(doc_lengths, ref_lengths, n_length_bins)
    pmat = np.ones((n, n_nodes))
    floor = np.zeros((n, n_nodes), dtype=bool)
    for b in np.unique(bins):
        in_b = bins == b
        ref_rows = in_b & (~is_fg)
        idx = np.nonzero(in_b)[0]
        for u in range(n_nodes):
            ref = P[ref_rows, u]
            if len(ref) == 0:
                continue
            p = _empirical_right_tail_p(P[idx, u], ref)
            pmat[idx, u] = p
            floor[idx, u] = p <= (1.0 / (len(ref) + 1.0) + 1e-12)
    discoveries = {}
    for q in q_grid:
        mask = np.zeros((n, n_nodes), dtype=bool)
        for u in range(n_nodes):
            mask[:, u] = _fdr_reject(pmat[:, u], q, method)
        discoveries[q] = mask
    return {"pmat": pmat, "floor": floor, "discoveries": discoveries, "bins": bins}
```

- [ ] **Step 4: Run — expect PASS.**
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "length_bins or per_node_discoveries" -q`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): length-conditioned per-node FDR discovery

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Zero-inflated-Beta vs empirical diagnostic

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `_zib_empirical_gap(values, *, zero_eps=1e-6) -> float` — fit a zero-inflated Beta (point mass pi0 at <= zero_eps + Beta MLE on the positive part via `scipy.stats.beta.fit(..., floc=0, fscale=1)`) and return the max absolute gap between the fitted mixture CDF and the empirical CDF (a KS-style statistic in [0,1]). Degenerate samples (all zero, <2 positive points) return `nan`. This is a REPORTED diagnostic; nothing downstream depends on it.

- [ ] **Step 1: Write the failing tests**

```python
from spark_vi.models.topic.dag_placement import _zib_empirical_gap

def test_zib_gap_small_for_beta_like_sample():
    rng = np.random.default_rng(0)
    pos = rng.beta(2.0, 8.0, size=4000)
    vals = np.concatenate([np.zeros(1000), pos])      # zero-inflated Beta by construction
    assert _zib_empirical_gap(vals) < 0.05

def test_zib_gap_large_for_non_beta_sample():
    rng = np.random.default_rng(1)
    # a bimodal positive part a single Beta cannot fit
    vals = np.concatenate([rng.uniform(0.05, 0.10, 2000), rng.uniform(0.85, 0.95, 2000)])
    assert _zib_empirical_gap(vals) > 0.15

def test_zib_gap_degenerate_returns_nan():
    assert np.isnan(_zib_empirical_gap(np.zeros(50)))
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k zib_gap -q`

- [ ] **Step 3: Implement** in `dag_placement.py` (add `from scipy import stats as _sps` at the top import block if not present):

```python
def _zib_empirical_gap(values, *, zero_eps=1e-6):
    """Max-CDF-gap between a fitted zero-inflated Beta and the empirical CDF of
    `values` (node-block mass in [0,1]). The mixture is pi0 * 1[x<=0] + (1-pi0) *
    Beta(a,b) with pi0 the mass at <= zero_eps and Beta MLE (scipy) on the
    positive part. Returns a KS-style statistic in [0,1]; nan if degenerate
    (all-zero or <2 positive points). Diagnostic only: it decides whether the
    exportable null (sub-project 2) can be a ~3KB parametric fit or must ship a
    tail-dense empirical grid; the FDR p-values never use it."""
    v = np.sort(np.asarray(values, dtype=float))
    v = np.clip(v, 0.0, 1.0)
    n = len(v)
    if n == 0:
        return float("nan")
    pos = v[v > zero_eps]
    if len(pos) < 2:
        return float("nan")
    pi0 = float(np.mean(v <= zero_eps))
    try:
        a, b, _, _ = _sps.beta.fit(pos, floc=0.0, fscale=1.0)
    except Exception:
        return float("nan")
    emp = (np.arange(1, n + 1)) / n                       # empirical CDF at sorted v
    fit = pi0 + (1.0 - pi0) * _sps.beta.cdf(v, a, b)      # mixture CDF (Beta.cdf(0)=0)
    return float(np.max(np.abs(emp - fit)))
```

- [ ] **Step 4: Run — expect PASS.**
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k zib_gap -q`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): zero-inflated-Beta vs empirical CDF-gap diagnostic

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Assemble the `fdr` block in `evaluate`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (`evaluate`, ~line 389; return dict ~line 489)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `per_node_discoveries` (Task 2), `_zib_empirical_gap` (Task 3), and the existing `node_pos` (per-node subtree-membership truth already built at ~line 431) and `fronts`.
- Produces: `evaluate(profiles, test_labels, lay, *, doc_lengths=None, fdr_q_grid=(0.05, 0.10, 0.20), n_length_bins=4)`. Adds a `"fdr"` key to the return dict; all prior keys unchanged. `doc_lengths=None` -> a single length bin (unconditioned). The `fdr` block: `{"q_grid": [...], "by_q": {q: {"n_discoveries", "precision", "recall"}}, "multimorbidity": {"mean_discoveries_per_multimorbid", "argmax_baseline_per_multimorbid"}, "saturation_rate": float, "zib_gap_mean": float, "zib_gap_max": float, "n_length_bins_effective": int}`. Precision/recall use `node_pos` (a discovery (i,u) is a true positive iff `node_pos[nodes[u]][i]`).

- [ ] **Step 1: Write the failing tests**

```python
from spark_vi.models.topic.dag_placement import evaluate

def _toy_lay():
    # 3-node flat layout: root 0 with children 1,2,3. DagLayout(parent_map,
    # n_bg, tpn); parent_map is child -> [parent], root 0 has no entry.
    from spark_vi.models.topic.dag_placement import DagLayout
    return DagLayout({1: [0], 2: [0], 3: [0]}, n_bg=2, tpn=1)

def test_evaluate_backward_compatible_and_fdr_block_present():
    lay = _toy_lay()
    profiles = [{u: (0.6 if u == 1 else 0.05) for u in lay.nodes} for _ in range(30)]
    labels = [{1} for _ in range(15)] + [set() for _ in range(15)]   # 15 cases on node 1
    out = evaluate(profiles, labels, lay)                            # no doc_lengths
    for k in ("mrr", "top2", "auc_by_depth", "detection", "recall_at_k"):
        assert k in out                                             # prior keys intact
    assert "fdr" in out and 0.1 in out["fdr"]["by_q"]
    assert out["fdr"]["by_q"][0.1]["n_discoveries"] >= 1

def test_evaluate_fdr_multimorbidity_beats_argmax():
    lay = _toy_lay()
    # patients truly on BOTH node 1 and node 2, with mass on both blocks.
    profiles = [{1: 0.4, 2: 0.4, 3: 0.02} for _ in range(20)] + \
               [{u: 0.02 for u in lay.nodes} for _ in range(200)]
    labels = [{1, 2} for _ in range(20)] + [set() for _ in range(200)]
    out = evaluate(profiles, labels, lay, doc_lengths=[10.0] * 220)
    mm = out["fdr"]["multimorbidity"]
    # argmax can credit at most one node per patient (<=1); FDR can credit both.
    assert mm["mean_discoveries_per_multimorbid"] > mm["argmax_baseline_per_multimorbid"]
```

- [ ] **Step 2: Run — expect FAIL** (`evaluate` has no `doc_lengths` kwarg / no `fdr` key).
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -k "evaluate_backward or evaluate_fdr" -q`

- [ ] **Step 3: Implement.** Change the `evaluate` signature to
`def evaluate(profiles, test_labels, lay, *, doc_lengths=None, fdr_q_grid=(0.05, 0.10, 0.20), n_length_bins=4):`
and, just before the `return` (after `detection` is built, ~line 488), insert:

```python
    # --- FDR readout: background-relative, per-node, multiple-testing corrected -
    # Post-hoc on P (node-block mass). Each (patient, node) is its own test; the
    # background docs are the empirical null (Efron two-groups); BH per node.
    node_list = lay.nodes
    lengths = (np.asarray(doc_lengths, dtype=float) if doc_lengths is not None
               else np.ones(len(fronts)))
    nlb = n_length_bins if doc_lengths is not None else 1
    q_grid = list(fdr_q_grid)
    disc = per_node_discoveries(P, is_fg, lengths, q_grid=q_grid,
                                n_length_bins=nlb)
    truth = np.array([[node_pos[u][i] for u in node_list]
                      for i in range(len(fronts))], dtype=bool)   # [n_docs x n_nodes]
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
    # multimorbidity payoff at the middle q: discoveries per truly-multimorbid
    # patient vs the argmax baseline (argmax credits <=1 node per patient).
    q_mid = q_grid[len(q_grid) // 2]
    mm_rows = np.array([len(f & set(node_list)) >= 2 for f in fronts])
    if mm_rows.any():
        m_mid = disc["discoveries"][q_mid]
        mean_disc = float(m_mid[mm_rows].sum(axis=1).mean())
        argmax_node = np.argmax(P[mm_rows], axis=1)
        argmax_tp = truth[mm_rows][np.arange(mm_rows.sum()), argmax_node]
        argmax_base = float(argmax_tp.mean())
    else:
        mean_disc = float("nan"); argmax_base = float("nan")
    gaps = [_zib_empirical_gap(P[~is_fg, u]) for u in range(len(node_list))]
    gaps = [g for g in gaps if not np.isnan(g)]
    fdr_block = {
        "q_grid": q_grid,
        "by_q": by_q,
        "multimorbidity": {
            "mean_discoveries_per_multimorbid": mean_disc,
            "argmax_baseline_per_multimorbid": argmax_base},
        "saturation_rate": float(disc["floor"][disc["discoveries"][q_mid]].mean())
            if disc["discoveries"][q_mid].any() else float("nan"),
        "zib_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "zib_gap_max": float(np.max(gaps)) if gaps else float("nan"),
        "n_length_bins_effective": int(len(np.unique(disc["bins"]))),
    }
```

Then add `"fdr": fdr_block` to the returned dict.

- [ ] **Step 4: Run — expect PASS**, then the whole file to confirm no regression:
Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -q`

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): fdr block in evaluate (per-node BH, multimorbidity, ZIB diagnostic)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Thread `doc_lengths` through the driver + report the block

**Files:**
- Modify: `analysis/cloud/dag_placement_cloud.py` (`profiles_from_scored_rows` ~line 26; inline-eval select ~line 264; the metrics print ~line 267; the `manifest` dict ~line 307)
- Test: `analysis/cloud/tests/test_dag_placement_cloud.py`

**Interfaces:**
- Consumes: `evaluate(..., doc_lengths=...)` (Task 4).
- Produces: `profiles_from_scored_rows(rows, lay) -> (profiles, test_labels, doc_lengths)` where `doc_lengths[i]` = sum of the row's `features` (BOW) counts; the inline eval selects `features` and passes `doc_lengths` to `evaluate`; the `fdr` block is printed and added to `manifest`.

- [ ] **Step 1: Update the existing test + add coverage** in `analysis/cloud/tests/test_dag_placement_cloud.py`. The existing `test_profiles_from_scored_rows_maps_affinity_and_frontier` must now unpack three values and assert token counts. Read that test first; change its unpack to `profiles, labels, lengths = profiles_from_scored_rows(rows, lay)` and add a `features` field to the synthetic rows (a `pyspark.ml.linalg.Vectors.sparse` or a lightweight stub with `.toArray()`), asserting `lengths` equals the per-row count sum. Add:

```python
def test_profiles_from_scored_rows_returns_token_lengths():
    from pyspark.ml.linalg import Vectors
    from dag_placement_cloud import profiles_from_scored_rows
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: [0], 2: [0]}, n_bg=1, tpn=1)
    rows = [{"nodeAffinity": Vectors.dense([0.3, 0.1]), "frontier": [1],
             "features": Vectors.sparse(5, {0: 2.0, 3: 1.0})}]
    profiles, labels, lengths = profiles_from_scored_rows(rows, lay)
    assert lengths == [3.0]                        # 2 + 1
```

- [ ] **Step 2: Run — expect FAIL** (3-tuple unpack / missing token sum).
Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -q`

- [ ] **Step 3: Implement.**
1. In `profiles_from_scored_rows`, also read `r["features"]` and accumulate `doc_lengths.append(float(r["features"].toArray().sum()))`; return `(profiles, test_labels, doc_lengths)`.
2. In the inline-eval block (~line 264), change the select to include features and unpack three values, then pass `doc_lengths`:

```python
            scored = model.transform(bundle.test_df).select(
                "nodeAffinity", "frontier", "features")
            rows = scored.collect()
            profiles, test_labels, doc_lengths = profiles_from_scored_rows(rows, lay)
            metrics = evaluate(profiles, test_labels, lay, doc_lengths=doc_lengths)
```

3. After the existing placement-metrics print, add an `fdr` summary print (follow the file's `[driver]   ...` convention), e.g.:

```python
            fdr = metrics["fdr"]
            print(f"[driver]   fdr: by_q={{q: (v['n_discoveries'], round(v['precision'],3), "
                  f"round(v['recall'],3)) for q, v in fdr['by_q'].items()}} "
                  f"multimorbidity={fdr['multimorbidity']} "
                  f"saturation={fdr['saturation_rate']:.3f} "
                  f"zib_gap_mean={fdr['zib_gap_mean']:.3f} "
                  f"bins={fdr['n_length_bins_effective']}", flush=True)
```

4. In the `manifest` dict (~line 307), add `"fdr": metrics["fdr"]`.

- [ ] **Step 4: Run — expect PASS**, plus a syntax check of the driver:
Run: `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -q`
Run: `.venv/bin/python -c "import ast; ast.parse(open('analysis/cloud/dag_placement_cloud.py').read()); print('ast OK')"`

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/dag_placement_cloud.py analysis/cloud/tests/test_dag_placement_cloud.py
git commit -m "feat(dag-placement-cloud): thread doc_lengths + report the fdr block

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Task 1 = FDR primitives (empirical null p + BH/BY, spec "the statistic, null, test"); Task 2 = length-conditioning + per-node BH discovery (spec "length-conditioned", "per-node BH"); Task 3 = ZIB-vs-empirical diagnostic (spec "the ZIB diagnostic", chooses SP2 export form); Task 4 = the `fdr` output block incl. precision/recall, multimorbidity payoff, saturation flag (spec "Outputs"); Task 5 = driver threading + reporting (spec "Driver threading"). Caveats (contaminated null, in-sample) are documented in the spec and the block reports the saturation flag; the contamination is inherent and not a code path. SP2 (exportable null) and the per-patient viewer are explicitly out of scope.

**Placeholder scan:** every code step carries complete code; the ZIB uses scipy (confirmed available, 1.17.1); the driver's BQ transform stays cluster-covered (only the arg surface + `profiles_from_scored_rows` are unit-tested).

**Type consistency:** `_empirical_right_tail_p`/`_fdr_reject` (Task 1) consumed by `per_node_discoveries` (Task 2); `per_node_discoveries` + `_zib_empirical_gap` (Task 3) consumed by `evaluate`'s `fdr` block (Task 4); `evaluate(doc_lengths=...)` consumed by the driver (Task 5); `profiles_from_scored_rows` returns a 3-tuple consumed by the inline eval. `P` is [n_docs x n_nodes] throughout (columns = `lay.nodes` order); precision/recall reuse `node_pos[u][i]` truth.

**Backward-mode safety:** `evaluate` keeps every prior return key and defaults `doc_lengths=None` -> single bin; Task 4 Step 4 runs the whole engine file to confirm no regression. The `profiles_from_scored_rows` 3-tuple is the one breaking change, fully contained to the driver + its own test (Task 5).
