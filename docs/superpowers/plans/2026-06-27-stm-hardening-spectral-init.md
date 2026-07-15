# STM Hardening (block-aware spectral init + Σ-regularization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden spark-vi's `OnlineSTM` so it recovers planted topics with a sane, bounded Σ *independent of `sigma_init`*, eliminating the collapse-vs-Σ-blowup instability characterized in insight [0029](../../insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md) — by giving it a deterministic, **block-aware** initialization instead of the random init that creates the instability.

**Architecture:** Build a committed local synthetic-instability harness with ground-truth topics (both non-gated and gated/planted-minority), then add the two stabilizers we lack as opt-in, model-agnostic features: **block-aware spectral (anchor-word) initialization** built directly against the `TopicBlockPartition` API (so the non-gated case is the `background_k=K` degenerate case and PLDA reuses it), and an optional **inverse-gamma Σ prior**. A local ablation determines which is necessary/sufficient and upgrades insight 0029 to Confirmed. The K−1 reference-topic parameterization is deferred to documented future work (its gating story is unresolved).

**Tech Stack:** Python 3.12, numpy, scipy, pytest. Pure-numpy core — nothing here needs the cluster.

## Global Constraints

- **spark-vi stays domain-agnostic** — no OMOP/EHR concept ids or names in `spark-vi/`. Synthetic data uses integer token ids only.
- **Build general, not special-case** — spectral init is written against the `TopicBlockPartition` API; non-gated (`topic_blocks=None`) routes through `_effective_partition()` (all-background, `background_k=K`) and must produce identical results to a hand-written global pass. No separate non-gated code path.
- **Opt-in, defaults OFF** — every new behavior is gated behind a constructor/`data_summary` argument whose default reproduces today's output exactly. Existing `spark-vi/tests/test_stm_*.py` MUST stay green after every task.
- **TDD** — failing test first, watch it fail, minimal code, watch it pass, commit.
- **No LaTeX in committed prose** — Unicode Greek (η, θ, Σ, Γ, ρ, β) + plain text only.
- **Ground-truth recovery is the load-bearing metric** — "recovers the planted topics (incl. the rare group's foreground) AND final Σ bounded," not ELBO or eyeballing.

---

## File Structure

- `spark-vi/tests/_stm_synth.py` (Create) — reusable synthetic corpora (non-gated + gated/planted-minority), in-process fit loop, planted-topic recovery metric.
- `spark-vi/tests/test_stm_hardening.py` (Create) — TDD tests + the xfail baseline characterization.
- `spark-vi/spark_vi/models/topic/spectral_init.py` (Create) — pure-numpy anchor-word building blocks + the block-aware orchestrator.
- `spark-vi/tests/test_spectral_init.py` (Create) — unit tests for the building blocks + gated orchestration.
- `spark-vi/spark_vi/models/topic/stm.py` (Modify) — `initialize_global` reads a spectral β from `data_summary`; M-step gains the optional Σ prior.
- `analysis/local/stm_ablation.py` (Create) — runnable ablation table.
- `docs/insights/0029-...md` (Modify) — fold in the ablation; upgrade Status.

---

## Task 1: Synthetic harness — non-gated + gated planted corpora + recovery metric

**Files:**
- Create: `spark-vi/tests/_stm_synth.py`
- Create: `spark-vi/tests/test_stm_hardening.py`

**Interfaces (Produces):**
- `synthetic_ehr_corpus(*, K_rare, V, D, doc_len, bg_frac, seed) -> (list[STMDocument], np.ndarray)` — non-gated; dense shared background over `[0:V//2]` + `K_rare` disjoint phenotype blocks in `[V//2:]`; each doc = `bg_frac` background tokens + one rare phenotype. Returns docs + planted β `(K_rare, V)`.
- `synthetic_gated_corpus(*, groups, fg_per_group, bg_k, V, D, doc_len, bg_frac, seed) -> (docs, planted_beta, partition)` — each doc is assigned a group (`doc.groups`), expresses the shared background plus ONLY its group's planted foreground block. Returns docs, planted β `(K, V)` laid out in `partition` slot order (background slots then per-group foreground slots), and the `TopicBlockPartition`.
- `fit_stm(docs, *, K, V, sigma_init, n_iter=250, batch=None, seed=42, partition=None, init_data=None, **model_kwargs) -> dict` — in-process fit. `batch=None` → full-batch ρ=1; else minibatch with ρ_t=(t+64)^−0.7 and full-corpus scaling of every stat key. `partition` → `topic_blocks`. `init_data` → passed to `initialize_global` (for spectral init).
- `planted_recovery(beta_hat, planted_beta, *, thresh=0.5) -> int` — count of planted phenotypes whose token block is captured (≥`thresh` mass) by some fitted topic.
- `foreground_recovers_group(beta_hat, partition, group, planted_beta, *, thresh=0.5) -> bool` — does some topic in `partition.block_indices(group)` capture that group's planted foreground block.
- `final_sigma_range(gp) -> (float, float)`.

- [ ] **Step 1: Failing test (generators + metrics)**

```python
# spark-vi/tests/test_stm_hardening.py
import numpy as np, pytest
from _stm_synth import (synthetic_ehr_corpus, synthetic_gated_corpus,
                        planted_recovery, foreground_recovers_group,
                        fit_stm, final_sigma_range)

def test_recovery_perfect_on_ground_truth():
    _, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=200, doc_len=30,
                                      bg_frac=0.7, seed=0)
    assert planted_recovery(planted, planted, thresh=0.5) == 8

def test_gated_corpus_shapes_and_groups():
    docs, planted, part = synthetic_gated_corpus(
        groups=("a", "b"), fg_per_group=2, bg_k=3, V=200, D=300,
        doc_len=30, bg_frac=0.6, seed=0)
    assert planted.shape[0] == part.K == 3 + 2 * 2
    assert any("a" in d.groups for d in docs) and any("b" in d.groups for d in docs)
    # ground-truth beta recovers each group's foreground by construction
    assert foreground_recovers_group(planted, part, "a", planted, thresh=0.5)
```

- [ ] **Step 2: Run, verify fail** — `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_hardening.py -q` → `ModuleNotFoundError: _stm_synth`.

- [ ] **Step 3: Implement `_stm_synth.py`**

```python
# spark-vi/tests/_stm_synth.py
"""Synthetic STM corpora (non-gated + gated) + in-process fit + ground-truth
recovery. Domain-agnostic: integer token ids only."""
from __future__ import annotations
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition

def _block_of(row, *, eps=1e-3):
    return np.where(row > eps)[0]

def planted_recovery(beta_hat, planted_beta, *, thresh=0.5):
    n = 0
    for k in range(planted_beta.shape[0]):
        if beta_hat[:, _block_of(planted_beta[k])].sum(axis=1).max() >= thresh:
            n += 1
    return n

def foreground_recovers_group(beta_hat, partition, group, planted_beta, *,
                              thresh=0.5):
    fg = partition.block_indices(group)
    # planted foreground rows for this group sit in the same slot indices
    for k in fg:
        block = _block_of(planted_beta[k])
        if len(block) and beta_hat[fg][:, block].sum(axis=1).max() >= thresh:
            return True
    return False

def final_sigma_range(gp):
    s = gp["Sigma"]; return float(s.min()), float(s.max())

def synthetic_ehr_corpus(*, K_rare, V, D, doc_len, bg_frac, seed=0):
    rng = np.random.default_rng(seed)
    BG_V = V // 2
    bg = np.full(V, 1e-4); bg[:BG_V] = rng.random(BG_V) + 0.1; bg /= bg.sum()
    bs = (V - BG_V) // K_rare
    planted = np.full((K_rare, V), 1e-4)
    for k in range(K_rare):
        planted[k, BG_V + k * bs: BG_V + (k + 1) * bs] += 1.0
    planted /= planted.sum(axis=1, keepdims=True)
    docs = []
    for _ in range(D):
        k = int(rng.integers(K_rare)); n_bg = int(rng.binomial(doc_len, bg_frac))
        toks = np.concatenate([rng.choice(V, size=n_bg, p=bg),
                               rng.choice(V, size=doc_len - n_bg, p=planted[k])])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64),
                                length=int(c.sum()), x=np.array([1.0])))
    return docs, planted

def synthetic_gated_corpus(*, groups, fg_per_group, bg_k, V, D, doc_len,
                           bg_frac, seed=0):
    rng = np.random.default_rng(seed)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                               foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    # Vocab layout: background region [0:V//2], then a disjoint region per
    # (group, fg-topic). planted[k] aligns with partition slot k.
    BG_V = V // 2
    rest = V - BG_V
    n_fg = len(groups) * fg_per_group
    fb = rest // max(n_fg, 1)
    planted = np.full((K, V), 1e-4)
    bg_rows = part.background_indices()
    for j, k in enumerate(bg_rows):           # background topics over [0:BG_V]
        planted[k, (j * (BG_V // bg_k)):((j + 1) * (BG_V // bg_k))] += 1.0
    fg_slot = 0
    for g in groups:                          # each group's foreground block
        for k in part.block_indices(g):
            lo = BG_V + fg_slot * fb
            planted[k, lo:lo + fb] += 1.0
            fg_slot += 1
    planted /= planted.sum(axis=1, keepdims=True)
    docs = []
    glist = list(groups)
    for _ in range(D):
        g = glist[int(rng.integers(len(glist)))]
        allowed = part.allowed_indices(frozenset({g}))
        # doc mixes background topics + this group's foreground topics
        bg_topic = bg_rows[int(rng.integers(len(bg_rows)))]
        fg_topics = part.block_indices(g)
        fg_topic = fg_topics[int(rng.integers(len(fg_topics)))]
        n_bg = int(rng.binomial(doc_len, bg_frac))
        toks = np.concatenate([
            rng.choice(V, size=n_bg, p=planted[bg_topic]),
            rng.choice(V, size=doc_len - n_bg, p=planted[fg_topic])])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, planted, part

def fit_stm(docs, *, K, V, sigma_init, n_iter=250, batch=None, seed=42,
            partition=None, init_data=None, **model_kwargs):
    m = OnlineSTM(K=K, vocab_size=V, P=1, sigma_init=sigma_init,
                  random_seed=seed, topic_blocks=partition, **model_kwargs)
    gp = m.initialize_global(init_data)
    if batch is None:
        for _ in range(n_iter):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
        return gp
    D = len(docs); rng = np.random.default_rng(seed); scale = D / batch
    for t in range(n_iter):
        idx = rng.choice(D, size=batch, replace=False)
        stats = m.local_update([docs[i] for i in idx], gp)
        scaled = {kk: (v * scale if isinstance(v, (np.ndarray, int, float)) else v)
                  for kk, v in stats.items()}
        gp = m.update_global(gp, scaled, learning_rate=(t + 64) ** -0.7)
    return gp
```

- [ ] **Step 4: Run, verify pass.** `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_hardening.py -q` → PASS.

- [ ] **Step 5: Add xfail baseline characterization**

```python
# append to test_stm_hardening.py
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="random-init STM is sigma_init-unstable; "
                   "fixed by spectral init (insight 0029)")
def test_baseline_random_init_is_unstable():
    docs, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30,
                                         bg_frac=0.7, seed=0)
    recos, smax = [], []
    for si in (1.0, 5.0, 20.0):
        gp = fit_stm(docs, K=40, V=300, sigma_init=si, batch=100, n_iter=250)
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recos.append(planted_recovery(beta, planted)); smax.append(final_sigma_range(gp)[1])
    assert min(recos) >= 6 and max(smax) < 1e3
```

- [ ] **Step 6: Run, verify xfail** — without `--runxfail` → `xfailed`; with → FAIL (baseline is unstable).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_stm_hardening.py
git commit -m "test(stm): synthetic harness (non-gated + gated) + ground-truth recovery; xfail baseline"
```

---

## Task 2: Block-aware spectral (anchor-word) initialization

**Files:**
- Create: `spark-vi/spark_vi/models/topic/spectral_init.py`
- Create: `spark-vi/tests/test_spectral_init.py`
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (`initialize_global` hook)
- Test: `spark-vi/tests/test_stm_hardening.py`

**Interfaces (Produces):**
- `word_cooccurrence(docs, V) -> np.ndarray` — V×V normalized same-document co-occurrence Q. Per doc with counts `n` and length `L>1`, accumulate `(outer(n,n) − diag(n)) / (L·(L−1))`; average over docs; normalize to sum 1. (Single-token docs contribute nothing — correct, they carry no co-occurrence.)
- `find_anchors(Q, n, *, seed_rows=None) -> list[int]` — greedy farthest-point on the row-normalized rows of Q. If `seed_rows` (word ids) is given, initialize the spanned basis with those rows (used for deflation) WITHOUT returning them; return the `n` new anchors farthest from the running span.
- `recover_beta(Q, anchors, rows=None) -> np.ndarray` — recover `len(anchors)×V` topic-word rows: express each word's Q̄ row as a non-negative convex combination of the anchor rows (NNLS) → P(topic|word); Bayes-flip with the word marginal `Q.sum(axis=1)` → P(word|topic); renormalize. `rows` optionally restricts which word rows are recovered against (default all).
- `spectral_init_beta(docs, partition, V) -> np.ndarray` — **the block-aware orchestrator**, K×V β in `partition` slot order:
  1. `Q_all = word_cooccurrence(docs, V)`; `bg_anchors = find_anchors(Q_all, partition.background_k)`; recover background rows from `Q_all`; place into `partition.background_indices()`.
  2. For each group `g` in `partition.groups`: `docs_g = [d for d in docs if g in d.groups]`; `Q_g = word_cooccurrence(docs_g, V)`; `fg_anchors = find_anchors(Q_g, len(partition.block_indices(g)), seed_rows=bg_anchors)`; recover those rows from `Q_g`; place into `partition.block_indices(g)`.
  Non-gated (`_effective_partition()`, `background_k=K`, no groups) → step 1 only → identical to a global pass.
- `initialize_global` accepts `data_summary={"spectral_beta": <KxV>}` → seeds `lambda = spectral_beta * gamma_shape` instead of random gamma; Γ, Σ unchanged. Default (None) unchanged.

- [ ] **Step 1: Failing unit tests for building blocks + gated orchestration**

```python
# spark-vi/tests/test_spectral_init.py
import numpy as np
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta, spectral_init_beta)
from _stm_synth import (synthetic_ehr_corpus, synthetic_gated_corpus,
                        planted_recovery, foreground_recovers_group)

def test_cooccurrence_normalized_square():
    docs, _ = synthetic_ehr_corpus(K_rare=4, V=40, D=100, doc_len=20, bg_frac=0.5, seed=1)
    Q = word_cooccurrence(docs, 40)
    assert Q.shape == (40, 40) and np.isclose(Q.sum(), 1.0, atol=1e-6)

def test_spectral_init_recovers_nongated_planted():
    from spark_vi.models.topic.partition import TopicBlockPartition
    docs, planted = synthetic_ehr_corpus(K_rare=6, V=120, D=800, doc_len=30,
                                         bg_frac=0.5, seed=2)
    part = TopicBlockPartition(group_var="", background_k=12, foreground=())
    beta0 = spectral_init_beta(docs, part, 120)
    assert beta0.shape == (12, 120)
    assert planted_recovery(beta0, planted, thresh=0.4) >= 4

def test_block_aware_init_recovers_rare_group_foreground():
    """The decisive gated property: a rare group's foreground anchor lands its
    planted phenotype at INIT, because it is found on the within-group Q
    (undiluted by the majority) and deflated against the background span."""
    docs, planted, part = synthetic_gated_corpus(
        groups=("maj", "rare"), fg_per_group=2, bg_k=3, V=240, D=1200,
        doc_len=30, bg_frac=0.6, seed=3)
    # make 'rare' a minority arm
    docs = [d for i, d in enumerate(docs) if ("rare" not in d.groups) or (i % 4 == 0)]
    beta0 = spectral_init_beta(docs, part, 240)
    assert foreground_recovers_group(beta0, part, "rare", planted, thresh=0.4)
```

- [ ] **Step 2: Run, verify fail** — `ModuleNotFoundError: spectral_init`.

- [ ] **Step 3: Implement `spectral_init.py`** (anchor-word, Arora et al. 2014; block-aware orchestrator per the Interfaces above). `find_anchors`: row-normalize Q; greedy — first anchor = row of max norm (or seed basis from `seed_rows`); each subsequent = row with max residual after projecting onto the span of chosen/seed rows (Gram–Schmidt). `recover_beta`: per word, NNLS of its row onto anchor rows → convex weights = P(topic|word); multiply by word marginal, normalize columns→rows to get P(word|topic).

- [ ] **Step 4: Run, verify pass** — `cd spark-vi && ../.venv/bin/python -m pytest tests/test_spectral_init.py -q` → PASS (esp. the rare-group test).

- [ ] **Step 5: `initialize_global` hook (failing test first)**

```python
# append to test_stm_hardening.py
from spark_vi.models.topic.spectral_init import spectral_init_beta
from spark_vi.models.topic.partition import TopicBlockPartition

@pytest.mark.slow
def test_spectral_init_makes_fit_init_independent():
    docs, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30,
                                         bg_frac=0.7, seed=0)
    part = TopicBlockPartition(group_var="", background_k=40, foreground=())
    beta0 = spectral_init_beta(docs, part, 300)
    recos = []
    for si in (1.0, 5.0, 20.0):
        gp = fit_stm(docs, K=40, V=300, sigma_init=si, n_iter=60,
                     init_data={"spectral_beta": beta0})
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recos.append(planted_recovery(beta, planted))
    assert min(recos) >= 6   # recovery no longer depends on sigma_init
```

- [ ] **Step 6: Implement the hook** — at the top of `OnlineSTM.initialize_global`:

```python
        if data_summary is not None and "spectral_beta" in data_summary:
            beta0 = np.asarray(data_summary["spectral_beta"], dtype=np.float64)
            return {"lambda": beta0 * self.gamma_shape, "eta": np.array(self.eta),
                    "Gamma": np.zeros((self.P, self.K)),
                    "Sigma": np.full(self.K, self.sigma_init)}
```

- [ ] **Step 7: Run, verify pass + existing green** — `cd spark-vi && ../.venv/bin/python -m pytest tests/test_spectral_init.py tests/test_stm_hardening.py tests/test_stm_math.py tests/test_stm_integration.py -q`.

- [ ] **Step 8: Commit**

```bash
git add spark-vi/spark_vi/models/topic/spectral_init.py spark-vi/tests/test_spectral_init.py spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_hardening.py
git commit -m "feat(stm): block-aware spectral (anchor-word) init against TopicBlockPartition (non-gated = degenerate case)"
```

---

## Task 3: Optional inverse-gamma Σ prior (model-agnostic toggle)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (constructor + Σ step of `update_global`, [stm.py:521-529](../../../spark-vi/spark_vi/models/topic/stm.py#L521-L529))
- Test: `spark-vi/tests/test_stm_hardening.py`

**Interfaces:** `OnlineSTM(..., sigma_prior_scale: float | None = None, sigma_prior_count: float = 0.0)`. Default None → today's behavior. Set → Σ target becomes `(residual_diag_k + sigma_prior_count·sigma_prior_scale) / (n_docs_k + sigma_prior_count)` (inverse-gamma posterior mean: shrink toward `sigma_prior_scale` with strength `sigma_prior_count` pseudo-docs). Orthogonal to gating — operates per-topic on the Σ vector regardless of blocks.

- [ ] **Step 1: Failing test**

```python
# append to test_stm_hardening.py
@pytest.mark.slow
def test_sigma_prior_reduces_blowup():
    docs, _ = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30, bg_frac=0.7, seed=0)
    off = fit_stm(docs, K=40, V=300, sigma_init=5.0, batch=100, n_iter=250)
    on = fit_stm(docs, K=40, V=300, sigma_init=5.0, batch=100, n_iter=250,
                 sigma_prior_scale=2.0, sigma_prior_count=50.0)
    assert final_sigma_range(on)[1] < final_sigma_range(off)[1] / 10
```

- [ ] **Step 2: Run, verify fail** — `unexpected keyword 'sigma_prior_scale'`.

- [ ] **Step 3: Constructor args** (after `self.sigma_ridge = ...`, plus signature):

```python
        if sigma_prior_scale is not None and sigma_prior_scale <= 0:
            raise ValueError(f"sigma_prior_scale must be > 0, got {sigma_prior_scale}")
        if sigma_prior_count < 0:
            raise ValueError(f"sigma_prior_count must be >= 0, got {sigma_prior_count}")
        self.sigma_prior_scale = None if sigma_prior_scale is None else float(sigma_prior_scale)
        self.sigma_prior_count = float(sigma_prior_count)
```

- [ ] **Step 4: Apply in the Σ M-step** — replace the `Sigma_target[present] = ...` line:

```python
        present = n_docs_per_topic > 0
        Sigma_target = Sigma_diag.copy()
        if self.sigma_prior_scale is None:
            Sigma_target[present] = (target_stats["residual_diag_stat"][present]
                                     / n_docs_per_topic[present])
        else:
            c0, s0 = self.sigma_prior_count, self.sigma_prior_scale
            Sigma_target[present] = (
                (target_stats["residual_diag_stat"][present] + c0 * s0)
                / (n_docs_per_topic[present] + c0))
```

- [ ] **Step 5: Run, verify pass + existing green** — `... pytest tests/test_stm_hardening.py::test_sigma_prior_reduces_blowup tests/test_stm_math.py tests/test_stm_contract.py tests/test_stm_integration.py -q`.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_hardening.py
git commit -m "feat(stm): optional inverse-gamma Sigma prior (model-agnostic) to damp residual-variance runaway"
```

---

## Task 4: Ablation runner + insight 0029 → Confirmed

**Files:**
- Create: `analysis/local/stm_ablation.py`
- Modify: `docs/insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md`

**Interfaces:** prints a table — rows `{random-init baseline, +Σ-prior, +spectral, +spectral+Σ-prior}` × cols `sigma_init ∈ {1,5,20}`, cells `(recovery/K_rare, Σ_max)`; plus a gated row (`synthetic_gated_corpus` with a minority arm) showing `foreground_recovers_group("rare")` for random-init vs spectral. No new public API.

- [ ] **Step 1: Write `analysis/local/stm_ablation.py`** — import the harness + `spectral_init_beta`; loop the configs; for spectral rows precompute `beta0 = spectral_init_beta(docs, part, V)` and pass `init_data={"spectral_beta": beta0}`; print recovery + `final_sigma_range(...)[1]`. Include one gated block comparing random vs block-aware-spectral on `foreground_recovers_group(..., "rare")`.
- [ ] **Step 2: Run it** — `cd <repo root> && .venv/bin/python analysis/local/stm_ablation.py`. Record which config achieves `recovery ≥ 6/8 for all sigma_init AND Σ_max < 1e3`, and whether spectral recovers the rare gated foreground where random init does not.
- [ ] **Step 3: Update insight 0029** — replace the "predicted, not yet run" caveats with the ablation table; note the non-monotonic `sigma_init` chaos (1 mediocre / 5 blowup / 20 clean) refining the two-basin framing; record that block-aware spectral init recovers the rare gated foreground at initialization; upgrade **Status: Observed → Confirmed**.
- [ ] **Step 4: Commit**

```bash
git add analysis/local/stm_ablation.py docs/insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md
git commit -m "test(stm): local ablation of hardening toggles; confirm insight 0029 with controlled results"
```

---

## Future work: K−1 reference-topic parameterization

Deferred (not in this plan), but **explicitly retained** so it isn't lost. The softmax translation degeneracy (`softmax(η) = softmax(η + c)`) leaves a likelihood-flat all-ones direction that, in the full-K parameterization, only the weak prior pins — a secondary driver of η drift. The published CTM/STM fix is the K−1 reference parameterization (fix one topic's η to 0). We defer it because:

- It is the most invasive change (η, Γ, Σ lose a dimension throughout the per-doc optimizer, Hessian, and M-step — validated core math).
- Its **gating story is unresolved**: each doc's softmax is over its *allowed* set, so "which topic is the reference?" is not obvious. A promising hypothesis to investigate when we pick this up: **fix one always-allowed background topic as the global reference** — since background is in every doc's allowed set, that single reference could pin the level for all docs. Needs to be worked through (and tested on `synthetic_gated_corpus`) before trusting it.
- It may prove unnecessary: if the Task-4 ablation shows block-aware spectral init (+ optional Σ-prior) already delivers init-independent recovery with bounded Σ, K−1 is hygiene, not a requirement — mirroring how `stm` leans on spectral init with Σ-shrinkage default-off.

Trigger to revisit: the ablation still shows η/Σ drift with spectral init on, OR PLDA work wants the cleaner identifiability. When revisited, spec it as a `reference_topic` toggle (non-gated first, then the background-reference hypothesis for gated) with a test asserting the translation drift is gone (e.g., the mean of per-doc MAP η does not wander and Σ stays bounded without a Σ-prior).

---

## Self-Review

**Spec coverage:** harness with gated planted-minority (Task 1) ✓; block-aware spectral init built against the partition API, non-gated as the degenerate case (Task 2) ✓; model-agnostic Σ prior (Task 3) ✓; ablation + insight confirmation incl. a gated row (Task 4) ✓; K−1 retained as documented future work with its gating hypothesis ✓.

**General-not-special-case:** spectral init has ONE code path (partition-driven); non-gated routes through `_effective_partition()` and the plan asserts it equals a global pass. No separate non-gated implementation.

**Toggle-default safety:** spectral init only activates when `data_summary` carries `spectral_beta`; the Σ prior only when `sigma_prior_scale` is set; both default to today's behavior, and every task re-runs the existing STM suite as a gate. The xfail baseline (Task 1) documents the failure without blocking CI.

**Type consistency:** `synthetic_gated_corpus -> (docs, planted, partition)`, `spectral_init_beta(docs, partition, V) -> KxV`, `data_summary={"spectral_beta": KxV}`, `sigma_prior_scale/count`, and the recovery metrics are used identically across Tasks 1–4 and the ablation. `find_anchors(..., seed_rows=)` is the deflation seam consumed by the gated orchestrator.

**Cluster independence:** every test and the ablation run in pure numpy via the local harness.

**Known risk:** the block-aware orchestrator's deflation (foreground anchors found on within-group Q, seeded against background anchors) is the novel part — Task 2's `test_block_aware_init_recovers_rare_group_foreground` is the load-bearing gate for it. `recover_beta`'s NNLS may need a small ridge for numerical stability on near-degenerate anchor sets; add it inside the implementation if the unit test shows instability.
