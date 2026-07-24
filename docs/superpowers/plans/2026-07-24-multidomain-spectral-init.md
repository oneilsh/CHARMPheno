# Multi-domain spectral init for case-finding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the anchor-word spectral init to multiple token domains (v1: conditions + drugs) so a DAG node's topic can be anchored by whichever domain is purest for it, each topic recovers a proper per-domain distribution, and a corroborating domain sharpens node-vs-background contrast (specificity).

**Architecture:** ONE joint co-occurrence Q over the concatenated vocab `[conditions ; drugs]` with the cross-block intact; ONE greedy anchor search on it; a per-domain candidate floor so a sparser domain still yields anchors; a post-recovery split/renormalize into per-domain bases sharing topic identity. Reuses `word_cooccurrence`, `find_anchors` (one new kwarg), `recover_beta` verbatim. The `split_domains` output is **both** the per-domain readout/validation artifact **and** the initializer seed for each per-domain λ_m in the SP2 model core (arc design `2026-07-24-multidomain-gated-lda-arc-design.md`): `initialize_global` will set λ_m = scale·β^m + prior. SP1 itself validates the init recipe against the existing single-domain Gibbs oracle (`fit_gated` over the concatenated vocab, drugs as extra tokens) as an early read; the true per-domain-β oracle is built in SP2. Validation is on planted two-domain synthetic corpora only — the domain-agnostic engine never sees clinical vocabulary.

**Tech Stack:** Python, numpy, scipy (`scipy.optimize.nnls`), pytest. No Spark in the v1 engine path (the Gibbs `fit_gated` oracle is the validator, matching the branch's placement-validation convention).

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **Engine stays domain-agnostic:** integer token ids and integer domain-boundary offsets only. NO clinical/OMOP/EHR vocabulary in `spark_vi/**` or its tests. The domain edge (which integers are drugs) lives in `analysis/cloud`, out of scope for this plan.
- **Domain-neutral naming in engine code AND tests (including comments/docstrings):** refer to domains as `0`/`1` (or `a`/`b`), never by clinical role. So: `V_a`/`V_b` not `V_cond`/`V_drug`, `b_only_node` not `drug_only_node`, `planted_a`/`planted_b`, "domain-1 signature" not "drug signature". Clinical semantics (conditions corroborated by drugs) appear ONLY in this plan's motivation prose and the L3/charmpheno layer — never inside `spark_vi/**` or `spark-vi/tests/**`.
- **Cite methods in docstrings:** anchor-word = Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, Zhu 2013 ("A Practical Algorithm for Topic Modeling with Provable Guarantees", ICML). Corroboration / anchor-and-learn phenotyping = Halpern, Horng, Choi, Sontag 2016 (JAMIA, "Electronic medical record phenotyping using the anchor and learn framework"). A method/default/constant from the literature must cite its source.
- **No LaTeX; Unicode Greek only** (α, β, θ, Σ, η, λ). The IDE does not render math delimiters.
- **TDD** (superpowers:test-driven-development): failing test first, minimal impl, green, commit.
- **Domain-boundary representation (fixed for the whole plan):** `domain_bounds` is a strictly-increasing sequence of cumulative column offsets starting at 0 and ending at V, so domain `d` spans columns `[domain_bounds[d], domain_bounds[d+1])`. Example: domain 0 spans `[0, V_a)`, domain 1 spans `[V_a, V_a + V_b)`, so `domain_bounds = [0, V_a, V_a + V_b]`. `None` means "single pooled domain" and MUST reproduce current behavior byte-for-byte. This generalizes to N domains at no extra cost.
- **Backward compatibility:** every new parameter is keyword-only with a default that reproduces existing behavior. Existing callers and existing tests must stay green untouched.
- **Load-bearing prerequisite (state in code/test docstrings, do not silently assume):** the cross-domain tie is `Q_01` (the domain-0×domain-1 co-occurrence block), which exists only from WITHIN-DOCUMENT cross-domain co-occurrence. If the two domains' tokens live in separate documents `Q_01 = 0` and the two hulls disconnect. The synthetic generator MUST place a topic's domain-0 tokens and domain-1 tokens in the SAME document.
- **Commit trailer EXACTLY:**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- **Push:** this branch does not auto-push. Do NOT `git push` unless explicitly asked.

## File Structure

- `spark-vi/spark_vi/models/topic/spectral_init.py` — MODIFY. `find_anchors` gains `domain_bounds` (per-domain candidate floor); new pure helper `split_domains`. `spectral_init_beta` gains `domain_bounds` passthrough.
- `spark-vi/tests/test_spectral_init.py` — MODIFY. Unit tests for the per-domain floor, `split_domains`, and the joint-recovery acceptance test.
- `spark-vi/tests/_stm_synth.py` — MODIFY. New two-domain planted generator `two_domain_dag_corpus`.
- `spark-vi/spark_vi/models/topic/dag_placement.py` — MODIFY. `fit_gated` gains `domain_bounds` passthrough to `find_anchors` so the case-finding seed surfaces the sparser domain's anchors.
- `spark-vi/tests/test_dag_placement.py` — MODIFY. Two-domain fit-path test + the FDR-delta specificity acceptance test.

## Out of scope / follow-on (do NOT build here)

- The multi-domain **model core** (per-domain λ_m, block-Dirichlet expectation, shared gated θ with ω_m, multi-domain ELBO). That is **SP2** (arc design `2026-07-24-multidomain-gated-lda-arc-design.md`), which consumes SP1's `split_domains` output as the λ_m seed. SP1 stops at producing the per-domain bases + validating the anchor recipe; it does not build the multi-domain fit.
- Threading `domain_bounds` through the production SVI path (`gated_init.spectral_block_aligned_lambda`, `spectral_init_scalable.py`, `GatedOnlineLDA`, `dag_placement_cloud.py`). SP1 validates via the Gibbs oracle; SVI wiring lands with SP2/SP3.
- The real-cohort FDR-delta ablation and any `analysis/cloud` Makefile target / drug-domain cohort builder. That is **SP4**; depends on separately-spec'd cluster drivers (condition/drug DAG builders).

---

### Task 1: Per-domain candidate floor in `find_anchors`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/spectral_init.py` (`find_anchors`, ~line 83-160)
- Test: `spark-vi/tests/test_spectral_init.py`

**Interfaces:**
- Consumes: existing `find_anchors(Q, n, *, seed_rows=None, min_marginal_frac=1.0)`.
- Produces: `find_anchors(Q, n, *, seed_rows=None, min_marginal_frac=1.0, domain_bounds=None)`. When `domain_bounds` is None, behavior is byte-identical to today. When given (a cumulative-offset sequence per the Global Constraints), the candidate floor `thr` is computed WITHIN each domain: a word in domain `d` is a candidate iff its marginal ≥ `min_marginal_frac × mean(nonzero marginals in domain d)`. The greedy geometry and return value (newly chosen anchor ids in selection order) are unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_spectral_init.py`:

```python
def test_find_anchors_domain_bounds_none_is_identical():
    """domain_bounds=None reproduces the pooled-floor behavior exactly."""
    import numpy as np
    from spark_vi.models.topic.spectral_init import find_anchors, word_cooccurrence
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    V = 20
    docs = []
    for _ in range(200):
        toks = rng.integers(0, V, size=8)
        u, c = np.unique(toks, return_counts=True)
        docs.append(SimpleNamespace(indices=u, counts=c.astype(float)))
    Q = word_cooccurrence(docs, V)
    assert find_anchors(Q, 5) == find_anchors(Q, 5, domain_bounds=None)
    assert find_anchors(Q, 5) == find_anchors(Q, 5, domain_bounds=[0, V])


def test_find_anchors_per_domain_floor_admits_sparse_domain_anchor():
    """A pure anchor in a sparse second domain clears its WITHIN-domain floor
    even though its marginal is below the pooled mean, so it can be selected.
    Under the pooled floor it is excluded; under the per-domain floor it is not."""
    import numpy as np
    from spark_vi.models.topic.spectral_init import find_anchors
    # Domain A = cols [0:4] (dense), domain B = cols [4:6] (sparse).
    # Build Q directly: dense block carries most mass; the domain-B anchor (col 4)
    # co-occurs purely with a single domain-A word (col 0) but at low total mass.
    V = 6
    Q = np.zeros((V, V))
    # dense domain-A co-occurrence
    Q[0, 1] = Q[1, 0] = 0.20
    Q[2, 3] = Q[3, 2] = 0.20
    Q[0, 2] = Q[2, 0] = 0.10
    # sparse domain-B anchor col 4 pairs only with its domain-A partner col 0, low mass
    Q[0, 4] = Q[4, 0] = 0.02
    # col 5 is domain-B noise, negligible
    Q[5, 5] = 1e-9
    Q = Q / Q.sum()
    domain_bounds = [0, 4, 6]
    pooled = find_anchors(Q, 4)                      # pooled floor
    per_dom = find_anchors(Q, 4, domain_bounds=domain_bounds)
    # The domain-B anchor (col 4) is below the pooled marginal mean and excluded
    # by the pooled floor, but clears the sparse-domain mean under per-domain.
    assert 4 not in pooled
    assert 4 in per_dom
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py::test_find_anchors_per_domain_floor_admits_sparse_domain_anchor -v`
Expected: FAIL (`domain_bounds` is an unexpected keyword argument).

- [ ] **Step 3: Implement the per-domain floor**

In `find_anchors`, change the signature and the `candidate` computation. Replace the current floor block:

```python
    marginal = Q.sum(axis=1)
    pos = marginal > 0
    thr = min_marginal_frac * marginal[pos].mean() if pos.any() else 0.0
    candidate = marginal >= thr                   # eligible to BE an anchor
```

with a domain-aware version:

```python
    marginal = Q.sum(axis=1)
    candidate = _domain_candidate_mask(marginal, min_marginal_frac, domain_bounds)
```

and add a module-level helper (near `_row_normalize`):

```python
def _domain_candidate_mask(marginal, min_marginal_frac, domain_bounds):
    """Boolean 'eligible to be an anchor' mask, floored WITHIN each domain.

    The candidate floor (find_anchors docstring) keeps sub-promille noise words
    from being picked as spurious hull vertices. On a multi-domain joint Q the
    pooled mean is dominated by the densest domain, so a sparser domain's real
    anchors fall below the pooled bar and no anchor ever comes from it (spec:
    'domain imbalance is the most likely thing to silently break init'). The fix
    is to compare each word only to the mean nonzero marginal of ITS OWN domain.

    domain_bounds is a strictly-increasing cumulative-offset sequence starting at
    0 and ending at V; domain d spans [domain_bounds[d], domain_bounds[d+1]).
    None -> a single pooled domain, reproducing the original pooled floor exactly.
    """
    import numpy as np
    V = marginal.shape[0]
    if domain_bounds is None:
        domain_bounds = [0, V]
    mask = np.zeros(V, dtype=bool)
    for lo, hi in zip(domain_bounds[:-1], domain_bounds[1:]):
        seg = marginal[lo:hi]
        pos = seg > 0
        thr = min_marginal_frac * seg[pos].mean() if pos.any() else 0.0
        mask[lo:hi] = seg >= thr
    return mask
```

Add `domain_bounds=None` as the final keyword-only parameter of `find_anchors`, and extend its docstring with a short paragraph explaining the per-domain floor and citing the spec's domain-imbalance risk. Keep everything else in `find_anchors` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k "domain_bounds or per_domain_floor"`
Expected: PASS (both new tests).

- [ ] **Step 5: Run the full spectral-init suite for regressions**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py tests/test_gated_init.py -q`
Expected: PASS (no existing test perturbed).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/spectral_init.py spark-vi/tests/test_spectral_init.py
git commit -m "feat(spectral-init): per-domain candidate floor in find_anchors

Multi-domain joint Q: pool mean is dominated by the densest domain, so a
sparser domain's real anchors fall below the pooled bar. Floor the anchor
candidate marginal WITHIN each domain (domain_bounds cumulative offsets);
None reproduces the pooled floor byte-for-byte.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `split_domains` post-recovery helper

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/spectral_init.py` (new function `split_domains`)
- Test: `spark-vi/tests/test_spectral_init.py`

**Interfaces:**
- Consumes: a joint β of shape `(K, V)` (from `recover_beta`) and the same `domain_bounds` sequence.
- Produces: `split_domains(beta, domain_bounds) -> list[np.ndarray]`. Returns one `(K, V_d)` matrix per domain, each ROW renormalized to sum 1 (a proper P(word | topic) within that domain). A topic whose slice is all-zero in a domain (never expresses it) gets a uniform row over that domain's vocab, so every returned matrix is row-stochastic. Order of the returned list matches domain order in `domain_bounds`.

- [ ] **Step 1: Write the failing tests**

```python
def test_split_domains_renormalizes_each_slice():
    import numpy as np
    from spark_vi.models.topic.spectral_init import split_domains
    # K=2 topics, V=5: domain A cols [0:3], domain B cols [3:5].
    beta = np.array([
        [0.4, 0.1, 0.1, 0.2, 0.2],   # topic 0
        [0.0, 0.0, 0.0, 0.5, 0.5],   # topic 1: zero over domain A
    ])
    A, B = split_domains(beta, [0, 3, 5])
    assert A.shape == (2, 3) and B.shape == (2, 2)
    # topic 0 domain-A slice [0.4,0.1,0.1] renormalized by 0.6
    np.testing.assert_allclose(A[0], np.array([0.4, 0.1, 0.1]) / 0.6)
    np.testing.assert_allclose(B[0], np.array([0.2, 0.2]) / 0.4)
    # every returned row sums to 1
    np.testing.assert_allclose(A.sum(1), 1.0)
    np.testing.assert_allclose(B.sum(1), 1.0)
    # topic 1 is zero over domain A -> uniform fallback, still stochastic
    np.testing.assert_allclose(A[1], np.full(3, 1.0 / 3))


def test_split_domains_single_domain_is_identity_up_to_renorm():
    import numpy as np
    from spark_vi.models.topic.spectral_init import split_domains
    beta = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    (only,) = split_domains(beta, [0, 3])
    np.testing.assert_allclose(only, beta)   # already row-stochastic
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k split_domains`
Expected: FAIL (`split_domains` not defined).

- [ ] **Step 3: Implement `split_domains`**

Add to `spectral_init.py`:

```python
def split_domains(beta, domain_bounds):
    """Split a joint K×V β into per-domain row-renormalized bases.

    Under the shared-topic multi-domain model a token drawn in domain 0 and a
    token drawn in domain 1 from one document share the same θ, so the joint
    co-occurrence factors as Q_01 = (B_0)ᵀ A (B_1) (spec) and ONE anchor defines
    the topic across both domains. After recover_beta returns the joint β over
    the concatenated vocab, slicing each topic row at the domain boundaries and
    renormalizing each slice to sum 1 gives the per-domain P(word | topic)
    matrices — the MixEHR-style bases (β^0, β^1) that share topic identity
    (Halpern, Horng, Choi, Sontag, JAMIA 2016, anchor-and-learn corroboration).

    domain_bounds: strictly-increasing cumulative offsets [0, ..., V]. Returns a
    list of (K, V_d) row-stochastic matrices in domain order. A topic that never
    expresses a domain (all-zero slice) falls back to a uniform row there so each
    returned matrix stays a valid stochastic matrix.
    """
    out = []
    for lo, hi in zip(domain_bounds[:-1], domain_bounds[1:]):
        sub = beta[:, lo:hi].copy()
        rs = sub.sum(axis=1, keepdims=True)
        zero = (rs[:, 0] <= 0)
        if zero.any():
            sub[zero] = 1.0 / (hi - lo)
            rs[zero, 0] = 1.0
        out.append(sub / rs)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k split_domains`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/spectral_init.py spark-vi/tests/test_spectral_init.py
git commit -m "feat(spectral-init): split_domains post-recovery per-domain bases

Slice the joint K×V beta at domain boundaries and row-renormalize each
slice -> the MixEHR-style per-domain P(word|topic) matrices sharing topic
identity. Uniform fallback keeps a never-expressed domain row stochastic.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Two-domain planted synthetic generator

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py` (new `two_domain_dag_corpus`)
- Test: `spark-vi/tests/test_spectral_init.py`

**Interfaces:**
- Produces: `two_domain_dag_corpus(*, parent, node_prev, V_a, V_b, doc_len, seed, b_only_node=None, ubiquitous_b=False) -> (docs, labels, domain_bounds, planted_a, planted_b, slot_of_node, node_codes)`.
  - `docs`: list of integer token arrays spanning `[0, V_a + V_b)`; a doc at node v emits a shared common pool + the domain-0 signatures along `closure(v)` AND the domain-1 signatures along `closure(v)`, in the SAME array (within-doc cross-domain co-occurrence — the load-bearing tie).
  - `domain_bounds = [0, V_a, V_a + V_b]`.
  - `planted_a` (K, V_a) and `planted_b` (K, V_b): the per-domain ground-truth signatures aligned to `DagLayout` slot order (reuse the `dag_placement_corpus` slotting convention).
  - `slot_of_node`: dict {node id -> DagLayout slot index}, so a caller can index `planted_a`/`planted_b` rows for a given node without re-deriving the slotting. (Consumed by Tasks 4/5/6.)
  - `b_only_node` (optional node id): that node's DOMAIN-0 signature is made ambiguous (shared with its parent, no unique domain-0 token) while its DOMAIN-1 signature stays unique — the "recovered from domain 1 alone" case the spec requires.
  - `ubiquitous_b` (bool): if True, add one domain-1 token emitted by EVERY document regardless of node (the universal-anchor control of Risk 1 / insight 0021) — used by the FDR-delta control in Task 6.
  - `node_codes`: exact per-node marker code dict (mirrors `dag_placement_corpus`), for `strip_dag_node_codes` at eval.
- Consumes: `DagLayout` from `dag_placement`; the slotting/common-pool conventions of the existing `dag_placement_corpus` (line ~579).

- [ ] **Step 1: Write the failing tests**

```python
def test_two_domain_corpus_within_doc_cross_domain():
    """Every doc's tokens span BOTH domains (Q_01 != 0 prerequisite) and a
    domain-1-only node's docs still carry its unique domain-1 signature."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    parent = {1: 0, 2: 1, 3: 1}     # root 0 -> node 1 -> leaves 2,3
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0},
        V_a=30, V_b=12, doc_len=24, seed=1, b_only_node=3)
    Va = domain_bounds[1]
    # at least half the docs carry a domain-0 token (<Va) AND a domain-1 token (>=Va)
    spanning = [d for d in docs if (np.asarray(d) < Va).any() and (np.asarray(d) >= Va).any()]
    assert len(spanning) > 0.5 * len(docs)
    # planted shapes
    assert pa.shape[1] == Va and pb.shape[1] == domain_bounds[2] - Va
    # b_only_node=3 has a nonzero unique domain-1 signature row (recovered in Task 4)
    assert pb[slot_of_node[3]].sum() > 0
```

(Keep the assertions to structural invariants the generator guarantees — spanning docs, shapes, a nonzero planted domain-1 signature. The recovery claim is Task 4.)

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k two_domain_corpus`
Expected: FAIL (`two_domain_dag_corpus` not defined).

- [ ] **Step 3: Implement the generator**

Model it on `dag_placement_corpus` (`_stm_synth.py` ~line 579). Concatenated vocab: domain 0 `[0, V_a)`, domain 1 `[V_a, V_a+V_b)`. Within each domain, reserve a shared common pool then per-node signature blocks (reuse the `C = V//3`, `sig` slotting idea separately per domain). For a doc at node v: emit common-pool tokens from BOTH domains + domain-0-signature tokens for every node in `closure(v)` (minus root) + domain-1-signature tokens for every node in `closure(v)`. For `b_only_node`, set that node's domain-0 signature block to its PARENT's domain-0 block (ambiguous — no unique domain-0 token) but keep a unique domain-1 block. For `ubiquitous_b=True`, append one extra domain-1 column index emitted by every doc. Build `planted_a`/`planted_b` as (K, V_d) rows aligned to `DagLayout` slot order exactly like `dag_placement_corpus` builds `node_sig`, and return `slot_of_node` mapping each node id to its slot index. Return `domain_bounds = [0, V_a, V_a + V_b]`. Docstring MUST state the within-doc co-occurrence prerequisite and that ids are integer/domain-agnostic.

- [ ] **Step 4: Run to verify pass**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k two_domain_corpus`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_spectral_init.py
git commit -m "test(synth): two-domain planted DAG corpus for multi-domain init

Two domains in ONE concatenated vocab, co-occurring WITHIN each document
(the Q_01 prerequisite). Supports a domain-1-only node (ambiguous in
domain 0, unique in domain 1) and a ubiquitous domain-1 token control.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: End-to-end joint-recovery acceptance test

**Files:**
- Test: `spark-vi/tests/test_spectral_init.py` (test-only; the acceptance gate for Tasks 1-3)

**Interfaces:**
- Consumes: `two_domain_dag_corpus` (Task 3), `word_cooccurrence`, `find_anchors(domain_bounds=...)` (Task 1), `recover_beta`, `split_domains` (Task 2).
- Produces: no new code — an integration test asserting the whole recipe recovers both domains, including a topic anchored from DOMAIN 1 alone.

- [ ] **Step 1: Write the acceptance test**

```python
def test_multidomain_init_recovers_both_domains_incl_b_anchored():
    """The joint recipe (one Q, one greedy with per-domain floor, one recover,
    split) recovers per-domain phenotypes for every node, INCLUDING a node whose
    domain-0 signature is ambiguous but whose domain-1 signature is unique."""
    import numpy as np
    from types import SimpleNamespace
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.spectral_init import (
        word_cooccurrence, find_anchors, recover_beta, split_domains)

    parent = {1: 0, 2: 1, 3: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0},
        V_a=40, V_b=16, doc_len=30, seed=3, b_only_node=3)
    V = domain_bounds[-1]
    counted = [SimpleNamespace(indices=np.unique(np.asarray(d)),
               counts=np.unique(np.asarray(d), return_counts=True)[1].astype(float))
               for d in docs]
    Q = word_cooccurrence(counted, V)
    K = pa.shape[0]
    anchors = find_anchors(Q, K, domain_bounds=domain_bounds)
    beta = recover_beta(Q, anchors)
    ba, bb = split_domains(beta, domain_bounds)

    # at least one anchor comes from domain 1 (id >= V_a)
    assert any(a >= domain_bounds[1] for a in anchors)
    # domain-1 recovery: node 3's unique planted domain-1 block is captured by
    # some recovered domain-1 topic (support-overlap mass), even though its
    # domain-0 signature is ambiguous.
    def _support(row, eps=1e-3):
        return np.where(row > eps)[0]
    node_b_support = _support(pb[slot_of_node[3]])
    assert bb[:, node_b_support].sum(axis=1).max() > 0.4
```

The node→slot mapping comes from the generator's `slot_of_node` return (Task 3) — no local helper needed.

- [ ] **Step 2: Run the acceptance test**

Run: `cd spark-vi && python -m pytest tests/test_spectral_init.py -v -k multidomain_init_recovers`
Expected: PASS. If the domain-1-anchored node does not clear the margin, STRENGTHEN THE PLANT (purer/denser unique domain-1 signature in the generator), never loosen the assertion (test-honesty). If a strengthened plant still fails, that is a real negative — stop and report.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_spectral_init.py spark-vi/tests/_stm_synth.py
git commit -m "test(spectral-init): multi-domain joint recovery acceptance

One joint Q -> per-domain-floored greedy -> recover -> split recovers both
per-domain bases, including a node anchored from domain 1 alone (ambiguous
in domain 0, unique in domain 1).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Thread `domain_bounds` through the fit path

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/spectral_init.py` (`spectral_init_beta`)
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (`fit_gated`, ~line 128-153)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- `spectral_init_beta(docs, partition, V, *, domain_bounds=None)`: passes `domain_bounds` to every `find_anchors` call (background pooled Q and each within-group Q_g). None reproduces current behavior. Per the spec, the multi-domain Q is built per group in step 2; `word_cooccurrence` already produces the joint Q, so only the floor threading is needed.
- `fit_gated(train_docs, train_labels, lay, V, *, beta_prior=0.02, n_iter=150, burn=80, rng=None, domain_bounds=None)`: passes `domain_bounds` to its `find_anchors(Q, K, ...)` seed call (line ~153). None reproduces current behavior byte-for-byte.

- [ ] **Step 1: Write the failing test**

```python
def test_fit_gated_domain_bounds_surfaces_b_seed():
    """With domain_bounds, fit_gated's spectral seed admits a domain-1 anchor for
    a domain-1-only node, and the fit still returns a valid (K,V) beta."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout, fit_gated
    parent = {1: 0, 2: 1, 3: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0},
        V_a=40, V_b=16, doc_len=30, seed=5, b_only_node=3)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    V = domain_bounds[-1]
    rng = np.random.default_rng(0)
    beta = fit_gated(docs[:900], labels[:900], lay, V, n_iter=40, burn=20,
                     rng=rng, domain_bounds=domain_bounds)
    assert beta.shape == (lay.K, V)
    assert np.isfinite(beta).all()
    np.testing.assert_allclose(beta.sum(1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -v -k domain_bounds_surfaces`
Expected: FAIL (`domain_bounds` unexpected kwarg).

- [ ] **Step 3: Implement the passthroughs**

In `spectral_init_beta` add `domain_bounds=None` keyword-only param; pass `domain_bounds=domain_bounds` to the background `find_anchors(Q_all, partition.background_k)` and each group's `find_anchors(Q_g, len(fg_idx), seed_rows=bg_anchors)`. In `dag_placement.fit_gated` add `domain_bounds=None` keyword-only param; change `beta0 = recover_beta(Q, find_anchors(Q, K))` to `beta0 = recover_beta(Q, find_anchors(Q, K, domain_bounds=domain_bounds))`. Update both docstrings to mention the multi-domain passthrough and cite the spec's joint-Q construction.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py tests/test_spectral_init.py -q`
Expected: PASS (new test green, all existing green).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/spectral_init.py spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): thread domain_bounds through fit path

spectral_init_beta and fit_gated accept optional domain_bounds and pass it
to find_anchors so the multi-domain seed surfaces the sparser domain's
anchors. None keeps existing behavior byte-for-byte.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: FDR-delta specificity acceptance test (the green light)

**Files:**
- Test: `spark-vi/tests/test_dag_placement.py` (test-only; the empirical specificity claim)

**Interfaces:**
- Consumes: `two_domain_dag_corpus` (Task 3, with `b_only_node` and `ubiquitous_b`), `fit_gated(domain_bounds=...)` (Task 5), `profile`, `strip_dag_node_codes`, `evaluate` (existing), `per_node_discoveries` / the `fdr_block` from `evaluate` (existing).
- Produces: no new engine code — an acceptance test that fits + profiles + evaluates the SAME corpus with domain 1 IN vs OUT (domain-1 columns dropped) and asserts the corroborating-domain node gains discoveries / lower per-node FDR at fixed q, while a ubiquitous domain-1 token does NOT.

- [ ] **Step 1: Write the acceptance test**

Follow the existing end-to-end pattern (`test_dag_placement.py` ~line 162-173): build the two-domain corpus with `b_only_node=<leaf>`; split train/test; `fit_gated` with `domain_bounds`; `profile(strip_dag_node_codes(d, codes), beta, lay, ...)` per held-out doc; `evaluate(profs, labels_test, lay)`. Then repeat with the domain-1 columns removed (domain-0-only vocab, `domain_bounds=None`). Compare the `fdr_block` / per-node discoveries for the domain-1-only node.

```python
def test_fdr_delta_corroborating_domain_raises_leaf_specificity():
    """Specificity claim (spec): a node-specific domain-1 signature lowers that
    leaf node's per-node FDR (more discoveries at fixed q) vs domain-0-only,
    because the corroborating domain sharpens node-vs-background contrast. A
    UBIQUITOUS domain-1 token (insight 0021 universal anchor) does NOT."""
    # ... build corpus, fit+profile+evaluate domain-1 IN and OUT ...
    # assert discoveries_in[b_only_node] >= discoveries_out[b_only_node]
    #   with a margin, at the SAME sensitivity/q
    # assert ubiquitous-domain-1 variant shows no such gain (control)
```

Make the plant STRONG (a domain-1 signature that fires only for `b_only_node`, dense unique block) so the direction is unambiguous. Assert a DIRECTIONAL inequality with margin at matched q — never a brittle exact number. Hold leakage fixed: `strip_dag_node_codes` removes the exact node-marker codes in BOTH arms so the FDR drop comes from corroboration, not from smuggling the label into features (spec Risk 2). Document that assertion explicitly in the test.

- [ ] **Step 2: Run the acceptance test**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -v -k fdr_delta_corroborating`
Expected: PASS. If the direction does not hold with a strong plant, STOP — that is a genuine negative on the specificity claim (the spec's own "value is measured, not assumed"). Report it; do not weaken the assertion to force green. If flaky across seeds, fix the seed and strengthen the plant (test-honesty: strengthen the plant, not loosen the gate).

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_dag_placement.py
git commit -m "test(dag-placement): FDR-delta specificity acceptance for cross-domain corroboration

Node-specific domain-1 signature lowers that leaf's per-node FDR (more
discoveries at fixed q) vs domain-0-only; a ubiquitous domain-1 token does
not. Leakage held fixed (node-marker codes stripped in both arms) so the
gain is corroboration, not label smuggling.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-plan wrap-up (controller, after Task 6)

- [ ] Whole-branch review (superpowers:requesting-code-review, most capable model) over `git merge-base main HEAD`..HEAD.
- [ ] Add an insights entry (`docs/insights/NNNN-*.md`) recording the FDR-delta result — positive (specificity gain quantified) or negative (drug corroboration null on synthetic) — per the maintain-the-insights-log convention.
- [ ] Do NOT merge or push; present the branch to the user for the merge/PR decision and the follow-on (SVI-path threading, real-cohort ablation).
