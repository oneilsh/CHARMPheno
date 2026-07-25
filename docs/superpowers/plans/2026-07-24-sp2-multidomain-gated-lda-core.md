# SP2 — Multi-domain gated LDA model core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the gated DAG LDA model multi-domain (MixEHR-style): each topic gets a per-domain topic-word distribution, all sharing one DAG-gated document-topic mixture θ, with a tuned per-domain modality weight ω_m to de-bias token-volume imbalance. Validated by a multi-domain Gibbs oracle + planted per-domain recovery + SVI≈Gibbs.

**Architecture:** The whole multi-domain change is generalizing the topic-word Dirichlet normalizer from a full row (`λ.sum(axis=1)`) to a per-domain block (`Σ_{w∈domain m} λ`), everywhere `expElogbeta` is formed. λ stays a single K×V array plus a `domain_bounds` cumulative-offset vector (domain-BLOCKED Dirichlet), so `combine_stats`/VIRunner/export are untouched and `domain_bounds=None` (a single [0,V] block) is byte-identical to today (the N=1 identity). The per-domain distributions are the `split_domains` view (SP1). Gate ⟂ domain: the gate acts on θ's support (allowed topics), the domain block acts on β's normalizer — orthogonal. ω_m weights the θ accumulation only; β sufficient-stats use true counts. Seeded by SP1's multi-domain spectral init.

**Tech Stack:** Python, numpy, scipy (`digamma`, `gammaln`), pytest. Validated locally via the collapsed-Gibbs oracle (the branch convention); no Spark in SP2 (shim/cloud is SP3/SP4).

## Global Constraints

- **Engine domain-agnostic + domain-neutral naming:** integer token ids and integer domain-boundary offsets ONLY. NO clinical/OMOP/EHR vocabulary anywhere in `spark_vi/**` or `spark-vi/tests/**` — code, comments, docstrings. Domains are named `0`/`1`/`m` (or `a`/`b`), never by clinical role. Clinical semantics live only in the arc-design/plan motivation prose and the L3/charmpheno layer.
- **`domain_bounds` semantics (same as SP1):** strictly-increasing cumulative column offsets `[0, …, V]`; domain d spans `[domain_bounds[d], domain_bounds[d+1])`. `None` == a single `[0, V]` block == current single-domain behavior, and MUST be byte-identical (the N=1 identity). Every new param is keyword-only with a behavior-preserving default.
- **ω weights θ, not β:** the per-domain modality weight scales the γ (doc-topic) accumulation only; the λ (topic-word) sufficient-stats, `phi_norm`, and the data-loglik use TRUE counts. ω default = 1.0 for every domain (faithful MixEHR — volume speaks). Document ω as a pseudo-likelihood/tempering weight (arc design), NOT the vanilla generative likelihood.
- **Cite:** MixEHR (Li, Nair, Lu et al. 2020, Nat. Commun. — per-modality topic-word, shared patient-topic mixture), Hoffman-Blei-Bach 2010 (Online LDA / SVI), Griffiths & Steyvers 2004 (the Gibbs oracle). No LaTeX; Unicode Greek only (α, β, θ, Σ, η, λ, ω, ψ, ρ).
- **v2 seam invariant:** the E-step must keep each token's domain available at the point responsibilities/γ are formed (a per-token domain lookup from `domain_bounds`), so v2 (generative π) is later a one-factor add. Do NOT collapse the E-step into a domain-agnostic gather that discards which domain a token came from.
- **TDD.** This branch does NOT auto-push — push only when asked.
- Commit trailer EXACTLY (last line of every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- `spark-vi/spark_vi/models/topic/lda.py` — MODIFY. New `dirichlet_expectation_blockwise(lam, domain_bounds)` helper; `OnlineLDA.__init__` gains `domain_bounds=None`; the 3 `expElogbeta` sites (`local_update`, `update_global`, `infer_local`) and `compute_elbo`'s global-KL use it; per-domain η.
- `spark-vi/spark_vi/models/topic/gated_lda.py` — MODIFY. `GatedOnlineLDA` uses the blockwise helper at its 2 `expElogbeta` sites; carries `domain_bounds`; multi-domain spectral-init seed; ω_m γ-weighting; per-domain θ-contribution instrumentation.
- `spark-vi/spark_vi/models/topic/dag_placement.py` — MODIFY. `fit_gated` (the Gibbs oracle) gains a per-domain collapsed word-topic factor (multi-domain oracle).
- `spark-vi/tests/test_lda.py`, `test_gated_lda.py` (or the existing gated test files), `test_dag_placement.py` — tests.

## Out of scope / follow-on

- The mllib shim wiring (concatenated `featuresCol` + `domainBounds` Param + split-on-ingest) and export of per-domain β — that is **SP3**.
- Real-cohort assembly + ω-swept FDR-delta on real data — **SP4**.
- v2 generative π_{k,m} — research extension after SP2; kept cheap by the v2-seam invariant (Task 8 guards it).

---

### Task 1: Block-Dirichlet expectation helper + thread through OnlineLDA

**Files:** Modify `spark-vi/spark_vi/models/topic/lda.py`; Test `spark-vi/tests/test_lda.py`.

**Interfaces:**
- Produces `dirichlet_expectation_blockwise(lam, domain_bounds=None) -> np.ndarray`: `exp(ψ(lam) − ψ(per-domain-block row sum))`. `None` → single `[0, lam.shape[1]]` block → byte-identical to `exp(digamma(lam) − digamma(lam.sum(axis=1, keepdims=True)))`.
- `OnlineLDA.__init__(..., domain_bounds=None)` keyword-only; stored as `self.domain_bounds`. Consumed by `local_update`, `update_global`, `infer_local`.

- [ ] **Step 1: Write the failing tests**

```python
def test_dirichlet_expectation_blockwise_none_is_full_row():
    import numpy as np
    from scipy.special import digamma
    from spark_vi.models.topic.lda import dirichlet_expectation_blockwise
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, size=(4, 10))
    full = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    np.testing.assert_allclose(dirichlet_expectation_blockwise(lam, None), full)
    np.testing.assert_allclose(dirichlet_expectation_blockwise(lam, [0, 10]), full)


def test_dirichlet_expectation_blockwise_normalizes_per_block():
    import numpy as np
    from scipy.special import digamma
    from spark_vi.models.topic.lda import dirichlet_expectation_blockwise
    rng = np.random.default_rng(1)
    lam = rng.gamma(2.0, size=(3, 7))           # domain 0 = cols [0:4], domain 1 = [4:7]
    out = dirichlet_expectation_blockwise(lam, [0, 4, 7])
    exp0 = np.exp(digamma(lam[:, :4]) - digamma(lam[:, :4].sum(axis=1, keepdims=True)))
    exp1 = np.exp(digamma(lam[:, 4:]) - digamma(lam[:, 4:].sum(axis=1, keepdims=True)))
    np.testing.assert_allclose(out[:, :4], exp0)
    np.testing.assert_allclose(out[:, 4:], exp1)


def test_online_lda_domain_bounds_none_byte_identical():
    """A full fit with domain_bounds=None reproduces the default fit exactly."""
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA
    from spark_vi.models.topic.types import BOWDocument
    rng = np.random.default_rng(3)
    docs = []
    for _ in range(40):
        idx = np.unique(rng.integers(0, 12, size=8)).astype(np.int32)
        docs.append(BOWDocument(indices=idx, counts=np.ones(len(idx)), length=len(idx)))
    def fit(db):
        m = OnlineLDA(K=4, vocab_size=12, random_seed=7, domain_bounds=db)
        gp = m.initialize_global(None)
        for _ in range(10):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
        return gp["lambda"]
    np.testing.assert_allclose(fit(None), fit([0, 12]), rtol=0, atol=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_lda.py -v -k "blockwise or domain_bounds_none_byte"`
Expected: FAIL (`dirichlet_expectation_blockwise` undefined / unexpected kwarg).

- [ ] **Step 3: Implement the helper + threading**

Add to `lda.py` (module level):

```python
def dirichlet_expectation_blockwise(lam, domain_bounds=None):
    """exp(E[log β]) with the Dirichlet normalizer taken PER DOMAIN BLOCK.

    Standard LDA normalizes each topic row over the whole vocabulary:
    exp(ψ(λ_kw) − ψ(Σ_w λ_kw)). Under the multi-domain (MixEHR-style) model
    each (topic, domain) slice is its OWN Dirichlet, so the normalizer runs over
    that domain's columns only: exp(ψ(λ_kw) − ψ(Σ_{w'∈domain m} λ_kw')). This is
    the single mechanism that turns the topic-word prior into MixEHR's
    per-modality Dirichlet (Li et al. 2020); the shared θ and the DAG gate are
    orthogonal and unchanged.

    domain_bounds: strictly-increasing cumulative offsets [0, …, V]; domain d
    spans [domain_bounds[d], domain_bounds[d+1]). None == a single [0, V] block,
    reproducing the full-row normalizer byte-for-byte (the N=1 identity).
    """
    import numpy as np
    from scipy.special import digamma
    V = lam.shape[1]
    if domain_bounds is None:
        domain_bounds = [0, V]
    out = np.empty_like(lam, dtype=np.float64)
    for lo, hi in zip(domain_bounds[:-1], domain_bounds[1:]):
        block = lam[:, lo:hi]
        out[:, lo:hi] = np.exp(digamma(block) - digamma(block.sum(axis=1, keepdims=True)))
    return out
```

Add `domain_bounds=None` as the final keyword-only param of `OnlineLDA.__init__`, store `self.domain_bounds = None if domain_bounds is None else list(domain_bounds)`. In `local_update`, `update_global`, and `infer_local`, replace each
`expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))`
with
`expElogbeta = dirichlet_expectation_blockwise(lam, self.domain_bounds)`.
(Leave `compute_elbo`'s global-KL for Task 2.) Update the module docstring's symbol table to note the per-domain normalizer.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd spark-vi && python -m pytest tests/test_lda.py -q`
Expected: PASS (new tests + all existing vanilla-LDA tests unchanged — the None default guarantees it).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/lda.py spark-vi/tests/test_lda.py
git commit -m "feat(lda): per-domain-block Dirichlet expectation (N=1 identity)

Generalize the topic-word normalizer from a full row to a per-domain block
(dirichlet_expectation_blockwise); OnlineLDA gains domain_bounds (None ==
single [0,V] block, byte-identical). This is MixEHR's per-modality Dirichlet
(Li 2020); shared theta + gate unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Per-domain η and per-domain global KL in the ELBO

**Files:** Modify `spark-vi/spark_vi/models/topic/lda.py` (`compute_elbo`, and η handling in `__init__`/`initialize_global`); Test `spark-vi/tests/test_lda.py`.

**Interfaces:**
- `OnlineLDA.__init__` accepts `eta` as a scalar (current) OR a length-(n_domains) sequence of per-domain concentrations. Stored so `initialize_global` can broadcast it to a length-V η vector (each column gets its domain's η_m). Scalar η with `domain_bounds=None` is byte-identical to today.
- `compute_elbo`'s global KL becomes Σ_k Σ_m KL(Dirichlet(λ[k, block_m]) || Dirichlet(η_m · 1_{V_m})) — a per-(topic, domain) Dirichlet KL.

- [ ] **Step 1: Write the failing tests**

```python
def test_per_domain_eta_broadcasts_to_vocab():
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA
    m = OnlineLDA(K=3, vocab_size=7, eta=[0.1, 0.5], domain_bounds=[0, 4, 7], random_seed=0)
    gp = m.initialize_global(None)
    # eta stored so the per-column prior is 0.1 on domain 0, 0.5 on domain 1
    eta_vec = m._eta_vocab_vector()          # helper: length-V prior
    np.testing.assert_allclose(eta_vec[:4], 0.1)
    np.testing.assert_allclose(eta_vec[4:], 0.5)


def test_compute_elbo_per_domain_kl_matches_manual():
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA, _dirichlet_kl
    m = OnlineLDA(K=2, vocab_size=5, eta=[0.2, 0.3], domain_bounds=[0, 3, 5], random_seed=1)
    gp = m.initialize_global(None)
    lam = gp["lambda"]
    # manual per-(topic,domain) KL
    manual = 0.0
    for k in range(2):
        manual += _dirichlet_kl(lam[k, :3], np.full(3, 0.2))
        manual += _dirichlet_kl(lam[k, 3:], np.full(2, 0.3))
    agg = {"doc_loglik_sum": np.array(0.0), "doc_theta_kl_sum": np.array(0.0)}
    elbo = m.compute_elbo(gp, agg)
    np.testing.assert_allclose(elbo, -manual)


def test_scalar_eta_none_bounds_elbo_unchanged():
    """Scalar eta + domain_bounds=None reproduces the pre-change global KL."""
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA, _dirichlet_kl
    m = OnlineLDA(K=2, vocab_size=5, eta=0.25, random_seed=1)
    gp = m.initialize_global(None); lam = gp["lambda"]
    manual = sum(_dirichlet_kl(lam[k], np.full(5, 0.25)) for k in range(2))
    agg = {"doc_loglik_sum": np.array(0.0), "doc_theta_kl_sum": np.array(0.0)}
    np.testing.assert_allclose(m.compute_elbo(gp, agg), -manual)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_lda.py -v -k "per_domain_eta or per_domain_kl or scalar_eta_none"`
Expected: FAIL (`_eta_vocab_vector` undefined; per-domain KL not implemented).

- [ ] **Step 3: Implement**

In `__init__`: accept `eta` scalar or length-(len(domain_bounds)-1) sequence. Validate: if `eta` is a sequence, `domain_bounds` must be set and lengths must match; all > 0. Store `self.eta` (scalar) or `self._eta_by_domain` (array). Add a method `_eta_vocab_vector()` returning a length-V array: scalar η → `np.full(V, η)`; per-domain η → each column filled with its domain's η_m via `domain_bounds`. In `initialize_global`, keep the stored `"eta"` as-is for the scalar case (byte-identical) but ALSO stash the vocab vector where `compute_elbo` can read it (e.g. compute on demand from `self`). In `compute_elbo`, replace the single-block loop with a per-(topic, domain) loop using `_eta_vocab_vector()` sliced per block. Ensure scalar-η + `domain_bounds=None` yields exactly the old single-block KL.

*(Note for implementer: keep the `"eta"` entry in `global_params` scalar-or-0d for the scalar case so `combine_stats`/`update_global`'s η path is unchanged; per-domain η is fixed (not optimized) in SP2 — `optimize_eta` stays unsupported in the gated engine.)*

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd spark-vi && python -m pytest tests/test_lda.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/lda.py spark-vi/tests/test_lda.py
git commit -m "feat(lda): per-domain eta prior + per-domain global KL

eta accepts a per-domain vector; compute_elbo's global KL becomes a
per-(topic,domain) Dirichlet KL. Scalar eta + domain_bounds=None is
byte-identical to the prior single-block KL.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Multi-domain GatedOnlineLDA (blockwise + multi-domain seed)

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py`; Test `spark-vi/tests/test_gated_lda.py` (create if absent; else the existing gated-lda test module).

**Interfaces:**
- `GatedOnlineLDA.__init__(lay, vocab_size, *, domain_bounds=None, **kw)` — forward `domain_bounds` to `OnlineLDA.__init__`; store `self.domain_bounds`.
- Its two `expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))` sites (`local_update`, `update_global`) become `dirichlet_expectation_blockwise(lam, self.domain_bounds)`.
- `initialize_global`: when a multi-domain spectral init is requested, seed λ from SP1's `spectral_init` recipe (joint Q → `find_anchors(domain_bounds=...)` → `recover_beta`), i.e. a per-domain-floored joint β over the concatenated vocab, scaled into λ. (Reuse SP1's `find_anchors`/`recover_beta`; the block structure is already correct because λ is the joint K×V matrix.)

- [ ] **Step 1: Write the failing test**

```python
def test_gated_lda_multidomain_fit_valid_and_blockwise():
    """A gated multi-domain fit over the two-domain corpus yields a (K,V) lambda
    whose per-domain expElogbeta blocks each normalize within their domain."""
    import numpy as np
    from scipy.special import digamma
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    parent = {1: 0, 2: 1, 3: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1., 2: 1., 3: 1.}, V_a=40, V_b=16,
        doc_len=30, seed=5, b_only_node=3)
    lay = DagLayout(parent, n_bg=2, tpn=1); V = domain_bounds[-1]
    gdocs = [GatedBOWDocument(indices=np.unique(d).astype(np.int32),
             counts=np.unique(d, return_counts=True)[1].astype(float),
             length=len(d), frontier=frozenset(f) if hasattr(f, "__iter__") else frozenset({f}))
             for d, f in zip(docs[:600], labels[:600])]
    m = GatedOnlineLDA(lay, vocab_size=V, domain_bounds=domain_bounds, random_seed=0)
    gp = m.initialize_global(None)
    for _ in range(20):
        gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=0.5)
    lam = gp["lambda"]
    assert lam.shape == (lay.K, V) and np.isfinite(lam).all()
    # each topic's per-domain beta slice is a proper distribution
    beta0 = lam[:, :domain_bounds[1]] / lam[:, :domain_bounds[1]].sum(1, keepdims=True)
    beta1 = lam[:, domain_bounds[1]:] / lam[:, domain_bounds[1]:].sum(1, keepdims=True)
    np.testing.assert_allclose(beta0.sum(1), 1.0); np.testing.assert_allclose(beta1.sum(1), 1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -v -k multidomain_fit_valid`
Expected: FAIL (unexpected `domain_bounds` kwarg).

- [ ] **Step 3: Implement**

Thread `domain_bounds` through `GatedOnlineLDA.__init__` to `super().__init__`; store `self.domain_bounds`. Replace the two inline `expElogbeta` computations (`local_update` ~line 126, `update_global` ~line 215) with `dirichlet_expectation_blockwise(lam, self.domain_bounds)` (import it from `lda`). For the multi-domain spectral seed in `initialize_global`, when `self.init` is a spectral strategy and `domain_bounds` is set, pass `domain_bounds` into the anchor search (SP1's `find_anchors(..., domain_bounds=domain_bounds)` / the block-aware init) so the seed λ surfaces the sparser domain's anchors. Update docstrings (cite MixEHR + the SP1 seed handoff).

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py tests/test_lda.py -q`
Expected: PASS (existing single-domain gated tests unchanged via the None default).

- [ ] **Step 5: Commit** (`feat(gated-lda): multi-domain blockwise expElogbeta + spectral seed`, trailer as above.)

---

### Task 4: Multi-domain Gibbs oracle (the validator)

**Files:** Modify `spark-vi/spark_vi/models/topic/dag_placement.py` (`fit_gated`); Test `spark-vi/tests/test_dag_placement.py`.

**Interfaces:**
- `fit_gated(..., *, domain_bounds=None)` — already threads `domain_bounds` into the spectral seed (SP1 Task 5). Now make the collapsed-Gibbs per-token conditional use a PER-DOMAIN word-topic factor: for a token w in domain m, `(n_kw + η_m) / (n_{k, ·∈m} + V_m·η_m)` instead of the pooled `(n_kw + β) / (n_k + V·β)`. `domain_bounds=None` reproduces the current single-domain sampler byte-for-byte.

- [ ] **Step 1: Write the failing test** — a recovery test: fit the multi-domain Gibbs oracle on the two-domain corpus with `domain_bounds`, assert both per-domain β blocks recover the planted signatures (using `slot_of_node`, mirroring SP1 Task 4's recovery assertion but through the Gibbs `beta_hat`). Also a `domain_bounds=None` byte-identical check against a fixed-seed current-oracle fit.

- [ ] **Step 2: Run to verify failure.** (`domain_bounds` per-domain normalizer not yet in the sampler.)

- [ ] **Step 3: Implement** — in `fit_gated`'s per-token Gibbs loop, precompute per-domain `n_{k,·∈m}` denominators and a per-token domain index (`np.searchsorted(domain_bounds, w, side="right") - 1`); replace the pooled `(n_kw + beta_prior)/(n_k + V*beta_prior)` with the per-domain factor using `η_m` (accept a per-domain `beta_prior` sequence too, defaulting to the scalar for all domains). `domain_bounds=None` → one domain → identical arithmetic. Cite Griffiths & Steyvers 2004 + the per-modality extension.

- [ ] **Step 4: Run to verify pass + regressions.** Full `tests/test_dag_placement.py`.

- [ ] **Step 5: Commit** (`feat(dag-placement): multi-domain per-domain Gibbs oracle`, trailer.)

---

### Task 5: Planted per-domain recovery (SVI) acceptance

**Files:** Test only — `spark-vi/tests/test_gated_lda.py` (acceptance gate composing Tasks 1-3).

- [ ] **Step 1: Write the acceptance test** — fit the multi-domain `GatedOnlineLDA` (SVI, `fit_gated_svi_local` from `_stm_synth.py`, or an inline batch-VB loop) on the two-domain corpus with `b_only_node`; split β via `split_domains`; assert BOTH β^0 and β^1 recover the planted per-domain signatures for every node, INCLUDING the domain-1-anchored node (`slot_of_node`), mirroring SP1 Task 4's recovery assertion but through the SVI model rather than the raw spectral recipe.
- [ ] **Step 2: Run.** If recovery is short of the gate, STRENGTHEN THE PLANT (via the SP1 generator's `b_only_signal_boost` / more docs / more iters), NEVER loosen the assertion. A strong-plant failure is a genuine negative — STOP and report.
- [ ] **Step 3: Commit** (`test(gated-lda): multi-domain SVI planted per-domain recovery`, trailer.)

---

### Task 6: SVI ≈ Gibbs equivalence on multi-domain

**Files:** Test only — `spark-vi/tests/test_gated_lda.py`.

- [ ] **Step 1: Write the equivalence test** — following the existing SVI≈Gibbs placement-equivalence pattern (the branch already validates single-domain this way; find it in the current gated tests / `svi_node_profiles` in `_stm_synth.py`): on a small two-domain corpus, fit BOTH the multi-domain Gibbs oracle (Task 4) and the multi-domain SVI model (Task 3), fold in held-out docs UNGATED to `node_affinity`, and assert the SVI and oracle placement readouts agree (rank correlation / MRR within a documented tolerance). This is the multi-domain analogue of the branch's oracle-validates-SVI gate.
- [ ] **Step 2: Run.** Tune iters/corpus size for a stable read; if SVI and Gibbs genuinely disagree beyond tolerance, that is a real finding — report it, do not loosen silently.
- [ ] **Step 3: Commit** (`test(gated-lda): multi-domain SVI-vs-Gibbs placement equivalence`, trailer.)

---

### Task 7: ω_m modality weight (θ-only) + v2 seam

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py` (and `lda.py`'s `_cavi_doc_inference` if the γ-weight is threaded there); Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- `GatedOnlineLDA.__init__(..., *, omega=None)` — `omega` is None (all 1.0, faithful MixEHR) or a length-(n_domains) sequence of nonnegative per-domain weights. Stored `self.omega`.
- The gated E-step weights the γ (doc-topic) accumulation per domain: γ = α + Σ_m ω_m Σ_{tokens∈m} count·φ. The λ sufficient-stats, `phi_norm`, and `doc_loglik` use TRUE counts (ω weights θ inference only). Implement by passing a per-token weight vector (ω of each token's domain, looked up from `domain_bounds`) into the γ recurrence — this per-token domain lookup IS the **v2 seam** (keep it live at φ-formation).
- `omega=None`/all-ones is byte-identical to Task 3.

- [ ] **Step 1: Write the failing tests**
  - `test_omega_ones_is_identity`: `omega=None` and `omega=[1,1]` reproduce the Task-3 fit exactly (fixed seed).
  - `test_omega_downweights_domain_shifts_theta`: on a corpus where domain 1 is high-volume, a small ω_1 (down-weight domain 1) shifts a held-out doc's θ toward the domain-0-explained topics vs ω=1 — a directional assertion (θ mass on the domain-0-driven block increases when domain 1 is down-weighted), not an exact number.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — add `omega` to `__init__` (validate length/nonneg). Build a per-token domain-weight vector `w_tok = omega[searchsorted(domain_bounds, indices, "right")-1]`. Thread an optional `gamma_count_weight` into `_cavi_doc_inference` (default = counts → identity) used ONLY in the γ recurrence's `eb_d @ (weight * counts / phi_norm)` term; `phi_norm`, `sstats_row`, and `doc_loglik` keep true `counts`. Document ω as a pseudo-likelihood/tempering weight on θ (arc design), default 1.0 = faithful MixEHR (Li 2020 uses raw volume). Keep the per-token domain map live at φ-formation (v2 seam).

- [ ] **Step 4: Run to verify pass + regressions.**

- [ ] **Step 5: Commit** (`feat(gated-lda): per-domain modality weight omega (theta-only, default 1)`, trailer.)

---

### Task 8: Per-modality θ-contribution instrumentation + v2-seam invariant test

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py`; Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- The E-step emits a per-domain θ-contribution stat: for each domain m, the total (ω-weighted) evidence mass domain m contributed to γ across the batch — so a caller can SEE whether one domain dominates θ (the volume-imbalance instrument the arc design requires). Returned in the `local_update` stats dict (e.g. `theta_contribution_by_domain`, length n_domains) and surfaced in `iteration_summary`/`iteration_diagnostics`.
- A v2-seam invariant test asserting the E-step exposes each token's domain at the point γ/φ are formed (e.g. the per-token domain-weight vector is computed from `domain_bounds` and has one entry per token), so v2 (generative π) remains a one-factor add.

- [ ] **Step 1: Write the failing tests**
  - `test_theta_contribution_by_domain_reported`: fit a step; assert the stat is length n_domains, nonnegative, and larger for the higher-volume domain under ω=1.
  - `test_v2_seam_per_token_domain_live`: assert a helper (e.g. `GatedOnlineLDA._token_domains(indices)`) returns the correct per-token domain index for a mixed-domain doc — the invariant that keeps v2 cheap.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — accumulate the per-domain θ-contribution during the E-step (sum of the ω-weighted per-token evidence, grouped by the per-token domain map already built in Task 7); return it in the stats dict and combine it in `combine_stats` (sum across partitions). Add the small `_token_domains(indices)` helper (the `searchsorted` lookup) and use it in the E-step so the seam is a named, tested unit. Surface the per-domain contribution in `iteration_summary`.

- [ ] **Step 4: Run to verify pass + regressions.**

- [ ] **Step 5: Commit** (`feat(gated-lda): per-domain theta-contribution instrument + v2 seam`, trailer.)

---

## Post-plan wrap-up (controller, after Task 8)

- [ ] Whole-branch review (superpowers:requesting-code-review, most capable model) over the SP2 commit range.
- [ ] Add an insights entry recording the multi-domain SVI≈Gibbs result + any ω_m θ-domination read (positive or null), per the insights-log convention.
- [ ] Do NOT merge or push; report SP2 status to the user. Next: SP3 (mllib shim) plan, just-in-time.
