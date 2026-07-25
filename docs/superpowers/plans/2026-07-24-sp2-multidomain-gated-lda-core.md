# SP2 — Multi-domain gated LDA model core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the gated DAG LDA model multi-domain (MixEHR-style): each topic gets a per-domain topic-word distribution stored as a literal per-domain dict, all sharing one DAG-gated document-topic mixture θ, with a tuned per-domain modality weight ω_m to de-bias token-volume imbalance. Validated by a multi-domain Gibbs oracle + planted per-domain recovery + SVI≈Gibbs.

**Architecture (LITERAL PER-DOMAIN DICT STORAGE, contained to GatedOnlineLDA):**
- **Stored parameters:** `global_params["lambda"]` becomes a literal dict `{m: (K, V_m)}` — one Dirichlet matrix per domain, each normalized on its own (type-safe: mis-normalization across domains is unrepresentable). This is what SP1's `split_domains` seeds, what readouts consume, and what SP3 will export.
- **Shared inference:** the E-step assembles a concatenated `expElogbeta` (K, ΣV_m) by normalizing each `lam_m` the standard full-row way and concatenating the blocks, then feeds the EXISTING gated CAVI (`_cavi_doc_inference`) over the concatenated `doc.indices`. One inference path, no second copy of the math.
- **Transient sufficient-stats stay CONCATENATED:** `lambda_stats` accumulates over the concatenated vocab as a single (K, ΣV_m) array, so the VIRunner's mini-batch scaling (`v * stats_scale`), `treeReduce(combine_stats)`, and Spark broadcast are ALL untouched — no base VIModel / runner / combine_stats change. `update_global` reads the dict λ, computes the concatenated natural-gradient target, and splits it back into a per-domain dict.
- **Backward compatible:** multi-domain is keyed on `domains` (per-domain vocab sizes). `domains=None` keeps the current single (K, V) array λ and every existing code path byte-for-byte. The base `OnlineLDA`, vanilla LDA, and HDP are NOT modified. All multi-domain logic lives in `GatedOnlineLDA` (which already overrides `initialize_global`/`local_update`/`update_global`).
- **Gate ⟂ domain:** the gate acts on θ's support (allowed topics); the per-domain matrices act on β's normalizer — orthogonal, unchanged gate. ω_m weights the θ accumulation only; β sstats use true counts. Seeded by SP1's `split_domains`.

**Tech Stack:** Python, numpy, scipy (`digamma`, `gammaln`), pytest. Validated locally via the collapsed-Gibbs oracle + in-memory batch VB (`fit_gated_svi_local`); no Spark in SP2 (shim/export/cloud is SP3/SP4).

## Global Constraints

- **Engine domain-agnostic + domain-neutral naming:** integer token ids / domain sizes ONLY. NO clinical/OMOP/EHR vocabulary anywhere in `spark_vi/**` or `spark-vi/tests/**` (code, comments, docstrings). Domains named `0`/`1`/`m` (or `a`/`b`), never clinical roles. Clinical semantics live only in the arc-design/plan motivation prose and L3/charmpheno.
- **Domain representation:** `domains` = a sequence of per-domain vocab sizes `[V_0, V_1, …]` (so `ΣV_m = V` and the cumulative offsets are `domain_bounds = [0, V_0, V_0+V_1, …]`, matching SP1). A concatenated token index `w` belongs to domain `searchsorted(domain_bounds, w, side="right") - 1`. `domains=None` == single-domain, and MUST be byte-identical to current behavior (the N=1 identity).
- **Literal dict storage:** `global_params["lambda"]` is `{m: (K, V_m)}` in multi-domain mode; a single (K, V) array when `domains=None`. Sufficient-stats (`lambda_stats`) stay a concatenated (K, V) array in BOTH modes (transient, not a stored parameter) so the runner/combine_stats are untouched.
- **ω weights θ, not β:** the per-domain modality weight scales the γ (doc-topic) accumulation only; the λ sstats, `phi_norm`, and data-loglik use TRUE counts. ω default = 1.0 every domain (faithful MixEHR — volume speaks). Document ω as a pseudo-likelihood/tempering weight (arc design), NOT the vanilla generative likelihood. Under ω≠1 the ELBO is the ω=1 bound (a convergence diagnostic), not an exact bound — state this.
- **Cite:** MixEHR (Li, Nair, Lu et al. 2020, Nat. Commun.), Hoffman-Blei-Bach 2010 (Online LDA/SVI), Griffiths & Steyvers 2004 (Gibbs oracle). No LaTeX; Unicode Greek only (α, β, θ, Σ, η, λ, ω, ψ, ρ).
- **v2 seam invariant:** the E-step keeps each token's domain available at the point γ/φ are formed (a per-token domain lookup from `domains`), so v2 (generative π) is later a one-factor add. Do NOT collapse the E-step into a domain-agnostic gather that discards which domain a token came from.
- **TDD.** Branch does NOT auto-push — push only when asked.
- Commit trailer EXACTLY (last line of every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- `spark-vi/spark_vi/models/topic/gated_lda.py` — MODIFY (the whole feature lives here). `domains` param; dict-λ `initialize_global` (random + SP1 spectral seed); `_assemble_expElogbeta(lam_dict)` + `_split_to_domains(concat, domains)` helpers; dict-aware `local_update`/`update_global`/`compute_elbo`/`infer_local`; ω_m γ-weighting; per-domain θ-instrument; `_token_domains` (v2 seam).
- `spark-vi/spark_vi/models/topic/lda.py` — MODIFY MINIMALLY, only if needed: thread an optional per-token `gamma_count_weight` into `_cavi_doc_inference` for ω (default → identity). The base `OnlineLDA` methods and single-domain behavior are otherwise untouched.
- `spark-vi/spark_vi/models/topic/dag_placement.py` — MODIFY. `fit_gated` (Gibbs oracle) gains a per-domain word-topic factor.
- `spark-vi/tests/test_gated_lda.py` (create if absent), `spark-vi/tests/test_dag_placement.py` — tests.

## Out of scope / follow-on

- Export/save of a dict-λ (VIResult per-domain sidecars) + the mllib shim (concatenated `featuresCol` + `domainBounds` Param + split-on-ingest) — **SP3**. (SP2 validates locally, no Spark save.)
- Real-cohort assembly + ω-swept FDR-delta — **SP4**.
- v2 generative π_{k,m} — research extension after SP2; kept cheap by the v2 seam (Task 8 guards it).

---

### Task 1: Per-domain dict-λ representation + assemble/split helpers + init

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py`; Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- `GatedOnlineLDA.__init__(lay, vocab_size, *, domains=None, **kw)` — store `self.domains` (None or a list of per-domain vocab sizes summing to `vocab_size`; validate the sum) and `self._domain_bounds` (cumulative offsets; None if single-domain).
- `_assemble_expElogbeta(lam_dict) -> (K, V)`: for each domain m, `exp(ψ(lam_m) − ψ(lam_m.sum(axis=1, keepdims=True)))`, concatenated in domain order. (Type-safe per-domain normalization.)
- `_split_to_domains(concat) -> {m: (K, V_m)}`: slice a concatenated (K, V) array at `_domain_bounds`.
- `initialize_global(data_summary)`: when `domains` is set, return `{"lambda": {m: (K, V_m)}, "alpha":…, "eta":…}` — random-Gamma per domain by default, or the SP1 spectral seed (`find_anchors(joint_Q, K, domain_bounds=…)` → `recover_beta` → `split_domains` → per-domain matrices scaled into λ_m). When `domains is None`, the current single-array path is returned UNCHANGED (byte-identical).

- [ ] **Step 1: Write the failing tests**

```python
def test_domains_none_is_single_array_unchanged():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=12, random_seed=7)          # domains=None
    gp = m.initialize_global(None)
    assert isinstance(gp["lambda"], np.ndarray) and gp["lambda"].shape == (lay.K, 12)


def test_multidomain_init_is_per_domain_dict():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=56, domains=[40, 16], random_seed=7)
    gp = m.initialize_global(None)
    lam = gp["lambda"]
    assert set(lam) == {0, 1}
    assert lam[0].shape == (lay.K, 40) and lam[1].shape == (lay.K, 16)


def test_assemble_split_round_trip():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], random_seed=1)
    lam = {0: np.abs(np.random.default_rng(0).normal(size=(lay.K, 4))) + .1,
           1: np.abs(np.random.default_rng(1).normal(size=(lay.K, 3))) + .1}
    eb = m._assemble_expElogbeta(lam)
    assert eb.shape == (lay.K, 7)
    # each block equals its own full-row normalization
    from scipy.special import digamma
    np.testing.assert_allclose(eb[:, :4], np.exp(digamma(lam[0]) - digamma(lam[0].sum(1, keepdims=True))))
    # split of a concatenated array round-trips the block shapes
    back = m._split_to_domains(eb)
    assert back[0].shape == (lay.K, 4) and back[1].shape == (lay.K, 3)


def test_domains_must_sum_to_vocab():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    with pytest.raises(ValueError):
        GatedOnlineLDA(lay, vocab_size=56, domains=[40, 10])   # 50 != 56
```

- [ ] **Step 2: Run to verify failure**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -v -k "domains or assemble_split"`
Expected: FAIL (`domains` unexpected kwarg / helpers undefined).

- [ ] **Step 3: Implement** — add `domains` (validate sum == vocab_size; store `self.domains`, `self._domain_bounds`). Add `_assemble_expElogbeta`/`_split_to_domains`. In `initialize_global`, branch on `self.domains`: None → current path (unchanged); set → build the dict λ (random-Gamma per domain; or, when a spectral strategy is requested, run SP1's joint recipe and `split_domains` into per-domain matrices, scaled like the existing seed). Cite MixEHR + the SP1 seed handoff.

- [ ] **Step 4: Run to verify pass + regressions**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py tests/test_lda.py -q`
Expected: PASS (single-domain gated + vanilla LDA unchanged).

- [ ] **Step 5: Commit** (`feat(gated-lda): per-domain dict-lambda storage + assemble/split + init`, trailer.)

---

### Task 2: Dict-aware E/M-step + per-domain η + multi-domain ELBO

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py`; Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- `local_update`: when `domains` is set, assemble `expElogbeta` via `_assemble_expElogbeta(lam_dict)`, run the SAME gated CAVI over `expElogbeta[allowed]`, accumulate a CONCATENATED `lambda_stats` (K, V) exactly as today. Single-domain path unchanged.
- `update_global`: when `domains` is set, assemble `expElogbeta`, compute the concatenated natural-gradient target `target = eta_vec + expElogbeta * lambda_stats`, then write back per-domain `new_lam[m] = (1−ρ)·lam[m] + ρ·target[:, block_m]`. `eta_vec` is per-domain (η_m broadcast over each block). Single-domain path unchanged.
- `eta` accepts a scalar (all domains) or a per-domain sequence; a `_eta_vocab_vector()` builds the length-V prior.
- `compute_elbo`: global KL becomes Σ_k Σ_m KL(Dirichlet(lam_m[k]) || Dirichlet(η_m·1_{V_m})). Single-domain scalar-η path byte-identical.

- [ ] **Step 1: Write the failing tests**

```python
def test_multidomain_fit_recovers_valid_per_domain_betas():
    """A gated multi-domain fit yields per-domain beta blocks that are proper
    distributions (each lam_m row-normalizes to 1)."""
    import numpy as np
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
             counts=np.unique(d, return_counts=True)[1].astype(float), length=len(d),
             frontier=frozenset(f) if hasattr(f, "__iter__") else frozenset({f}))
             for d, f in zip(docs[:600], labels[:600])]
    m = GatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], random_seed=0)
    gp = m.initialize_global(None)
    for _ in range(20):
        gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=0.5)
    for md in (0, 1):
        lam_m = gp["lambda"][md]
        beta_m = lam_m / lam_m.sum(1, keepdims=True)
        np.testing.assert_allclose(beta_m.sum(1), 1.0)
        assert np.isfinite(lam_m).all()


def test_single_domain_fit_byte_identical():
    """domains=None reproduces the current gated fit exactly (fixed seed)."""
    import numpy as np
    # ... build a small single-domain gated corpus, fit with domains=None twice,
    # and (if a pre-change reference is captured) assert byte-identical; at minimum
    # assert the domains=None path returns an ndarray lambda and runs unchanged.
```

(Include a per-domain-η ELBO test mirroring the arc design: manual Σ_k Σ_m KL matches `compute_elbo` for a per-domain η.)

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the `domains`-branch in `local_update`/`update_global`/`compute_elbo`/`infer_local` + per-domain η. Keep the single-domain branch literally the current code (byte-identical). ELBO note: token-loglik + shared θ-KL are unchanged (they operate over the concatenated vocab); only the global β-KL becomes per-(topic, domain).
- [ ] **Step 4: Run to verify pass + regressions** (`tests/test_gated_lda.py tests/test_lda.py`).
- [ ] **Step 5: Commit** (`feat(gated-lda): dict-aware multi-domain E/M-step + per-domain eta/ELBO`, trailer.)

---

### Task 3: Multi-domain Gibbs oracle (the validator)

**Files:** Modify `spark-vi/spark_vi/models/topic/dag_placement.py` (`fit_gated`); Test `spark-vi/tests/test_dag_placement.py`.

**Interfaces:**
- `fit_gated(..., *, domain_bounds=None, beta_prior=…)` — the per-token collapsed conditional uses a PER-DOMAIN word-topic factor: for token w in domain m, `(n_kw + η_m) / (n_{k,·∈m} + V_m·η_m)`. Accept a per-domain `beta_prior` sequence (default: the scalar for all domains). `domain_bounds=None` reproduces the current single-domain sampler byte-for-byte. (SP1 already threads `domain_bounds` into `fit_gated`'s spectral seed; this adds the per-domain sampler factor.)

- [ ] **Step 1: Write the failing test** — recovery: fit the multi-domain Gibbs oracle on the two-domain corpus with `domain_bounds`, assert both per-domain β blocks recover the planted signatures (via `slot_of_node`, mirroring SP1 Task 4's assertion through the Gibbs `beta_hat`). Plus a `domain_bounds=None` byte-identical check vs a fixed-seed current-oracle fit.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — precompute per-domain denominators `n_{k,·∈m}` and a per-token domain index (`searchsorted(domain_bounds, w, "right")−1`); replace the pooled factor with the per-domain one; `domain_bounds=None` → one domain → identical arithmetic. Cite Griffiths & Steyvers 2004 + the per-modality extension.
- [ ] **Step 4: Run to verify pass + regressions** (`tests/test_dag_placement.py`).
- [ ] **Step 5: Commit** (`feat(dag-placement): multi-domain per-domain Gibbs oracle`, trailer.)

---

### Task 4: Planted per-domain recovery (SVI) acceptance

**Files:** Test only — `spark-vi/tests/test_gated_lda.py`.

- [ ] **Step 1: Write the acceptance test** — fit the multi-domain `GatedOnlineLDA` (SVI, via `fit_gated_svi_local` or an inline batch-VB loop) on the two-domain corpus with `b_only_node`; take per-domain β from `gp["lambda"][m]` normalized; assert BOTH β^0 and β^1 recover the planted per-domain signatures for every node, INCLUDING the domain-1-anchored node (`slot_of_node`) — the SVI analogue of SP1 Task 4.
- [ ] **Step 2: Run.** If short of the gate, STRENGTHEN THE PLANT (SP1 generator's `b_only_signal_boost` / more docs / iters), NEVER loosen the assertion; a strong-plant failure is a genuine negative — STOP and report.
- [ ] **Step 3: Commit** (`test(gated-lda): multi-domain SVI planted per-domain recovery`, trailer.)

---

### Task 5: SVI ≈ Gibbs equivalence on multi-domain

**Files:** Test only — `spark-vi/tests/test_gated_lda.py`.

- [ ] **Step 1: Write the equivalence test** — following the branch's existing single-domain SVI≈Gibbs placement-equivalence pattern (`svi_node_profiles` in `_stm_synth.py` + the current equivalence test): on a small two-domain corpus fit BOTH the multi-domain Gibbs oracle (Task 3) and the multi-domain SVI model, fold in held-out docs UNGATED to `node_affinity`, assert the placement readouts agree (rank correlation / MRR within a documented tolerance).
- [ ] **Step 2: Run.** Tune iters/size for a stable read; a genuine SVI-vs-Gibbs disagreement beyond tolerance is a real finding — report it, do not loosen silently.
- [ ] **Step 3: Commit** (`test(gated-lda): multi-domain SVI-vs-Gibbs placement equivalence`, trailer.)

---

### Task 6: ω_m modality weight (θ-only) + v2 seam

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py` (+ `lda.py`'s `_cavi_doc_inference` for the γ-weight); Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- `GatedOnlineLDA.__init__(..., *, omega=None)` — None (all 1.0, faithful MixEHR) or a length-(n_domains) sequence of nonnegative per-domain weights.
- The gated E-step weights the γ accumulation per domain: γ = α + Σ_m ω_m Σ_{tokens∈m} count·φ. `phi_norm`, the λ sstats, and `doc_loglik` use TRUE counts (ω weights θ only). Implement via a per-token weight vector `w_tok = omega[_token_domains(indices)]` passed into the γ recurrence — this per-token domain lookup IS the v2 seam.
- `omega=None`/all-ones is byte-identical to Task 2.

- [ ] **Step 1: Write the failing tests**
  - `test_omega_ones_is_identity`: `omega=None` and `omega=[1,1]` reproduce the Task-2 fit exactly (fixed seed).
  - `test_omega_downweights_domain_shifts_theta`: on a corpus where domain 1 is high-volume, a small ω_1 shifts a held-out doc's θ toward the domain-0-explained block vs ω=1 — a directional assertion, not an exact number.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — add `omega` (validate). Thread an optional per-token `gamma_count_weight` into `_cavi_doc_inference` (default → identity) used ONLY in the γ recurrence `eb_d @ (w_tok * counts / phi_norm)`; `phi_norm`/`sstats`/`doc_loglik` keep true `counts`. Document ω as a pseudo-likelihood tempering weight on θ (arc design; Li 2020 uses raw volume), default 1.0. Note the ELBO is the ω=1 bound under ω≠1. Keep the per-token domain map live at φ-formation (v2 seam).
- [ ] **Step 4: Run to verify pass + regressions.**
- [ ] **Step 5: Commit** (`feat(gated-lda): per-domain modality weight omega (theta-only, default 1)`, trailer.)

---

### Task 7: Per-modality θ-contribution instrument + v2-seam invariant test

**Files:** Modify `spark-vi/spark_vi/models/topic/gated_lda.py`; Test `spark-vi/tests/test_gated_lda.py`.

**Interfaces:**
- `local_update` emits a per-domain θ-contribution stat (length n_domains): the total (ω-weighted) evidence mass each domain contributed to γ across the batch — the volume-imbalance instrument the arc design requires. Returned in the stats dict (`theta_contribution_by_domain`), summed across partitions (default `combine_stats` handles it since it's a flat array), surfaced in `iteration_summary`.
- `_token_domains(indices) -> np.ndarray`: the per-token domain index (`searchsorted`), the named+tested v2-seam unit.

- [ ] **Step 1: Write the failing tests**
  - `test_theta_contribution_by_domain_reported`: fit a step; assert the stat is length n_domains, nonnegative, larger for the higher-volume domain under ω=1.
  - `test_v2_seam_per_token_domain_live`: `_token_domains` returns the correct per-token domain for a mixed-domain doc.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — accumulate the per-domain θ-contribution in the E-step (grouped by `_token_domains`); return + surface it. Add `_token_domains` and use it in the E-step so the seam is a named unit.
- [ ] **Step 4: Run to verify pass + regressions.**
- [ ] **Step 5: Commit** (`feat(gated-lda): per-domain theta-contribution instrument + v2 seam`, trailer.)

---

## Post-plan wrap-up (controller, after Task 7)

- [ ] Whole-branch review (superpowers:requesting-code-review, most capable model) over the SP2 commit range.
- [ ] Add an insights entry recording the multi-domain SVI≈Gibbs result + any ω_m θ-domination read (positive or null), per the insights-log convention.
- [ ] Do NOT merge or push; report SP2 status to the user. Next: SP3 (mllib shim + dict-λ export) plan, just-in-time.
