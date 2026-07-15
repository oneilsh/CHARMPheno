# Block-wise Unit-Diagonal Correlation Σ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-covariance-with-`pd_complete` Σ M-step in `OnlineSTM` with a block-wise unit-diagonal (correlation-matrix) estimator that fixes the variance runaway by construction and retires the completion machinery from the fit path.

**Architecture:** The M-step standardizes the observed per-pair scatter to correlations, lazy-keeps unsupported pairs, and pins the diagonal to 1 — no `pd_complete`. Under single-label gating each E-step only inverts a fully-observed marginal `Σ[bg ∪ one-group]`, so no completion is needed; the full Σ is left block-structured (cross-group at 0) and never inverted whole. Reporting is unchanged (`topic_correlation_identified` already NA's cross-group). `pd_complete`/`min_frobenius_psd_completion` remain as tested linalg utilities (they have direct test consumers), only the M-step call is removed.

**Tech Stack:** Python 3.12, numpy, scipy, pytest. `spark-vi` package. No new dependencies.

## Global Constraints

- Domain-agnostic in `spark-vi`: integer topic/token indices only; no OMOP/EHR/cohort names in the library. (Cohort names live in `analysis/` + docs.)
- No LaTeX in docstrings/docs; Unicode Greek (Σ η θ μ Γ ρ) + plain text.
- Cite literature for any method/default; label heuristics as heuristics.
- Markdown-linkable code refs in docs: `[name](path#Lstart-Lend)`.
- Unit-diagonal is exact: `Σ_ii = 1` for all i after every M-step.
- `min_pair_support` keeps its meaning: `N_ij >= min_pair_support` ⇒ observed (estimated); else lazy-kept.
- Standardization formula (verbatim): `R_ij = (S_ij/N_ij) / sqrt((S_ii/N_ii)·(S_jj/N_jj))`; empirical variances used only to standardize, never stored as the prior.
- `pd_complete` / `min_frobenius_psd_completion` stay in `_linalg.py` (utilities with direct test consumers); only the M-step call + its logging are removed.
- **`sigma_init` decision (resolved here): KEEP it, unchanged.** The caller audit found downstream readers (`mllib/topic/stm.py:132,177,279` + the drivers), so removal is invasive churn. The default `sigma_init=1.0` already makes `initialize_global`'s `np.eye(K)*sigma_init` an identity, and the M-step pins the diagonal to 1 from the first step onward — so `sigma_init` is vestigial but harmless and needs no code change. No task modifies `initialize_global`.
- Docs (ADR + insight/ADR amendments) are sequenced LAST, after all code tests are green.

---

### Task 1: Block-wise unit-diagonal M-step (engine core)

Replaces the covariance-assembly + `pd_complete` block of `update_global` with the block-wise standardized unit-diagonal estimator, removes the now-dead completion plumbing, adds the block-wise unit tests, and migrates the `test_stm_full_sigma` tests that assert the removed completion behavior. Leaves the full suite green.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — M-step Σ block (lines ~692-747), imports (lines 34-35, 44, 47)
- Test (create): `spark-vi/tests/test_stm_blockwise.py`
- Test (migrate): `spark-vi/tests/test_stm_full_sigma.py` (4 completion tests)

**Interfaces:**
- Consumes: `OnlineSTM.update_global(global_params, target_stats, learning_rate) -> dict`; `target_stats["residual_outer_stat"]` (K×K scatter S), `target_stats["n_pairs_stat"]` (K×K support N); `global_params["Sigma"]` (K×K, unit diagonal after first M-step); `self.min_pair_support: int`.
- Produces: `update_global` returns `global_params` with `"Sigma"` a K×K unit-diagonal correlation matrix (supported off-diagonals = standardized correlations, unsupported lazy-kept, diagonal exactly 1.0). No `"Sigma"` completion; no `_pd_info`/pd_complete log line.

- [ ] **Step 1: Write the failing block-wise M-step unit tests**

Create `spark-vi/tests/test_stm_blockwise.py`:

```python
import numpy as np
import pytest
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.partition import TopicBlockPartition


def _drive_mstep(m, gp, S, N, lr=1.0):
    """Call update_global with a planted scatter S and support N (the only Sigma
    inputs), returning the new Sigma. Other stats are zero-shaped placeholders."""
    K, V, P = m.K, m.V, m.P
    stats = {
        "residual_outer_stat": S,
        "n_pairs_stat": N,
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.zeros((P, P)),
        "XtX_groups": [np.zeros((P, P)) for _ in m._effective_partition().groups],
        "XtMu": np.zeros((P, K)),
        "n_docs_per_topic": np.ones(K),
    }
    return m.update_global(gp, stats, learning_rate=lr)["Sigma"]


def test_mstep_output_is_unit_diagonal():
    m = OnlineSTM(K=4, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.eye(4) * 3.0 + 0.5          # arbitrary PSD-ish scatter
    N = np.full((4, 4), 100.0)
    Sig = _drive_mstep(m, gp, S, N)
    np.testing.assert_allclose(np.diag(Sig), 1.0, atol=1e-12)


def test_mstep_supported_offdiag_is_standardized_correlation():
    m = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    N = np.full((3, 3), 50.0)
    Sig = _drive_mstep(m, gp, S, N)          # lr=1 -> full move to target
    # r_01 = (S01/N)/sqrt((S00/N)(S11/N)) = 1/sqrt(4*9) = 1/6
    assert abs(Sig[0, 1] - (1.0 / 6.0)) < 1e-9
    assert abs(Sig[1, 0] - (1.0 / 6.0)) < 1e-9


def test_mstep_unsupported_pair_is_lazy_kept():
    m = OnlineSTM(K=3, vocab_size=8, P=1, min_pair_support=10, random_seed=0)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    S = np.eye(3) * 2.0
    N = np.full((3, 3), 50.0)
    N[0, 2] = N[2, 0] = 0.0                   # pair (0,2) unsupported
    Sig = _drive_mstep(m, gp, S, N)
    assert abs(Sig[0, 2] - 0.3) < 1e-9        # kept at previous value, not zeroed


def test_mstep_pins_runaway_variance_to_one():
    # A scatter whose free-variance MLE would inflate topic 1's diagonal to 500;
    # block-wise pins it to 1 regardless.
    m = OnlineSTM(K=2, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.array([[1.0, 0.0], [0.0, 500.0]])
    N = np.full((2, 2), 20.0)
    Sig = _drive_mstep(m, gp, S, N)
    assert abs(Sig[1, 1] - 1.0) < 1e-12       # pinned, not 500/20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise.py -q`
Expected: FAIL — current M-step returns a completed covariance (non-unit diagonal), so `test_mstep_output_is_unit_diagonal` and the standardized-correlation test fail.

- [ ] **Step 3: Replace the M-step Σ block with the block-wise estimator**

In `spark-vi/spark_vi/models/topic/stm.py`, replace the entire Σ block (from the `# Σ: full-covariance M-step via maximum-determinant PD completion.` comment through the `_log.info(...)` call — currently lines ~692-747) with:

```python
        # Σ: block-wise unit-diagonal correlation M-step (no completion).
        #
        # Standardize the observed per-pair scatter to correlations and pin the
        # diagonal to 1. Under single-label gating every E-step inverts only a
        # fully-observed marginal Σ[bg ∪ one-group] (safe_inverse), so nothing needs
        # the cross-group block completed; it stays at its 0 init and is never
        # inverted whole. Pinning Σ_ii = 1 removes the variance degree of freedom that
        # drives the softmax-saturation runaway (insight 0033). min_pair_support is the
        # observed/lazy threshold; unsupported pairs (incl. cross-group, N=0) are
        # lazy-kept (ADR 0027), so an absent group's entries do not decay.
        S = target_stats["residual_outer_stat"]
        N = target_stats["n_pairs_stat"]
        supported = N >= self.min_pair_support
        with np.errstate(invalid="ignore", divide="ignore"):
            mle = np.where(supported, S / np.where(N > 0, N, 1.0), 0.0)
        # empirical std used ONLY to standardize (never stored as the prior variance)
        diag_var = np.diag(mle).copy()
        std = np.sqrt(np.where(diag_var > 0.0, diag_var, 1.0))
        R = mle / np.outer(std, std)
        R_target = np.where(supported, R, Sigma)          # lazy-keep unsupported
        new_Sigma = (1.0 - learning_rate) * Sigma + learning_rate * R_target
        np.fill_diagonal(new_Sigma, 1.0)                  # exact unit diagonal
```

The existing `return {...}` block immediately below (currently lines ~749-755) is
**unchanged** — it already references `new_Sigma` (`"Sigma": new_Sigma`) and
`target_stats["n_pairs_stat"]` (`"n_pairs": ...`), both still produced above. Do not
touch it. `Sigma = global_params["Sigma"]` earlier in the method (line ~656) supplies
the previous Σ used for the lazy-keep and ρ-blend.

Then update the imports:
- Line 44: change `from spark_vi.models.topic._linalg import pd_complete, safe_inverse` to `from spark_vi.models.topic._linalg import safe_inverse`.
- Remove line 34 `import logging`, line 35 `import time`, and line 47 `_log = logging.getLogger(__name__)` (all three were used only by the removed M-step completion logging — verified: `grep -n "_log\.\|time\.\|logging\." stm.py` returns only the removed block).

- [ ] **Step 4: Run the block-wise tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Migrate the `test_stm_full_sigma.py` completion tests**

Run `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -q` first to see which fail. Then edit `spark-vi/tests/test_stm_full_sigma.py`:

- `test_free_offdiagonal_gets_zero_precision_completion` and `test_thin_cross_group_completed_with_zero_precision`: these assert `pd_complete` fills a free/thin cross pair with the zero-precision value in the M-step. Under block-wise the free/thin pair is **lazy-kept at its current Σ value** (0 for a never-supported cross pair). Rewrite each to assert the free/thin off-diagonal **equals its prior Σ value (unchanged)**, e.g. replace the zero-precision-completion assertion with:
  ```python
  # free/thin cross pair is lazy-kept (block-wise: no completion), not filled
  assert Sig[a, b] == gp["Sigma"][a, b]
  ```
- `test_assembled_sigma_is_spd_when_cross_block_inconsistent`: the premise (M-step returns an SPD full Σ) no longer holds — the full Σ is intentionally block-structured and need not be SPD. Rewrite to assert (a) the diagonal is unit (`np.allclose(np.diag(Sig), 1.0)`) and (b) each **marginal** `Sig[np.ix_(allowed, allowed)]` for `allowed = bg ∪ group` is SPD (`np.linalg.eigvalsh(...).min() > 0`), replacing the full-matrix SPD assertion. Rename to `test_marginals_are_spd_block_structured_full_sigma`.
- `test_cross_group_covariance_from_comorbid_docs`: with supported comorbid scatter injected, the cross-group pair IS observed (N ≥ floor) so it is estimated as a standardized correlation. Keep the "nonzero cross-group entry" assertion but change the expected value from the raw covariance to the standardized correlation `S_ab/sqrt(S_aa S_bb)` (compute from the injected stats); assert `abs(Sig[a,b]) <= 1.0`.
- `test_nongated_recovers_planted_correlated_sigma`, `test_nongated_e2e_offdiagonal_sign`, `test_initialize_global_sigma_is_matrix`, `test_topic_correlation_matrix_from_sigma`, and the E-step tests (`test_full_precision_*`, `test_gated_prior_uses_marginal_subblock_not_conditional`): re-run; these should still pass (unit-diagonal preserves off-diagonal sign/structure and E-step is unchanged). If a numeric golden shifted because variances are now 1, update the golden to the recomputed value and add a one-line comment noting the unit-diagonal parameterization.

- [ ] **Step 6: Run the full spark-vi STM suite to verify green**

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise.py tests/test_stm_full_sigma.py tests/test_stm_contract.py -q`
Expected: PASS. (`test_stm_contract`'s `maxvar` test reads `Σ_var` which is now all-1 for the max-variance topic — confirm it still passes; the max-variance topic is now arbitrary among ties, so if that test asserted a specific topic index under a planted non-unit Σ it stays valid because it sets `gp["Sigma"]` directly.)

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_blockwise.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(spark-vi): block-wise unit-diagonal correlation M-step (retire pd_complete from the fit path)"
```

---

### Task 2: Port the gated logistic-normal generator + integration test

Adds the ground-truth gated generator to the synthetic helper and an end-to-end test proving no runaway + sub-phenotype recovery + within-group correlation recovery under the block-wise M-step. Ports the validated scratchpad prototype.

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py` (add `gated_ln_corpus`)
- Test (create): `spark-vi/tests/test_stm_blockwise_integration.py`

**Interfaces:**
- Consumes: `OnlineSTM`; `planted_recovery(beta_hat, planted_beta) -> int`, `foreground_recovers_group(beta_hat, partition, group, planted_beta) -> bool` (existing in `_stm_synth.py`); `TopicBlockPartition`; `spark_vi.models.topic._linalg.pd_complete` (used only to build a PD ground-truth Σ_true in the generator, not in the fit).
- Produces: `gated_ln_corpus(*, group_weights, fg_per_group, bg_k, V, D, doc_len, seed=0) -> (docs, partition, Sigma_true, beta)` — a single-label gated corpus sampled from a known unit-diagonal Σ_true.

- [ ] **Step 1: Write the failing integration test**

Create `spark-vi/tests/test_stm_blockwise_integration.py`:

```python
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from _stm_synth import (gated_ln_corpus, planted_recovery,
                        foreground_recovers_group)


def _fit(cls, docs, part, V, beta, n_iter=150, batch=120, seed=42):
    m = cls(K=part.K, vocab_size=V, P=1, random_seed=seed,
            topic_blocks=part, min_pair_support=5, reference_topic=True)
    gp = m.initialize_global({"spectral_beta": beta})     # aligned init
    D = len(docs); rng = np.random.default_rng(seed); scale = D / batch
    max_var = 0.0
    for t in range(n_iter):
        idx = rng.choice(D, size=batch, replace=False)
        stats = m.local_update([docs[i] for i in idx], gp)
        scaled = {k: (v * scale if isinstance(v, np.ndarray) and k.endswith("stat")
                      else v) for k, v in stats.items()}
        gp = m.update_global(gp, scaled, learning_rate=(t + 64) ** -0.7)
        max_var = max(max_var, float(np.diag(gp["Sigma"]).max()))
    return gp, max_var


def test_blockwise_no_runaway_and_recovers_subphenotypes():
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.92, "B": 0.08}, fg_per_group=4, bg_k=6,
        V=200, D=2000, doc_len=80, seed=0)
    gp, max_var = _fit(OnlineSTM, docs, part, 200, beta)
    # (1) no variance runaway: diagonal pinned at 1 throughout
    assert max_var <= 1.0 + 1e-9, f"variance ran away to {max_var}"
    # (2) sub-phenotype recovery holds, including the thin minority arm
    bhat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
    assert planted_recovery(bhat, beta) >= part.K - 1
    assert foreground_recovers_group(bhat, part, "B", beta)
    # (3) within-group correlations are recovered (unit-diagonal => Σ IS R)
    Sig = gp["Sigma"]
    a, b = part.block_indices("A")[0], part.block_indices("A")[1]
    assert Sig[a, b] > 0.05, "within-group A correlation not recovered"


def test_blockwise_marginals_are_pd():
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.92, "B": 0.08}, fg_per_group=4, bg_k=6,
        V=200, D=2000, doc_len=80, seed=1)
    gp, _ = _fit(OnlineSTM, docs, part, 200, beta)
    Sig = gp["Sigma"]; bg = list(part.background_indices())
    for g in part.groups:
        allowed = sorted(set(bg) | set(part.block_indices(g)))
        assert np.linalg.eigvalsh(Sig[np.ix_(allowed, allowed)]).min() > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise_integration.py -q`
Expected: FAIL with `ImportError: cannot import name 'gated_ln_corpus'`.

- [ ] **Step 3: Add `gated_ln_corpus` to `_stm_synth.py`**

Append to `spark-vi/tests/_stm_synth.py` (import `pd_complete` at the top of the file: `from spark_vi.models.topic._linalg import pd_complete`):

```python
def gated_ln_corpus(*, group_weights, fg_per_group, bg_k, V, D, doc_len, seed=0):
    """Single-label gated logistic-normal corpus with a KNOWN unit-diagonal Sigma_true.

    Each doc belongs to ONE group (sampled by group_weights, so a minority arm is
    thin); its allowed set is background ∪ that group's foreground block. eta over the
    allowed set ~ N(0, Sigma_true[A,A]); theta = softmax(eta); words ~ theta·beta.
    Planted correlations (bg-bg 0.10, bg-fg 0.25, within-fg 0.30) are made PD via the
    max-det completion of the cross-foreground free block (pd_complete used ONLY to
    build ground truth, not in the fit). Domain-agnostic: integer token ids only."""
    rng = np.random.default_rng(seed)
    groups = tuple(group_weights)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                              foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    sig = max(1, (V // 2) // K)
    C = max(1, min(sig, V - K * sig))
    beta = np.full((K, V), 1e-3)
    for k in range(K):
        beta[k, 0:C] += rng.random(C) + 0.1
        lo = C + k * sig
        beta[k, lo:lo + sig] += 5.0
    beta /= beta.sum(axis=1, keepdims=True)

    bg = part.background_indices()
    Sigma_true = np.eye(K); obs = np.eye(K, dtype=bool)
    for a in bg:
        for b in bg:
            if a != b:
                Sigma_true[a, b] = 0.10; obs[a, b] = True
    for a in bg:
        for g in groups:
            for c in part.block_indices(g):
                Sigma_true[a, c] = Sigma_true[c, a] = 0.25
                obs[a, c] = obs[c, a] = True
    for g in groups:
        blk = part.block_indices(g)
        for i in blk:
            for j in blk:
                if i != j:
                    Sigma_true[i, j] = 0.30; obs[i, j] = True
    Sigma_true = pd_complete(Sigma_true, obs)

    gl = list(groups)
    wts = np.array([group_weights[g] for g in gl], float); wts /= wts.sum()
    docs = []
    for _ in range(D):
        g = gl[int(rng.choice(len(gl), p=wts))]
        allowed = sorted(part.allowed_indices(frozenset({g})))
        eta = rng.multivariate_normal(np.zeros(len(allowed)),
                                      Sigma_true[np.ix_(allowed, allowed)])
        theta = np.zeros(K)
        theta[allowed] = np.exp(eta - eta.max()); theta /= theta.sum()
        toks = rng.choice(V, size=doc_len, p=theta @ beta)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, part, Sigma_true, beta
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise_integration.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_stm_blockwise_integration.py
git commit -m "test(spark-vi): gated logistic-normal generator + block-wise no-runaway/recovery integration test"
```

---

### Task 3: Non-gated characterization + migrate the recovery-invariant test

Confirms the non-gated path reduces to a directly-standardized full correlation matrix, and updates the one M-step-driven test in `test_stm_pd_completion_conditioning.py` (the pd_complete-isolation tests there stay — they test the retained utility).

**Files:**
- Test (add to): `spark-vi/tests/test_stm_blockwise_integration.py`
- Test (migrate): `spark-vi/tests/test_stm_pd_completion_conditioning.py`

**Interfaces:**
- Consumes: `gated_ln_corpus` (Task 2); `synthetic_gated_corpus_overlap`, `fit_stm`, `planted_recovery` (existing `_stm_synth.py`).
- Produces: no new public symbols.

- [ ] **Step 1: Write the failing non-gated characterization test**

Add to `spark-vi/tests/test_stm_blockwise_integration.py`:

```python
def test_nongated_sigma_is_unit_diagonal_correlation():
    """Non-gated (no groups): every pair is observed, so Sigma is the directly
    standardized full correlation matrix — unit diagonal, all |offdiag|<=1, PD."""
    from _stm_synth import synthetic_ehr_corpus
    docs, planted = synthetic_ehr_corpus(K_rare=4, V=80, D=400, doc_len=60,
                                         bg_frac=0.4, seed=0)
    m = OnlineSTM(K=4, vocab_size=80, P=1, random_seed=42, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(30):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    Sig = gp["Sigma"]
    np.testing.assert_allclose(np.diag(Sig), 1.0, atol=1e-12)
    assert np.all(np.abs(Sig) <= 1.0 + 1e-9)
    assert np.linalg.eigvalsh(Sig).min() > -1e-9    # PD (no free pairs to break it)
```

- [ ] **Step 2: Run to verify it passes** (behavior already implemented in Task 1)

Run: `cd spark-vi && python -m pytest tests/test_stm_blockwise_integration.py::test_nongated_sigma_is_unit_diagonal_correlation -q`
Expected: PASS. (If it fails because `reference_topic=True` fixes topic 0's row to `[1,0,0,0]`, that is correct — the reference row is unit-diagonal with zero off-diagonals; adjust the `|offdiag|<=1` assertion to exclude the reference row only if it actually trips, noting why.)

- [ ] **Step 3: Migrate `test_recovery_invariant_to_full_sigma_condition_number`**

In `spark-vi/tests/test_stm_pd_completion_conditioning.py`, the test `test_recovery_invariant_to_full_sigma_condition_number` asserts recovery is invariant while the assembled full-Σ condition number spans orders of magnitude. Under the unit-diagonal M-step the full Σ is block-structured (not a completed covariance), so "full-Σ cond spanning orders of magnitude" is no longer the right framing. Replace the assertion block: keep the multi-seed fit + `planted_recovery`, drop the `cond` spread assertion, and assert instead that recovery is stable AND the diagonal is unit across seeds:
```python
    # Unit-diagonal M-step: Sigma is a correlation matrix each seed; recovery is
    # stable and the diagonal is pinned to 1 (no variance runaway to track).
    assert min(recs) >= max(recs) - 1, recs
    for gp in gps:                       # collect gp per seed in the loop
        np.testing.assert_allclose(np.diag(gp["Sigma"]), 1.0, atol=1e-12)
```
Leave the pd_complete-isolation tests (`test_blockarrow_naive_pin_illconditioned_pd_complete_fixes`, `test_gated_shape_naive_indefinite_pd_complete_well_conditioned`, `test_overlap_generator_actually_shares_terms`) unchanged — they test the retained `pd_complete` utility directly.

- [ ] **Step 4: Run the conditioning + hardening suites to verify green**

Run: `cd spark-vi && python -m pytest tests/test_stm_pd_completion_conditioning.py tests/test_stm_hardening.py -q`
Expected: PASS. (If `test_stm_hardening.py::test_baseline_random_init_is_unstable` now fails because unit-diagonal removed the Σ blowup it asserted, update it to assert random-init instability via **topic recovery** — `planted_recovery(random_init) < planted_recovery(spectral_init)` — instead of a Σ-scale blowup, and note the runaway is now structurally prevented.)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/test_stm_blockwise_integration.py spark-vi/tests/test_stm_pd_completion_conditioning.py spark-vi/tests/test_stm_hardening.py
git commit -m "test(spark-vi): non-gated unit-diagonal characterization + migrate recovery-invariant/hardening tests"
```

---

### Task 4: Full-suite + reporting regression sweep

Runs the whole `spark-vi` suite plus the charmpheno correlation export and dashboard tests to confirm the reporting layer is unchanged (cross-group NA, within-group identified) and nothing else regressed. No code change expected; fixes any straggler that assumed completion-filled Σ.

**Files:**
- Verify: `spark-vi/tests/` (full), `charmpheno/` correlation export tests, `dashboard/` correlation tests
- Modify (only if a test regresses): the specific failing test file

**Interfaces:**
- Consumes: `charmpheno` `build_correlation_json`, dashboard `Correlation` loader (unchanged).
- Produces: none.

- [ ] **Step 1: Run the full spark-vi suite**

Run: `cd spark-vi && python -m pytest tests/ -q`
Expected: PASS (all). Investigate any failure; the only legitimate changes are tests that asserted the removed completion behavior (fix per the Task 1/3 patterns — lazy-keep instead of completion, marginal-PD instead of full-PD).

- [ ] **Step 2: Run the charmpheno correlation-export tests**

Run: `cd charmpheno && python -m pytest tests/ -k correlation -q`
Expected: PASS. `build_correlation_json` already NA's cross-group (n_pairs=0 < floor) and keeps within-group — unchanged by the unit-diagonal Σ (which only makes `topic_correlation(Σ) ≈ Σ`). If a fixture assumed a completed cross-group value, replace it with `null` (NA) and note the block-wise semantics.

- [ ] **Step 3: Run the dashboard correlation tests**

Run: `cd dashboard && npm test -- correlation` (or the project's test command)
Expected: PASS — the NA = `r === null || !identified` logic is unchanged.

- [ ] **Step 4: Commit (only if a straggler was fixed)**

```bash
git add -A
git commit -m "test: reporting regression sweep for block-wise unit-diagonal Sigma"
```
If nothing changed, skip the commit and note "reporting green, no changes" in the progress ledger.

---

### Task 5: exp 0027 validation doc

Adds the cluster experiment that validates the block-wise model on the real cohort. Doc-only (a fit config); no code.

**Files:**
- Create: `docs/experiments/0027-stm-comorbid-blockwise-unit-diagonal.md`

**Interfaces:**
- Consumes: the experiment frontmatter schema (see `docs/experiments/0026-stm-comorbid-fullcov-runaway-diagnosis.md` for the exact keys); `scripts/run_experiment.py` resolves it via `find_by_id`.
- Produces: a `make exp ID=27`-runnable experiment.

- [ ] **Step 1: Create the experiment doc**

Copy the frontmatter of `docs/experiments/0026-stm-comorbid-fullcov-runaway-diagnosis.md` verbatim, changing only `id: 27`, `slug: stm-comorbid-blockwise-unit-diagonal`, `status: pending`, and `max_iter: 100` (a full validation run, not the diagnostic's 40). Body sections:
- **Title:** `# STM comorbid (cancer + dementia): block-wise unit-diagonal correlation Σ`
- **Why:** validates the design of `docs/superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md` on the real cohort — the runaway (insight 0033) fixed by construction, the completion machinery retired.
- **Success criteria:** max `Σ_var` ≈ 1 throughout (no runaway — watch the `maxvar[...]` line); Alzheimer's/amnestic + vascular dementia sub-phenotypes preserved (insight 0032 Finding 2); within-group + bg↔fg correlation matrix trustworthy with cross-group NA; ELBO recovered near −1.59e6; per-iter wall-clock free of the `M-step pd_complete` cost (that log line is gone).

- [ ] **Step 2: Verify the runner resolves it**

Run: `cd <repo root> && python -c "import sys; sys.path.insert(0,'scripts'); import run_experiment as R; from pathlib import Path; fm=R.read_frontmatter(R.find_by_id(Path('docs/experiments'),27)); R.validate_frontmatter(fm); print('exp 27 valid:', fm['slug'], fm['max_iter'])"`
Expected: `exp 27 valid: stm-comorbid-blockwise-unit-diagonal 100`

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/0027-stm-comorbid-blockwise-unit-diagonal.md
git commit -m "docs(exp): 0027 block-wise unit-diagonal Sigma cluster validation"
```

---

### Task 6 (LAST — gated on Tasks 1-5 green): ADR + insight/ADR amendments

Records the decision and amends the superseded docs. Sequenced last so the docs describe shipped, test-green behavior.

**Files:**
- Create: `docs/decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md`
- Modify: `docs/decisions/0033-stm-full-covariance-sigma.md` (amendment note), `docs/insights/0032-...md` (Resolution cross-link), `docs/insights/0033-...md` (Implication: the fix shipped)

**Interfaces:**
- Consumes: ADR numbering (next is 0034); the spec.
- Produces: ADR 0034.

- [ ] **Step 1: Write ADR 0034**

Create `docs/decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md` (follow the house ADR format — Status: Accepted + Implemented; Context / Decision / Alternatives / Consequences / References). Content: block-wise unit-diagonal correlation Σ replaces full-covariance-with-completion for gated single-label STM. Decisions: (1) unit-diagonal correlation parameterization (runaway fixed by construction — insight 0033; ν→∞/scale-1 variance-anchor limit, scale 1 load-bearing per insight 0030); (2) single-label gating (cross-group correlations foreclosed/NA); (3) block-wise M-step, no completion (`pd_complete`/Dykstra retired from the fit path — kept as utilities); (4) E-step marginal inversion unchanged, `safe_inverse` the per-doc guard. Alternatives considered + rejected: keep completion + post-hoc standardize (option 1 — extra step, keeps the fallback/cost); reparameterize-in-correlation-space (option 2 — |r|>1 hazard, verified); IW/diag-shrink variance levers (falsified, insight 0032 Findings 4-6). Cite Blei & Lafferty 2007, LKJ (Lewandowski et al. 2009), insight 0029/0030/0032/0033, ADR 0027/0033.

- [ ] **Step 2: Amend ADR 0033 and insights 0032/0033**

- `docs/decisions/0033-stm-full-covariance-sigma.md`: add a dated amendment note at the top of the relevant decision — the full-covariance-with-completion M-step is superseded by ADR 0034 (block-wise unit-diagonal) for the gated single-label case; link ADR 0034.
- `docs/insights/0032-...md`: in the Resolution, add a one-line cross-link that the runaway it left open (via insight 0033) is resolved by the block-wise unit-diagonal design (ADR 0034).
- `docs/insights/0033-...md`: append an Implication line that the fix shipped as ADR 0034 / exp 0027 (block-wise unit-diagonal).

- [ ] **Step 3: Verify links + LaTeX-clean**

Run: `grep -rn "E\[" docs/decisions/0034*.md docs/insights/0033*.md` (should be empty — use `E(β)` not `E[β]`); visually confirm all `[text](path#Ln)` links resolve.
Expected: no LaTeX-reference-label warnings; links valid.

- [ ] **Step 4: Commit**

```bash
git add docs/decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md docs/decisions/0033-stm-full-covariance-sigma.md docs/insights/0032-*.md docs/insights/0033-*.md
git commit -m "docs(adr): 0034 block-wise unit-diagonal correlation Sigma; amend ADR 0033 + insights 0032/0033"
```

---

## Notes for the executor

- **After all tasks:** the final whole-branch review (subagent-driven-development) should verify (a) no remaining M-step caller of `pd_complete`, (b) `pd_complete`/`min_frobenius_psd_completion` still have their isolation tests and pass, (c) the full `spark-vi` + `charmpheno` + dashboard suites are green, and (d) `make exp ID=27` resolves.
- **Do not** delete `pd_complete`/`min_frobenius_psd_completion` — they are retained utilities with direct test consumers (verified by the caller audit: `test_linalg_pd_complete.py`, `test_stm_pd_completion_conditioning.py` isolation tests).
- **Reference prototype:** `scratchpad/blockwise_prototype.py` and `scratchpad/stm_runaway_repro.py` (this session) are the validated source of the M-step code and the integration test.
