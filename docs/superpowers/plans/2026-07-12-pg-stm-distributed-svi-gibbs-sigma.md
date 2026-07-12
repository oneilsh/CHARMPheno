# Distributed PG-STM (SVI + Gibbs-Σ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the distributed PG-STM engine (Spark) that runs the decisive Σ-runaway-cure test at scale on a duplicate of exp-0027, plus a distributed exact-Gibbs Σ pass for the comorbidity correlation read-out — buttoned up for a cluster run without touching exp-0027.

**Architecture:** Two phases sharing the `mapPartitions(_local).treeReduce(_combine)` sufficient-statistic idiom already used across `spark_vi/mllib/topic/stm.py`. Phase 1 = distributed PG-SVI (the runaway test, `sigma_mode` mle|iw). Phase 2 = distributed exact-Gibbs Σ refinement over the converged β/Γ. The per-doc math is already pure functions in `spark_vi/models/topic/pg_stm.py`; this plan refactors the M-step accumulation into distributable moment form and wraps it in a streaming driver, so full-batch StreamingPGSTM == the validated `PGSTMVI.fit`.

**Tech Stack:** Python, NumPy, PySpark 3.5.8 (local session-scoped `spark` fixture in `tests/conftest.py`), `polyagamma==2.0.2`, SciPy.

## Global Constraints

- `polyagamma==2.0.2` is the only new dependency; it must be added to the Dataproc cluster image (ships wheels). Do NOT vendor a sampler.
- Per-doc math stays PURE in `spark_vi/models/topic/pg_stm.py` (single source of truth); the streaming driver in `spark_vi/mllib/topic/pg_stm.py` only orchestrates + reduces.
- `StreamingPGSTM(batch=all, rho=1).fit` MUST reproduce `PGSTMVI.fit`'s β/Γ/Σ to near-float (the equivalence gate). Any divergence is a bug.
- Reduce only SMALL, global-shaped arrays to the driver: word-topic stats (K,V), XᵀX (P,P), XᵀM (P,K−1), block-scatter S (K−1,K−1), group_counts, D. NEVER collect per-doc M (D,K−1) or the corpus.
- Σ M-step uses the F1 PD-completed `_assemble_sigma` (block-IW; `pd_complete` for the never-co-active cross-blocks).
- Hash document/person IDs (SHA-256, truncated) before any row-level `.show()`/print in drivers; aggregates + probabilities are fine raw.
- Cite any method/default/constant from the literature in docstrings; no LaTeX in prose (Unicode Greek OK).
- Do NOT modify exp-0027 (`docs/experiments/0027-*.md`); DUPLICATE its config into new experiments.

---

### Task 1: Refactor the PG M-step into pure, distributable sufficient-stat pieces

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py` (add stat/M-step helpers; refactor `PGSTMVI.fit` to use them)
- Test: `spark-vi/tests/test_pg_stm_sufficient_stats.py` (new)

**Interfaces:**
- Produces:
  - `pg_gamma_ridge_moments(XtX, XtM, *, ridge) -> Gamma (P, K-1)` = `solve(XtX + ridge·I, XtM)` (moment form of `gamma_ridge`, mathematically identical).
  - `class PGSuffStats` (plain dataclass or dict) with fields `wts (K,V)`, `XtX (P,P)`, `XtM (P,K-1)`, `S (K-1,K-1)`, `group_counts (dict[str,int])`, `D (int)`.
  - `pg_empty_stats(K, V, P, groups) -> PGSuffStats` (zeros).
  - `pg_accumulate_doc(stats, doc, estep_out, *, K) -> None` (in-place add one doc's contribution; `estep_out = (m, Vd, phi, active, allowed, mu_active)` from `_e_step_doc`, plus `doc.x`). Adds: `wts[:, doc.indices] += (phi*counts).T`; `XtX += outer(x,x)`; `XtM[:, active] += outer(x, m)`; `S[active,active] += outer(m-mu, m-mu)+Vd`; `group_counts[g]+=1`; `D+=1`.
  - `pg_combine_stats(a, b) -> PGSuffStats` (pure sum, treeReduce combiner).
  - `pg_mstep(stats, *, beta_eta, gamma_ridge, sigma_mode, Psi0_scale, nu0, partition, layout) -> (beta, Gamma, Sigma)` — calls `beta_dirichlet_mean`, `pg_gamma_ridge_moments`, and `_assemble_sigma` (moved to a module function `assemble_sigma(S, bg_sticks, group_counts, D, *, sigma_mode, Psi0_scale, nu0, partition, layout)` or kept as a staticmethod the driver can call).
- Consumes: existing `_e_step_doc` (unchanged), `beta_dirichlet_mean`, `sigma_iw_posterior_mean`, `pd_complete`.

- [ ] **Step 1: Write the failing test** — moment-form Γ equals the stacked-form Γ, and `PGSuffStats` round-trips one doc.

```python
import numpy as np
from spark_vi.models.topic.pg_stm import gamma_ridge, pg_gamma_ridge_moments

def test_gamma_ridge_moment_form_matches_stacked():
    rng = np.random.default_rng(0)
    D, P, Km1 = 50, 3, 5
    X = rng.normal(size=(D, P)); M = rng.normal(size=(D, Km1))
    ref = gamma_ridge(M, X, ridge=1e-6)
    got = pg_gamma_ridge_moments(X.T @ X, X.T @ M, ridge=1e-6)
    assert np.allclose(ref, got, atol=1e-10)
```

- [ ] **Step 2: Run it, watch it fail** (`pg_gamma_ridge_moments` undefined).
- [ ] **Step 3: Implement** `pg_gamma_ridge_moments`, `PGSuffStats`, `pg_empty_stats`, `pg_accumulate_doc`, `pg_combine_stats`, `pg_mstep`, and a module-level `assemble_sigma(...)` (extract the body of `PGSTMVI._assemble_sigma`; keep the method as a thin wrapper).
- [ ] **Step 4: Refactor `PGSTMVI.fit`** to build a `PGSuffStats` in its per-doc loop via `pg_accumulate_doc` and call `pg_mstep` — behavior-preserving. Keep `psi_mean`/`psi_var` outputs as they are (driver-only diagnostics; NOT part of the reduced stats).
- [ ] **Step 5: Run the FULL existing pg-stm suite** — `test_pg_stm_{link,conditionals,updates,assignment,nested,vi,gibbs,assembly,runaway,stick_native}.py` — all must stay green (the refactor is behavior-preserving). Plus the new `test_pg_stm_sufficient_stats.py`.
- [ ] **Step 6: Commit** `refactor(pg-stm): extract distributable PG sufficient-stats + moment-form M-step`.

---

### Task 2: `StreamingPGSTM` — distributed full-batch + minibatch PG-SVI

**Files:**
- Create: `spark-vi/spark_vi/mllib/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_streaming.py` (new; uses the `spark` fixture)

**Interfaces:**
- Consumes: Task-1 `pg_empty_stats`/`pg_accumulate_doc`/`pg_combine_stats`/`pg_mstep`, `_e_step_doc` (via a small module wrapper `pg_estep_doc(doc, layout, log_beta, Gamma, Sigma, partition)` if `_e_step_doc` is a method — expose a free function), `stick_layout`.
- Produces:
  - `class StreamingPGSTM(K, V, partition, *, P, beta_eta=0.1, gamma_ridge=1e-6, sigma_mode="iw", Psi0_scale=1.0, nu0=None, seed=0)`.
  - `.fit(doc_rdd, *, max_iter=100, batch="all", tau0=64.0, kappa=0.7, depth=2, on_iteration=None) -> {"beta","Gamma","Sigma","sigma_max_trace"}`. `doc_rdd` is an RDD of `STMDocument`. `batch="all"` → full-batch (ρ=1 every iter, one treeReduce over all docs). A float in (0,1] → minibatch fraction with Robbins-Monro `ρ_t=(t+tau0)^-kappa`, natural-param blend of the reduced stats (scale minibatch stats by D/|batch|), mirroring `StreamingSTM.fit`.

**Distributed iteration (both modes):** broadcast `(log_beta, Gamma, Sigma, layout, partition)`; `work_rdd.mapPartitions(_local).treeReduce(_combine, depth)` where `_local` folds `pg_accumulate_doc(pg_estep_doc(...))` over the partition's docs into one `PGSuffStats`; driver runs `pg_mstep` (full-batch) or blends into the running natural params with ρ (minibatch); append `max|Σ|` to `sigma_max_trace`.

- [ ] **Step 1: Write the failing equivalence test** — full-batch StreamingPGSTM reproduces `PGSTMVI.fit`.

```python
import numpy as np
from spark_vi.models.topic.pg_stm import StreamingPGSTM
from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus

def test_streaming_fullbatch_matches_single_machine(spark):
    docs, part, _St, _b = gated_ln_corpus(group_weights={"A":0.6,"B":0.4},
        fg_per_group=1, bg_k=3, V=40, D=120, doc_len=30, seed=0)
    P = docs[0].x.shape[0]
    ref = PGSTMVI(K=part.K, V=40, partition=part, P=P, n_iter=30, seed=0).fit(docs)
    rdd = spark.sparkContext.parallelize(docs, 4)
    got = StreamingPGSTM(K=part.K, V=40, partition=part, P=P, seed=0).fit(
        rdd, max_iter=30, batch="all")
    assert np.allclose(got["beta"], ref["beta"], atol=1e-8)
    assert np.allclose(got["Gamma"], ref["Gamma"], atol=1e-8)
    assert np.allclose(got["Sigma"], ref["Sigma"], atol=1e-8)
```

- [ ] **Step 2: Run it, watch it fail** (module/class absent). Note: the E-step draws no randomness (deterministic mean-field), so full-batch equivalence is exact up to float reduction order; use `atol=1e-8` and `depth=1` if reduction-order drift exceeds it — document the tolerance.
- [ ] **Step 3: Implement** `StreamingPGSTM` + `pg_estep_doc` free-function wrapper.
- [ ] **Step 4: Run the equivalence test — passes.**
- [ ] **Step 5: Add a minibatch convergence test** — `batch=0.25, max_iter=150` on a synthetic gated corpus recovers the planted β (`planted_recovery >= part.K-2`) and returns a bounded (`max|Σ|<1e2`) PD Σ.
- [ ] **Step 6: Commit** `feat(pg-stm): StreamingPGSTM distributed PG-SVI (full-batch == single-machine)`.

---

> **BUILD NOTE (Task 3, as-built):** a fully-distributed exact-Gibbs Σ sampler was
> prototyped but drifts to the wrong-sign correlation from any init (unpinned
> label-switching across the shared background block). Since Σ is a single small
> (K−1)×(K−1) global, it does NOT need full-corpus distribution — the as-shipped
> `pg_stm_sigma_readout` collects a driver-side **subsample** and runs the VALIDATED
> single-machine `pg_stm_gibbs` (recovers the planted correlation; the F4 positive
> control). The distributed exact sampler is deferred as a future optimization.

### Task 3: Distributed exact-Gibbs Σ pass (comorbidity read-out)

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/pg_stm.py` (add `pg_stm_gibbs_sigma_rdd`)
- Test: `spark-vi/tests/test_pg_stm_streaming.py` (add)

**Interfaces:**
- Produces: `pg_stm_gibbs_sigma_rdd(doc_rdd, *, beta, Gamma, Sigma0, partition, layout, n_sweeps=40, burn=20, Psi0_scale=1.0, nu0=None, seed=0, depth=2) -> {"Sigma", "Sigma_samples"}`. Holding β/Γ FIXED, each sweep: broadcast current Σ; each worker draws per doc `omega ~ PG` then `psi ~ N(m_d, V_d)` EXACTLY (full-covariance draw: `m,V = psi_posterior(...)`, `psi = m + chol(V) @ z`), accumulates the block scatter `S` + group_counts via `treeReduce`; driver draws `Sigma ~ block-IW(S)` (reuse the `pg_stm_gibbs` Σ step: `assemble_sigma` with an IW *draw* not the mean — factor a `draw_block_iw(...)` shared helper, or call `invwishart.rvs` per block). Average post-burn `Sigma_samples`.
- Consumes: `omega_sample`, `psi_posterior`, `_e_step_doc`'s inner responsibilities for the token counts (the fixed-β/Γ per-doc counts), `stick_layout`, the `pg_stm_gibbs` block-IW draw.

- [ ] **Step 1: Write the failing test** — the distributed Gibbs-Σ recovers the planted background correlation on the STICK-NATIVE corpus (the F4 positive control), matching the single-machine `pg_stm_gibbs` within sampling tolerance.

```python
import numpy as np
from spark_vi.models.topic.pg_stm import pg_stm_gibbs_sigma_rdd
from spark_vi.models.topic.pg_stm import pg_stm_gibbs, stick_layout
from tests._stm_synth import gated_ln_corpus_stick

def _corr(S):
    d = np.sqrt(np.diag(S)); return S / np.outer(d, d)

def test_distributed_gibbs_sigma_recovers_planted(spark):
    docs, part, St, beta = gated_ln_corpus_stick(group_weights={"A":0.5,"B":0.5},
        fg_per_group=1, bg_k=3, V=60, D=1000, doc_len=40, seed=2)
    P = docs[0].x.shape[0]
    # single-machine reference sampler for beta/Gamma to hold fixed:
    gb = pg_stm_gibbs(docs, K=part.K, V=60, partition=part, P=P, n_iter=600, burn=300, seed=0)
    rdd = spark.sparkContext.parallelize(docs, 4)
    out = pg_stm_gibbs_sigma_rdd(rdd, beta=gb["beta"], Gamma=gb["Gamma"],
        Sigma0=np.eye(part.K-1), partition=part, layout=stick_layout(part),
        n_sweeps=60, burn=30, seed=0)
    r_true = _corr(St)[0,1]; r_dist = _corr(out["Sigma"])[0,1]
    assert r_true > 0 and r_dist > 0 and abs(r_dist - r_true) < 0.15
```

- [ ] **Step 2: Run it, watch it fail** (function absent).
- [ ] **Step 3: Implement** `pg_stm_gibbs_sigma_rdd` (+ shared `draw_block_iw` helper if extracted from `pg_stm_gibbs`).
- [ ] **Step 4: Run — passes** (Gibbs recovers the identified planted correlation, per insight 0044).
- [ ] **Step 5: Commit** `feat(pg-stm): distributed exact-Gibbs Sigma pass (comorbidity read-out)`.

---

### Task 4: Cloud driver + `run_experiment` `pg_stm` dispatch

**Files:**
- Create: `analysis/cloud/pg_stm_bigquery_cloud.py`
- Modify: `scripts/run_experiment.py` (`validate_frontmatter`, `build_fit_driver_path`, `build_fit_args`, add `build_pg_stm_args`)
- Test: `scripts/tests/test_run_experiment.py` (add pg_stm cases); `analysis/cloud/tests/` (a driver-arg/smoke test mirroring `test_stm_driver_partition.py`)

**Interfaces:**
- `analysis/cloud/pg_stm_bigquery_cloud.py`: mirror `stm_bigquery_cloud.py` — reuse `_corpus_load`, `_covariates_load`, the cohort-def cache, `_driver_common`. Build the `STMDocument` RDD + covariate/group columns exactly as the STM driver does (same gating partition from `background_k`/`foreground`/`group_var`), then:
  - Phase 1: `StreamingPGSTM(sigma_mode=<--sigma-mode>).fit(rdd, max_iter=<--max-iter>, batch=<--subsampling-rate or "all">)`.
  - Phase 2 (if `--gibbs-sweeps > 0`): `pg_stm_gibbs_sigma_rdd(rdd, beta, Gamma, Sigma, ...)`.
  - Write `beta`, `Gamma`, `Sigma` (Phase 1), `Sigma_gibbs` (Phase 2), `sigma_max_trace`, and metadata (incl. `sigma_mode`, `model_class="pg_stm"`) to the out dir, matching the STM artifact layout the downstream tools expect.
- `run_experiment.py`:
  - `validate_frontmatter`: allow `model_class == "pg_stm"` (same `covariate_formula`/`categorical_cols`/`continuous_cols` requirement as stm).
  - `build_fit_driver_path`: `pg_stm` → `analysis/cloud/pg_stm_bigquery_cloud.py`.
  - `build_pg_stm_args`: reuse `build_stm_args` output (cohort, K, background_k, foreground, group_var, covariate formula, cache, max_iter, etc.) plus `--sigma-mode <iw|mle>` (frontmatter `sigma_mode`, default iw) and `--gibbs-sweeps <int>` (frontmatter `gibbs_sweeps`, default 0).

- [ ] **Step 1: Write failing tests** — `validate_frontmatter` accepts a `pg_stm` frontmatter; `build_fit_driver_path` returns the pg_stm driver; `build_pg_stm_args` emits `--sigma-mode iw` and `--gibbs-sweeps N` from frontmatter.
- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement** the dispatch + `build_pg_stm_args`, then the cloud driver (mirroring the STM driver; keep it thin — orchestration only).
- [ ] **Step 4: Run** the run_experiment tests + a driver import/arg smoke test. (A full BigQuery fit is NOT run locally — the cluster does that.)
- [ ] **Step 5: Commit** `feat(exp): pg_stm model_class dispatch + cloud driver (SVI + Gibbs-Sigma)`.

---

### Task 5: Duplicate exp-0027 into new experiments (do NOT overwrite) + exp-0050 record

**Files:**
- Create: `docs/experiments/0050-pg-stm-distributed-iw-cancer-dementia.md` (frontmatter cloned from 0027, `model_class: pg_stm`, `sigma_mode: iw`, `gibbs_sweeps: 40`)
- Create: `docs/experiments/0051-pg-stm-distributed-mle-cancer-dementia.md` (same, `sigma_mode: mle`, `gibbs_sweeps: 0` — the un-regularized contrast arm)
- Modify: none of exp-0027.

**Interfaces:** the frontmatter fields `run_experiment.py` reads (`id, slug, cohort, cohort_def, model_class, prior_obs_days, person_mod, doc_unit, covariate_formula, categorical_cols, continuous_cols, random_seed, cache_uri, K, background_k, foreground, group_var, max_iter`) copied verbatim from exp-0027, changing only `id`, `slug`, `model_class`, and adding `sigma_mode` / `gibbs_sweeps`.

- [ ] **Step 1:** Copy exp-0027 frontmatter into `0050-*.md`; set `id: 50`, `slug: pg-stm-distributed-iw-cancer-dementia`, `status: planned`, `model_class: pg_stm`, `sigma_mode: iw`, `gibbs_sweeps: 40`. Keep K=50, background_k=30, foreground `cancer:10,dementia:10`, `~ C(sex) + age`, group_var source_cohort, cohort cancer_or_dementia — identical to 0027 so the runaway regime is reproduced.
- [ ] **Step 2:** Write the exp-0050 body = the pre-registered success criteria from the design spec (IW keeps Σ bounded+PD where mle blows up / loses PD; sub-phenotypes preserved; Phase-2 Gibbs-Σ is a valid comorbidity matrix). Leave a "Results (to fill on cluster run)" section.
- [ ] **Step 3:** Create `0051-*.md` as the `sigma_mode: mle` contrast arm (`id: 51`, `gibbs_sweeps: 0`), body pointing at 0050 for the shared design and stating its role = the un-regularized arm expected to reproduce the insight-0033 runaway.
- [ ] **Step 4:** Run `python scripts/run_experiment.py` in dry/validate mode (or the frontmatter validator) on both new configs to confirm they parse and dispatch to the pg_stm driver without error. Do NOT launch a cluster job.
- [ ] **Step 5: Commit** `docs(exp): 0050/0051 — pg_stm distributed runaway-cure arms (exp-0027 duplicate, iw vs mle)`.

---

## Post-build (cluster, user-gated)
- Add `polyagamma==2.0.2` to the Dataproc image/init.
- `make exp ID=50` and `make exp ID=51`; compare `sigma_max_trace` / Σ eigmin / max|Σ| / Cholesky across the arms; fill exp-0050 Results; write the insight if IW cures the runaway at scale.

## Risks
- Full-batch equivalence tolerance (Task 2): float reduction-order drift across partitions — pin `depth`/tolerance; if it exceeds `1e-8`, document why and assert the looser bound with a comment (still a real equivalence).
- Minibatch ρ blend correctness (Task 2): mirror `StreamingSTM`'s validated natural-gradient blend exactly; the full-batch path is the safety net (equivalence-tested).
- Gibbs-Σ cost at K=50 (Task 3): 49×49 block draws are cheap; per-doc sampling is the cost, bounded by `n_sweeps` and embarrassingly parallel.
- Driver artifact layout (Task 4): match the STM driver's output keys so downstream (sub-project 3) can consume it.

## Critical files
- `spark-vi/spark_vi/models/topic/pg_stm.py` (pure primitives + Task-1 stats refactor)
- `spark-vi/spark_vi/mllib/topic/pg_stm.py` (NEW — StreamingPGSTM + Gibbs-Σ)
- `spark-vi/spark_vi/mllib/topic/stm.py` (the treeReduce + StreamingSTM template to mirror)
- `analysis/cloud/pg_stm_bigquery_cloud.py` (NEW), `analysis/cloud/stm_bigquery_cloud.py` (template)
- `scripts/run_experiment.py` (dispatch)
- `docs/experiments/0050,0051-*.md` (NEW), `docs/experiments/0027-*.md` (template — DO NOT MODIFY)
