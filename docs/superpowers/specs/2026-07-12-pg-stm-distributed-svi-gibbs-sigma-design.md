# Sub-project 2 — Distributed PG-STM (SVI + Gibbs-Σ) design

**Status:** design (brainstormed 2026-07-12, three core decisions locked via questions)
**Branch:** `pg-stm` (continues from milestone-1 core, HEAD after F4 = `bb855e2`)
**Predecessors:** sub-project 1 (single-machine PG-VI core + Gibbs cross-check, milestone-1
CLOSED — exp 0049); F1–F4 (pd_complete assembly fix, honest reframe, stick-native VI-vs-Gibbs
gate). **Successor:** sub-project 3 (dashboard export + LKJ prior).

## Why

Milestone-1 established the PG/full-Bayes machinery works and that a proper IW Σ posterior is
better-conditioned than the un-regularized point estimate — but it did NOT decisively test
"does IW cure THE runaway," because the insight-0033 10^10 blow-up is a scale / real-data
phenomenon that the K=6/D=1000 toy corpus never reproduced (the toy "MLE indefinite" contrast
turned out to be a block-Σ zero-fill artifact, now fixed). The decisive test lives on the real
**exp-0027 cancer_or_dementia** corpus (K=50, background_k=30, foreground cancer:10 / dementia:10,
`~ C(sex) + age`, group_var source_cohort) — the corpus that produced the original 10^10 runaway.

Separately, F4 / insight 0044 proved that **mean-field VI cannot recover the Σ correlation** (it
reads the wrong sign even on an identified corpus, where exact Gibbs recovers it). Since the Σ
correlation read-out is the comorbidity deliverable, the distributed engine needs a second path
for it. Hence two engines.

## Locked decisions (brainstorm, 2026-07-12)

1. **Two engines:** distributed PG-SVI carries β/Γ/gating (+ IW Σ for the runaway test); a
   separate exact-Gibbs pass reports the Σ correlations (comorbidity read-out).
2. **Two-phase, not interleaved:** SVI runs to convergence with the cheap mean-field Σ (fine for
   the runaway MAGNITUDE/PD question), THEN one exact-Gibbs Σ refinement pass over the converged
   β/Γ. Gibbs cost paid once; the runaway test and the correlation read-out are distinct
   deliverables with distinct success criteria.
3. **Controlled mle-vs-iw contrast at scale:** run the SAME distributed SVI kernel on exp-0027
   twice, toggling ONLY `sigma_mode` (un-regularized `scatter/n` vs IW posterior). Decisive
   estimator isolation on the identical corpus/E-step — shows mle reproduces the 10^10 (or
   non-PD) blow-up while IW stays bounded+PD. ~2× cluster cost, accepted.

## Architecture

Both phases reuse the existing distributed sufficient-statistic idiom
(`mapPartitions(_local).treeReduce(_combine, depth=2)`, globals broadcast via default-arg
closures) already used across `spark_vi/mllib/topic/stm.py` — only small global-shaped arrays
reach the driver, so it scales to any D. The per-doc math is ALREADY pure functions in
`spark_vi/models/topic/pg_stm.py` (Tasks 1–6), so the E-step and stat accumulation port verbatim.

### Phase 1 — Distributed PG-SVI (= the runaway-cure test)

New `spark_vi/mllib/topic/pg_stm.py :: StreamingPGSTM`, mirroring `StreamingSTM`
(stm.py:1271) but wrapping the PG-VI kernel:

- **Minibatch loop** (Robbins-Monro ρ_t = (t + τ0)^−κ, the same schedule `StreamingSTM.fit`
  uses): sample a minibatch RDD; each worker runs the per-doc E-step
  (`omega_expectation`/`psi_posterior` → `token_responsibilities` → per-doc β/Γ/Σ sufficient
  stats) and the partition accumulates them; `treeReduce` sums; the driver forms the natural-
  gradient global update and applies it with ρ_t.
- **IW block-Σ M-step** via the milestone-1 `_assemble_sigma` (now PD-completed, F1) —
  `sigma_mode ∈ {"iw","mle"}` for the controlled contrast. The full-batch VI we validated IS
  this kernel at batch=all, so Phase 1 is "same updates, minibatched + ρ."
- **Deliverable:** does IW keep Σ bounded + PD across SVI iterations on exp-0027 where mle
  reproduces the runaway? Plus β/Γ/gating at scale (sub-phenotype recovery preserved).

### Phase 2 — Distributed exact-Gibbs Σ pass (= comorbidity read-out)

New `pg_stm_gibbs_sigma_rdd` (distributed): holding the converged (β, Γ) FIXED, run `n_sweeps`
Gibbs sweeps for Σ only:

- Each worker draws, per doc, ω_d | ψ_d (PG) and ψ_d | data_d, β, Γ, Σ **exactly** — the FULL
  multivariate Gaussian conditional `N(m_d, V_d)`, V_d = (Σ⁻¹ + diag(ω_d))⁻¹, NOT the mean-field
  point estimate — then accumulates the ψ scatter (centered on Γᵀx_d) via `treeReduce`.
- Driver draws Σ | scatter ~ block-IW (the milestone-1 `pg_stm_gibbs` Σ step, block-assembled +
  PD-completed). Average Σ over post-burn sweeps.
- **Deliverable:** the trustworthy-correlation Σ that mean-field VI botches (per insight 0044) —
  the comorbidity correlation matrix for the dashboard (built in sub-project 3).

## Components / files

- **Create** `spark-vi/spark_vi/mllib/topic/pg_stm.py` — `StreamingPGSTM` (Phase 1) +
  `pg_stm_gibbs_sigma_rdd` (Phase 2), both reusing `models/topic/pg_stm.py` primitives and the
  `treeReduce` idiom.
- **Create** `analysis/cloud/pg_stm_bigquery_cloud.py` — cloud driver mirroring
  `stm_bigquery_cloud.py`: loads the exp-0027 corpus + covariates (reuse `_corpus_load`,
  `_covariates_load`, the cohort-def cache), runs Phase 1 (mle then iw) + Phase 2, writes the
  Σ-trace / β / Γ / final Σ + Gibbs-Σ artifacts. Wire into `scripts/run_experiment.py`
  (`model_class: pg_stm`).
- **Cluster image:** add `polyagamma==2.0.2` (ships wheels) to the Dataproc init/image.
- **Out of scope (sub-project 3):** dashboard export of the Gibbs-Σ correlations + LKJ prior.

## Validation ladder (synthetic, BEFORE the cluster) — TDD

New `spark-vi/tests/test_pg_stm_streaming.py`, using the existing synthetic corpora
(`gated_ln_corpus`, `gated_ln_corpus_stick`) and a local Spark context (as `test_mllib_stm.py`
does):

1. **SVI == full-batch at batch=all:** `StreamingPGSTM(batch=all, ρ=1)` reproduces the validated
   `PGSTMVI.fit` β/Γ/Σ (near-byte). Gates that the distributed kernel is the same math.
2. **Minibatched SVI converges** to the same β/Γ (planted recovery non-regression) and a
   bounded+PD Σ on a synthetic gated corpus.
3. **Distributed Gibbs-Σ == single-machine `pg_stm_gibbs`** Σ on the STICK-NATIVE corpus:
   recovers the planted background correlation (the F4 positive control), confirming the
   distributed scatter+IW draw matches the reference sampler.
4. **Runaway isolation at synthetic scale:** the mle-vs-iw contrast reproduces the milestone-1
   direction under the streaming kernel (mle at least as ill-conditioned as iw; both PD post-fix
   — the SEVERITY is what needs real scale, asserted only on the cluster).

## Cluster experiment (exp 0050)

`docs/experiments/0050-pg-stm-distributed-runaway-cure-exp0027.md`. Run on exp-0027:

- **Phase 1 ×2** (sigma_mode mle, iw). Record `sigma_max_trace`, final Σ eigmin / max|Σ|,
  Cholesky status, β/topic quality (NPMI, sub-phenotype recovery vs the STM baseline).
- **Success criteria (the decisive gate):** the un-regularized mle reproduces the insight-0033
  pathology at scale (Σ → 10^10 OR loses PD), while the IW posterior over the identical E-step
  stays bounded (max|Σ| = O(1–10)) AND PD, with sub-phenotypes preserved. THAT is "IW cures the
  runaway," established on the real corpus the toy scale couldn't.
- **Phase 2:** the Gibbs-Σ correlation matrix — sanity that it is a valid, interpretable
  comorbidity structure (symmetric, PD, in-range), distinct from (and more trustworthy than) the
  mean-field VI Σ.

## Error handling / risks

- **polyagamma on the cluster image** — the one hard external dependency; verify with a smoke fit
  before the full run.
- **Gibbs-Σ cost / convergence** — bounded by `n_sweeps` (paid once, Phase 2 only); the per-doc
  draws are embarrassingly parallel. Monitor the Σ trace across sweeps for burn-in.
- **PD at scale** — `_assemble_sigma`'s `pd_complete` (F1) runs on the driver each Σ update; K=50
  (K−1=49 dims) is small, so the completion cost is negligible.
- **ρ schedule / minibatch size** — reuse `StreamingSTM`'s validated defaults; expose as knobs.
- **Runaway on the mle arm** — if mle genuinely diverges to 10^10, guard the driver so the arm
  fails loud with the recorded trace rather than NaN-crashing the job (the divergence IS the
  result to capture).

## Non-goals

Dashboard export, LKJ Σ prior, and the structured/collapsed variational posterior (a possible
future alternative to the two-engine split) are explicitly out of scope for sub-project 2.
