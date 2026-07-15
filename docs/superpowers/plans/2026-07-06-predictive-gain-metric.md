# Predictive-gain presence / depth / prominence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the θ̂-threshold "prevalence" readout AND the θ̂ "topic-mass / prominence" histogram
with three predictive-gain quantities (presence, depth, prominence) computed from leave-one-topic-out
held-out predictive gain — one principled contrast in place of a tower of point-posterior functionals.

**Architecture:** Per document, split tokens visible/held-out (reuse `heldout_split`); infer the gated
mode over the allowed set at the calibrated scale c* (reuse `_stm_doc_inference`); score held-out
predictive LL (reuse `_predictive_loglik`). For each allowed topic k, re-score after removing k
(`Δ_k(d) = LL(allowed) − LL(allowed\{k})`). Aggregate per-topic across the corpus (distributed
`mapPartitions`/`treeReduce`), emit ONLY per-topic aggregates. Build the correct-but-slow COLD version
first (reuses `_stm_doc_inference` over reduced allowed sets — zero novel numerics); add the
Hessian-downdate one-Newton-step optimization second, validated against COLD. Wire into both dashboard
builders + the frontend; re-export.

**Tech Stack:** Python/NumPy/SciPy + PySpark (spark-vi); the existing gated logistic-normal STM E-step;
Svelte/TS dashboard.

## Global Constraints (every task inherits these)

- **Calibrated scale, always.** All per-document inference — the mode AND the Laplace covariance —
  uses `Sigma_inv_allowed = (1/c*)·inv(Σ_allowed)` where c* is the bundle's held-out `eta_scale`, NEVER
  the unit fit scale. This is the ADR 0034 addendum (`docs/decisions/0034-*.md`). Pass c* in; fall
  back to 1.0 only if the bundle lacks it, and log that.
- **Two inference/scoring conventions, never conflated:** INFERENCE uses `expElogbeta =
  exp(digamma(λ) − digamma(λ.sum))`; SCORING held-out predictive LL uses `beta_prob = λ/λ.sum` (E[β])
  via `_predictive_loglik`. (The sweep docstring is explicit; conflating them miscalibrates.)
- **Export is aggregate-only (terms of service).** Only per-topic aggregates leave — the (presence,
  depth) scalars, an aggregate binned prominence histogram, and per-topic diagnostic scalars. NO
  per-document quantities ever exported.
- **Within-group denominators for foreground topics.** A foreground topic's presence/depth denominator
  is its own group's documents; background topics use the whole corpus. Never rank foreground and
  background on a shared raw denominator.
- **COLD is the correctness oracle.** The downdate optimization must agree with the cold re-inference
  within tolerance on fixtures before it is used; a real-data discrepancy diagnostic is emitted.
- **Two caveats are load-bearing** (from the design doc §5): (1) for a high-mass topic (θ̂_k above a
  small bound) the one-Newton-step downdate may under-capture the covering — take 2–3 steps / a
  convergence check there; (2) depth-share is aggregated as Σ-numerator / Σ-denominator across
  documents, NEVER as a per-document ratio.
- **Cloud + local builder blocks stay byte-parallel** (project convention). TDD; frequent commits; no
  magic constants without a justifying comment.

---

## File Structure

- **Create** `spark-vi/spark_vi/mllib/topic/predictive_gain.py` — the estimator (single-doc gain +
  distributed RDD + accumulator). Imports the E-step primitives from
  `spark_vi.models.topic.stm` (`_stm_doc_inference`) and `spark_vi.mllib.topic.stm`
  (`_gated_mode_theta`) and eval (`heldout_split`, `_predictive_loglik`).
- **Create** `spark-vi/tests/test_predictive_gain.py` — fixtures + cold/downdate agreement + accumulation.
- **Modify** `analysis/cloud/build_dashboard_cloud.py`, `analysis/local/build_dashboard.py` — a new
  distributed phase (after the eta_scale hoist, reusing its doc_rdd + c*) that computes the per-topic
  aggregates and threads them into the export.
- **Modify** `charmpheno/charmpheno/export/dashboard.py` (+ `model_adapter.py` / `DashboardExport`) —
  carry the new per-topic aggregate fields into `phenotypes.json`.
- **Modify** `dashboard/src/lib/types.ts`, `dashboard/src/lib/store.ts`,
  `dashboard/src/lib/atlas/CodePanel.svelte`, `dashboard/src/lib/atlas/TopicMap.svelte` — the
  (presence, depth) readout + the prominence distribution, replacing the τ readout + θ̂ histogram.
- **Modify** `docs/insights/` (new entry) + ADR pointer.

---

# PHASE 1 — Backend estimator (spark-vi). COLD first, then the downdate optimization.

### Task 1: Single-document predictive gain (COLD) — the reference implementation

**Files:**
- Create: `spark-vi/spark_vi/mllib/topic/predictive_gain.py`
- Test: `spark-vi/tests/test_predictive_gain.py`

**Interfaces:**
- Consumes: `_stm_doc_inference(indices, counts, expElogbeta, Gamma, Sigma_inv_allowed, x,
  max_iter, tol, allowed, reference) -> (eta_hat, nu_d, iters)`; `_gated_mode_theta(eta_hat, allowed,
  K)`; `heldout_split(doc, holdout_frac, seed) -> (visible_doc, held_indices, held_counts)`;
  `_predictive_loglik(theta, beta_prob, held_indices, held_counts) -> float` (confirm exact signature
  in `spark_vi/eval/topic/concentration_recovery.py` and match it).
- Produces: `doc_predictive_gain(doc, gp, partition, *, c, reference, holdout_frac, seed, ...) ->
  DocGain` where `DocGain` carries: `allowed` (indices), `delta` (len-|allowed| Δ_k in nats, aligned
  to `allowed`), `ll_full` (float), `n_held` (int), `theta_full` (len-K, for the high-mass step-count
  trigger and the prominence value), and `dedup_delta` (Δ recomputed with held counts capped at 1).

- [ ] **Step 1: Write the failing test — a hand-checkable two-topic corpus.**
  Build a tiny gated fixture (reuse `tests/_stm_synth.py` helpers or a minimal inline model): 2
  disjoint-vocab topics, documents whose tokens are all from one topic. Assert: for a document made
  only of topic-0 tokens, `delta[0]` (gain of the topic that generated it) is clearly positive and
  `delta[1]` ≈ 0 (removing a topic the document has no held-out tokens for costs ~nothing) — the
  auto-floor property. Assert `delta` is finite and aligned to `allowed`.

- [ ] **Step 2: Run it — expect ImportError/NameError (function absent).**
  `cd spark-vi && python -m pytest tests/test_predictive_gain.py -k cold -v`

- [ ] **Step 3: Implement `doc_predictive_gain` (COLD).**
  For the document: `visible, held_idx, held_cnt = heldout_split(...)`; skip (return None) if the split
  is degenerate (mirror the sweep's skip rule — too few tokens / empty visible). Compute `allowed =
  partition.allowed_indices(doc.groups)`, `rinv = safe_inverse(Σ[allowed,allowed])`,
  `Sigma_inv_allowed = (1/c)·rinv`. Infer `eta_hat, nu_d, _ = _stm_doc_inference(visible.indices,
  visible.counts, expElogbeta, Gamma, Sigma_inv_allowed, x, allowed=allowed, reference=reference)`;
  `theta_full = _gated_mode_theta(eta_hat, allowed, K)`; `ll_full = _predictive_loglik(theta_full,
  beta_prob, held_idx, held_cnt)`. Then FOR EACH position `p, k` in `allowed`:
  - `allowed_k = np.delete(allowed, p)`; if the reference topic is being removed or `allowed_k` is
    empty, set `delta_k = 0.0` and continue (removing the pinned reference is undefined; document with
    a single allowed topic has no contrast).
  - `rinv_k = safe_inverse(Σ[allowed_k, allowed_k])`; re-infer `eta_k, _, _ =
    _stm_doc_inference(..., Sigma_inv_allowed=(1/c)·rinv_k, allowed=allowed_k, reference=reference)`;
    `theta_k = _gated_mode_theta(eta_k, allowed_k, K)`; `ll_k = _predictive_loglik(theta_k, beta_prob,
    held_idx, held_cnt)`; `delta[p] = ll_full − ll_k`.
  Build `dedup_delta` identically but with `held_cnt` replaced by `np.ones_like(held_cnt)` in the two
  scorings (unique-token variant). Return the `DocGain`.
  (Correctness note: this is O(|allowed|) full inferences per document — deliberately simple and
  exact; the downdate in Task 4 makes it fast.)

- [ ] **Step 4: Run the test — expect PASS.** Same command.

- [ ] **Step 5: Add an auto-floor + finiteness test and a within-a-topic sanity test**, then commit.
  `git commit -m "feat(spark-vi): cold single-doc predictive gain (LOO held-out) for STM"`

### Task 2: Permuted-topic null band (the presence threshold, model-generated)

**Files:** Modify `predictive_gain.py`; Test `test_predictive_gain.py`.

**Interfaces:** Produces `null_delta(doc, gp, partition, *, c, reference, n_perm, rng_seed, ...) ->
float[]` — for each of `n_perm` permutations, shuffle one topic's β row across the vocabulary and
compute that permuted topic's Δ against the real allowed set (same held-out split); the null band is
the distribution of Δ a topic that explains nothing produces. Return the per-doc null Δ samples (to be
accumulated into a corpus null band).

- [ ] **Step 1: Failing test** — a permuted topic's Δ has near-zero mean and small spread relative to a
  real signature topic's Δ on the same documents.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `null_delta` — reuse the Task-1 ablation machinery with a β row permuted
  under a seeded RNG (vary permutation by `rng_seed + perm_index`; note in a comment that determinism
  is via the seed, not `np.random` global). Keep `n_perm` small (default 4) — it feeds a corpus-level
  band, not a per-doc estimate.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit.** `feat(spark-vi): permuted-topic null band for predictive-gain presence`

### Task 3: Distributed corpus aggregation (`corpus_predictive_gain_gated_rdd`)

**Files:** Modify `predictive_gain.py`; Test `test_predictive_gain.py`.

**Interfaces:** Produces `corpus_predictive_gain_gated_rdd(doc_rdd, global_params, partition, *,
c, reference=None, holdout_frac=0.5, seed=0, sample_cap=200_000, n_perm=4, n_bins=50, depth=2) ->
dict` with per-topic aggregate arrays (length K, indexed by topic id): `mean_gain` (Σ Δ_k / count_k,
within-group count), `depth_num` (Σ Δ_k) and `depth_den` (Σ_d Σ_j Δ_j attributed to k's documents) so
`depth = depth_num/depth_den` is formed at the end (NEVER a per-doc ratio), `presence` (fraction of
k's documents whose Δ_k clears the corpus null band), `prominence_hist` (K × n_bins, binned Δ_k over
documents — the aggregate distribution replacing the θ̂ histogram), `length_corr` (per-topic
correlation of per-doc Δ_k with document length, via streaming sums), `dedup_mean_gain`, and
`null_band` summary + `count_k`, `n_docs`.

- [ ] **Step 1: Failing test** — on a synthetic gated corpus (`_stm_synth`), the distributed result's
  per-topic `mean_gain` for a planted signature topic exceeds a non-signature topic's; `depth` is
  formed from summed num/den; foreground topics' `count_k` equals their group's document count (within-
  group denominator); shapes are correct.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** the `mapPartitions(_local).treeReduce(_combine)` pass mirroring
  `corpus_theta_gated_rdd` / `corpus_concentration_stm_rdd` (broadcast gp + partition via default-arg
  closures; sample to `sample_cap` first via `doc_rdd.sample`; log N/N'/frac). `_local` accumulates,
  per topic: Σ Δ_k, count_k (within-group), the Δ_k histogram bin counts, streaming (Σ len, Σ Δ_k·len,
  Σ len², Σ Δ_k, Σ Δ_k², count) for `length_corr`, the dedup Σ, the document's Σ_j Δ_j into each of its
  topics' `depth_den` share, and the null-Δ samples. `_combine` sums all arrays (functional, treeReduce-
  safe). Form `depth`, `presence` (vs the reduced null band), `mean_gain`, `length_corr` at the end.
  Guard empty-corpus with a raise. Foreground accumulation is WITHIN the doc's group.
- [ ] **Step 4: Run, expect pass** (+ the existing sweep/stm tests still green).
- [ ] **Step 5: Commit.** `feat(spark-vi): distributed corpus predictive-gain aggregation (gated STM)`

### Task 4: The Hessian-downdate one-Newton-step ablation (optimization) + real-data check

**Files:** Modify `predictive_gain.py`; Test `test_predictive_gain.py`.

**Interfaces:** A `fast=True` path in `doc_predictive_gain` that, instead of a cold re-inference per k,
warm-starts from the full mode and takes ONE Newton step over `allowed\{k}` reusing the grad/Hessian
(the reference-row/col-deletion pattern `_stm_doc_inference` already documents — expose or replicate
the grad+Hessian at the mode). For a topic with `theta_full[k]` above a small bound (default 0.2 —
heuristic, comment it), take up to 3 steps / until the step norm falls below tol (caveat 1).

- [ ] **Step 1: Failing test — downdate agrees with COLD.** On the Task-1 fixture and a random gated
  document, `delta_fast` matches `delta_cold` within an absolute tolerance (e.g. 1e-2 nats) for all
  topics, and within a tighter tolerance for high-mass topics (verifying the multi-step trigger).
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** the warm-start downdate. If exposing the mode's grad/Hessian from
  `_stm_doc_inference` is cleaner than recomputing, add an optional return or a small helper in
  `models/topic/stm.py` that returns them — keep `_stm_doc_inference`'s existing return tuple
  unchanged (add a sibling, don't break callers). Delete k's row/col (like the reference), one Newton
  step, re-score. The multi-step trigger for high-mass topics.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Add a real-data discrepancy hook** — a `predictive_gain_downdate_audit(doc_rdd, ...,
  n_sample)` that runs BOTH cold and fast on a small sample and returns the per-topic max/mean |Δ
  discrepancy| (a per-topic aggregate — ToS-safe). This is the real-data cold-solve check the
  validation basis rests on; it is called during the export phase and logged. Commit.
  `feat(spark-vi): downdate one-Newton-step ablation + cold-solve real-data audit`

### Phase 1 verification
`cd spark-vi && python -m pytest tests/test_predictive_gain.py -v` green; existing
`tests/test_corpus_theta_gated.py`, sweep/stm/scale tests still green. The downdate audit function
returns small discrepancies on a synthetic corpus.

---

# PHASE 2 — Export + frontend + re-export (GATED: run Phase 1 on the cluster and look at the real
# (presence, depth), the prominence histogram, the null band, and the downdate audit BEFORE building
# the frontend. The exact aggregate schema + frontend encoding finalize here, informed by the numbers.)

### Task 5: Thread the aggregates into the bundle export (both builders, parity)

**Files:** Modify `analysis/cloud/build_dashboard_cloud.py`, `analysis/local/build_dashboard.py`,
`charmpheno/charmpheno/export/dashboard.py`, `charmpheno/charmpheno/export/model_adapter.py`.

**Approach:** A new distributed phase placed AFTER the hoisted eta_scale phase (so c* is known — reuse
`eta_scale` and the phase's `doc_rdd`/`stm_cov_df` exactly as the θ-histogram phase does) that calls
`corpus_predictive_gain_gated_rdd(doc_rdd, gp, partition, c=eta_scale or 1.0, reference=...)`, runs the
downdate audit on a sample, and writes the per-topic aggregates into `phenotypes.json` (new per-topic
fields: `presence`, `depth`, `prominence_hist` + shared `prominence_bin_edges`, and the diagnostic
scalars `length_corr`, `dedup_gain`, `null_band`). Aggregate-only; within-group denominators labeled.
`DashboardExport` gains the fields (frozen dataclass → `dataclasses.replace`, as in Fix B). Guard on
STM+gated; enhancement-only try/except (failure omits the fields). Cloud+local byte-parallel; py_compile
cloud; extend the local end-to-end export test (`charmpheno/tests/test_theta_histogram_stm_build.py`
sibling) to assert the new fields serialize. TDD per the builder-test pattern.

### Task 6: Frontend — (presence, depth) readout + prominence distribution

**Files:** Modify `dashboard/src/lib/types.ts`, `store.ts`, `atlas/CodePanel.svelte`,
`atlas/TopicMap.svelte`, and copy in `lib/copy.ts`.

**Approach:** Replace the τ-threshold `prevalenceReader` headline with **presence** (fraction clearing
the null band) and add **depth** as the second axis (the bubble map encodes presence × depth — position
or size for one, color/second-channel for the other; finalize the encoding against the real numbers).
Replace the `CodePanel` θ̂ "topic mass distribution" histogram (`theta_histogram`) with the
**prominence distribution** (`prominence_hist` + `prominence_bin_edges`). Drop the `tauThreshold`
control from the headline (keep θ̂ histogram code paths only if a bundle lacks the new fields —
backward compat). Update `copy.ts` wording (presence = "how widely, in predictive nats"; depth = "how
much / unique contribution"). Vitest for the new readers; `npm run check` no new errors; `npm run
build` clean. TDD.

### Task 7: Re-export #3 + validation

**Approach (mostly operational, not code):** `make -C analysis/cloud build-covariates EXP=0028
FORCE=1` + `build-dashboard-exp ID=28`; ingest to `dashboard/public/data/population_cancer`; transfer
labels (fit unchanged); verify the new fields + the downdate-audit log; run the **read-the-documents
audit** (sample documents at high/middle/low presence for a niche foreground, a broad background, and a
mid-breadth topic; human-judge the ranking) — the validity gate; then commit the bundle. Record an
insight-log entry (the predictive-contrast pivot; the auto-floor/uncertainty/saturation dissolution;
the real numbers vs the θ̂-threshold version).

---

## Risks (confirm during execution)
- **Cold cost.** O(|allowed|) inferences/doc — tractable on the sampled corpus but slow; if Phase-1
  cluster timing is too high, the downdate (Task 4) is the mitigation, and `sample_cap` can drop.
- **Downdate fidelity.** The one-step softmax-renormalization approximation — Task-4 fixture agreement +
  the real-data audit gate it; fall back to cold if the audit shows material discrepancy.
- **Null-band cost.** `n_perm` ablations per doc add to the per-doc cost; keep `n_perm` small (feeds a
  corpus band).
- **Depth denominator.** Must be summed across documents, never a per-doc ratio (caveat 2) — a defect
  a reviewer should specifically check in Task 3.
- **`_predictive_loglik` signature** — confirm the exact argument order/shape against
  `concentration_recovery.py` before Task 1 (the plan assumes `(theta, beta_prob, held_idx, held_cnt)`).

## Critical files
- `spark-vi/spark_vi/mllib/topic/predictive_gain.py` (NEW), `tests/test_predictive_gain.py` (NEW)
- `spark_vi/models/topic/stm.py` (`_stm_doc_inference` — maybe a grad/Hessian sibling for Task 4),
  `spark_vi/mllib/topic/stm.py` (`_gated_mode_theta`, the RDD idiom),
  `spark_vi/eval/topic/concentration_recovery.py` (`heldout_split`, `_predictive_loglik`)
- `analysis/{cloud/build_dashboard_cloud,local/build_dashboard}.py`,
  `charmpheno/charmpheno/export/{dashboard,model_adapter}.py`
- `dashboard/src/lib/{types.ts,store.ts,atlas/CodePanel.svelte,atlas/TopicMap.svelte,copy.ts}`
