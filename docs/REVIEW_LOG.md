# Code Review Log

This log records code-walkthrough and refactor sessions after initial structuring and test implementations.

This is a **living document** in the same sense as `docs/architecture/`: append
each new review session as its own dated `##` section at the top of the log
(newest first). Each session entry briefly notes which areas were reviewed,
which refactors shipped, which pre-existing issues were caught, which docs and
ADRs changed, and any threads parked for later. Keep entries impersonal and
project-scoped — pedagogical content and per-contributor preferences belong
elsewhere.

---

## 2026-04-22 to 2026-04-29 — Bootstrap walkthrough and refactor sessions

A bottom-up walkthrough of the post-bootstrap codebase, accompanied by four
refactor detours triggered by issues surfaced during review. Two new ADRs
(0005, 0006) document the largest changes; multiple pre-existing
documentation/code drift issues were caught and fixed.

### Areas reviewed

**Lesson 1 — Math foundations of variational inference.**
Bayesian inference from a coin flip; conjugacy (Beta-Bernoulli, Dirichlet-Categorical,
Normal-Normal); MCMC and Gibbs sampling as motivation for VI; ELBO derivation
end-to-end via Jensen's inequality; mode-seeking reverse KL and mean-field
independence as two flavors of underestimated uncertainty; local/global structure
in hierarchical models (D, V, K, N_d shapes for LDA); CAVI updates; sufficient
statistics under Fisher-Neyman factorization; natural gradient for conjugate-exp
families and the `λ̂ - λ` collapse; Robbins-Monro stochastic approximation; map
from formulas to `VIModel` contract methods.

**Lesson 2 — `CountingModel` proof-of-life.**
Beta-Bernoulli math in framework form; how each `VIModel` method "lights up" for
the toy model; the ELBO computation; the test suite as executable contract spec.

**Lesson 3 — `VIRunner` distributed loop.**
Closure default-args for serialization safety; broadcast → `mapPartitions` →
`treeReduce` pattern; Robbins-Monro step schedule; broadcast lifecycle
(`unpersist` discipline; driver-side handle vs executor block-manager caches);
mini-batch sampling cache discipline; auto-checkpoint hook placement;
`resume_from` semantics; `start_iteration` Robbins-Monro continuity invariant;
the mock-wrapping test pattern in `test_broadcast_lifecycle.py`.

**Lesson 4 — `VIConfig`, `VIResult`, persistence.**
Three validation idioms (range checks, coupled-fields invariants via XOR-of-
None, type-vs-value separation); `VIResult` dual-purpose semantics
(completed-run vs in-progress checkpoint); JSON + per-name `.npy` over pickle;
`format_version` as cheapest forward-compat handle.

**Lesson 5 — `charmpheno` clinical layer.**
One-way dependency invariant (`charmpheno → spark_vi`, never reverse); canonical
4-column OMOP shape; widened `(IntegerType, LongType)` validator for fixture/
real-CDR symmetry; the loader-family contract anchored by `load_omop_parquet`;
stub-as-design-tool pattern (`load_omop_bigquery`, `OnlineHDP`); `CharmPhenoHDP`
wrapper as composition (has-a OnlineHDP) and translation-layer slot for clinical
terminology.

**Sidebar — RDD vs DataFrame.**
Where the bridge lives ([analysis/local/fit_charmpheno_local.py:60](../analysis/local/fit_charmpheno_local.py#L60)
in the driver script, not the clinical layer); why model-specific reshaping
belongs in drivers, not loaders.

**Sidebar — Hungarian topic alignment.**
Where recovery-vs-ground-truth machinery would live (the empty
`charmpheno/evaluate/` subpackage); permutation invariance of LDA/HDP topics;
split/merge as the dominant real-world failure mode beyond simple ordering;
why HDP's discovered `K_fit` makes the matching problem rectangular and the
unmatched-fitted-topics output the interesting signal.

**Lesson 6 — Data pipeline.**
HF dataset streaming as memory (not bandwidth) optimization; top-K filter as
power-law-faithful compression; per-topic renormalization; Poisson clamps
(`max(1, ...)`) as a deliberate departure from the pure generative process;
LDA generative process in numpy (vectorized Dirichlet, three nested sampling
loops); `true_topic_id` oracle column and the convention that training code
must not consume it; `.meta.json` sidecar for reproducible experiment artifacts.

**Lesson 7 — End-to-end + project infrastructure.**
The smoke driver and integration test as proof of life (hermetic-by-construction
fixture, structural-only assertions); three-Makefile orchestration with `$(MAKE)
-C` delegation and `[ -d ]` partial-checkout robustness; the JAVA_HOME detection
song; three-tier test ladder (`test`, `test-all`, `test-cluster`); pre-commit
as a layer over git's native `.git/hooks/`; the four hooks each guarding a
specific catastrophic failure (PHI leak, history bloat, broken-conflict commits,
notebook output churn); architecture docs (living) vs ADRs (append-only) vs
AGENTS.md (orientation) and why the trio is non-redundant; notebook-as-thin-driver
discipline; the `tutorials/` runbook + future `02_*` conceptual-tutorial slots.

### Refactor detours that shipped

**Detour 1 — Mini-batch sampling implementation (ADR 0005).**
Triggered by comparison against Apache Spark MLlib's `OnlineLDAOptimizer`.
Added `VIConfig.mini_batch_fraction`, `sample_with_replacement`, `random_seed`.
Per-iteration `RDD.sample` + `persist(MEMORY_AND_DISK)` + realized `count()` +
empty-batch guard in `VIRunner.fit`. Pre-scale by `corpus_size / batch_size` to
match MLlib's canonical pattern (chosen over the cheaper `1/fraction`).
`VIModel.update_global` parameter renamed `aggregated_stats` → `target_stats` to
disambiguate from the raw `aggregated_stats` passed to `compute_elbo`. 7 new
tests; new entry in `RISKS_AND_MITIGATIONS.md`.

**Detour 2 — Real Beta-Bernoulli ELBO in `CountingModel`.**
Replaced surrogate-hack ELBO with the textbook closed-form
`ELBO(q) = E_q[log p(x|p)] - KL(q || prior)` via digamma and `betaln`. Tightness
at the analytic posterior provides the strongest correctness check available
for this model. Replaced one hack-specific test with three correctness tests
(tightness at posterior, lower-bound property when q is off, monotone progress
toward posterior).

**Detour 3 — `collect()` → `treeReduce` aggregation.**
Replaced `mapPartitions().collect()` + Python-side fold with
`mapPartitions().treeReduce(model.combine_stats)` to bound driver memory and
match MLlib's pattern. Pre-existing inconsistency resolved: the runner module
docstring already claimed `treeAggregate` while the code did `collect()`. New
"Partition-stats aggregation" entry in `RISKS_AND_MITIGATIONS.md`.

**Detour 4 — Tier 3 persistence cleanup (ADR 0006).**
Three coupled issues fixed in one stroke:
1. Eliminated duplicate save/load implementations. `spark_vi/diagnostics/`
   (entire directory + `checkpoint.py` + `__init__.py`) deleted.
   `save_checkpoint` / `load_checkpoint` removed from public API. `VIResult` is
   now the canonical record for both completed runs and in-progress checkpoints
   (`converged=False` covers both "ran out of iterations" and "interim
   checkpoint").
2. Wired the dead `VIConfig.checkpoint_interval` field. Added
   `VIConfig.checkpoint_dir`, coupled with `checkpoint_interval` (both-or-
   neither, enforced in `__post_init__`). When set, `VIRunner.fit` auto-saves
   a `VIResult` to `checkpoint_dir` every N iterations.
3. Added clean `resume_from=path` kwarg on `fit()`, eliminating the previous
   monkey-patch idiom. Loaded `VIResult` seeds `global_params`, `elbo_trace`,
   and `start_iteration` automatically.

Also: manifest gains `format_version: 1` (load raises `ValueError` for unknown
versions). `n_iterations` in returned `VIResult` now correctly includes
`start_iteration` offset (pre-existing silent bug). Test count: 44 spark-vi +
14 charmpheno + 1 integration = 59 tests, all passing.

### Pre-existing issues caught and fixed

- `SPARK_VI_FRAMEWORK.md` documented `VIResult` with `model` and `history`
  fields that don't exist in the dataclass. Corrected to actual fields
  (`global_params`, `elbo_trace`, `n_iterations`, `converged`, `metadata`).
- `SPARK_VI_FRAMEWORK.md` had `update_global` / `global_update` signature
  drift between the doc and the code. Aligned to `update_global`.
- `VIConfig.learning_rate_kappa` docstring did not mention the
  Robbins-Monro convergence guarantee range `(0.5, 1]` — only the validation-
  accepted range `(0, 1]`. Clarified that values in `(0, 0.5]` are permitted
  but not guaranteed to converge.
- `VIRunner.fit` returned a `VIResult` whose `n_iterations` did not include
  the `start_iteration` offset — silent bug because no test asserted on this
  value for resumed runs. Now correctly reflects total iterations including
  any resume offset.
- `runner.py` module docstring claimed `treeAggregate` while the code did
  `collect()` + Python fold. Resolved by Detour 3 (now actually uses
  `treeReduce`, docstring updated).

### New ADRs

- [0005 — Mini-batch sampling matching MLlib `OnlineLDAOptimizer`](decisions/0005-mini-batch-sampling.md)
- [0006 — Unified persistence format: `VIResult` as canonical state](decisions/0006-unified-persistence-format.md)

### Doc updates

- `SPARK_VI_FRAMEWORK.md`: `kappa` convergence range; `update_global`
  signature; `treeReduce` aggregation note; `VIResult` field correction;
  `checkpoint_dir`; SVI / checkpointing moved from "Future Directions" to
  "Implemented".
- `RISKS_AND_MITIGATIONS.md`: mini-batch sampling entry added; partition-stats
  aggregation entry added; "No built-in checkpointing" marked **Resolved as of
  ADR 0006**; Robbins-Monro entry refined.
- `TOPIC_STATE_MODELING.md`: Joint Estimation paragraph for patient-as-
  partition (conditional Dirichlet); two systematic-review citations added.
- `test_broadcast_lifecycle.py`: rich code comments added explaining the
  transparent-proxy mock pattern, recursion-avoidance, scoped patching, and
  per-iteration broadcast accounting math.

### Open threads parked

These are not regressions or known bugs — they are deferred opportunities noted
during review:

- **Vanilla LDA as a `VIModel` for realistic recovery validation.** Would use
  the existing `simulate_lda_omop.py` synthetic data + the real β to test
  full topic recovery end-to-end. A meaningful step beyond `CountingModel`'s
  scalar bias. Deferred until the real `OnlineHDP` lands.
- **`elbo_eval_interval` field.** Currently every iteration calls
  `compute_elbo`. A first-class skip-cadence kwarg would let expensive ELBOs
  be computed less often without forcing models to return NaN as a workaround.
- **Combined mini-batch + auto-checkpoint integration test.** Both features
  work individually with passing tests; the cross-feature interaction has not
  been explicitly tested. Probably correct but worth pinning down.
- **Empty `charmpheno/` subpackages.** `evaluate/`, `profiles/`, `export/` are
  committed as empty namespace markers. `evaluate/` has concrete planned
  content (recovery-metric machinery — see split/merge discussion under
  Lesson 6 sidebar). The other two are speculative; flattening them is a
  defensible YAGNI move pending a follow-on spec.
