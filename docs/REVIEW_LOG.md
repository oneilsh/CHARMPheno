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

## 2026-05-01 — Vanilla LDA branch walkthrough

A bottom-up walkthrough of the `vanilla-lda` branch after its initial
implementation entry, focused on framing the design choices for future
maintainers and surfacing methodology lessons from the head-to-head with
Spark MLlib. Five lessons; several small refactor detours shipped during
review; one substantive simulator extension (asymmetric-prior generation
from upstream U-fractions).

### Areas reviewed

**Lesson 1 — `VanillaLDA` math.**
[`spark_vi/models/lda.py`](../spark-vi/spark_vi/models/lda.py) walked end-
to-end. CAVI implicit-φ recurrence (`gamma_d`, `expElogthetad`, `phi_norm`
of length n_unique rather than n_tokens — the Lee/Seung 2001 trick).
Sufficient-statistic accumulation pattern in `local_update`: `expElogbeta`
precomputed once per partition, sparse `lambda_stats[:, doc.indices] +=`
write, three-term ELBO accumulation inline. The post-aggregation
`expElogbeta * target_stats` multiplication in `update_global` as the
deferred second factor of the implicit-φ — MLlib's `*:* expElogbeta.t`
in `OnlineLDAOptimizer.submitMiniBatch` does the same thing structurally.
ELBO three-term decomposition (per-doc data likelihood, per-doc Dirichlet
KL, global Dirichlet KL) and the placement convention (per-record terms
in `local_update`, global-only terms in `compute_elbo`).

**Lesson 2 — Runner contract and capability hooks.**
[`spark_vi/core/model.py`](../spark-vi/spark_vi/core/model.py) /
[`runner.py`](../spark-vi/spark_vi/core/runner.py) optional-capability
pattern: `infer_local(self, row, global_params)` defaults to raise
NotImplementedError with class name; `VIRunner.transform` orchestrates
broadcast → `mapPartitions` → unpersist for any model that supports it.
Pure-function contract (`self` may be read for hyperparameters but never
post-fit state) and why model-vs-result split is what makes
checkpoint/export/resume work cleanly.

**Lesson 3 — Broadcast lifecycle and serialization.**
[`tests/test_broadcast_lifecycle.py`](../spark-vi/tests/test_broadcast_lifecycle.py)
transparent-proxy approach (delegate `.value` to inner real broadcast,
intercept `.unpersist` for counting) with exact-count assertions on the
three runner paths (max-iter fit, convergence-early-exit fit, transform).
Default-arg closure-capture as the Spark-safe convention for
`mapPartitions` closures, with both failure modes named in code (free-var
mutation between def and pickling, cloudpickle nested-scope quirks).

**Lesson 4 — Topic prep + alignment evaluation.**
[`charmpheno/omop/topic_prep.py`](../charmpheno/charmpheno/omop/topic_prep.py)
wrapping `pyspark.ml.feature.CountVectorizer` (string cast, alphabetical-
by-frequency vocab order, dual return for downstream concept-id
reattachment — the `bow_df` + `vocab_map` shape consumed by both
implementations). [`evaluate/topic_alignment.py`](../charmpheno/charmpheno/evaluate/topic_alignment.py)
JS divergence as symmetric-KL, prevalence ordering for the biplot,
`ground_truth_from_oracle` empirical-β pivot from the simulator's
`true_topic_id` column with OOV / out-of-range guards.

**Lesson 5 — MLlib head-to-head and configuration blindness.**
[`charmpheno/evaluate/lda_compare.py`](../charmpheno/charmpheno/evaluate/lda_compare.py)
parity harness; the slow `test_vanilla_lda_matches_mllib_on_well_separated_corpus`
test as the rigorous math-regression gate (~0.01 nats observed, 0.20
threshold). [`analysis/local/compare_lda_local.py`](../analysis/local/compare_lda_local.py)
as iteration driver, not benchmark — three-panel JS biplot (ours vs truth,
mllib vs truth, ours vs mllib) as diagnostic decomposition. The
methodology lesson — math identity → hyperparameter identity → RNG/float
effects, in that order — and the τ₀/κ schedule discovery that fell out of
applying it.

### Refactor detours that shipped

**Detour 1 — Asymmetric-prior simulator (049c084 + e1988b4).**
The HF `lda_pasc` topic_name string `T-<rank> (U <usage>%, H <uniformity>,
C <coherence>)` carries per-topic upstream metadata that the original
`fetch_lda_beta.py` discarded. Extended `fetch_lda_beta.py` with
`parse_topic_metadata` + sidecar `data/cache/lda_topic_metadata.parquet`,
and `simulate_lda_omop.py` with optional `--topic-metadata` flag that
switches θ's prior from symmetric `α_k = θ_α` to asymmetric `α_k = K · θ_α
· Ũ_k` (Ũ = upstream usage renormalized over topics present in β). Total
concentration `α_0 = K · θ_α` invariant preserved so `theta_alpha` keeps
its per-topic meaning. Initial ship had the U/H/C field semantics wrong
(`coherence_h`, `baseline_delta_c`); follow-up commit corrected to
`uniformity_h`, `coherence_c` per upstream methods documentation. Wallach,
Mimno, McCallum 2009 ("Rethinking LDA: Why Priors Matter") cited as the
canonical motivation in [ADR 0008](decisions/0008-vanilla-lda-design.md)
and [TOPIC_STATE_MODELING.md](architecture/TOPIC_STATE_MODELING.md), with
the mechanism difference noted (they learn α via empirical Bayes; we feed
in an external fixed base measure).

**Detour 2 — Match MLlib's learning-rate schedule (ec8e538).**
On long-tailed asymmetric-prior corpora, our default `tau0=1.0, kappa=0.7`
(Hoffman 2013 general SVI) recovered noticeably more rare topics than
MLlib's `tau0=1024, kappa=0.51` (Hoffman 2010 LDA-tuned). Initial
interpretation was an implementation-quality difference. Reading
`OnlineLDAOptimizer.scala` line-by-line confirmed the math is identical;
the divergence was entirely the schedule. Pinned τ₀=1024, κ=0.51 in
`compare_lda_local.py` and the slow parity test for apples-to-apples.
Inline comments in the driver document the empirical regime where the
schedules diverge and the prescription to "tune τ₀/κ per workload at the
call site if you need it."

**Detour 3 — Hungarian re-render for biplots (0325e32).**
On nearly-uniform-prevalence runs (symmetric Dirichlet(0.1·1_K) prior),
the prevalence-ordered biplot's diagonal looks spurious because prevalence
ordering is noise-dominated. Added `optimal_match_reorder(js_matrix)`
using `scipy.optimize.linear_sum_assignment` for post-hoc Hungarian
matching when the prevalence signal is flat. Lazy import of scipy keeps
the base evaluate path scipy-free. Two tests: known-permutation recovery,
and brute-force-over-all-perms minimization check.

**Detour 4 — Doc + test clarifications across spark-vi (ad86d7b).**
Inline comments on default-arg closure-capture in
[`runner.py`](../spark-vi/spark_vi/core/runner.py#L131); ELBO-term
placement pattern paragraph in
[`SPARK_VI_FRAMEWORK.md`](architecture/SPARK_VI_FRAMEWORK.md) and
[`core/model.py`](../spark-vi/spark_vi/core/model.py); attribution of
`gamma_shape=100` to Hoffman 2010's `onlineldavb.py` (and MLlib's adoption
of the same value as a private constant) in
[`lda.py:initialize_global`](../spark-vi/spark_vi/models/lda.py); and a
new test `test_vanilla_lda_update_global_uses_input_lambda_for_expElogbeta`
in [`test_lda_contract.py`](../spark-vi/tests/test_lda_contract.py) that
breaks the lr=1 special case with non-uniform input λ to isolate the
reference frame of the `expElogβ` factor (the ADR-0008 bug regression
guard, sharper than the earlier surrogate test).

### Methodology lessons surfaced

**Configuration blindness in head-to-head comparisons.** The single
biggest takeaway: when comparing two reference implementations of the
same algorithm, hyperparameters left at *default* on each side are
silent confounders. For SVI-LDA the relevant invisible knobs are the
learning-rate schedule (τ₀, κ), `optimizeDocConcentration`,
`optimizeTopicConcentration`, `gammaShape`, and the RNG seed. Each is
defensible as a default in isolation; defaults from different libraries
combined produce different objectives without a single line of code that
looks wrong. Procedural fix: walk the full reference parameter API once,
classify each knob as matched-explicitly / left-default-on-purpose / not-
applicable, before reading the comparison output.

**Math identity → config identity → numerics, in that order.** When
implementations disagree, reading the reference's source code (here,
`OnlineLDAOptimizer.scala`) to settle "is the math the same?" is a five-
minute exercise that prevents hours of speculation about implementation-
quality differences. Skipping straight to "the implementations differ in
some deep way" is a seductive failure mode because the hypothesis is
*interesting*; it's almost never the right answer. Math identity is
cheap to check and decisive.

**Aggressive early SVI steps can beat warmup on long-tailed corpora.**
Counter-intuitive but observed and now documented: on asymmetric-prior
data, a τ₀=1 schedule (which fully replaces λ on iteration 0) gives rare
topics more chance to differentiate before the loss surface settles, vs.
τ₀=1024 (which lets dominant topics consume rare topics' evidence during
the gentle warmup). The parity test pins MLlib's schedule because the
contract is apples-to-apples; recovery-quality runs in
`compare_lda_local.py` should pick the schedule per workload.

### Doc updates

- [ADR 0008 — Vanilla LDA design](decisions/0008-vanilla-lda-design.md):
  Wallach 2009 reference added to the asymmetric-α deferral section, with
  the empirical-Bayes-vs-fixed-base-measure mechanism distinction
  explicit.
- [TOPIC_STATE_MODELING.md](architecture/TOPIC_STATE_MODELING.md):
  Wallach 2009 in References → Topic Models alongside Blei 2003 and
  Hoffman 2010.
- [SPARK_VI_FRAMEWORK.md](architecture/SPARK_VI_FRAMEWORK.md): ELBO-term
  placement pattern paragraph under `compute_elbo`.

### Open threads parked

- **MLlib Estimator/Transformer compatibility shim** — slated as the next
  major work item, before OnlineHDP. The shim should let users pass a
  DataFrame with a `features` column and receive a `Pipeline`-shaped
  fitted model, exposing MLlib-named hyperparameters (`docConcentration`,
  `topicConcentration`, `learningOffset`, `learningDecay`,
  `subsamplingRate`, etc.) as pass-through. Nothing inside `VIModel` or
  `VIRunner` needs to change; the shim is a wrapper layer.
- **Empirical Bayes on α** (still per ADR 0008). The Newton step on the
  Dirichlet-concentration log-likelihood has a diagonal-plus-rank-1
  Hessian (Minka 2000), so the K-dimensional update is O(K) per step
  with Sherman-Morrison. Cheap enough to interleave with each SVI batch
  if we choose to ship it — but adding it requires the framework's
  `update_global` to accommodate non-conjugate gradient updates
  alongside the existing closed-form-conjugate ones, which is a
  meaningful contract change.
- **Concentration parameters as variational random variables** in
  OnlineHDP — γ and α are model-complexity-controlling and can't
  reasonably be left fixed in a non-parametric model, so OnlineHDP will
  fold q(γ), q(α) into the SVI ELBO via Gamma-hyperprior + non-conjugate
  natural-gradient steps (Wang, Paisley & Blei 2011). The framework
  extension for non-conjugate updates flagged above is a prerequisite.

---

## 2026-04-30 — Vanilla LDA implementation

A real multi-parameter VIModel ships, exercising the framework end-to-end
against synthetic data with known ground truth and a head-to-head
comparison against Spark MLlib's reference implementation.

### Components shipped

- **`spark_vi/models/lda.py`** — Hoffman 2010 Online LDA + Lee/Seung 2001
  implicit-phi trick. Symmetric alpha. Hyperparameters default-matched to
  MLlib's `pyspark.ml.clustering.LDA` for fair comparison.
- **`spark_vi/core/types.py`** — `BOWDocument` canonical bag-of-words row
  type for topic-style models.
- **`spark_vi/core/model.py`** + **`runner.py`** — optional `infer_local`
  capability + `VIRunner.transform` orchestrator. See ADR 0007.
- **`charmpheno/omop/topic_prep.py`** — `to_bow_dataframe` (OMOP -> BOW
  via `pyspark.ml.feature.CountVectorizer`).
- **`charmpheno/evaluate/topic_alignment.py`** — JS divergence,
  prevalence ordering, biplot data, `ground_truth_from_oracle`.
- **`charmpheno/evaluate/lda_compare.py`** — `run_ours` / `run_mllib`
  head-to-head harness.
- **`analysis/local/fit_lda_local.py`** + **`compare_lda_local.py`** —
  drivers; comparison driver renders three-panel JS biplot.

### New ADRs

- [0007 — VIModel inference capability](decisions/0007-vimodel-inference-capability.md)
- [0008 — Vanilla LDA design choices](decisions/0008-vanilla-lda-design.md)

### Doc updates

- `SPARK_VI_FRAMEWORK.md` — `VanillaLDA` entry, `infer_local` documented,
  `VIRunner.transform` paragraph.
- `RISKS_AND_MITIGATIONS.md` — "MLlib parity expectations" entry plus
  "Small-corpus topic collapse in SVI" entry.

### What broke and how we caught it

Initial integration testing surfaced what looked like generic small-
corpus seed-fragility: `lambda.sum(axis=1)` an order of magnitude too
high, several seeds producing 0-2 collapsed topics, best-permutation JS
divergence ~0.25 nats. Pulled out of auto mode for diagnosis.

Root cause was a missing factor in `update_global`: the CAVI implicit-
phi parameterization is `phi_dnk ∝ expElogthetad[k] * expElogbeta[k, w_dn]`,
and our per-doc accumulation in `local_update` captured only the first
factor. The aggregated sufficient statistic must be re-multiplied by
`expElogbeta` (computed from the *current* lambda) before the Robbins-
Monro step. MLlib does this with a single post-aggregation
`*:* expElogbeta.t` in `OnlineLDAOptimizer.submitMiniBatch`; we now match.

Lesson: small-synthetic-corpus topic collapse is real but is also exactly
the failure mode a math regression mimics. The MLlib parity test (Task 15)
is the rigorous gate that distinguishes the two: with matched hyperparameters
and `optimizeDocConcentration=False`, our diagonal mean JS vs MLlib runs
~0.01 nats. The fragility-prone synthetic-recovery test originally proposed
in Task 12 was dropped in favor of an ELBO-trend smoke test plus the parity
gate.

Captured in [ADR 0008](decisions/0008-vanilla-lda-design.md) and
[`RISKS_AND_MITIGATIONS.md`](architecture/RISKS_AND_MITIGATIONS.md).

### Open threads parked

- Asymmetric alpha + `optimizeDocConcentration` Newton-Raphson update.
- Per-iteration ELBO trace from MLlib.
- LDA notebook tutorial.
- Several Type-hint / test-hygiene minor items captured in the Task 22
  / final cleanup notes of the implementation plan; non-blocking.
- The real `OnlineHDP` (this was the warm-up).

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
