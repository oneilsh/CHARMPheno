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

## 2026-05-04 to 2026-05-05 — Dataproc/BigQuery cluster bring-up walkthrough

A six-lesson walkthrough of the cluster bring-up workstream that landed
between the prior log entry and this one — the MLlib Estimator/Transformer
shim, the BigQuery OMOP loader, the cloud diagnostics tooling, and the
per-iteration callback hook. Outside-in framing (cluster reality first,
framework contract last) chosen because the user had just exercised the
pipeline live and the lessons could anchor on observed behavior. Several
small fix-ups shipped from a code review that ran at the start of the
session; one new architecture-doc section added.

### Areas reviewed

**Lesson 1 — Cluster bring-up + diagnostics.**
The `--py-files` deployment shape and the driver/executor split that makes
it load-bearing (executor closures fail to import local packages without
the zip ship); why `transform`/UDF is the louder smoke than `fit`/aggregate.
The `spark-submit` invocation flags walked one by one (`--master yarn`,
`--deploy-mode client` and why client mode is required for live UI / port
4040 access, `--driver-memory 4g` motivated by the `concept` join OOM,
`--py-files` zip vs `.py` accepted forms, `bq-smoke` deliberately *not*
using `--py-files` to isolate connector-vs-deployment failures). The two
REST APIs the inspector consumes (Spark `/api/v1/...` on driver port 4040+
or History Server 18080; YARN ResourceManager 8088), the zombie-app
problem (OOM-killed drivers leave `completed: false` event logs forever
since they never write the completion event), the History Server's
GCS-flush ~5 s lag relative to the live driver UI. The polling loop's
three layers (discover → identify → render-on-loop), per-tick app-id
re-detection so the inspector follows new submissions automatically,
`state = {}` reset on app-id change so delta metrics don't span jobs,
`_safe_get` fail-soft pattern catching only `_NETWORK_ERRORS` not
`Exception`. **Side discussion** — `SparkSession.builder.getOrCreate()`
as the universal idiom that works in raw spark-submit and managed
notebooks alike, and the YARN-vs-K8s split where YARN is one of several
Spark cluster managers and the cluster outlives any individual session.

**Lesson 2 — Spark mechanics in production.**
Persist mechanics: lazy evaluation + actions, the
`MEMORY/DISK/DESERIALIZED/replication` storage-level taxonomy, default
`MEMORY_AND_DISK` for DataFrames vs `MEMORY_ONLY` for raw RDDs,
[`_log_persist`](../analysis/cloud/lda_bigquery_cloud.py#L90-L110) as a
truthing pattern. Caveat surfaced — `df.storageLevel` reports the
*requested* level, not actual block materialization; the truer check goes
through `getRDDStorageInfo().numCachedPartitions()`. Broadcast lifecycle
in production: what a broadcast actually is (driver `BroadcastManager` +
per-executor block-manager copies), the leak class (~32 MB/iter for
K=20, V=2000 on each executor; scales superlinearly), why it doesn't
crash with OOM (block manager has its own budget; spills silently to
disk), the `prior_bcast` ratchet, the convergence-path explicit
unpersist, `unpersist(blocking=False)` semantics. Tree-reduce vs
collect: three aggregation shapes (collect+fold, reduce, treeReduce) and
their driver-memory profiles, why associativity *and* commutativity are
required for tree-shape, why `mapPartitions` returns `[stats]`
(one-element list) not bare dict.

**Lesson 3 — The BigQuery connector.**
Connector mechanics: Storage Read API gRPC path vs the Query API path,
partition-per-stream where BQ chooses parallelism rather than Spark,
predicate pushdown gotchas (esp. `MOD` not reliably pushing down in
older connector versions, `concept_id != 0` reliably does), `parentProject`
for the data-vs-billing-project split common in restricted-data
environments. Join shape: three strategies (broadcast hash, shuffle hash,
sort-merge), `autoBroadcastJoinThreshold` (10 MB default), why explicit
`F.broadcast(concept)` OOM'd the driver (forces collect-to-driver before
broadcast), `F.broadcast` as "I know better than the planner" hint that
should be used only when measured. Boundary discipline:
`select(*CANONICAL_COLUMNS, ...)` + `validate()` as the loader's final
act, schema-drift firewall pattern.

**Lesson 4 — MLlib Estimator/Model contract.**
Why MLlib splits Estimator/Model: pipelines, persistence, type safety;
for our shim only pipelines load-bear in v1 (persistence deferred per
ADR 0009). Param system: class-level descriptors with type converters,
why MLlib uses them (introspection, cross-language consistency,
persistence), `HasFeaturesCol/HasMaxIter/HasSeed` mixin pattern, shared
`_VanillaLDAParams` between Estimator and Model so getters match.
DataFrame ↔ RDD bridge:
[`dataset.select.rdd.map(_vector_to_bow_document)`](../spark-vi/spark_vi/mllib/lda.py#L268-L274),
the UDF-on-executors story for `_transform` (and why `--py-files` is
load-bearing here too). `setOnIteration` as instance attribute not
Param: callables aren't pickleable; the design rule "Param for identity,
instance attr for incidentals (diagnostics, hooks)"; never copied to
the Model in `_fit`. **Clarification** — the deferral in ADR 0009 is
*MLlib-style* `Pipeline.save` / `MLWritable` / `MLReadable`, not
framework persistence; the framework's `save_result` / `load_result`
(ADR 0006) ships and works.

**Lesson 5 — Callback as contract extension.**
The three layers (driver factory → shim instance attr → runner kwarg)
walked as a case study. Contract shape `(iter_num, global_params,
elbo_trace)` framework-level only — domain richness rides via closure
capture. Kwarg-on-fit beats method-on-`VIModel` because the callback is
per-invocation observation, not model state — keeps math
diagnostic-free, allows different fits to opt in differently. Mutation
hazard + why no defensive copy (deep-copy of (K, V) lambda each iter is
too expensive for a diagnostic path; document-the-contract is the
chosen tradeoff). The factory pattern in
[`_make_topic_evolution_logger`](../analysis/cloud/lda_bigquery_cloud.py#L62-L92)
captures domain context (vocab map, concept names, throttle cadence) by
closure rather than widening the framework signature.

**Lesson 6 — The driver as orchestration.**
The driver as the gluing layer (composes spark-vi + charmpheno + Spark +
BQ + MLlib; originates almost no logic). The
[`_phase`](../analysis/cloud/lda_bigquery_cloud.py#L46-L59) context
manager for wall-time attribution as a debugging primitive — 12 lines
that pay for themselves the first time a run is unexpectedly slow. How
the vocab/concept-name dicts thread driver-side through closure capture
into the topic-evolution logger (three small dicts, framework never
sees them, interpretation reconstructed at the boundary only where
humans look). The driver as the implicit end-to-end integration test —
the last reasonable point at which a regression in any layer below can
hide before the user notices.

### Refactor detours that shipped

**Detour 1 — Code-review fix-ups across the bring-up workstream.**
Independent review pass on the diff between the prior log entry and the
walkthrough start surfaced six should-fix items, all small. Type
annotation added to
[`VanillaLDAEstimator.setOnIteration`](../spark-vi/spark_vi/mllib/lda.py#L243-L256)
to match the runner's `Callable[[int, dict, list[float]], None] | None`.
Mutation-safety caveat added to both
[`runner.fit`](../spark-vi/spark_vi/core/runner.py#L73-L85) and the
shim's `setOnIteration` docstring (callback must not mutate
`global_params` since the same dict feeds the next iteration's
broadcast). The driver's `_on_iter` swallow tightened from `*_` to
explicit `_: list[float]` so a contract change would break loudly
rather than silently. AQE-vs-broadcast comment in
[`bigquery.py`](../charmpheno/charmpheno/omop/bigquery.py#L94-L97)
corrected to mention shuffle-hash as a possibility and to anchor on the
explicit-broadcast OOM as the *why*. BQ predicate-pushdown comment
softened from "pushed down" to "depends on connector version" since we
hadn't verified. `_NETWORK_ERRORS` tuple in
[`inspect_app.py`](../analysis/cloud/inspect_app.py#L92) hoisted above
`find_app_id` and reused in place of the inline catch tuple.

**Detour 2 — Closure-capture / kwarg-on-fit pattern documented inline.**
Triggered by the user request after Lesson 5: capture the contract
extension pattern in code so future readers don't have to reverse-
engineer it. Added a short paragraph to
[`_make_topic_evolution_logger`](../analysis/cloud/lda_bigquery_cloud.py#L62-L72)
explaining factory-vs-bare-def as closure capture for narrow framework
contracts. Extended the `on_iteration` parameter doc in
[`runner.fit`](../spark-vi/spark_vi/core/runner.py#L73-L85) with the
kwarg-on-fit-rather-than-`VIModel`-method rationale and the deliberate
no-defensive-copy choice. Both edits are docstring-only, no behavior
changes.

**Detour 3 — `Data Sources: BigQuery` section in framework doc.**
Triggered during Lesson 3. Added a five-subsection block to
[`SPARK_VI_FRAMEWORK.md`](architecture/SPARK_VI_FRAMEWORK.md):
Storage Read API path; read-side billing routing via `parentProject`;
predicate pushdown coverage (what does and doesn't reliably push down,
verification via connector INFO logs); partitioning and clustering
awareness for OMOP-shaped event tables and what that implies for sampler
design; a short "what we don't currently use" pointer (BQ-side
pre-aggregation, materialized views, `INFORMATION_SCHEMA.JOBS` for cost
attribution). Framing kept generic — restricted-data-environment
patterns rather than naming any specific provider.

### Pre-existing issues caught and noted

- The shim's earlier `setOnIteration` signature was effectively untyped;
  static checkers and IDE tooltips would silently accept any object.
  Now annotated.
- `bigquery.py` claimed predicate pushdown without verification; corrected
  to a softer claim, with a recipe for verifying via connector INFO logs.
- `bigquery.py` AQE comment overstated the broadcast lockout; broadcast
  is still possible at runtime via post-shuffle stats, just not via the
  explicit hint. Comment now reads correctly.

### Open threads parked

These are not regressions or known bugs — they are deferred opportunities
noted during this session.

- **Drop the History Server fallback from `inspect_app.py`.** The user
  decided live-driver-only is the right surface for the inspector;
  filtering zombies via the `_ZOMBIE_STALE_MS` recency check patches a
  symptom of consulting the History Server at all. Removing the fallback
  collapses `discover_spark_base`, deletes `_ZOMBIE_STALE_MS` and the
  staleness-filter logic, and simplifies `find_app_id`. Deferred until
  the cluster is back up to test against.
- **Promote `_log_persist` to a strict precondition.** The current diagnostic
  reports the *requested* persist level, not whether blocks materialized;
  a `.persist()` followed by no action passes the check while leaving the
  cache empty. For a loop-heavy training workload, silent persist failure
  causes the upstream lineage (including BigQuery reads) to re-run every
  iteration. Replace with a check on `getRDDStorageInfo().numCachedPartitions()`
  that raises before fit begins. Placement decision pending: probably
  belongs in `spark-vi` as a new `diagnostics/persist.py` so the runner can
  enforce on its own inputs, with the driver script calling it for
  upstream DataFrames as well. Caveat: catches "forgot the action" but
  not mid-fit eviction (executor death, memory-pressure block-manager
  eviction); a re-check between iterations would close that gap but is
  more invasive.
- **MLlib-style persistence (`MLWritable` / `MLReadable` / `Pipeline.save`).**
  Deferred per [ADR 0009](decisions/0009-mllib-shim.md). Implementation
  is mostly translation — walk every `Param`, serialize to MLlib's JSON
  layout, write the trained `VIResult` alongside (or point at a
  `save_result` artifact), implement the matching `_load`. Not blocking
  any current workflow; pick up when someone needs `Pipeline.save` to
  succeed on a pipeline containing `VanillaLDAEstimator`. The diagnostic
  callback's instance-attr-not-Param design keeps it cleanly outside the
  persistable surface either way.

### Doc updates

- [`SPARK_VI_FRAMEWORK.md`](architecture/SPARK_VI_FRAMEWORK.md): new
  "Data Sources: BigQuery" section with five subsections (Storage Read
  API, `parentProject`, predicate pushdown, partitioning/clustering
  awareness, what we don't currently use). TOC updated.
- [`runner.py`](../spark-vi/spark_vi/core/runner.py#L73-L85):
  `on_iteration` parameter doc extended with kwarg-on-fit rationale and
  no-defensive-copy explanation; mutation-must-not-happen rule already
  present, reinforced.
- [`mllib/lda.py`](../spark-vi/spark_vi/mllib/lda.py#L243-L256):
  `setOnIteration` typed; mutation rule noted in docstring.
- [`lda_bigquery_cloud.py`](../analysis/cloud/lda_bigquery_cloud.py#L62-L72):
  `_make_topic_evolution_logger` docstring extended with closure-capture
  pattern paragraph.
- [`bigquery.py`](../charmpheno/charmpheno/omop/bigquery.py#L85-L97):
  predicate-pushdown and AQE-broadcast comments corrected per review.

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
