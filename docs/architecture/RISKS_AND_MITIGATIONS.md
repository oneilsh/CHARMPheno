# Risks and Mitigations

This document summarizes known technical risks identified during design review,
organized by component. It also captures a key strategic observation: the milestone
deliverables do not depend on the OU process or continuous-time dynamics, giving
flexibility to simplify Stage 2 without jeopardizing the delivery plan.

---

## Strategic Context

The milestones (C3.T3b / C3.T4b) require "unsupervised phenotype discovery and
outcome characterization" using a "Bayesian state modeling framework." The HDP topic
model + spark-vi framework satisfy these requirements directly. The OU process and
temporal dynamics are valuable research directions but are not load-bearing for any
deliverable. This means:

- The framework and HDP can be implemented with confidence as the core deliverables.
- Stage 2 can be as simple as clustering on θ vectors or logistic regression on
  θ-derived features for outcome characterization.
- More ambitious dynamics models (OU, VAR, nonlinear drift) can be pursued
  opportunistically if the simpler work lands early, or reserved for publications.

The one recurring milestone phrase that needs attention is **"outcome
characterization"** — connecting discovered phenotypes to clinical outcomes
(readmission, mortality, etc.). This does not require a generative dynamics model;
supervised methods on HDP-derived features suffice.

---

## Component Risk Summary

| Component | Confidence | Notes |
|---|---|---|
| spark-vi framework | **High** | Proven distributed pattern (MLlib OnlineLDA), minimal abstractions |
| Online HDP (Stage 1) | **High** | Published method, existing implementations, known to work at scale |
| Two-stage architecture | **High** | Clean interface (θ vectors), stages independently swappable |
| Sparse OU (Stage 2, as designed) | **Moderate** | Implementable but multiple sharp edges compound |
| VB with horseshoe/spike-and-slab on OU | **Lower** | Research-level inference problem, no off-the-shelf recipe |
| Per-document θ quality → Stage 2 signal | **Lower** | Silent load-bearing assumption of the whole pipeline |

---

## Framework Risks

### Sparse gradient dictionaries don't match the framework contract

**Risk:** The `local_update` return type is `dict[str, np.ndarray]` with default
`combine_stats` doing elementwise array summation. The OU model's sparse gradient
dictionaries (keyed by topic-pair tuples) don't fit this contract.

**Impact:** Only affects OU or similarly sparse models. The HDP uses dense sufficient
statistics and fits the contract naturally.

**Mitigation:** Either override `combine_stats` for sparse models, or use scipy sparse
matrices that support addition. A minor contract clarification, not a redesign.

### Partition-stats aggregation

**Risk:** Naively pulling all per-partition outputs to the driver via `RDD.collect()`
puts driver memory under pressure proportional to `N_partitions × per_partition_size`.
For an HDP-scale `(K, V)` matrix at ~84 MB across 10 partitions, that is ~840 MB on
the driver — marginal at typical Spark driver sizes (1 GB) and infeasible on smaller
drivers.

**Impact:** OOM on the driver during long runs at production scale; silent slowness
even when memory does not run out (the driver-side fold is single-threaded Python
arithmetic over potentially many partition outputs).

**Mitigation:** `VIRunner` uses `RDD.treeReduce(model.combine_stats)` rather than
`collect()` + driver-side fold. Driver memory is bounded by a single merged stats
dict rather than the sum of all partition outputs; the reduction itself happens
cluster-side in `O(log N_partitions)` rounds. This matches MLlib `OnlineLDAOptimizer`'s
`treeAggregate` pattern. Custom `combine_stats` implementations must remain
associative and commutative — the existing contract requirement for additive
sufficient statistics.

### Broadcast lifecycle

**Risk:** Each iteration creates a new Spark broadcast variable (~84MB for HDP at
T=150, V=70K). Without explicit `unpersist()` of old broadcasts, they accumulate and
eventually cause OOM.

**Impact:** Affects any model with large broadcast parameters during long training runs.

**Mitigation:** Add `unpersist()` call in the training loop before creating each new
broadcast. Standard PySpark practice — just needs to be in the framework code.

### Robbins-Monro schedule assumes stochastic updates

**Risk:** The default decaying learning rate is justified by stochastic VI theory
(Hoffman et al., 2013), which assumes updates on data subsets. Models that process all
data every iteration (full-batch) will see the step size decay toward zero, stalling
before convergence.

**Impact:** Affects models run with `mini_batch_fraction=None` (e.g., the OU model
over all patients). HDP and other stochastic-VI models opt in to mini-batching via
`VIConfig.mini_batch_fraction`, which puts them in the regime the Robbins-Monro
schedule was designed for; see ADR 0005.

**Mitigation:** Two paths. (1) Set `VIConfig.mini_batch_fraction` to a non-None
value to make iterations stochastic, validating the schedule's assumptions. (2) For
full-batch models, override `kappa` toward 1.0 to slow the decay, or eventually
extend the framework to support model-specified learning-rate schedules.

### Mini-batch sampling: variance, reproducibility, and overhead

**Risk 1 (semantics):** With `sample_with_replacement=True` (default, matching
MLlib `OnlineLDAOptimizer`), a single document can appear multiple times in one
batch. This is mathematically correct for SGD on i.i.d. samples but can surprise
readers used to dataset traversal semantics.

**Risk 2 (reproducibility):** Per-iteration sampling makes runs non-reproducible
unless `VIConfig.random_seed` is set. Even with a seed, Spark partition ordering
and floating-point summation order can introduce small numerical variance across
runs on different cluster shapes — reproducibility is best-effort, not bit-exact.

**Risk 3 (compute and memory):** Each mini-batch iteration calls `batch.count()`
(one extra Spark action) and persists the sampled batch RDD with
`StorageLevel.MEMORY_AND_DISK` so that `count()` and the subsequent
`mapPartitions` reuse the same partitions. Without persistence the sample
lineage would recompute twice per iteration. The cache cost is bounded by the
size of one batch; for very large batches Spark may spill to disk, which is
slower than memory-only but does not fail.

**Impact:** Low for HDP at planned scale. Mini-batches of ~5% of a multi-million
visit corpus fit comfortably in worker memory; the extra `count()` action is
small relative to the `mapPartitions` work that follows.

**Mitigation:** Set `random_seed` whenever reproducibility matters during
development. For production runs, accept the slight per-iteration variance —
it is the same regime MLlib has run in production for years.

### `data_summary` for `initialize_global` is underspecified

**Risk:** The pre-pass that computes `data_summary` has no declared interface — models
can't specify what summary statistics they need.

**Impact:** Low. Each model will need different summaries (HDP needs vocab_size, OU
needs dimensionality), but the implementation can simply compute a superset or let
models define a `compute_data_summary` method.

**Mitigation:** Define during implementation. Not an architectural issue.

---

## MLlib parity expectations

**Risk:** Head-to-head comparisons of `OnlineLDA` against
`pyspark.ml.clustering.LDA` may produce numerically different topic-word
matrices and document-topic distributions even when the math is
implemented correctly. Treating numerical equality as a correctness gate
would generate false alarms.

**Sources of legitimate divergence:**

- Different RNG state and seeding semantics (Mersenne Twister vs.
  numpy default; different per-partition seed derivations).
- Different mini-batch sampling implementations (Spark's `RDD.sample`
  vs. our wrapper).
- Different CAVI per-document iteration counts when `cavi_tol` is hit at
  slightly different rates.
- Different float precision in places (Breeze BLAS vs. NumPy BLAS).
- Asymmetric alpha optimization is on by default in some MLlib paths;
  ours always uses symmetric alpha.

**Mitigation:** The agreement gate is prevalence-aligned topic similarity:
rank topics in each implementation by their corpus-level total mass
(sum of theta over docs), align by descending prevalence, and compute
the mean Jensen-Shannon divergence on the diagonal of the K_ours x K_mllib
matrix. Diagonal-dominance after this ordering = topic agreement; the
threshold for "comparable" is a domain decision (current rule of thumb:
< 0.15 nats on the synthetic recovery test). Off-diagonal smear localizes
split/merge differences and is informative, not pass/fail.

**See also:** [ADR 0008](../decisions/0008-vanilla-lda-design.md),
`charmpheno/charmpheno/evaluate/topic_alignment.py`.

## Small-corpus topic collapse in SVI

**Risk:** Stochastic Variational Inference for LDA (and HDP) is prone to
empirical topic collapse on small synthetic corpora — entire topics whose
lambda barely moves from the prior, while one or two topics absorb most of
the corpus mass. This is a known SVI characteristic, not a math bug:
Hoffman 2010 §4 evaluates on corpora of 100K-352K documents for a reason.

**Symptoms:** several `lambda.sum(axis=1)` rows near `eta * V` (the prior),
with one or two rows much larger; mean diagonal JS divergence > 0.20 nats
on a synthetic recovery test that should comfortably pass.

**Mitigation:** Do NOT use synthetic recovery on a tiny fixture as the
correctness gate; small-corpus collapse is seed-fragile and produces false
alarms. The MLlib parity test in `charmpheno/tests/test_lda_compare.py` is
the rigorous gate — running both implementations on the same input under
matched hyperparameters localizes any regression to our side. For real
analysis, use corpora of D >= 1000 patients per topic and a hot init
(set `gamma_shape >> K`) if collapse is observed.

**See also:** [ADR 0008](../decisions/0008-vanilla-lda-design.md);
`spark-vi/tests/test_lda_integration.py` for the design choice that
dropped the synthetic-recovery test in favor of an ELBO-trend smoke test.

---

## HDP Risks

### Per-document θ estimates are noisy with short documents

**Risk:** With 5-20 diagnosis codes per visit and K=300 topics, individual θ posteriors
are diffuse and dominated by the prior. Point estimates of θ are unreliable for any
single visit.

**Impact:** Global HDP parameters (topic-word distributions) converge fine — they see
aggregated statistics across millions of visits. But per-document θ vectors passed to
Stage 2 carry substantial noise. If Stage 2 is a dynamics model, this noise inflates
the estimated diffusion and can wash out real signal. If Stage 2 is simple clustering,
the impact is mitigated by aggregating θ across visits per patient.

**Mitigation:**
- For simple Stage 2 approaches (clustering, regression): average θ across visits per
  patient, which reduces noise substantially.
- For dynamics models: propagate posterior uncertainty from HDP (sample multiple θ
  draws, or incorporate variational variance into the observation model). This is noted
  as future work in the research design.

---

## OU Process Risks (Stage 2, if pursued)

These risks are relevant only if the OU process is implemented. They do not affect the
framework or HDP. All are confined to the OU model's `local_update` and
`global_update` — the framework contract does not need to change.

### ILR transform with near-zero topic proportions

**Risk:** The ILR transform computes log-ratios of θ entries. Sparse θ vectors (most
entries near zero) produce extreme negative values after the log, dominating the
transformed vector and destabilizing OU estimation. This is the well-known "zero
problem" in compositional data analysis. Additionally, ILR-transformed sparse
compositions are heavily non-Gaussian, violating the OU transition density assumption.

**Impact:** High. This is the most numerically fragile point in the pipeline and could
force a rethink of the transform approach.

**Mitigation options:**
- **Multiplicative/additive replacement** before the log — standard but introduces a
  tuning parameter that affects downstream dynamics.
- **Transform only active topics per patient** — aligns with the sparse subblock
  approach but puts different patients in different coordinate subspaces.
- **Skip ILR entirely** — use a dynamics model that operates on the simplex directly
  (Dirichlet diffusion) or on raw θ (discrete-time VAR).
- **CLR instead of ILR** — same zero problem but preserves original topic coordinates,
  simplifying interpretation.

### ILR basis choice affects sparsity assumptions

**Risk:** Sparsity of A under one ILR basis does not correspond to sparsity under
another. The assumption that A is sparse is actually an assumption about interaction
structure in a specific, unspecified coordinate system.

**Impact:** Medium. Could produce spurious sparsity patterns or miss real interactions
depending on the (arbitrary) basis choice.

**Mitigation:** Use CLR (which preserves the original topic coordinates) or document
and justify the basis choice.

### Matrix exponential instability during estimation

**Risk:** `expm(A * dt)` requires all eigenvalues of A to have negative real parts. During
iterative estimation, intermediate A values may have positive eigenvalues. Combined
with large dt (years-long gaps between visits), this produces overflow, NaN gradients,
and cascading optimization failure.

**Impact:** Medium-high during estimation; manageable after fitting.

**Mitigation:** Project A onto the stable cone after each global update —
eigendecompose, clamp positive real parts to a small negative value, reconstruct. One
eigendecomposition per iteration on the driver, negligible cost.

### VB vs. L1-penalized MLE inconsistency

**Risk:** The research design argues for variational Bayes with horseshoe/spike-and-slab
priors. The technology stack table and the framework document describe L1-penalized
MLE. These are fundamentally different methods with different sufficient statistics,
different global updates, and different outputs.

**Impact:** High for planning. An implementer following both documents gets
contradictory specifications.

**Mitigation:** Resolve before implementation. L1-penalized MLE is substantially safer
to implement first. VB with continuous shrinkage priors on OU drift matrices is a
research-level problem with no off-the-shelf recipe — pursue as an upgrade after
validating the pipeline with MLE.

### "Active topics per patient" is underestimated

**Risk:** The documents use 5-20 active topics per patient, but this is per-visit
sparsity. A patient's active set is the union across all visits — easily 40-60 for
patients with long histories. This inflates subblock sizes and matrix exponential costs.

**Impact:** Medium. Per-patient computation is ~4-16x more expensive than estimated,
but still parallelizable.

**Mitigation:** Measure on real HDP output before committing to complexity assumptions.
The subblock approach still works, just with larger blocks.

### Sparse subblock is an approximation

**Risk:** The subblock approach ignores interactions where topic j influences topic i
but j is never active for a given patient. The documents present this as exact
computation.

**Impact:** Low-medium. A reasonable approximation — topics that are never observed for
a patient contribute no likelihood signal anyway — but should be documented as an
approximation, not exact.

### Sparse gradient aggregation scales poorly

**Risk:** As gradient dictionaries merge up the treeAggregate tree, they grow toward the
union of all keys. The final merged dictionary may approach dense K² entries,
contradicting the "nothing large moves across the network" claim. Python dict merging
is also much slower than NumPy array addition.

**Impact:** Medium. Could be the actual bottleneck of Stage 2.

**Mitigation:** Use scipy sparse matrices instead of Python dicts for gradient
accumulation. Or accept a dense gradient for the global A update — at K=300, a
90K-entry array is ~700KB, which is fine.

### Scaling estimates are optimistic

**Risk:** The documents characterize matrix exponential computation as "negligible" and
network transfer as "nothing large." Actual estimates: ~140M expm calls per iteration
(~1400 CPU-seconds single-threaded), gradient dicts growing toward dense during
aggregation. Stage 2 could be 10-100x slower than implied.

**Impact:** Not a blocker on a cluster, but sets wrong expectations for development
iteration speed and cost planning.

**Mitigation:** Set realistic expectations. Profile early on synthetic data.

---

## Azure Synapse Platform Risks

The implementation target is Truveta's environment on Azure Synapse Analytics. The
core PySpark pattern (broadcast → mapPartitions → treeAggregate) is fully supported —
Synapse runs standard Apache Spark 3.3/3.4 and does not restrict RDD-level operations.
NumPy and SciPy are pre-installed in the Synapse Spark runtime. The risks are
operational, not API-level.

### Session stability for long training runs

**Risk:** Synapse Studio notebooks lose browser WebSocket connections after 2-4+ hours
of continuous execution. The Spark job continues running on the cluster, but live
notebook output is lost — defeating the notebook-first diagnostics design. Reconnecting
shows results after completion but not the live training display.

**Impact:** High for development workflow. The framework's live training display
(convergence plots, ELBO tracking) is designed for interactive notebook use. Training
runs of 100+ iterations on billions of visits will likely exceed the stability window.

**Mitigation:** The framework's explicit batch mode (`output="batch"`) routes metrics to
Python logging, which is unaffected by browser disconnects. Batch Spark job submission
(Spark Job Definitions, Livy API) is a standard Synapse feature but likely not
available in Truveta's managed environment. The primary mitigation is robust
checkpointing (see below) so that session drops don't lose progress. Use
`client.stop_session()` to release resources promptly after runs complete.

### No built-in checkpointing

**Risk:** If a Spark session or kernel dies mid-training, all Python state (including
learned global parameters) is lost. Synapse does not provide automatic checkpointing of
application-level state.

**Impact:** High for long training runs. A multi-hour training run that fails at
iteration 90 of 100 loses all progress.

**Status: Resolved as of ADR 0006.**

- `VIConfig.checkpoint_interval` paired with `VIConfig.checkpoint_dir` triggers
  `VIRunner.fit` to auto-save a `VIResult` every N iterations during the run. The
  two fields are coupled — `__post_init__` raises if exactly one is set.
- `VIRunner.fit(rdd, resume_from=path)` loads a saved `VIResult` (whether written
  by an explicit `save_result` call or by the auto-checkpoint mechanism) and
  continues training with the Robbins-Monro counter preserved, so the resumed run
  produces the same final state as a continuous run of equal total length.
- The on-disk format is platform-agnostic — `manifest.json + params/*.npy` to any
  filesystem path. On Truveta, this path can be under
  `study.get_artifacts_path(fs=True)`, which presents study-level persistent
  storage as a regular filesystem path. On other platforms, any local or mounted
  path works.

### Truveta resource pool constraints

**Risk:** Truveta provides per-user Spark resource pools of up to 10 VMs, each with
8 cores and 56GB usable memory. The "Premium" tier is likely 1 driver + 9 executors
(72 cores, ~500GB total executor memory). This is a fixed ceiling — there is no option
to scale beyond 10 nodes.

**Impact:** Medium. For the HDP with stochastic mini-batches (the intended online VI
approach), 9 executors is sufficient — mini-batches of 50K-500K visits process in
seconds, and the stochastic natural gradient converges over many fast iterations. For
full-batch models like the OU process (if pursued), processing all patients every
iteration on 9 executors would be slow but not infeasible. The larger concern is
development iteration speed: 3-5 minute cold starts per session, and full-corpus passes
take minutes even with mini-batching.

**Mitigation:** During development, start with small T (150 vs. 300) and moderate V
(10-20K frequent codes) to keep broadcast sizes small and iterations fast. Scale up for
production runs. Mini-batch stochastic VI is essential — full-pass iterations on
billions of visits would be too slow on this cluster.

### Development environment strategy

**Context:** Most development and debugging will happen locally (e.g., MacBook Pro)
using PySpark in local mode against small public datasets like MIMIC. Truveta's
Synapse environment is for production-scale runs on real clinical data.

**Risk:** Low. PySpark's local mode uses the same API surface as a distributed cluster
— `broadcast`, `mapPartitions`, `treeAggregate` all work identically. The framework's
pure-PySpark design (no Scala, no JVM jars, no cluster-specific dependencies) means
code developed locally translates directly to Synapse without modification.

**Remaining gaps:**
- Performance characteristics differ: operations that are instant locally (broadcast of
  a small test dataset) may reveal overhead at scale (120MB broadcast × 100 iterations).
  Profiling on Synapse with realistic data sizes is still necessary before production
  runs.
- Truveta-specific setup (SDK initialization, data loading, package deployment) belongs
  in notebook headers, not in the framework code itself.

### Broadcast variable GC pressure in tight loops

**Risk:** Repeatedly calling `unpersist()` and `sc.broadcast()` on 84-120MB arrays
every iteration can cause garbage collection pauses, especially with Synapse's default
JVM memory settings.

**Impact:** Low-medium. Manifests as occasional slow iterations rather than failure.

**Mitigation:** Use `broadcast.unpersist(blocking=True)` to ensure cleanup completes
before re-broadcasting. Monitor GC time in Spark UI. If problematic, increase driver
memory via `%%configure` or pool settings.

### Custom package deployment

**Risk:** Truveta restricts outbound network access — `pip install` from PyPI is not
available. NumPy and SciPy are pre-installed in the runtime, but deploying the
`spark-vi` package itself requires a workaround.

**Impact:** Low. A one-time setup concern, not an ongoing risk.

**Mitigation:** Package `spark-vi` as a `.zip` archive, upload to study artifacts, and
load via `sys.path.insert` + `spark.sparkContext.addPyFile`. This pattern is proven —
it was used successfully for the `brpmatch` package (see `BRPMatch.ipynb`):

```python
import sys
path = study.get_artifacts_path() + '/public/spark_vi.zip'
sys.path.insert(0, path)
spark.sparkContext.addPyFile(path)  # ships zip to all executors
from spark_vi import VIRunner, OnlineHDP, VIConfig
```

The `addPyFile` call ensures workers can import the package, not just the driver. The
framework should be structured so that a simple zip of the source directory works (flat
package layout, no compiled extensions, no build-time code generation).

---

## Milestone Risk Assessment

| Milestone | Target | Risk | Notes |
|---|---|---|---|
| 27mo | Framework + strategy design | **Low** | Substantially complete (this document set) |
| 30mo | Framework + initial models + synthetic validation | **Low-Medium** | Framework + HDP are safe; OU synthetic validation depends on resolving ILR/VB questions, but OU is not required |
| 33mo | Additional models + benchmarking | **Low** | A clustering or mixture model on θ exercises the framework without OU complexity |
| 36mo | Apply to clinical data | **Medium** | θ noise issue becomes visible here; outcome characterization needs at least supervised methods on θ features |
| 39mo | Integration architecture design | **Low** | Model export format is well-designed |
| 42mo | Model hosting integration | **Low** | Standard deployment engineering |
| 45mo | Outcome prediction + on-device serving | **Medium** | On-device runtime is underspecified |
| 48mo | Hardened platform | **Low** | Depends on scope decisions made at 45mo |

**Overall:** The critical path runs through the framework and HDP, both of which are
high-confidence. Stage 2 complexity is elective — the milestones can be met with
simpler approaches, preserving ambitious dynamics modeling as optional research upside.



## Novelty and Prior Art Risk

### Reviewer comparison to existing distributed VI frameworks

**Risk:** A reviewer familiar with the AMIDST toolbox (Masegosa et al., 2019) or
general-purpose PPLs (Pyro, NumPyro) could argue that spark-vi duplicates existing
work. AMIDST implemented the same broadcast→aggregate→update pattern on Spark/Flink
for conjugate exponential family models (GMMs, HMMs, Kalman filters, LDA) and
published in *Knowledge-Based Systems*.

**Impact:** Medium. Could weaken the contribution narrative if the framework is
presented as the primary novelty.

**Mitigation:** AMIDST has been dormant since 2018 and its creator has moved to other
research (learning theory, variational structure learning). More substantively, AMIDST
cannot host the models in this pipeline: it lacks nonparametric models (HDP) and
cannot accommodate non-VI models (sparse OU via penalized MLE) — its architecture is
restricted to directed graphical models with VMP-compatible distributions. The
framework should be positioned as *enabling* the clinical modeling pipeline, not as the
primary contribution. The scientific contribution is the two-stage pipeline (HDP → OU)
and its clinical application; the framework is the vehicle. See the "Prior Art &
Positioning" section in SPARK_VI_FRAMEWORK.md for a detailed comparison.

### Reviewer comparison to Bayesian latent-class EHR phenotyping (Hubbard et al., PCORI)

**Risk:** A reviewer familiar with prior PCORI-funded EHR phenotyping work
— particularly Hubbard et al. (2021), *Estimating Patient Phenotypes and
Outcome-Exposure Associations in the Presence of Missing Data in Electronic
Health Records*, developed within PEDSnet and applied to pediatric type 2
diabetes — could see surface overlap: both projects use unsupervised Bayesian
methods for EHR phenotyping, both target pediatric populations, and both
reject rule-based Boolean "computable phenotypes" in favor of probabilistic
approaches. A reviewer who knows this work may question whether CharmPheno
duplicates it.

**Impact:** Low-to-medium. The overlap is largely in vocabulary and
high-level framing; the technical targets are meaningfully different. But
it's worth being explicit about the distinction in proposal narrative to
defuse the comparison.

**Mitigation:** Hubbard et al. develop a **Bayesian latent-class model that
estimates a single pre-specified phenotype** (e.g., T2DM yes/no) more
accurately than Boolean rules by modeling informative missingness in EHR
data. The phenotype to estimate is specified in advance; the contribution is
in the estimator's handling of MAR/MNAR missingness and in its downstream
use in exposure–outcome association analyses. CharmPheno does something
categorically different: it **discovers the set of phenotypes** from
co-occurrence structure in the data without specifying them in advance, and
produces per-patient mixture profiles over the learned phenotype set. The
two approaches sit at different points in the phenotyping pipeline — in
principle a CharmPheno-derived phenotype definition could serve as the
latent-class target for a Hubbard-style estimator — and they should be
positioned as complementary rather than competing. The proposal narrative
should name Hubbard et al. explicitly in the related-work framing and state
this distinction in one sentence.

**Known methodological gap:** Hubbard et al.'s central technical contribution
is the treatment of informative missingness in EHR data (a code's absence
often carries signal: a test wasn't ordered, a condition wasn't looked for).
CharmPheno as currently designed follows the standard topic-modeling
convention of treating absent codes as simply absent, which is a known
simplification in the clinical context. This isn't a fatal flaw — most
unsupervised EHR phenotyping work has the same simplification — but
extending CharmPheno to model informative missingness is a natural direction
for follow-on work and would address the one technical dimension where
Hubbard et al. have something the current CharmPheno design does not.

---
