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

## 2026-05-07 — HDP concentration-parameter optimization (ADR 0013)

The γ/α/η optimization deferred from ADR 0011 (and again from ADR 0012's
`optimize*`-flag punt) lands as ADR 0013. The math turned out simpler
than ADR 0010's LDA-α templates implied: HDP's γ and α are scalar
concentrations on Beta(1, β) priors, not Dirichlet vectors, and the
ELBO contribution `N · log β + (β − 1) · S` admits a closed-form
maximizer `β* = -N/S` — exact in one step, no Hessian, no iteration.
η stays as scalar Newton (Hoffman 2010 §3.4, reused from LDA). All
three are gated by opt-in flags; γ and α default on (γ optimization
*is* the headline appeal of HDP), η defaults off (matches LDA — least
stable in SVI per Hoffman). A small follow-on arc surfaced
`--tau0`/`--kappa` through both cloud drivers because the MLlib
defaults (τ₀=1024, κ=0.51 → ρ_0 ≈ 0.029) are designed for
D≫100k corpora and make γ/α optimization crawl on smaller datasets.
Mid-walkthrough we caught and fixed an honesty problem in the
helper-lift commit (back-compat aliases hiding the real public surface).
Three `--no-ff` merge bubbles total.

### What shipped

- [ADR 0013](decisions/0013-hdp-concentration-optimization.md), new.
  Records the math derivation (why Beta(1, β) closes when LDA's α
  doesn't), the closed-form-vs-Newton-vs-self-consistency choice, the
  γ-on/α-on/η-off default flag rationale, the `global_params`-vs-`self`
  decision, and the new `spark_vi.inference` package home.
- [`spark_vi.inference`](../spark-vi/spark_vi/inference/) package, new.
  First module is
  [`concentration_optimization.py`](../spark-vi/spark_vi/inference/concentration_optimization.py)
  with three pure helpers: `alpha_newton_step` (asymmetric Dirichlet
  Sherman-Morrison Newton), `eta_newton_step` (scalar Dirichlet
  Newton), `beta_concentration_closed_form` (exact M-step for Beta(1, β)
  concentrations). All three return raw steps / closed-form maximizers;
  the caller applies ρ_t damping and the 1e-3 floor. Intended as the
  long-term home for shared variational primitives — first concrete
  consumer is HDP, but LDA already imports the Newton helpers from
  here too.
- [`OnlineHDP`](../spark-vi/spark_vi/models/online_hdp.py) gains
  `optimize_gamma`, `optimize_alpha`, `optimize_eta` constructor flags
  (defaults `True, True, False`). γ, α, η move from `self` instance
  attributes into `global_params` so they round-trip through broadcast
  and persistence the same way λ, u, v already do. `local_update`
  accumulates `s_alpha = Σ_d Σ_k [ψ(b_jk) − ψ(a_jk + b_jk)]` per-doc
  when `optimize_alpha` is on. `update_global` runs the four-step
  sequence: λ/u/v natural-gradient updates (using *old* γ, η), then γ
  closed-form on the *just-updated* (u, v), then α closed-form using
  the corpus-scaled `s_alpha`, then η Newton on the *just-updated* λ.
  `compute_elbo`, `infer_local`, `iteration_summary` all read γ/α/η
  from `global_params`; `iteration_summary`'s diagnostic line gains
  current γ/α/η values for live-fit visibility.
- [`OnlineHDPEstimator`](../spark-vi/spark_vi/mllib/hdp.py) gains three
  Params: `optimizeDocConcentration` (default True),
  `optimizeCorpusConcentration` (default True),
  `optimizeTopicConcentration` (default False). The validator's
  ADR-0012 "no optimize* flags" assertion is dropped. Trained
  accessors (`trainedAlpha()`, `trainedCorpusConcentration()`,
  `trainedTopicConcentration()`) now read from
  `result.global_params` instead of fit-time scalars, so post-fit
  introspection reflects optimization. `OnlineHDPModel.__init__` no
  longer takes alpha/gamma/eta kwargs — they live in the trained
  globals dict.
- Cloud drivers gain `--tau0`/`--kappa` and (HDP only — LDA already
  had two of three) `--[no-]optimize-{corpus,doc,topic}-concentration`
  argparse flags. Both
  [`hdp_bigquery_cloud.py`](../analysis/cloud/hdp_bigquery_cloud.py)
  and [`lda_bigquery_cloud.py`](../analysis/cloud/lda_bigquery_cloud.py)
  now plumb the SVI step parameters through to
  `learningOffset`/`learningDecay` on the Estimator.
  [`analysis/cloud/Makefile`](../analysis/cloud/Makefile) help text
  documents the override pattern (`HDP_BQ_ARGS='--tau0 16 --kappa 0.7'`)
  and the rationale (default ρ_0 ≈ 0.029 is glacial on small corpora).
- 18 new tests (176 total, was 158): 6 in
  [`test_concentration_optimization.py`](../spark-vi/tests/test_concentration_optimization.py)
  (closed-form recovery on synthetic Beta draws, validation rejections,
  helper-import smoke checks); 9 in
  [`test_online_hdp_unit.py`](../spark-vi/tests/test_online_hdp_unit.py)
  (s_alpha emission/omission, γ/α/η movement under each optimize_*
  flag, floor enforcement, compute_elbo reading γ/η from global_params,
  iteration_summary surfacing); 2 in
  [`test_online_hdp_integration.py`](../spark-vi/tests/test_online_hdp_integration.py)
  (γ/α-moves and η-on smoke against synthetic D=200 corpora); 4
  shim-side covering Param defaults, translation, and trained-accessor
  behavior under both optimize-on and optimize-off paths.

### Walkthrough lessons

**Lesson 1 — The math.** Why Beta(1, β) gets a closed form when LDA's
α can't. The β-dependent ELBO contribution of N independent Beta(1, β)
factors is `L(β) = N·log β + (β − 1)·S` where `S = Σ E_q[log(1−W)]` is
the sufficient statistic from `q(W) = Beta(u, v)`. Derivative `N/β + S`
has a
single root at `β* = -N/S`; second derivative `-N/β² < 0` confirms it's
a maximum. The contrast with LDA: LDA's α prior is Dirichlet, whose
log-partition `log Γ(Σ α_k)` couples all components through the sum,
forcing Newton with a structured Hessian. HDP's β concentration is a
single scalar on a Beta whose normalizer collapses to `1/β` (because
Γ(1) = 1), leaving only `log β` and `(β−1)·S` — quadratic enough that
the root is exact. The SVI wrapper `β_new = (1−ρ)·β_old + ρ·β*` does
double duty: noise damping AND optimal step direction in the
natural-gradient sense (Hoffman 2013 — same form as the conjugate
exponential-family natural-gradient update used for λ, u, v).

**Lesson 1 detour — γ's stochasticity is sneakier than α's.** α's
sufficient statistic is summed over docs in the *minibatch* (then
corpus-scaled), so it's classic-shape minibatch noise. γ's sufficient
statistic is computed from the current *global* (u, v), no minibatch
dependence — given (u, v), S_γ is deterministic. Yet γ still gets ρ_t
damping. Reason: (u, v) themselves only just moved by ρ_t; if γ jumped
to the closed-form fixed point of those partially-updated globals, it'd
be conditioning on a state that's still mid-flight. ρ_t on γ keeps it
at the same "completion fraction" as the (u, v) it's responding to —
everything moves together at the same SVI cadence.

**Lesson 2 — Helper-lift refactor.** Three options for sharing the
Newton helpers: cross-import between sibling models, duplicate, or
hoist to a new module. Picked option 3. The package didn't have a
home for "math primitives shared across models" — created
`spark_vi.inference/` for variational-inference primitives, intended
as the future home for natural-gradient updates and ELBO computations
shared across models too. The leading-underscore-as-`as`-alias trick
used in the initial commit was caught mid-walkthrough as an honesty
problem (see Detour 1).

**Lesson 3 — HDP model wiring.** The two sequencing decisions in
`update_global`: (1) λ/u/v use *old* γ and η — block coordinate
ascent rule, when updating block A freeze block B at its current
value; (2) γ uses *just-updated* (u, v), η uses *just-updated* λ —
when updating concentration X, condition on the most-recent state
of what X is a prior over. The two rules look opposite but are
consistent under "condition on the most recent state of what the
operation depends on." LDA's η-on-just-updated-λ is the same pattern.
α's per-doc sufficient stat is genuinely different from γ and η: it's
the only concentration whose M-step needs per-doc data, so the
accumulation lives in `local_update` alongside the other per-doc
suff-stats. None-sentinel pattern (`s_alpha = 0.0 if optimize_alpha
else None`) keeps the no-optimize path free of the digamma cost and
gives `update_global` a clean `if "s_alpha" in target_stats` guard.

**Lesson 4 + 5 — Shim Params and trained accessors.** Three new Params
named for MLlib parity (`optimizeDocConcentration`,
`optimizeTopicConcentration` match LDA verbatim;
`optimizeCorpusConcentration` is HDP-specific, parallels the existing
`corpusConcentration` value-Param). Default flag values: γ on (HDP's
headline appeal — auto-discovers effective topic count), α on (matches
LDA), η off (Hoffman 2010 §3.4 — least stable in SVI). The
`global_params` migration done in Lesson 3 is what made the trained
accessors work cleanly: methods read from
`self._result.global_params["alpha"]` etc. so post-optimization values
are surfaced; the Model is reconstructible from `(VIResult, T, K)`
alone (closer to persistence-ready than before, which still defers).
The `trained` prefix on accessor names is the ADR 0012 collision-fix
preserved unchanged. **Default-behavior change flagged**: pre-ADR-0013
`OnlineHDPEstimator()` produced γ/α equal to constructor inputs;
post-ADR-0013 produces γ/α that reflect the data. Acceptable for a v1
model with no published artifacts to compare against.

**Lesson 6 — Cloud-driver `--tau0`/`--kappa` and ρ_t intuition.** The
Robbins-Monro schedule `ρ_t = (τ₀ + t + 1)^(-κ)` and what each knob
does: bigger τ₀ → smaller ρ throughout (especially early), bigger κ →
faster decay. Concrete numbers: MLlib's defaults (τ₀=1024, κ=0.51)
give ρ_0 ≈ 0.029, designed for D ≫ 100k. On D ~ 1k–10k corpora the
γ/α optimization crawls — the user reported "starts at ρ ≈ 0.0291"
which is exactly `1025^(-0.51)`. After 30 iterations at MLlib defaults,
γ has covered about 58.7% of the distance to its closed-form target.
Smaller-corpus-friendly settings: τ₀ ~ 10–64, κ = 0.7. Convergence
condition: κ ∈ (0.5, 1] for the Robbins-Monro guarantees Σρ_t = ∞ and
Σρ_t² < ∞ to both hold. Default κ = 0.51 is *just* inside the bound;
0.7 is a safer choice on small corpora with marginal step-size loss.

### Refactor detours that shipped

**Detour 1 — Drop the back-compat aliases.** During Lesson 2 the user
flagged that the helper-lift's `from spark_vi.inference.concentration_optimization
import alpha_newton_step as _alpha_newton_step` trick in
[lda.py](../spark-vi/spark_vi/models/lda.py) was avoiding a
6-import-site refactor. The aliases-as-rename pattern made the names
falsely look "module-private LDA helpers" when the function actually
lived in the public `spark_vi.inference` module — two paths to the
same function, leading-underscore signal contradicting reality, and
test files reading as "LDA-internal-detail tests" when they were
testing shared helpers. Dropped the aliases; updated 6 import sites
(5 in tests, 1 in probes). Lda.py's import line now reads honestly:
`from spark_vi.inference.concentration_optimization import
alpha_newton_step, eta_newton_step`. Single canonical import path
everywhere.

### Pre-existing issues caught

- **Back-compat aliases hiding the real public surface** — caught
  mid-walkthrough by the user pushing on whether the underscore aliases
  in [lda.py](../spark-vi/spark_vi/models/lda.py) were actually
  necessary or just dodging a small refactor. They were dodging. Honest
  assessment: the leading underscore signaled "module-private LDA
  helper" to readers but the function genuinely lived in a public
  shared module. Three resulting problems — duplicate API paths,
  misleading underscore convention, and test files reading as
  LDA-internal when they were testing the shared module — none of
  which were caught by the test suite (everything still passed).
  General principle reinforced: when a "small refactor" feels like
  it's avoiding a pattern violation, the honest thing is usually to
  do the refactor.
- **MLlib's defaults bite small corpora** — surfaced when the user
  reported ρ ≈ 0.0291 and asked whether that was expected. MLlib's
  τ₀=1024, κ=0.51 are tuned for the canonical Hoffman 2010 D ≈ 100k+
  setting; on D ~ 1k–10k corpora the schedule is glacial. The
  `--tau0`/`--kappa` surfacing fixes the symptom (caller can
  override); the underlying default is unchanged because we still
  want MLlib parity for callers running on cluster-scale data.

### Doc updates

- [ADR 0013](decisions/0013-hdp-concentration-optimization.md), new.
  Full math derivation, the closed-form-vs-Newton choice rationale,
  and the alternative-rejected list (Newton instead of closed form,
  per-stick adaptive ρ_t, asymmetric per-vocab η, keeping γ/α/η on
  `self`, skipping the closed-form derivation for code uniformity).
  Includes the Wang AISTATS 2011 deviation note: that paper holds γ/α
  fixed in experiments and doesn't derive an optimization rule; the
  closed-form M-step here comes from the variational-EM stick-breaking
  literature (Blei & Jordan 2006 for DP mixtures, naturally extending
  to HDP's two-stick structure).
- [ADR 0011](decisions/0011-online-hdp-design.md) and
  [ADR 0012](decisions/0012-hdp-mllib-shim.md) are unchanged. Their
  γ/α/η-deferral text is now superseded by ADR 0013 for the
  optimization story but the ADRs themselves are append-only; ADR
  0013's Context section names the deferrals it resolves.

### Empirical run

User triggered a 25-iter cloud fit on AoU OMOP at T=100, K=20,
person-mod 50, subsamplingRate=0.1, **τ₀=16, κ=0.7** with γ/α
optimization on. Observations:

- **ρ schedule honored:** iter 1 ρ=0.1376 (matches `17^(-0.7) = 0.1376`
  to four decimals); iter 25 ρ=0.0743. ~5× larger steps than MLlib
  defaults would have given.
- **γ optimization moving steadily upward:** 1.0 → 1.92 over 25 iters,
  monotone increase. Direction means the closed-form maximizer γ* keeps
  landing above the current value — the data wants more topics than
  initial γ=1.0 implied. Linear extrapolation suggests γ → ~2.5+ before
  the natural-gradient fixed point.
- **α optimization moving up too:** 1.0 → 1.43. Per-doc topic
  concentration rising; reasonable for clinical data where multimorbid
  patients hit several phenotype clusters.
- **5–6 active topics out of T=100** (active-mass threshold 0.95). HDP
  doing what it's supposed to: truncation gives headroom but only
  meaningful atoms grow. Topic IDs are stable (truncation indices); the
  display filters to top-5-by-`E\[β\]`, so atoms shuffle in/out of the
  display as their ranking changes (e.g., topic 4's renal-transplant
  cluster dropped below topic 5's substance-use cluster at iter 12).
- **Topic content is clinically coherent:** topic 0 is the
  metabolic-syndrome / general-chronic-disease background (HTN+T2DM+HLD,
  `E\[β\]`≈0.55–0.63); topic 1 is cardiovascular (atherosclerosis+AFib);
  topic 2 is mental health + chronic pain; topic 3 is pregnancy + PTSD
  (mixed because both populations get complex charts); topic 4 is renal
  transplant; topic 5 is substance use disorders. All recognizable
  phenotype clusters.
- **Topic 0 background dominance** (`E\[β\]`≈0.6) is the same "stopword
  phenomenon" flagged in the previous review log entry — clinical
  data's most-frequent concept_ids (HTN, T2DM, HLD) form a giant
  absorbing topic. Mitigation options: accept; pre-filter top-N
  most-common concepts; tf-idf-style weighting in CountVectorizer.
  Parked thread (see below).
- **ELBO trace noisy but improving:** -2.48M → -1.67M (low at iter 19)
  → bouncing in -1.7M to -1.9M range. Smoothed-endpoint trend clearly
  upward; minibatch noise expected at subsamplingRate=0.1.
- **λ row-sum spread climbing:** 111 → 6415. Active topics accumulating
  evidence; inactive ones near the η floor. Healthy "rich get richer."

### Open threads parked

- **Topic 0 stopword mitigation** — well-known clinical-data issue;
  three concrete options identified (accept; pre-filter top-N most
  frequent concept_ids before vectorizing; tf-idf-style weighting).
  Cheapest experiment is option (b) — add a `--drop-top-n` flag to
  the vocab builder. Probably its own small spec.
- **Wang reference cross-check fixture** — still deferred from ADR
  0011 ("considered and deferred"). The closed-form γ/α optimization
  is a deliberate deviation from Wang anyway (he holds them fixed),
  so a bit-match cross-check would have to be against the no-optimize
  path. Less urgent than it was for v1.
- **MLWritable / Pipeline.save persistence** — still deferred per
  ADR 0009 / 0012. The `global_params` migration moved the model
  closer to persistable (no fit-time-only attrs to handle), so when
  persistence lands it'll be straightforward.
- **Lazy-λ sparse-vocabulary update** — ADR 0011 v2 punt unchanged.
  Only matters if profiling shows full-V digamma is the bottleneck.
- **`on_iteration` callback shape** — same friction as flagged in the
  previous review log entry; the iteration_summary surfacing of γ/α/η
  in this arc partially mitigates by giving cloud-driver loggers
  visibility into the trained values without needing to reconstruct
  a model. Fuller refactor still parked.
- **Per-stick adaptive ρ_t for the M-steps** — ADR 0013 alternative
  rejected for v3 (no evidence we need separate damping). Revisit if
  γ optimization shows oscillation on production runs.

---

## 2026-05-07 — HDP MLlib shim arc and post-shim cleanups

The HDP shim deferred from ADR 0011 lands as `OnlineHDPEstimator` /
`OnlineHDPModel`, paralleling `VanillaLDAEstimator`. The first walkthrough
detour exposed a latent property/Param-name collision in the LDA shim
that the LDA test suite happened never to exercise; backported the fix.
Subsequent walkthrough chunks surfaced four more "funny smells" — a
deletable wrapper layer, a fixed-truncation threshold, duplicated
multi-step math, and inconsistent local-driver API styles — each
addressed in its own focused arc. Six `--no-ff` merge bubbles in one
day; the inner model from v1 is unchanged.

### What shipped

- [`spark_vi.mllib.hdp.OnlineHDPEstimator` / `OnlineHDPModel`](../spark-vi/spark_vi/mllib/hdp.py)
  — the MLlib-shaped Estimator/Model pair. Param surface mirrors
  `VanillaLDAEstimator` for the shared subset (`k`, `maxIter`,
  `featuresCol`, `learningOffset`/`learningDecay`, `subsamplingRate`,
  `topicConcentration`); HDP-specific extras are `docTruncation` (K),
  `corpusConcentration` (γ), `gammaShape`, `caviMaxIter`, `caviTol`.
  Reject-list at fit time: `optimizer != "online"`, vector
  `docConcentration`, vector `topicConcentration`. No `optimize*` flags
  surfaced — γ/α/η optimization deferred per ADR 0011 §"Hold γ and α
  fixed at user-set values," and adding `False`-only flags would
  mislead.
- HDP-specific Model accessors: `corpusStickWeights()` returns the E[β_t]
  vector under mean-field q (exact for the variational mean — see
  "Pre-existing issues caught" below for the docstring correction);
  `activeTopicCount(mass_threshold=0.95)` returns the count of topics
  whose top-ranked weights cover the threshold's worth of corpus mass
  (truncation-independent).
- [`spark_vi.mllib._common._vector_to_bow_document`](../spark-vi/spark_vi/mllib/_common.py)
  — hoisted out of [lda.py](../spark-vi/spark_vi/mllib/lda.py) so the
  HDP shim doesn't import privates from the LDA sibling.
- Local + cloud driver pair:
  [`analysis/local/fit_hdp_local.py`](../analysis/local/fit_hdp_local.py)
  goes through the shim (catches shim regressions on a 10-doc parquet
  before paying for cloud submits);
  [`analysis/cloud/hdp_bigquery_cloud.py`](../analysis/cloud/hdp_bigquery_cloud.py)
  parallels `lda_bigquery_cloud.py` with HDP-native diagnostics
  (E[β_t]-ranked snapshots, active-topic filtering by cumulative mass).
- [`analysis/cloud/Makefile`](../analysis/cloud/Makefile) gains
  `hdp-bq-smoke` target with `HDP_BQ_ARGS` override; ran on AoU
  Dataproc with T=100, K=20, person-mod 50, subsamplingRate 0.1 — see
  "Empirical run" below.
- Makefile JAVA_HOME selection robustified across
  [root](../Makefile), [spark-vi/](../spark-vi/Makefile), and
  [charmpheno/](../charmpheno/Makefile): candidate list now includes
  `/opt/homebrew/opt/openjdk` (any Java≥17, not just `openjdk@17`),
  and `JAVA_HOME` only exports when a candidate is found — empty
  `JAVA_HOME=""` is worse than unset because tooling sees a nominally-set
  value and skips PATH-default discovery, which on developer machines
  was pinning Java 11 and tripping PySpark 3.5+'s unconditional
  `-Djava.security.manager=allow` flag.

### Walkthrough lessons

**Lesson 1 — ADR 0012 + the refactor.** Why the helper hoist (`_common.py`
hosts shared shim helpers; siblings don't import privates from each
other). The `k=T` Param-naming decision: MLlib's `k` is "exact topic
count" but HDP's T is a truncation upper bound — the docstring carries
the semantic correction, and `docTruncation` covers K. The
`optimize*`-flag deferral cascade: ADR 0011's v1 scope call propagates
to the shim's Param surface; adding `False`-only flags would mislead
about model capability. The "siblings, not generic adapter" call
answering ADR 0009's open question — the audit of shared code (~130
lines) vs. divergent shape (Param schemas, validators, translation,
trained accessors, model-specific methods) showed an adapter would
delegate to user-provided callables for every meaningful component;
that's not abstraction, it's plumbing-as-API.

**Lesson 2 — Param surface + validators.** `_OnlineHDPParams` mixin
declares 13 Params once, inherited by both Estimator and Model
(mirrors MLlib's `_LDAParams`). `docConcentration` is typed
`toListFloat` despite being scalar-only because MLlib's
`docConcentration` accepts vectors; we keep the type identical for
Pipeline composability and reject the vector case at validate time
instead of at type-conversion time. `corpusConcentration` is a
brand-new Param outside MLlib's vocabulary — first one we've added,
sets the convention for future shims. Validator's three rejections
(non-online optimizer, vector `docConcentration`, list/tuple
`topicConcentration`) — defensive against the typed-as-float-but-be-safe
case for `topicConcentration`. Notably *not* validated by the shim:
`docTruncation >= 2`, `corpusConcentration > 0`, `k >= 2` — those
constraints live on `OnlineHDP.__init__`, single source of truth.

**Lesson 2 detour — Wallach 2009 analogue for HDP.** Wallach's
empirical finding ("asymmetric α, symmetric η") carries over
asymmetrically to HDP. **Symmetric η** carries directly — same role,
same recommendation. **Asymmetric α** does *not* carry as vector α —
HDP's α is scalar in Wang/Paisley/Blei 2011 and the canonical
references; the variational `q(W) = Beta(1, α)` is a single-scalar
distribution. The asymmetric-topic-prior structure is encoded in HDP's
**corpus stick β**, asymmetric *by construction* via stick-breaking.
Optimizing γ (the Beta concentration on β) is the v3 analogue of
Wallach's asymmetric-α finding — it lets the model learn the shape of
the asymmetric corpus prior. ADR 0010's η-Newton machinery directly
templates the future γ-Newton (both are global scalars whose
sufficient statistics are computable from a global parameter — η from
λ, γ from u/v).

**Lesson 3 — `_fit` path.** Five-line tour of [hdp.py:296-315](../spark-vi/spark_vi/mllib/hdp.py):
vocab size discovered from data (DataFrame-side `head(1).size`, MLlib
parity), persist precondition load-bearing because `dataset.rdd.map(...)`
builds an uncached RDD even when the upstream DataFrame is cached
(VIRunner's `assert_persisted` is strict; we persist + `count()` to
materialize and pay the BOWDocument-conversion cost once),
try/finally for unpersist (crash-safe broadcast cleanup),
`on_iteration` callback as instance attr not Param (callables aren't
MLlib-serializable, would break `Pipeline.save`), param-copy loop
two-branch dance preserves the `isSet`/`hasDefault` semantic on the
Model.

**Lesson 4 — methods-vs-properties bug.** The most instructive piece
of the arc, because it caught a latent LDA-shim bug. First draft of
the HDP shim used the LDA pattern verbatim (`@property
topicConcentration`, `@property alpha`). Three tests crashed with
`RecursionError`: the property body called `self.isSet(...)`, which
internally does `getattr(self, name)` → property body → recursion.
LDA dodged this because *its* property body reads
`self._result.global_params["eta"]` directly without touching the
Params API. Fixed the recursion by stashing scalars in `__init__`,
which then revealed bug 2: MLlib's `_set` does
`p = getattr(self, name); p.typeConverter(value)`. The property
returns a float, not a Param descriptor, so `typeConverter` raises
`AttributeError`. LDA dodged this one too — but only because no LDA
test path explicitly sets the colliding Params on the Estimator. Fix:
rename the trained-scalar accessors to methods with `trained` prefix
(`trainedAlpha()`, `trainedTopicConcentration()`,
`trainedCorpusConcentration()`) — no shadowing, matches MLlib's
actual `pyspark.ml.clustering.LDAModel.topicConcentration()` method
idiom. The LDA-shim backport (separate arc) brings the LDA shim into
pattern-consistency and locks in a regression test
(`test_explicit_topic_concentration_through_fit_path`).

**Lesson 5 — Model surface.** `topicsMatrix()` returns full T topics
(not active-only — filtering is the caller's call), Fortran-flatten
for MLlib's `DenseMatrix(numRows, numCols, values)` constructor.
`describeTopics(maxTermsPerTopic)` schema matches MLlib's LDA. The
HDP-specific accessors are where the math meets the API:
`corpusStickWeights()` thin-wraps the stick-breaking-mean function;
`activeTopicCount(mass_threshold)` thin-wraps the cumulative-mass
function. `_transform()` UDF reconstructs the OnlineHDP instance on
each executor from Params + globals — the Model carries no reference
to its training-time Python instance, so it's reconstructible from
VIResult alone. Returns θ length T (corpus topics), not the doc-stick
π length K — the conversion happens inside `infer_local` via
`theta = pi_doc @ var_phi`.

**Lesson 6 — drivers + Makefile + JAVA_HOME.** Local driver goes
through the shim (validates the same code path the cloud driver runs).
Cloud driver's HDP-native diagnostics differ from LDA's: E[β_t] sort
key (vs λ row-sum), active-topic filtering in mid-fit snapshots
(unfiltered would print 100 nearly-empty rows), three lenses per
topic (`E[β]`, `Σλ`, `peak`) instead of one. Defensive duplication of
`expected_corpus_betas` in the cloud driver was later eliminated
(post-walkthrough cleanup). The JAVA_HOME fix details: `JAVA_HOME=""`
is *worse* than unset because tooling sees a nominally-set empty value
and skips fallback discovery. The conditional-prefix pattern
(`$(if $(JAVA_HOME),JAVA_HOME=$(JAVA_HOME) ,)`) is the reliable way
to express "set this only if we have a value."

### Refactor detours that shipped

**Detour 1 — Drop `CharmPhenoHDP` wrapper.** During Lesson 1's
`k=T`-naming discussion, surfaced that `CharmPhenoHDP` introduced a
*third* name-translation layer between `OnlineHDP` and the shim
(`max_topics`/`max_doc_topics` renames atop the shim's
`k`/`docTruncation` renames). LDA has no equivalent wrapper. The
wrapper's only concrete value-add was the rename + RDD-shaped
fit/transform; future hooks (phenotype labelers, export hooks) were
aspirational, never implemented. Deleted in
[`chore(charmpheno): drop CharmPhenoHDP wrapper`](d50c324). Removes
~200 lines (model + tests). Clinical concerns now live in driver
scripts and `charmpheno.evaluate` / `charmpheno.omop`.

**Detour 2 — LDA-shim trained-accessor backport.**
`fix(mllib): apply trained* accessor pattern to LDA shim`
([ee3de4d](ee3de4d)). The latent bug discovered in Lesson 4: LDA's
`@property alpha` and `@property topicConcentration` had the same
shape of bug as the HDP first-draft. No LDA test exercised the
explicit-set path, so the bug was dormant. Backport renames to
`trainedAlpha()` / `trainedTopicConcentration()` methods, adds
[`test_explicit_topic_concentration_through_fit_path`](../spark-vi/tests/test_mllib_lda.py)
to lock in the regression. Pattern-consistency between sibling shims
restored.

**Detour 3 — Cumulative-mass active-topic-count.**
`feat(hdp): cumulative-mass active-topic-count + correct E[β] docstring`
([a9435d2](a9435d2)). Replaces the truncation-dependent `1/(2T)`
threshold with parameterized `mass_threshold=0.95` (PCA's
"explained-variance" analog). Truncation-independent — same answer for
any T ≥ effective topic count. Plumbed through
`OnlineHDPModel.activeTopicCount`,
`OnlineHDP.iteration_summary`, and the cloud driver's
`--active-mass-threshold` argparse arg. Three new unit tests cover
the parameterization, monotonicity, and truncation invariance.

**Detour 4 — Consolidate `expected_corpus_betas`.**
`refactor(hdp): consolidate expected_corpus_betas in online_hdp.py`
([8dc708f](8dc708f)). The same E[β_t] math lived inline in three
places (model `iteration_summary`, shim `mllib/hdp.py`,
cloud driver). Lifted to a public function on `online_hdp.py`. Single
source of truth lives with the model module; shim and cloud driver
import.

**Detour 5 — Backport `fit_lda_local.py` to the LDA shim.**
`refactor(analysis): backport fit_lda_local.py to VanillaLDAEstimator`
([107c19e](107c19e)). The HDP local driver was written through the
shim; the LDA local driver was older and used `VanillaLDA + VIRunner`
directly. Brought into parity — both local drivers now exercise the
same code path their cloud counterparts use. Drops checkpoint args
(shim doesn't surface them; matches `fit_hdp_local.py`); aligns
default `--kappa` to 0.51.

**Detour 6 — Lift `topic_count_at_mass`.**
`refactor(hdp): lift topic_count_at_mass to module-level on online_hdp.py`
([72f903d](72f903d)). The cumulative-mass logic (sort + cumsum +
threshold + fp-slop guard) was duplicated in three places — same
shape as Detour 4 but for a different function. Lifted to a public
function on `online_hdp.py` with a generic parameter name
(`topic_weights`, not `E_beta`) so the primitive is reusable beyond
HDP. Cloud driver's post-fit summary now uses
`model.activeTopicCount(...)` directly instead of recomputing — the
shim's accessor is the right level of abstraction for that caller.

### Pre-existing issues caught

- **Latent property/Param-name collision in LDA shim** —
  caught by writing the HDP sibling and copying the pattern, then
  having an HDP test that explicitly set `topicConcentration` on the
  Estimator (the codepath no LDA test exercised). Generalizable
  methodology: when copying a pattern from a sibling, the parts you
  don't have a test for in the original are exactly where latent bugs
  hide.
- **`_expected_corpus_betas` "Jensen biased low" docstring** — wrong
  reasoning. The plug-in formula `E[W] · ∏ E[1-W]` is *exact* for the
  variational E[β_t] under the mean-field q (independent Beta
  factors let expectation distribute through the product). Jensen
  bias would apply if we were computing a non-linear functional of β
  via the means, but here we're computing E[β] itself. Docstring
  corrected; the right caveat is the standard mean-field
  underestimated-uncertainty issue, not Jensen.
- **Cumulative-mass logic duplicated three places** — only partially
  addressed at first lift (`expected_corpus_betas`); the cumulative-mass
  block (`np.argsort` → `np.cumsum` → threshold → fp-slop) was still
  inline in three places. Caught when the user noted "low-level math
  shouldn't live in usage scripts." Resolved with the second lift
  (Detour 6).
- **`CharmPhenoHDP` wrapper as deletable layer** — caught while
  discussing the `k=T` naming decision; the wrapper's renames
  introduced a third translation step with no offsetting value-add.

### Doc updates

- [ADR 0012 — HDP MLlib shim](decisions/0012-hdp-mllib-shim.md), new.
  Records the design choices: `k=T`, `docTruncation`=K, new
  `corpusConcentration` for γ, no `optimize*` flags (deferral cascade
  from ADR 0011), siblings-not-adapter answer to ADR 0009's open
  question, methods-vs-properties decision (with the latent-LDA-bug
  context). Edited in place twice during the day's arcs (drop-CharmPhenoHDP
  refs after Detour 1; cumulative-mass + exact-mean docstring after
  Detour 3) — both edits authorized by the user as cleanliness
  exceptions because the ADR was hours old, not yet historicized. The
  append-only convention re-applies for any future edits.

### Open threads parked

- **HDP γ/α/η optimization** — picked up next session. ADR 0010's
  Newton machinery templates: γ uses η's global-stat shape (computable
  from u/v), α uses LDA-α's per-doc shape, optional η reuses LDA's
  symmetric Newton directly. Cloud-test feedback (active=8/100 with
  γ=1.0 fixed, dominant background topic 0 carrying ~47% mass)
  motivates the work — γ is HDP's high-value tunable analogous to
  Wallach 2009's asymmetric α.
- **`on_iteration` callback shape** — currently
  `(iter_num, global_params, elbo_trace)`. Friction surfaced when
  cloud-driver loggers had to import stateless math from the model
  module instead of calling `model.method(global_params)`. Refactor
  would touch the runner contract and the LDA shim too; deferred as
  not-worth-the-blast-radius. Alternative posed: a "training state"
  abstraction that exposes diagnostic methods. Deferred.
- **LDA `iteration_summary` / cloud-driver dedup** — same shape as
  the HDP cumulative-mass duplications, but LDA's "math" is one-line
  numpy primitives (`lam.sum(axis=1)`, peak ratios). Wrapping
  primitives in named functions doesn't add abstraction; deferred as
  not-worth-the-payoff.
- **MLWritable / Pipeline.save persistence** — still deferred per
  ADR 0009 / 0012. Lands when a concrete user materializes; today
  callers persist via `VIResult.export_zip` and reconstruct.
- **Empirical cloud run** — first cloud HDP fit on AoU OMOP at
  T=100, K=20, person-mod 50 surfaced healthy training behavior:
  active topic count grew 6→7→8 over the 12-iter window, λ row-sum
  spread grew 266→855 (topics differentiating), top-3 sticks
  cumulatively held ~87% of mass (geometric-decay shape). Topic 0
  was the classic "general adult comorbidities" diffuse background
  (47% mass, peak=0.022) — expected on clinical data, mitigatable
  via per-visit documents or larger η. Topic 6 had questionable
  coherence (alcohol/cocaine + macular degeneration) — flagged as
  one to watch. ELBO noise (~30% range iter-to-iter) was expected
  from 10% subsamplingRate; the smoothed corpus-scale trend should
  be readable through it but the unsmoothed log isn't.

---

## 2026-05-06 to 2026-05-07 — Online HDP v1 walkthrough and merge

The `feat/online-hdp-v1` branch lands on `main` after a five-lesson
bottom-up walkthrough of the Online HDP implementation (Wang/Paisley/Blei
2011). The walkthrough surfaced three substantive fixes — an
`iteration_summary` truncation guard, a count-weighting deviation in the
doc-Z ELBO term, and a `CharmPhenoHDP.transform` delegation gap — all of
which shipped before the merge. The math (HDP as two-level
stick-breaking with doc-stick → corpus-topic pointer table) was covered
in a dedicated detour up front, including the deliberate vocabulary shift
from "atom" to topic / slot to disentangle global vs. doc-level usage.

### What shipped

- [`OnlineHDP.iteration_summary`](../spark-vi/spark_vi/models/online_hdp.py#L551-L595)
  — three-line live-training diagnostic: active-topic count
  `#{t : E[β_t] > 1/(2T)}`, top-3 corpus stick weights descending,
  λ row-sum spread (max/min). The pre-merge fix
  ([b0c85e1](b0c85e1)) added a `T < 3` guard so the
  "top-3" slice doesn't index past the truncation; relevant for the
  smallest unit-test fixtures.
- [`CharmPhenoHDP.transform`](../charmpheno/charmpheno/phenotype/charm_pheno_hdp.py#L82-L112)
  — wrapper-level frozen-globals inference. The pre-merge fix
  ([b0c85e1](b0c85e1)) wired the wrapper's
  `transform` to delegate through the underlying `OnlineHDP.infer_local`
  with a managed broadcast (same lifecycle discipline as
  `VIRunner.transform`); previously the wrapper had a stub that wasn't
  paired with a real implementation.
- [`_doc_e_step` doc_z_term ELBO block](../spark-vi/spark_vi/models/online_hdp.py#L200-L216)
  — count-weighted token-token ELBO contribution, with an inline
  DEVIATION FROM REFERENCE comment block now anchoring the choice. The
  count weighting is a deviation from Wang's reference Python *and* the
  intel-spark Scala port (both omit it, matching the AISTATS paper as
  printed); without it the (a, b) update maximizes a different objective
  than the reported per-iter ELBO. Caught originally during Task 5
  implementation by `test_doc_e_step_per_iter_elbo_nondecreasing`; this
  walkthrough turned the inline justification into an ADR-anchored note.

### Walkthrough lessons

**Detour up front — Stick-breaking foundations.** Re-derived
`GEM(γ)` via Sethuraman: `β_k = W_k · ∏_{l<k}(1−W_l)` with
`W_k ∼ Beta(1, γ)`. Concrete numerical example with the "first stick gets
half on average; second stick takes half of what's left" intuition. Then
the two-level structure: corpus stick `β` over discoverable topics;
per-doc stick `π` over **slots**; doc slots point into corpus topics via
the variational table `var_phi` (K, T). Truncation in the variational
posterior: `q(W_T = 1) = 1` (degenerate at the last position), so the
last topic / last slot absorbs whatever residual mass the truncation
leaves. The lesson replaced the "atom" jargon from Wang's paper with
**topic** (corpus level, T) / **slot** (doc level, K) throughout the
walkthrough, on user request — disentangles two different things the
paper conflates.

**Lesson 1 — Helpers and module shape.** Walked the three helpers as
the building blocks for everything downstream:
[`_log_normalize_rows`](../spark-vi/spark_vi/models/online_hdp.py#L37-L48)
(stable per-row log-softmax),
[`_expect_log_sticks`](../spark-vi/spark_vi/models/online_hdp.py#L50-L70)
(Sethuraman's `E[log W] + Σ E[log(1−W_l)]` for the variational Beta
factors), and
[`_beta_kl`](../spark-vi/spark_vi/models/online_hdp.py#L73-L103)
(closed-form `KL(Beta(a, b) ‖ Beta(a₀, b₀))` via digamma / lgamma).
Module shape note: `OnlineHDP.local_update` does the partition E-step,
`update_global` the SVI step, `compute_elbo` the ELBO assembly,
`infer_local` the held-out doc inference — same VIModel surface as
VanillaLDA, with the addition of `(u, v)` corpus-stick globals alongside
λ.

**Lesson 2 — Doc-level CAVI.** Walked
[`_doc_e_step`](../spark-vi/spark_vi/models/online_hdp.py#L106-L233)
end-to-end. The three updates:
`var_phi` (K, T) — slot-to-topic responsibility, softmax over corpus
topics for each slot;
`phi` (W_unique, K) — token-to-slot responsibility, softmax over slots
for each unique vocabulary type in the doc;
`(a, b)` (K-1) — variational doc-stick Beta parameters.
Two structural points anchored: (1) per-unique-type `phi` storage with
`* counts[:, None]` count-weighting at every consumption site (the
deviation captured in the doc_z_term comment block); (2) the
"`iter < 3`" warmup trick — drop prior-correction terms in
`var_phi`/`phi` updates for the first three iterations of each doc's
CAVI. Undocumented in the AISTATS paper but preserved in both Wang's
Python and intel-spark's Scala port; ADR 0011 records the choice and
parks an ablation for v2.

**Lesson 3 — SVI wiring.** Walked
[`local_update`](../spark-vi/spark_vi/models/online_hdp.py#L328-L403)
and
[`update_global`](../spark-vi/spark_vi/models/online_hdp.py#L405-L439).
Per-partition E-step structure: precompute `Elogβ` and
`Elog_sticks_corpus` once per partition, scatter per-doc sufficient
stats into a partition-level (T, V) `lambda_stats` accumulator, sum
ELBO scalar contributions. The natural-gradient SVI step on the driver:
λ̂ = η + (corpus-rescaled) `lambda_stats`, mixed via
`λ ← (1−ρ_t)·λ + ρ_t·λ̂`; the (u, v) sticks update via the Wang Eq 22-23
form using `Σ_t var_phi[k, t]` and the suffix sum thereof. Only
λ persists as a "topic-word" object across iterations; per-doc state
(var_phi, phi, a, b) is rebuilt every minibatch by design.

**Lesson 4 — ELBO and held-out inference.**
[`compute_elbo`](../spark-vi/spark_vi/models/online_hdp.py#L441-L489)
assembles the bound from per-doc aggregates (`doc_loglik_sum`,
`doc_z_term_sum`, `doc_c_term_sum`, `doc_stick_kl_sum`) plus two
driver-side KL terms: corpus stick KL (Beta(u_t, v_t) ‖ Beta(1, γ),
summed t) and topic Dirichlet KL (Dir(λ_t) ‖ Dir(η · 1_V), summed t).
In minibatch mode the per-doc piece is the minibatch sum (not
corpus-rescaled), so the reported ELBO is a noisy unbiased estimator;
the integration test gate uses smoothed-endpoint comparison to absorb
the variance. [`infer_local`](../spark-vi/spark_vi/models/online_hdp.py#L491-L549)
is the real frozen-globals HDP doc-CAVI for `transform()` —
deliberately not Wang's `infer_only` which collapses HDP into LDA-shape
for prediction. ADR 0011 §"Real frozen-globals HDP doc-CAVI" records the
three reasons (held-out evaluation accuracy, future patient-train /
visit-infer split, predictive modeling).

**Lesson 5 — Wrapper, integration tests, active-topic count.**
[`CharmPhenoHDP`](../charmpheno/charmpheno/phenotype/charm_pheno_hdp.py)
as thin clinical wrapper: clinical-user-facing names (`max_topics` → T,
`max_doc_topics` → K) translate to the inner OnlineHDP truncation
parameters; vocab handling stays in the wrapper. Three integration
gates: ELBO-finite + smoothed-endpoint ELBO trend
([`test_online_hdp_integration.py`](../spark-vi/tests/test_online_hdp_integration.py))
and synthetic recovery on D=2000 with Hungarian matching
(scipy's `linear_sum_assignment`) for label-switching invariance — the
HDP-rectangular variant where unmatched fitted topics are themselves a
signal (model discovered topics the synthetic generator didn't seed).
Active-topic count `#{t : E[β_t] > 1/(2T)}` introduced as the cheap
"this corpus topic carries half-uniform mass" proxy reused in
`iteration_summary` and (subsequently) on the MLlib shim.

### Pre-existing issues caught

- **`iteration_summary` T<3 indexing** ([b0c85e1](b0c85e1))
  — the `top3 = np.sort(E_beta)[::-1][:3]` slice silently produced a
  shorter array when T < 3, then the format string broke at runtime
  rather than at construction. Smallest unit-test fixtures with T=2 hit
  the path; production-sized T=150 never would. Guard added.
- **`CharmPhenoHDP.transform` delegation** ([b0c85e1](b0c85e1))
  — the wrapper had been exposing a stub `transform` that didn't actually
  drive frozen-globals inference. Wired through `OnlineHDP.infer_local`
  with a managed broadcast (mirrors `VIRunner.transform`'s lifecycle
  discipline). Smoke test added.
- **doc_z_term count-weighting** (deviation from Wang reference, fixed
  pre-walkthrough; documented during walkthrough) — Wang's reference
  Python and intel-spark Scala port both omit the `* counts[:, None]`
  factor when summing the per-token ELBO entropy + cross-entropy over
  the per-unique-word `phi` storage. Without the factor the (a, b)
  update (which uses count-weighted `phi_sum`) maximizes a different
  objective than the reported ELBO, causing post-convergence drift.
  Caught by `test_doc_e_step_per_iter_elbo_nondecreasing` during Task 5.
  Walkthrough added the inline DEVIATION FROM REFERENCE comment block
  and the matching ADR 0011 entry.

### Doc updates

- [ADR 0011 — Online HDP v1 design decisions](decisions/0011-online-hdp-design.md)
  ([a265fd8](a265fd8)) records six load-bearing scope/design choices:
  skip lazy-lambda sparse-vocab update in v1, hold γ and α fixed, no
  in-loop `optimal_ordering`, real frozen-globals HDP doc-CAVI for
  `transform()` (not LDA-collapse), keep the iter<3 warmup trick, and
  match-LDA Gamma init for λ. The doc_z_term deviation from Wang gets
  its own §; v2 work (MLlib shim + drivers) is explicitly deferred.
- [`docs/architecture/TOPIC_STATE_MODELING.md`](architecture/TOPIC_STATE_MODELING.md)
  ([ee84bc0](ee84bc0)) — clarified the T (corpus) / K (doc) naming
  convention up front and corrected the doc_z_term spec to match the
  count-weighted implementation.
- **Scratch-space cleanup** ([1517d4e](1517d4e)) — removed
  `docs/superpowers/` references from durable code (3 files) and ADRs
  (5 files: 0007-0011), per the project rule that committed artifacts
  should not link into per-session scratch space. Replaced with
  self-contained explanations or pointers to canonical ADR / spec
  locations.

### Open threads parked

- **MLlib Estimator/Transformer shim for OnlineHDP and cloud driver
  scripts** — explicitly deferred to v2 in ADR 0011, following the
  ADR 0009 precedent (LDA shim added *after* the model was built and
  validated). The "second data point" question ADR 0009 raised about
  whether to write a generic `VIModel → MLlib` adapter gets resolved
  when this work lands.
- **γ and α concentration optimization** — deferred to a follow-on
  ADR + spec pair after v1. The Newton machinery from ADR 0010 templates
  to HDP (γ uses η's global-stat shape; α uses LDA-α's per-doc shape).
- **`warmup_iters=0` ablation** — ADR 0011 keeps the iter<3 warmup as
  default; v2 will run an ablation to check whether it earns its keep,
  since neither Wang's paper nor the reference port documents the
  rationale.
- **Wang-reference parity fixture** ([eba0b58](eba0b58)) — Task 16
  recorded the deferral of a bit-match test against Wang's reference
  Python (would require running blei-lab/online-hdp on a fixed seed and
  capturing intermediates). Useful future regression gate; not gated on
  v1.
- **Held-out perplexity track** for both LDA and HDP — discussed at the
  merge boundary as a likely v3 branch alongside coherence metrics
  (UMass / NPMI / C_v) and topic diversity. Topic-modeling literature
  (Chang et al. 2009 "Reading Tea Leaves") finds perplexity negatively
  correlates with human-judged coherence, so coherence + diversity are
  the more useful evaluation signal beyond ELBO.

---

## 2026-05-05 to 2026-05-06 — LDA concentration optimization + post-implementation walkthrough

Empirical-Bayes Newton-Raphson optimization for the Dirichlet
concentration parameters lands on the `lda-concentration-opt` branch:
asymmetric vector α via Blei 2003 Appendix A.4.2 (closed-form
Sherman-Morrison Newton on the diagonal-plus-rank-1 Hessian, O(K) per
step); symmetric scalar η via Hoffman 2010 §3.4 scalar Newton. Defaults
flip to match MLlib parity (`optimizeDocConcentration=True`,
`optimizeTopicConcentration=False`); the v0 divergence recorded in
ADR 0009 is gone. A six-lesson post-implementation walkthrough and a
D=10k empirical α-recovery probe followed; the probe surfaced one
substantive numerical bug (a digamma-underflow cascade in the α
sufficient-statistic accumulator that catastrophically collapsed every
α component to the floor on certain random inits) which got fixed in
the same arc.

### What shipped

- [`_alpha_newton_step`](../spark-vi/spark_vi/models/lda.py#L107-L152)
  — pure-function closed-form Newton step using the matrix-inversion
  lemma applied to `H = c·1·1ᵀ − diag(d)`. Caller does ρ_t damping and
  the post-step floor; helper does the math.
- [`_eta_newton_step`](../spark-vi/spark_vi/models/lda.py#L155-L186) —
  scalar Newton helper. Mirrors α's separation-of-concerns.
- [`VanillaLDA.local_update`](../spark-vi/spark_vi/models/lda.py#L273-L352)
  conditionally accumulates `e_log_theta_sum` (length-K, per-doc Dirichlet
  expected log) when `optimize_alpha=True`. Stat key is absent from the
  return dict when off, so `optimize_alpha=False` pays zero overhead and
  the framework's `combine_stats` reducer transparently ignores the
  missing key.
- [`VanillaLDA.update_global`](../spark-vi/spark_vi/models/lda.py#L354-L438)
  optionally fires `_alpha_newton_step` on the corpus-rescaled
  `target_stats["e_log_theta_sum"]` and `_eta_newton_step` on a stat
  computed inline from the just-updated λ. Both apply the shared ρ_t
  schedule from λ (Hoffman 2010 §3.3) and clip at 1e-3.
- [`VanillaLDAEstimator`](../spark-vi/spark_vi/mllib/lda.py) — the
  ADR 0009 validator's two rejections (`optimizeDocConcentration=True`,
  vector `docConcentration`) removed; new `optimizeTopicConcentration`
  Param mirrors `optimizeDocConcentration`'s shape and help text.
  `_build_model_and_config` plumbs both flags through to the model.
  `VanillaLDAModel` gains `alpha` / `topicConcentration` accessors that
  read from the fitted `VIResult` rather than the constructor Param —
  important because `transform` must use the *trained* α.
- [`VIModel.iteration_summary`](../spark-vi/spark_vi/core/model.py)
  optional hook returning a model-defined string appended to the
  runner's per-iteration log line. Default `""`. VanillaLDA returns a
  compact `α[min/max/mean], η, Σλ_k[min/max]` summary.
- Per-topic α / Σλ / peak prefix on the existing topic-evolution
  callback in
  [`analysis/cloud/lda_bigquery_cloud.py`](../analysis/cloud/lda_bigquery_cloud.py),
  sorted by Σλ descending so the heaviest topics are read first.

### Walkthrough lessons

**Lesson 1 — ELBO architecture.** The `L = L_α(α; γ) + L_η(η; λ) +
L_other(γ, λ, φ)` decomposition. Holding the variational posteriors
`q(θ_d)=Dir(γ_d)` and `q(φ_k)=Dir(λ_k)` fixed, `L_α` and `L_η` are
classical Dirichlet MLE-from-pseudocounts objectives — the same shape
as Blei 2003's offline α-estimator, just with the pseudocounts coming
from variational expectations rather than EM-style soft assignments.

**Lesson 2 — α Newton step.** The gradient `g_k = D[ψ(Σα) − ψ(α_k)] + Σ_d
E[log θ_dk]` and Hessian `H = c·1·1ᵀ − diag(d)` with `c = D·ψ'(Σα)`,
`d_k = D·ψ'(α_k)`. The diagonal-plus-rank-1 structure admits a
matrix-inversion-lemma closed-form (Blei A.2): `Δα_k = (g_k − b)/d_k`
where `b = Σ(g_j/d_j) / (Σ 1/d_j − 1/c)` — the rank-1 "everyone moves
together" coupling. Walked line-by-line against
[`_alpha_newton_step`](../spark-vi/spark_vi/models/lda.py#L107-L152).
Discovered en route: Blei 2003 Appendix A.4.2's *printed* Hessian has a
transcription sign/factor error (missing `M` on the second term); a
re-derivation gives the negative-definite `H` above and matches MLlib's
`OnlineLDAOptimizer.updateAlpha`. Documented at
[lda.py:128-130](../spark-vi/spark_vi/models/lda.py#L128-L130).

**Lesson 3 — η Newton step.** Scalar collapse of α's machinery: gradient
and Hessian become 1-D, no inversion lemma needed. Walked
[`_eta_newton_step`](../spark-vi/spark_vi/models/lda.py#L155-L186) in
three lines. The interesting structural point: η's sufficient statistic
`Σ_t Σ_v E[log φ_tv]` depends only on λ (a global parameter on the
driver), not on per-doc state. So the entire η update lives inside
`update_global` with no contribution from `local_update` and no
mini-batch corpus rescale. This is the asymmetry that templates HDP's
two concentration updates: γ on the corpus stick reuses η's global-stat
shape; α on the doc stick reuses LDA-α's per-doc shape.

**Lesson 4 — Wiring α through the SVI loop.** Five-stop pipeline:
per-doc accumulation inside the partition loop → conditional return
from `local_update` → cross-partition `treeReduce` via `combine_stats`
→ runner's `D / |batch|` rescale to corpus-equivalent → driver-side
Newton step + ρ_t damping + floor. The transparent stat-key pattern
(model adds a key, runner doesn't change) is how this lands without a
framework change.

**Lesson 5 — Wiring η through the SVI loop.** Stops 1-4 of α's
pipeline collapse to nothing because η's stat is computable from λ.
Driver-only update; no `local_update` involvement. Computed against
the `new_lam` (just-updated) rather than the `lam` (entry value), to
keep η and λ co-current — small choice with a clean justification.

**Lesson 6 — Test design.** Three-tier pyramid: helper unit tests with
idealized inputs (
[`test_alpha_newton_step_recovers_known_alpha_on_synthetic`](../spark-vi/tests/test_lda_math.py),
[`test_eta_newton_step_recovers_known_eta_on_synthetic`](../spark-vi/tests/test_lda_math.py));
ELBO-trend integration gate (existing
[`test_vanilla_lda_elbo_smoothed_endpoints_show_overall_improvement`](../spark-vi/tests/test_lda_integration.py));
end-to-end smoke gate
([`test_alpha_optimization_runs_end_to_end_without_regression`](../spark-vi/tests/test_lda_integration.py)).
Each tier closes a regression class the others can't see. The
originally-planned "α drifts toward truth" test was cut at D=200
because of topic-collapse pinning components to floor independent of
optimization quality; this lesson contextualized the cut, and a
follow-up D=10k probe walked it back (see below).

### Refactor detours that shipped

**Detour 1 — Citation correction (396f32c).** During Lesson 2, fetched
the Blei 2003 JMLR PDF to confirm the §5.4 reference cited in 10 sites
across the spec, plan, ADR 0010, and code comments. §5.4 is titled
"Smoothing" and is about the prior on β, not about α optimization. The
Newton derivation actually lives in **Appendix A.4.2** (LDA-specific
specialization) layered on **Appendix A.2** (general structured-Hessian
Newton via matrix-inversion lemma). Corrected all 10 sites; documented
A.4.2's printed-Hessian transcription error in the helper docstring so
a future reader doesn't try to "fix" the negative-definite version.

**Detour 2 — ELBO test rename (7399cd3).** Empirical probe across
multiple window sizes and iteration counts showed the smoothed ELBO
trace has ~50% positive consecutive differences even on healthy fits —
i.e., the trace is *not* monotonic under any window. The original test
name `test_vanilla_lda_elbo_smoothed_trend_is_non_decreasing` implied
monotonicity it never asserted. What the test actually checks is the
endpoint trend: `smoothed[-1] > smoothed[0]`. Renamed to
`test_vanilla_lda_elbo_smoothed_endpoints_show_overall_improvement`
and rewrote its docstring to make explicit what survives (gross
sign-flips that drag the bound down overall) and what doesn't (subtler
mid-trace regressions that preserve the endpoints). Spec / plan / ADR
0010 cross-references updated.

**Detour 3 — Per-iteration logging hook (9bee128 + dc9aa2e).** Added
optional `VIModel.iteration_summary(global_params) -> str` returning a
short string the runner appends to its per-iter log line. Default is
empty. VanillaLDA returns scalar globals (η, Σα, Σλ row range) on the
inline runner line; the cloud script's existing topic-evolution
callback picks up per-topic α / Σλ / peak as a prefix on each topic
row, sorted by Σλ descending. Native topic index is preserved as the
label so a topic moving in the ranking is itself a signal.

### Pre-existing issues caught (digamma-underflow cascade)

A D=10k synthetic α-drift probe surfaced a numerical bug that wasn't
visible in the unit suite. The sequence:

1. With asymmetric `true_α=[0.1, 0.5, 0.9]` and random λ init, some
   seeds produce a fitted-topic mass distribution where one topic is
   starved (Σλ_k ≈ 600 vs ~200K-400K for siblings). Standard SVI
   topic-collapse driven by the asymmetric truth.
2. Iter 1's α-Newton step correctly drives the starved topic's α
   toward zero — clipping at the 1e-3 floor.
3. Iter 2 CAVI initializes γ_d for the orphan topic component near
   `α[k]` = 1e-3. ψ(1e-3) ≈ -1000, and `exp(ψ(γ_dk) − ψ(Σγ))`
   underflows to zero.
4. The α-suff-stat accumulator at
   [models/lda.py](../spark-vi/spark_vi/models/lda.py) was computing
   `e_log_theta_sum += np.log(expElogthetad)` — `log(0) = -inf`
   silently propagated.
5. -inf reached `_alpha_newton_step` and corrupted *every* component
   of Δα via the rank-1 Hessian coupling
   `b = Σ(g/d) / (Σ 1/d − 1/c)`. All α components clipped to floor on
   iter 2.

Pre-fix sweep over five seeds at D=10k: 3/5 catastrophically collapsed.
β recovery in those runs was unaffected (cosine ≥ 0.99 across seeds);
only α died. Post-fix: 0/5 collapse, drift to truth +43.6% to +62.5%.

The fix replaces `log(exp(...))` with the equivalent
`digamma(γ) − digamma(γ.sum())` directly. Costs one extra digamma
call per CAVI exit (length-K vector, sub-microsecond at K=20).
The original code's TODO comment at lda.py:338-341 chose the
`log(exp(...))` form for "one log/doc beats two digammas/doc on the
Spark hot path" — sound when expElogthetad isn't near zero, which is
the regime *until* α hits the floor. Neither the original
implementation review nor the post-implementation review caught this;
it surfaced only when the long-running probe stress-tested seeds the
default unit-test scale couldn't reach. Diagnostic methodology
captured at
[`probes/diagnose_collapse_in_spark.py`](../spark-vi/probes/diagnose_collapse_in_spark.py).

### Tests

- **New unit tests:**
  [`test_alpha_newton_step_recovers_known_alpha_on_synthetic`](../spark-vi/tests/test_lda_math.py)
  and
  [`test_eta_newton_step_recovers_known_eta_on_synthetic`](../spark-vi/tests/test_lda_math.py)
  — synthetic recovery against idealized variational inputs (delta
  posteriors at the truth) for the closed-form helpers.
  [`test_alpha_newton_step_floors_at_1e-3`](../spark-vi/tests/test_lda_math.py)
  — pathological gradient direct test of the floor logic.
- **New contract tests** in
  [`test_lda_contract.py`](../spark-vi/tests/test_lda_contract.py): the
  optimize-flag plumbing, `local_update` conditionally emitting
  `e_log_theta_sum`, `update_global` Newton-step wiring (both α and
  η), and — added in the cascade-fix detour —
  `test_local_update_alpha_stat_finite_when_alpha_at_floor` which
  constructs the pathological state by hand and asserts the suff-stat
  stays finite. Verified to fail on the reverted code (returns
  `[-inf, -0.69, -0.69]`).
- **New slow integration test:**
  [`test_alpha_optimization_runs_end_to_end_without_regression`](../spark-vi/tests/test_lda_integration.py)
  — D=200 smoke gate (wiring fires, no NaN, floor honored, no
  blow-up). Originally drafted with an "α drifts toward truth"
  assertion at this scale; reframed during implementation when
  topic-collapse made truth-recovery untestable at D=200 (Hoffman 2010
  §4 used D=100K-352K to validate recovery). The reframe is documented
  in the test docstring and ADR 0010.
- **New slow recovery test (post-cascade-fix):**
  [`test_alpha_optimization_drifts_toward_corpus_truth_at_D10k`](../spark-vi/tests/test_lda_integration.py)
  — fits at D=10k for 300 iters, Hungarian-aligns fitted topics to
  true topics by β cosine similarity (`scipy.optimize.linear_sum_assignment`
  on negated cosine), asserts ≥30% L1 reduction toward
  `true_α=[0.1, 0.5, 0.9]`. Empirical sweep across five seeds showed
  +43.6% to +62.5% drift; 30% leaves margin for cross-platform
  numerical drift. Runtime ~43 s. This resurrects the recovery
  ambition the original test design was forced to abandon at smaller
  scales.
- **One-off probes** kept in tree under
  [`spark-vi/probes/`](../spark-vi/probes/) as empirical justification
  for the recovery threshold and the cascade fix:
  `alpha_drift_probe.py` (D=10k seed sweep + 2000-iter long fit),
  `diagnose_collapse.py` (offline numpy reproducer — eliminated the
  obvious causes), `diagnose_collapse_in_spark.py` (in-Spark per-iter
  dump — identified the actual mechanism). Not test code; not
  invoked from the test suite.

Final unit suite: 67 passing (up from 65 on initial implementation).
Slow integration suite: 4 passing.

### Methodology lessons surfaced

**Stage-1 vs stage-2 failure-mode separation.** The post-fix sweep
shows topic starvation (stage 1 — random init orphans a fitted topic)
still happens at D=10k for some seeds, but the consequence is now
mild: one α component pinned at floor, others healthy. The cascade
(stage 2 — numerical contagion through the rank-1 Hessian) was the
load-bearing failure mode; the orphan-topic mode is a tolerable SVI
characteristic. Naming the two stages separately during the diagnosis
is what made the one-line fix obvious — without that decomposition
the bug looks like "LDA collapses sometimes," which is
under-actionable.

**Hidden RNG state defeats reproducibility.** [models/lda.py:262
and :305](../spark-vi/spark_vi/models/lda.py#L262) use `np.random.gamma`
against the global numpy RNG state, not seeded by `cfg.random_seed`.
The first probe runs got drift +60% then -55% at the same `cfg.random_seed`
because the global RNG state differed run-to-run. The recovery test
seeds the global RNG explicitly as a workaround; the real fix is the
TODO at
[lda.py:302-304](../spark-vi/spark_vi/models/lda.py#L302-L304) (derive
per-doc seed from `(cfg.random_seed, doc_key)` deterministically),
parked.

**Test design vs corpus scale.** Recovery testing has a corpus-scale
floor below which the test is uninformative regardless of optimizer
quality. The original "α drifts to truth at D=200" test was empirically
unattainable; "α drifts to truth at D=10k" is. The threshold question
is corpus-scale, not implementation-quality — the literature reference
(Hoffman 2010 §4) was telling us this all along; we didn't internalize
it until the empirical probe forced the issue.

### Doc updates

- [ADR 0010 — Concentration parameter optimization](decisions/0010-concentration-parameter-optimization.md):
  ships the decision, including the asymmetric-α / symmetric-η pattern
  per Wallach 2009, the shared ρ_t damping (Hoffman 2010 §3.3), the
  1e-3 floor, the test reframe, and the corrected Blei 2003 citation.
- [Spec: 2026-05-05-lda-concentration-optimization-design](superpowers/specs/2026-05-05-lda-concentration-optimization-design.md):
  the design walkthrough; updated mid-arc with the citation
  correction, the test reframe, and the renamed ELBO endpoint test.
- [Plan: 2026-05-05-lda-concentration-optimization](superpowers/plans/2026-05-05-lda-concentration-optimization.md):
  the 13-task TDD plan executed via subagent-driven development; final
  verification block updated to reference the renamed ELBO test.

### Open threads parked

- **Per-doc deterministic RNG.** TODO at
  [`lda.py:302-304`](../spark-vi/spark_vi/models/lda.py#L302-L304):
  derive each doc's `gamma_init` seed from `(cfg.random_seed,
  doc_key)` so SVI fits are reproducible end-to-end without relying
  on the global numpy RNG state. The recovery test currently seeds
  `np.random` as a workaround.
- **Asymmetric η** (per-vocabulary). MLlib doesn't do this, mini-batch
  SVI is least stable on η, and Wallach 2009 argued symmetric η is
  the right default. Parking-lot per ADR 0010.
- **Concentration-specific learning rate.** Empirical observation:
  ρ_t is calibrated for λ stability (high-dimensional, mini-batch
  noisy) and is overkill for α (low-dimensional, scalar Newton for η).
  At D=10k, 300 iters reaches +62% drift; 2000 iters reaches +90% with
  Σα still climbing. A faster α/η-specific schedule would close the
  gap, at the cost of breaking the unified-schedule contract MLlib
  follows. Park unless empirical demand appears.
- **MLWritable round-trip of optimized α/η.** Persistence of the
  fitted concentrations remains an ADR 0009 v1 punt; once the shim
  implements `MLWritable`, both `model.alpha` and
  `model.topicConcentration` need to round-trip.
- **Online HDP** is the next major work item. The two HDP
  concentration parameters (γ on the corpus stick, α on the doc
  stick) reuse this branch's machinery directly: γ ≅ η (global stat,
  scalar Newton, lives in `update_global`), α (HDP) ≅ α (LDA)
  (per-doc stat, structured-Hessian Newton, lives in `local_update` +
  rescale). Building both flavors here was the de-risking step.

---

## 2026-05-05 — Strict persist precondition for VIRunner

A parking-lot item from the prior session, picked up after a successful
live LDA fit on the cluster confirmed the surrounding pipeline was
healthy. The work promotes the soft `_log_persist` diagnostic in the
driver script to a hard precondition inside `VIRunner.fit`, with a
small companion module `spark_vi.diagnostics.persist` housing the check.
Outcome: the framework now refuses to enter its iteration loop on
uncached input rather than silently re-executing the upstream lineage
(BigQuery scan + CountVectorizer) every iter — a multi-minute regression
class that was previously only catchable by reading per-iter wall-time
trends.

### What shipped

- [`spark_vi/diagnostics/persist.py`](../spark-vi/spark_vi/diagnostics/persist.py)
  with a polymorphic `assert_persisted(target, name)`. RDD path queries
  `getRDDStorageInfo()` keyed by `rdd.id()` and requires
  `numCachedPartitions > 0` — the rigorous block-manager truth check.
  DataFrame path checks `df.storageLevel != NONE` on the public surface.
- [`VIRunner.fit`](../spark-vi/spark_vi/core/runner.py#L103-L108) now
  strict-asserts on `data_rdd` after the resume/init block and before
  the loop. Failure raises `RuntimeError` with an actionable message
  pointing at the missing `.persist()` or `.count()` step.
- [`VanillaLDAEstimator._fit`](../spark-vi/spark_vi/mllib/lda.py#L271-L291)
  persists + counts the derived `bow_rdd` before constructing the
  runner, with a `try/finally` unpersist after the fit returns. Needed
  because `dataset.select(...).rdd.map(...)` always yields a fresh
  uncached RDD even when `dataset` is DataFrame-cached upstream — the
  shim is the only place that can satisfy the runner's precondition on
  the actual RDD it sees.
- Driver script: [`_log_persist`](../analysis/cloud/lda_bigquery_cloud.py)
  helper deleted; the two call sites (`omop`, `bow_df`) now use
  `assert_persisted`.
- Test surface: ~10 existing `runner.fit(rdd)` call sites in
  [`test_runner.py`](../spark-vi/tests/test_runner.py),
  [`test_checkpoint.py`](../spark-vi/tests/test_checkpoint.py),
  [`test_lda_integration.py`](../spark-vi/tests/test_lda_integration.py),
  and [`test_broadcast_lifecycle.py`](../spark-vi/tests/test_broadcast_lifecycle.py)
  updated to pre-persist+count their input RDDs. The pattern is
  intentionally inline rather than fixture-wrapped so the contract is
  visible at every call site. New
  [`test_persist_check.py`](../spark-vi/tests/test_persist_check.py)
  pins the six failure modes (cached vs forgotten-action vs
  never-persisted, for both RDD and DataFrame; plus type rejection).
- All 95 spark-vi tests + 27 charmpheno tests pass.

### Pre-existing issues caught (mid-walkthrough)

A short post-implementation walkthrough surfaced one finding worth
acting on: the original `_assert_df_persisted` had two checks
(`df.storageLevel != NONE` followed by a JVM cache-manager probe via
`spark._jsparkSession.sharedState().cacheManager()`). Tracing
PySpark's `Dataset.storageLevel` to its Scala source showed it is
*itself* implemented via that same cache manager — so the second probe
was strictly redundant with the first, and reaching through `private[sql]`
internals to obtain the same answer was net-negative. Simplified
[`_assert_df_persisted`](../spark-vi/spark_vi/diagnostics/persist.py#L88-L101)
to use only `df.storageLevel`; comment at lines 89-94 records the
redundancy reasoning so a future reader doesn't re-introduce the probe.

### Design rationale captured in code

- **Spot-cluster tolerance.** The check is `numCachedPartitions > 0`,
  not `== numPartitions`, so partial preemption loss between persist
  and fit-entry doesn't false-positive — Spark transparently recomputes
  lost partitions on next access. A stricter check would routinely raise
  on healthy spot clusters. Documented in the module docstring at
  [lines 15-20](../spark-vi/spark_vi/diagnostics/persist.py#L15-L20).
- **Private-bridge access in the RDD path.** `sc._jsc.sc().getRDDStorageInfo()`
  reaches through PySpark's private Java handle to call a public Scala
  method that has no Python equivalent. The risk note at
  [lines 67-74](../spark-vi/spark_vi/diagnostics/persist.py#L67-L74)
  documents that a Spark major-version rename would break loudly
  (AttributeError at fit entry) rather than silently — failure mode is
  itself a signal.
- **DataFrame check is intentionally weaker.** PySpark exposes no public
  way to verify materialized blocks for DataFrames; the simpler check
  catches "forgot persist" and "accidentally unpersisted" but not
  "registered but no blocks materialized." Adequate because the
  shim's downstream `bow_rdd` persist+count materializes upstream
  caches via lineage anyway. Documented in the function's leading
  comment.
- **One-shot precondition, not a continuous guarantee.** Mid-fit
  eviction is intentionally not detected. Re-checking inside the loop
  would be invasive and the value would be observability rather than
  correctness — Spark's recompute-and-recache already handles eviction
  correctly. Documented in the module docstring at line 19.

### Threads parked

- **MLlib-style persistence (Pipeline.save / MLWritable / MLReadable)**
  per ADR 0009 — still open from the prior session; not advanced here.
  The ADR called out the deferral ("v1 ships without persistence") and
  that remains the standing decision.
- **OMOP concept-hierarchy rollup** to collapse near-duplicate concepts
  like the two "Type 2 diabetes mellitus" entries observed in topic 6
  during the live fit. User explicitly deferred ("after we do HDP or
  other fancy models") — the rollup is a data-prep change touching the
  vocab path, and the framework-side investment is more valuable while
  the model surface is still expanding.

### Memory cleanup

Two completed parking-lot memories removed: the strict-persist promotion
landed in this session, and the inspect_app history-server drop landed
in the previous session. `MEMORY.md` index pruned to match.

### Out of scope

- Any change to the Makefile, `setup_workspace.py`, or the cluster-side
  inspect dashboard.
- Mid-iteration cache integrity checks; intentionally deferred (see
  rationale above).
- Touching `VanillaLDAEstimator`'s instance-attr `_on_iteration` design
  — it's still outside the Param surface for the same ADR 0009 reason.

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
