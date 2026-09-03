---
id: 110
slug: native-mondo-label-space
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE LABEL-SPACE RUN. 0109's front matter with ONE knob changed: dag_source goes from
# `mondo` to `mondo_native`, and `dag_collapse` disappears (the native build applies
# the splice itself, so the flag would only ask for it twice — the driver pins it off).
# Window, strip, mask, vocab, head params, seed and spark_conf are copied verbatim so
# 0104's and 0109's recorded numbers stay legitimate controls: the corpus DOCUMENTS are
# identical, only the label space over them changes.
#
# ID NOTE: main also has an 0109 (`mondo-hpo-dual-axis`); this branch's 0109 is
# `whole-mondo-collapsed-dag`. The collision is real and predates this run — the two
# histories have no merge base. This experiment takes 0110 on both sides so it cannot
# collide either way; the dual-0109 gets resolved (rename ours, or accept the dual with
# a cross-reference) when the histories unify. Do not renumber this to 0109.
#
# What the knob does: the label space stops being a hierarchy RECONSTRUCTED from anchor
# covers and becomes Mondo's own graph. Three changes, one flag:
#   1. ATTESTATION — a doc attests the MOST SPECIFIC Mondo terms its in-window codes
#      map or climb to (standard-exact, else nearest mapped SNOMED ancestor via
#      concept_ancestor, tie-reduced to most-specific). 0104/0109 attested EVERY
#      powered ancestor of every code.
#   2. POWERING — a node is kept when >= min_positives distinct persons fall in its
#      is-a CLOSURE. "Directly coded" becomes a node property, not a node type, so
#      terminals and class covers collapse into one rule.
#   3. THE DAG — Mondo's is-a order transitively reduced over the kept set, then the
#      0109 splice as a thin-chain post-pass. In a transitive reduction a node cannot
#      be a sibling of its own descendant, which is what makes 0104's 619 subsumed
#      category-anchors structurally impossible rather than merely rarer.
#
# C AND K ARE EXPECTED TO GROW, and are MEASURED at build time, not guessed: closure
# support >= direct support, and mid-level Mondo terms carrying no code of their own
# now qualify. The build prints both receipts before any fit. Do not size the cluster
# off 0104's C=3,820 / K=3,827 without reading them.
#
# PRE-INDEX CLOSURE (E1, `preindex_closure: true`). The corpus additionally carries
# a per-document sparse `R_d` column: the closure of what the patient already
# carried BEFORE the index — the label definition evaluated on the FEATURE window
# instead of the label window, over the same kept-node DAG and the same provider.
# It is what makes incident eligibility (`c ∉ R_d`, spec D2) computable at eval
# time, and it is a CORPUS property: computed once at build time, stored with the
# corpus, reused byte-identically by every run compared against this one, with
# nothing a fit produces entering it. The immediate consumer is the E-census
# GO/NO-GO probe (`make diag-incident-census`), which must run on THIS corpus
# before any incident metric is built.
#
# IT MOVES THIS RUN'S BUNDLE CACHE KEY. The flag folds into the key only when on
# (the `dag_collapse` discipline), so 0104's and 0109's keys — and every
# flag-off key — are byte-identical; but a 0110 bundle built before this line was
# added is under a DIFFERENT key and will not be found. That is the intended
# trade: the census is a property of 0110's corpus and has to be measured on the
# corpus the record run reports, so the column ships WITH that corpus rather than
# in a second, separately-keyed one. Cost is one extra full-history condition scan
# plus one attestation pass at build time; a cache HIT pays nothing.
dag_source: mondo_native
preindex_closure: true
readout_mode: distributed
readout_theta_topm: 256
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
# COPIED VERBATIM FROM 0104. This block encodes a week of cluster forensics (run log
# 08-26 through 08-29: the "kill swarms" were executor JVM heap OOMs self-masked by
# Dataproc's -XX:OnOutOfMemoryError='kill %p'), and it is the ONLY geometry that has
# survived a full whole-Mondo solve. Do not re-derive it; do not trim it.
spark_conf:
  # THE week-of-kill-swarms root cause (run log 08-29): executor JVM heap OOM,
  # self-masked by Dataproc's -XX:OnOutOfMemoryError='kill %p' (every heap OOM
  # presents upstream as "killed by external signal"). At K≈3,827 a tree-combine
  # task holds several ~117 MB (C,K) pickled partials at once; 4 concurrent tasks
  # against the 6g default heap (shared with the cached packed corpus) blow up
  # thousands of stages in. THIS config — 2 concurrent tasks against an 8g heap,
  # 11g container — survived 449 passes / iter 50 with ZERO executor deaths
  # (08-29), and is the ONLY one of the two candidates that fits the cluster:
  # yarn.scheduler max container is 13,544 MB on these workers (the 12g+6g
  # variant was refused at submit). On beefier workers a wider shape (cores=4 +
  # memory=12g + overhead=6g) is worth re-trying via GPR_SPARK_CONF first.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  # Dynamic allocation OFF, fixed executors. DA is confirmed enabled on the
  # cluster, and the 08-28 evening relaunch hardened the case that DA idle-release
  # IS the kill mechanism: containers were killed "on request" / "by external
  # signal" on PRIMARY workers (w-2 here; w-0/w-1 in earlier waves) — hostnames
  # GCP cannot preempt — right after the driver-side reconstruct/transform gap
  # where every executor idles past DA's 60s timeout, and the kill/task-launch
  # races then feed excludeOnFailure until the scheduler starves (stage 3, 4 min
  # in). A minExecutors=8 floor did NOT stop it. A long batch solve wants a
  # deterministic executor set anyway: DA off + an explicit instance count sized
  # to the cluster (12 nodes x 1 executor of 4 cores/6g+4g). YARN grants what
  # exists and holds the rest pending, so a smaller cluster still runs — adjust
  # instances when the cluster shape changes. If kills persist on primaries even
  # with DA off, the mechanism is platform-level (autoscaler/YARN), not Spark.
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 12
  # Preemption-wave exclusion starvation (08-28 evening: the solve died 9,112s in
  # when a retry of task 0 "cannot run anywhere due to node and executor
  # excludeOnFailure" aborted the TaskSet). At the 1h default, every spot kill wave
  # adds hosts to the app-level exclude list and none age out inside a multi-hour
  # solve, so the schedulable set only shrinks. Dataproc secondary workers come back
  # under the SAME hostnames, so an aged-out entry is a REUSABLE node, not a stale
  # one — 10m is long enough to route around a genuinely bad host and short enough
  # that a preemption wave's collateral is forgiven before the next wave lands.
  spark.excludeOnFailure.timeout: 10m
  # Driver-disk leak #2 mitigation (0109 run log, 08-31/09-01): a second ENOSPC
  # consumer survives every ADR 0047 destroy fix — ~100 GB of master disk over a
  # few hours with the Python broadcasts provably released. The suspect is
  # ContextCleaner-gated JVM-side driver state (torrent-broadcast pieces for task
  # binaries, shuffle metadata), which Spark frees only when the DRIVER JVM
  # garbage-collects, and a large mostly-idle driver heap may never trigger a GC
  # across a multi-hour solve. A forced periodic GC bounds the backlog at a cost
  # of one full GC per interval on an idle heap. Evidence pending: the in-band
  # disk telemetry (analysis/cloud/disk_telemetry.py) now prints one
  # `disk_telemetry:` line every 120s into the persisted job log, so the next run
  # says whether this is the mechanism or only a partial mitigation.
  spark.cleaner.periodicGC.interval: "5min"
---

# 0110 — Native Mondo label space: map-and-roll labels, keep what's supported

**Why.** Two experiments and a week of predict-refute-diagnose have decomposed exp
0104's 763 degenerate label heads (C=3,820) exactly:

| bucket | n | fixed by |
|---|---:|---|
| root | 1 | structural, by design |
| structural only-child class nodes | 143 | exp 0109's splice-to-fixpoint |
| **subsumed category-anchors** | **619** | nothing yet — this run |

The 619 are the deep one, and they are a property of the CONSTRUCTION, not the data.
Three facts compose into them:

- the climb attests **every powered ancestor** of a coded concept;
- `reduce_to_anchor_hierarchy` nests anchors only under class covers, so a
  category-anchor sits as a **sibling of its own descendants**;
- closure masking hands out negatives only via siblings.

So a category is co-attested on every doc that fires a "sibling" and is never observed
as a negative: an all-positive cell, a constant head, on 619 common clinical categories.

The obvious next patch — nest anchor-under-anchor — is a hand-rolled local
approximation of what Mondo's own graph does globally. In a **transitive reduction** of
the real ontology restricted to the kept nodes, a subsumed sibling cannot exist: the
redundant sibling edge is exactly what the reduction deletes. The limit of the patch
sequence is to stop reconstructing Mondo's hierarchy from covers and use Mondo's
hierarchy. That is this run.

## Hypothesis

1. **Degenerate trajectory 763 → 620 → ~1.** The headline structural claim. Post-build
   the degenerate set should be the root plus a small, **named** thin-chain residue
   (coded terms that are genuine only children — the splice protects them because
   patients really attest them). Anything beyond that refutes the account and must be
   enumerated before any other number here is believed. This is a CORPUS property, so
   `diag-sibling-support` settles it off the bundle alone — no record fit needed to
   learn it. See the protocol.
2. **Macro AUC/AP on the shared node set is comparable to the 0104 record
   (0.6978 / 0.4845).** Not obviously up: the nodes 0104 scored well are mostly
   terminals, and they keep their positives. What changes for them is a cleaner sibling
   contrast (their categories are now ancestors, not siblings) and a differently-sized
   K. Flat-to-slightly-up is the expectation; a material drop is a real finding.
3. **The deliverable claim is the FULL label space, not the shared subset.** Every
   formerly-constant category head that becomes genuinely scoreable counts —
   "hypertensive disorder" moving from a constant column to a real AUC is the point of
   the run, and it cannot appear in a shared-node comparison at all.
4. **Detection is a real AUC.** The constant-column fix is always-on since 0109; with
   ~1 constant column left there is nothing for it to exclude, which is its own
   confirmation of (1).

## What changes, precisely

One flag, three coupled changes (`analysis/cloud/mondo_native_dag.py` has the full
argument; this is the summary):

| | 0104 / 0109 (`dag_source: mondo`) | 0110 (`dag_source: mondo_native`) |
|---|---|---|
| attestation | every powered ancestor of every code | the **most specific** mapped Mondo term(s) per code |
| node identity | OMOP concept id (terminal) / synthetic negative (class) | Mondo term id, uniformly |
| powering | terminal patient count ≥ 100, class nodes taken from covers | **is-a closure support** ≥ 100, uniformly |
| "directly coded" | a node TYPE (terminal vs class) | a node PROPERTY |
| parents | single nearest superset cover (a tree) | Mondo's is-a order, transitively reduced (a DAG) |
| thin chains | `dag_collapse: true`, opt-in | intrinsic to the build |

Everything else — the documents, the windows, the vocabularies, the strip, the mask
mode, the head — is byte-identical to 0109.

## C and K must be MEASURED, not guessed

Closure support ≥ direct support, and a mid-level Mondo term with no code of its own can
now clear the bar on its descendants' patients. Main's shipped `source_climb` stats
(9,927 mapped terms, 4,822 used, 3,063 with >20 persons) say the candidate pool is
larger than 0104's 3,820-node DAG, and the powering rule is more generous, so **C is
expected to grow and K = n_bg + tpn·C with it**. Two receipts print before any fit:

```
[mondo-native] powering: <N> standard code(s) resolve to <M> Mondo term(s); <P> term(s)
  carry closure support, <Q> clear min_positives=100 (smallest kept support <S>);
  <R> code(s) attest the final DAG
[mondo-native] label DAG (native-mondo-v1): <Q> powered term(s) (<C> directly coded);
  induced Hasse <H> node(s), <X> with >1 parent; splice removed <A> thin-chain +
  <B> childless in <P> pass(es) -> <F> node(s) (<Y> multi-parent);
  predicted residual degenerate = <D>
```

**Record Q, F, X and Y in the run log before committing cluster time.** The
`spark_conf` block below is the only geometry that has survived a whole-Mondo solve at
K≈3,827; if F comes back materially larger, the executor-heap arithmetic in its
comments (a tree-combine task holding several ~117 MB (C,K) pickled partials) has to be
redone before a record fit, and `min_positives` is the dial that buys it back.

`X` and `Y` — the multi-parent counts — are also the first real measurement of whether
Mondo's ~50% multi-parenthood survives into a label DAG. 0104's accidental tree meant
no diamond was ever exercised in the layout, the closure mask or the conditional
readout. The pre-flight checklist for that is unit-tested (`DagLayout.closure` /
`allowed` visit a diamond once; `frontier_to_label`'s sibling expansion unions over ALL
parents; `_dag_children_and_depth` lists a child under both parents), but a nonzero Y
is what makes those tests load-bearing rather than hypothetical.

## Two places the design was not implementable as specified

Recorded here because they change what the run measures, not just how it is built. Both
are argued at length in `analysis/cloud/mondo_native_dag.py`'s module docstring.

1. **`nearest_mapped_parents` is NOT a transitive reduction**, though main's docstring
   calls its output "the induced Hasse edges". Nearest-*per-branch* admits a distant
   ancestor as a parent whenever an intermediate is unpowered (A→B→D, A→C→D, B dropped
   ⇒ D gets {A, C}, and C is an ancestor of D) — which rebuilds the subsumed-sibling
   shape inside the construction meant to make it impossible. The reduction is applied
   explicitly (`induced_hasse_parents`) and the acceptance property — *no kept node is
   a sibling of its own descendant* — is a unit test over every kept subset of a
   diamond. Had this gone unnoticed, hypothesis (1) would have failed on the cluster
   for a reason nobody would have found quickly.
2. **The source-exact rung of the ladder is OFF, and not by choice.** Rung 1
   discriminates two rows sharing a standard concept but differing in source code, so
   it needs a frame carrying `condition_source_concept_id`. The corpus's label frame is
   built from `condition_era`, which has no source-concept column at all (this is why
   main's own driver refuses the source spaces unless `--source-table
   condition_occurrence`). Adding it means editing `cohorts.py`, the
   maximum-blast-radius hashed module. So the ladder here is two rungs (standard-exact,
   then climb), and the SAME two-rung resolution both powers and attests — deliberately,
   because a node powered by evidence the labels cannot express would be kept and then
   never attested, manufacturing exactly the `no_pos` degenerates this run is trying to
   drive to zero. Consequence for the numbers: coverage sits between main's measured
   2.1% unmapped (3 rungs) and 5.7% (standard-space only).

A third, smaller one: **engine ids are ints, not Mondo curies.** The plan called for
`int2cid` to carry curie strings; `attach_frontiers` hard-casts attested ids with
`int()`, the attestation column is typed `array<bigint>`, and the drift gate re-reads
the map as ints — all in source-hashed modules. Nodes are therefore keyed by the
curie's numeric part (`MONDO:0004995` → `4995`), which is injective, stable across
builds (unlike `mondo_dag`'s enumeration-order synthetic negatives) and reversible via
`mondo_curie()`. Per-node reports read `name_by_id`, so they are unaffected.

## Comparison protocol

Same documents, same fit knobs, same seed, same `spark_conf` — a different label space
and nothing else. 0104 (macro AUC 0.6978 / AP 0.4845) and 0109 are the controls.

1. **The two build receipts**, above. Q/F/X/Y into the run log before any fit.
2. **`diag-sibling-support` BEFORE the record fit.** Degeneracy is a corpus property,
   so it needs only the bundle — the diagnostic reads the cached train labels, not the
   model. (The driver has no build-only mode, so in practice it runs off the cheapest
   bundle-producing launch; see the Run section.) Expect
   `root: 1`, `allpos_no_sibling_support` ≈ 0, `allpos_with_supported_sibling` = 0. A
   nonzero `no_pos` bucket is the signal that the powering rule (whole-population
   closure support) and the corpus (a sampled, windowed label frame) have drifted
   apart — a real finding about `min_positives`, not about the DAG.
3. **The readout banner: 763 → 620 → ~1.** The falsifiable claim, restated against the
   fit. If the degenerate count is not ≈1 + the named residue, enumerate the remainder
   before believing anything else here.
4. **Detection AUC is real**, and the constant-column exclusion line reports ~0.
5. **Macro AUC/AP on the SHARED node set.** Join `results_readout.json`'s `per_node`
   rows against 0104's and 0109's, restrict to node ids present in all three, and
   compare there — the headline macros are over different label spaces and a difference
   between them could be pure composition. **The join key needs care**: 0104/0109 rows
   are keyed by OMOP concept id and 0110's by Mondo term id, so the intersection has to
   be taken through the mapping (`mondo_to_omop_mapping.build_mondo_to_omop`'s
   `standard_concept_id ↔ mondo_id`), not on the raw id. Nodes that map many-to-one in
   either direction are excluded from the shared set rather than resolved.
6. **Macro AUC/AP on the FULL new label space** — the deliverable claim (hypothesis 3).
   Report alongside the count of heads that were constant in 0104 and are scoreable
   here.
7. **Conditional `P(child | parent)` metrics.** The edge set changes completely (real
   is-a edges, multi-parent), so expect a different, larger edge count. This is the
   first run where a child can have two kept parents, which is what the conditional
   readout's cohort construction has never been exercised on.
8. **Cost.** K moves with C; if C grows, per-iteration fit cost and the readout's C·K
   L-BFGS state grow with it. Compare against 0109's wall clock with the C ratio in
   hand.

## Cache / reproducibility notes

`mondo_native` is a **fold-when-on, versioned** corpus input, the same discipline
`dag_collapse` established. The bundle key folds `mondo_native`,
`mondo_native_version` (`native-mondo-v1`) and the source hashes of
`mondo_native_dag`, `mondo_usage_core` and `mondo_collapse` **only when the flag is
selected**, so every SNOMED key and every `dag_source: mondo` key — exp 0104's record
bundle and exp 0109's — is byte-identical to what it was before this existed and both
reproduce unchanged. The four pinned tripwire hashes in
`tests/scripts/test_case_finding_cache_mondo.py` are unmoved.

This run therefore takes a cache MISS on its first launch and assembles its own bundle
(~20 min of BigQuery plus the native build's own `concept_ancestor` climb and closure
aggregation), which is correct: it is a different corpus.

`dag_collapse` is pinned OFF here in both the spec builder and the manifest recovery —
the splice is part of the native construction, so the flag could only apply it twice.
The recovery path has its own guard: a native fit's ids are ints, so
`mondo_spec_mismatch`'s `MONDO:`-prefix witness cannot see it, and
`native_spec_mismatch` reads the manifest's own `dag_source` instead and exits 2 on a
contradicting override rather than rebuilding a different label space under a wrong key.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/gated-conditional-voi

# 1. BUILD the native bundle and read C/K off the receipts. There is no build-only
#    mode in the driver, and `--diag-only` still fits — so the cheapest path to a
#    cached bundle + a manifest the diagnostic can key from is a DEV run with
#    `diag_only: true` temporarily added to the front matter below: 30 iterations,
#    no theta collect, no readout, no baselines. THE TWO BUILD RECEIPTS PRINT
#    DURING ASSEMBLY, before iteration 1 — watch for them and kill the job there if
#    C came back materially larger than 0104's 3,820 (see "C and K must be
#    MEASURED"); the bundle is already written through by then, so nothing is lost.
CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=110
#   -> record: [mondo-native] powering: ... / [mondo-native] label DAG: ...
#              [driver]   corpus: V=(...) K=... C=...  and the [cost] block

# 2. THE DEGENERACY GATE, on the corpus, before any RECORD fit. Degeneracy is a
#    corpus property, so it needs the bundle, not a fit — which is why it runs here
#    rather than after step 3. It requires a cache HIT (step 1 wrote it through).
make -C analysis/cloud diag-sibling-support ID=110
#   -> expect root: 1, allpos_no_sibling_support ~ 0,
#              allpos_with_supported_sibling = 0, and a NAMED thin-chain residue

# 2b. THE INCIDENT-ELIGIBILITY CENSUS (E-census) — the GO/NO-GO gate for the
#    incident evaluation program. Also a CORPUS property, also before any record
#    fit, also a cache HIT (step 1's bundle carries E1's `preindexClosure` column
#    because this run's front matter sets `preindex_closure: true`).
make -C analysis/cloud diag-incident-census ID=110
#   -> prints the count of nodes clearing 20/20 on BOTH incident classes against
#      the spec's ~300 bar, the C2.1 population (train-degenerate heads that
#      acquire incident negatives), and the constant-head fate breakdown; writes
#      incident_census.json into the run dir.
#   -> RECORD THE COUNTS AND THE GO/NO-GO CALL IN THE RUN LOG BELOW. The tool
#      reports; the human decides. On NO-GO the incident macro is not a
#      deliverable, E2/E3/E4 are not built, and that run-log entry is the
#      program's terminal record for them. Given 0109's root prevalence 0.9609
#      this is a live outcome, not a formality — and the numbers are recorded as
#      a finding either way.

# 3. The 30-iter dev smoke proper (only if step 2 clears): drop `diag_only` from the
#    front matter and re-run. The bundle is a HIT now, so this is fit + readout only.
CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=110
make -C analysis/cloud report ID=110

# 4. Record run — only after the smoke's numbers are in the run log AND the executor
#    arithmetic has been re-checked against the measured C. Flip `status:` to
#    `running` first.
CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=110
make -C analysis/cloud report ID=110

# Recovery re-readout on a SAVED fit. The fit records dag_source: mondo_native, so no
# override is needed; passing a contradicting --dag-source exits 2 by design.
make -C analysis/cloud gated-pc-readout ID=110
```

The front matter's `spark_conf:` is injected into both commands automatically — see
0104's note; it is copied here verbatim from 0109 and must not be retyped into
`CHARM_SPARK_CONF` / `GPR_SPARK_CONF`.

## Run log

**2026-09-01 — smoke attempt 1: the gated_pc arm LANDED IN FULL, then the driver died
asking an unsupervised fit for its co-fit head.** Everything through the headline was
clean: fit, distributed readout over 1,855 nodes, detection, and the
conditional-sharpening table all printed. The driver then raised
`[UNRESOLVED_COLUMN.WITH_SUGGESTION] ... probability` out of the co-fit-head block —
`_collect_lean_proba(test_scored, C, score_col="probability")`. Root cause is a config
contract, not a corrupt frame (the SAME `test_scored` had just served the gated_pc arm's
collect on `topicDistribution`): `OnlinePCLDAModel._transform` appends `probability`
ONLY when `weightY != 0`, and the whole-Mondo mainline — 0104, 0109 and this run alike —
fits at `weight_y: 0`. There is no co-fit head at weight_y=0; its weights are at the
zero seed and `predictProbability` refuses outright for exactly that reason.

Why this run and not 0104/0109, which share the front matter: the block has been
unguarded since `c938c5b` (2026-08-20), and `a7f724d` (08-21) fixed precisely this crash
in the RE-READOUT tool (`gated_pc_readout`) while leaving the fit driver's copy
unconditional. 0104's numbers came out of that re-readout ("co-fit arm skipped
(weight_y=0, as designed)"), and 0109 never reached the block — both of its readouts
died earlier on the driver-disk ENOSPC. **This was the first run whose readout survived
far enough inside the fit driver to reach the line.** Nothing in `fc679e6`/`96d871b`
changed the block or a guard on it.

Fixed (same commit as this entry): the driver skips the co-fit head arm on the same two
witnesses as the re-readout tool — `weight_y == 0` or no `probability` column — with a
printed SKIP line. And `_retry_spark_action` now fails FAST on `AnalysisException`: a
schema error is decided during query analysis before a task is submitted, so the
60/120/240s of backoff this run spent on it was guaranteed waste that also dressed a
code bug up as cluster trouble.

**Front-matter audit (same day): 0110 vs 0109 differ ONLY in `id` / `slug` / `status` /
`dag_source` / `dag_collapse`.** Every model and corpus knob — `weight_y: 0.0`,
`readout_theta_topm`, head params, window, vocab, mask, seed, and the whole `spark_conf`
block — is byte-identical, and 0104's differs from 0109's only by `dag_collapse`. The
dropped `dag_collapse: true` is a no-op rather than a silent change of intent:
`multidomain_corpus_spec` pins `dag_collapse` to False whenever `dag_source != "mondo"`,
so the native path could not have honoured the flag anyway (the splice is intrinsic to
that build). **The smoke's numbers are therefore mainline-comparable to 0104/0109 as
designed** — the crash cost the secondary head arm only, and the gated_pc arm above it
is a legitimate reading.

**2026-09-01 — build receipts, dev-smoke numbers, and the degeneracy verdict.**
(The smoke fit preceded the diag because the protocol's steps ran as one chained
command — harmless, the diag is a corpus property and the cache was hot; the record
run should keep the diag-first order.)

**Build receipts (native DAG, 83.2 s):** 31,016 standard codes → 8,894 Mondo terms;
6,032 with closure support; **2,984 clear min_positives=100**; induced Hasse 2,985
nodes (1,378 multi-parent); splice removed 256 thin-chain + 15 childless in 2 passes
→ **C=2,714 (1,277 multi-parent, 47%)**; 2,501 directly coded, so 483 kept terms
carry no direct codes — "directly coded" is a node property now, as designed.
Builder's structural prediction: residual degenerate = 1. Corpus:
V=(5,000 cond + 5,000 meas + 1,601 drug), **K=2,721** (8 bg + 2,713×1 tpn). C and K
SHRANK vs 0109 (3,677/3,684): the growth worry never materialized — the old space's
1,306 class covers were the bulk. The 0104/0109 memory geometry now has slack.

**Dev smoke (30-iter fit, 60-iter readout budget, weightY=0.0 on the fit banner):**

| 60-iter readout budget | macro AUC | macro AP | scored | detection AUC | constant cols |
|---|---|---|---|---|---|
| 0109 smoke (C=3,677) | 0.6894 | 0.4747 | 2,106 | 0.5622 | 619 |
| **0110 smoke (C=2,714, native)** | **0.7288** | **0.3975** | **1,855** | **0.6203** | **176** |

Comparison-protocol discipline applies: DIFFERENT node sets, so the macro delta is
directional, not a verdict — the shared-node-set comparison comes from the two runs'
per-node rows. The AP drop is composition: the native space scores deep rare nodes
(marginal AP at depth 7 ≈ 0.03) the old space never surfaced. Solver conditioning
transformed: 292 passes for 60 iters (≈4.9 passes/iter vs 15–25 deep-phase on
0104/0109), 6 line-search failures, 1 stalled, ~44 s/iter at 20 executors.
First-ever real multi-parent conditional readout: pooled ECE 0.0235, cond-AUC
0.65–0.77 by depth, top-1 beats majority broadly at depths 2–3 (immune 0.688 vs
0.369; connective tissue 0.548 vs 0.356), with the tool's own caveat that per-node
ECE (mean 0.132) ≫ pooled.

**Degeneracy verdict (`diag-sibling-support`, train split): 177/2,714 — root 1,
no_pos 0, allpos_no_sibling_support 167, allpos_with_supported_sibling 9.** The
trajectory, honestly booked: **763 → 620 → 177**, and the TRAP CLASS the arc chased
— subsumed siblings — went **619 → 9**. The builder's "residual = 1" held for the
mechanism it modeled and missed two residue classes it never modeled:

- **The 9 are tie-map twins, not the old trap.** The giveaway is IDENTICAL n_obs
  between sibling pairs (bilateral/unilateral renal agenesis both 113;
  adolescent/juvenile idiopathic scoliosis both 56): `reduce_tie_map` keeps tie
  LEAVES, so a code tied between sibling terms attests both, they co-fire on every
  doc, and neither is ever observed negative against the other.
- **The 167 (0/0 siblings) are the only-child class in a new costume.** The native
  splice protects TERMINALS, and directly-coded nodes are terminal — so a
  directly-coded only-child (esophageal disorder, galactorrhea, `human disease` as
  the lone depth-1 node) survives the splice and is then observed only when it
  fires under sibling-only closure masking. 0109's splice never met this class:
  the old space's only-children were abstract covers, never terminal-protected.

**The principled fix for the 167 is NOT another splice patch**: an only-child's
missing negatives are exactly "parent fires without the child" — the D5
local-negative definition in the incident-episode eval spec
(`docs/superpowers/specs/2026-09-01-incident-episode-eval-program.md`). The masking
rule and the eval program converge on the same repair; it is a train-time masking
change and stays deferred with it. Residual constant-head rate: **6.5% of C**, vs
16.9% (0109) and 20% (0104). `no_pos: 0` — closure-support powering leaves nothing
label-starved.

**2026-09-02 — RECORD RUN LANDED; census GO; experiment closed.** (First record
attempt died executor-side on a missing `--py-files` entry for
`preindex_closure.py` — the module's UDF pickles by reference; fixed in 9f7f002
and shipped on the fit and readout-recovery submits. The relaunch went
end-to-end.)

**Record numbers (full budgets, 20 executors, fit total 36,337 s ≈ 10.1 h):**

| record budget | macro AUC | macro AP | scored | detection AUC | constant cols |
|---|---|---|---|---|---|
| 0104 record (C=3,820, anchor space) | 0.6978 | 0.4845 | 2,106 | (pre-fix artifact) | 619 |
| **0110 record (C=2,714, native)** | **0.7350** | **0.4074** | **1,855** | **0.6475** | **176** |

P@R0.5/0.8/0.9 = 0.397/0.285/0.253; R@FDR0.1/0.25/0.5 = 0.144/0.259/0.401.
Smoke→record gap small (+0.006 AUC, +0.010 AP over the 60-iter smoke) — the
solver was already near its answer at smoke budget, consistent with the improved
conditioning. Node sets differ across the spaces (and across id SYSTEMS — OMOP
anchors vs Mondo terms), so the macro delta is directional; the shared-node
comparison per §Comparison protocol requires the xref mapping and is the one
analysis still owed.

Conditional sharpening (held-out isotonic): pooled ECE raw 0.0508 → **calibrated
0.0176**; cond-AUC 0.67–0.78 by depth; top-1 beats majority at essentially every
mid-depth parent (immune 0.695 vs 0.369, syndromic 0.667 vs 0.451). Per-node ECE
mean 0.132 ≫ pooled — the tool's own pooling-flatters caveat stands. The per-node
domain λ-mass table is the first look at multi-domain attribution in the native
space: some heads are drug-dominated (cystitis 0.977 drug — antibiotic signature),
others near-pure condition (postsurgical hypothyroidism 1.000).

**Solver note for the optimization backlog:** the CALIBRATION solve (75% split,
warm-started) consumed ~8 h of the 10 — full 200 iterations, 5,884 passes,
2,734 line-search failures, 328 stalled nodes, deep-phase ~30 passes/iter over
~96 nodes/pass. The line-search backtrack cap (offered, never requested) and
ship-only-searching-rows partials are now the dominant wall-clock lever.

**Diag (record corpus): identical to the smoke — 177 = 1 root + 167
splice-protected only-children + 9 tie-map twins; no_pos 0.** Reproduced under
the new bundle key with the preindexClosure column present.

**CENSUS (E-census): GO.** 2,222/2,714 nodes clear min_count=20 on BOTH incident
classes (bar ~300); positives-only 157, negatives-only 285, neither 50, zero
nodes without eligible docs. **C2.1 population = 0**: every train-degenerate head
stays all-positive under incident eligibility (still-all-positive 177 /
acquired-negatives 0), so the forced-0.5 macro-inflation channel is EMPTY on this
corpus — the R2.1 constant-column guard ships anyway as belt-and-braces.
Decision recorded: **GO — E2/E3/E4 (WP4/WP5/WP6) proceed.**

**2026-09-02 (cont.) — E4 future-conversion: the PU channel-1 FLOOR.**
`diag-conversion-analysis ID=110 --deciles off`, over incident negatives (index
had no prior attestation of closure(c); label window 365d):

| horizon | nodes | incident negatives (obs) | later convert | rate |
|---|---|---|---|---|
| 365d  | 2394 | 13,876,314 | 132,824 | **0.0096** |
| 730d  | 2354 | 10,910,423 | 193,419 | **0.0177** |
| 1095d | 2320 |  8,539,451 | 216,189 | **0.0253** |

A LOWER BOUND on PU channel 1 (label noise from a not-yet-made diagnosis);
channels 2 (never diagnosed) and 3 (diagnosed in care this CDR can't see) stay
unmeasured — read each rate as "at least this many of these negatives are
wrong." The right-censoring gate on `observation_period_end_date` (R4.4) is
visibly working: denominators shrink with the horizon. Interpretation: the floor
is LOW, so the incident-negative labels are mostly clean and the incident metrics
are not badly PU-deflated — the "negatives aren't really negatives" worry is
bounded small on the measurable channel. CAVEAT: pooled across nodes weighted by
negative count (common nodes dominate; per-node detail in the workspace-internal
`conversion_analysis.json`), and this is a CONTAMINATION FLOOR, not case-finding
validation — a pooled 0.96%/yr is not yet distinguished from background
incidence. The score-decile enrichment that WOULD distinguish them (R4.8: do the
model's top-scoring negatives convert materially above its bottom-scoring ones?)
is pending the `--deciles on` run, which needs the persisted readout heads from a
readout under commit 10bdf1d (`make gated-pc-readout ID=110`).

**2026-09-02 (cont.) — E2 incident metrics + E4 decile validation: the eval program's payoff.**
(Both off the saved fit; the readout persisted `readout_heads_gated_pc.npz`, so
the deciles scored with no re-fit and no disk death — the 10bdf1d/6bbd9ad fix
working.)

**Tracking vs prediction (spec E2 / D7 — a PREVALENT-fit model on an INCIDENT
cohort; discrimination, never prospective). On the SAME 1,599 shared nodes:**

| arm / node set | AUC | AP | nodes |
|---|---|---|---|
| prevalent / full | 0.7350 | 0.4074 | 1855 |
| prevalent / shared | 0.7412 | 0.4361 | 1599 |
| **incident / shared** | **0.6741** | **0.2429** | 1599 |

Excluding prior carriers of closure(c) from both classes — leaving only genuine
new-onset — costs **~0.067 AUC**: that much of the prevalent headline was
tracking (autocorrelated re-coding of an already-known condition). The true
forward-prediction signal is 0.6741, well above chance. AP nearly halves
(0.436→0.243) because incident cohorts are far lower-prevalence once prior
carriers leave, and AP is prevalence-sensitive where AUC is not. Skipped columns
(counted separately, never summed): degenerate 192, small (<20 incident) 923,
**CONSTANT 0** — the census-predicted-empty C2.1 population confirmed; R2.1's
constant-column guard is present and correctly fired zero times.

**Case-finding validation (spec E4/R4.8): decile-stratified conversion of
incident negatives — MONOTONIC, top ≈3× bottom, at every horizon.**

| horizon | d0 | … | d9 | top−bottom |
|---|---|---|---|---|
| 365d  | 0.006 | ↗ | 0.020 | +0.0140 (3.3×) |
| 730d  | 0.011 | ↗ | 0.036 | +0.0241 (3.3×) |
| 1095d | 0.017 | ↗ | 0.049 | +0.0323 (2.9×) |

The PU floor (0.96%/1.77%/2.53% pooled, prior entry) could not, alone, be
distinguished from background incidence. This decile gradient distinguishes it:
background incidence would be FLAT across deciles; instead the model's own
"negatives" convert to the real diagnosis at a rate climbing monotonically with
the score it gave them. The label-noise channel is concentrated exactly where the
model points — the contamination IS the model surfacing not-yet-diagnosed cases.
Prospective-within-retrospective case-finding validation, no chart review. At 3y a
top-decile "negative" carries ~1-in-20 future-diagnosis odds vs ~1-in-60 at the
bottom.

**The incident-episode eval program (audit → spec → plan → E1..E4) is complete
and delivered on the 0110 native corpus.** E5 (episode-anchored sampling) remains
0111, deferred by design. Still owed to the record: the shared-node comparison
against 0104's anchor-space record (needs the OMOP↔Mondo xref to align id systems).

---

**2026-09-03 — derived work, run closed.** Everything that scoped exp 0111 ran
OFF THIS RUN'S PERSISTED ARTIFACTS with zero re-fits and zero corpus
re-assembly — the persisted-heads npz and the E4 sidecar paying for three
analyses they weren't built for:

- **0111 scout** (`docs/reports/2026-09-02-0111-scouting-window-depth.md`):
  eval-side horizon sweep REFUTED window-widening (AUC falls 0.6003→0.5857;
  the AP rise is a base-rate artifact — insight 0077); depth breakout is a
  reporting characterization (shallow-concentrated, present everywhere).
- **Episode probes** (`docs/reports/2026-09-02-0111-episode-probe-results.md`,
  `diag-episode-probe` off this run's sidecar): 2,583/2,714 nodes reach ≥20
  gated onset episodes vs this run's 923 starved — episode anchoring un-starves
  the label space; 66.2% of FIRST episodes die to the 365d prior-obs gate
  (insight 0078). Decisions locked for 0111: gap 90d, cap 3, distributed eval
  as a precondition.

Insights promoted from this run: **0075** (tracking vs prediction, ~0.067 AUC),
**0076** (prospective-within-retrospective decile validation), **0077** (window
refutation), **0078** (episode probe). Spec for the successor:
`docs/superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md`. This
run's numbers are the last of their doc unit that future work compares against
directly — 0111 changes the document unit and carries its own random arm as
control (insight 0010); 0110 continues as the *artifact* supplier (bundle,
sidecar, heads), not as a numeric baseline for episode arms.
