---
id: 109
slug: whole-mondo-collapsed-dag
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE STRUCTURAL-DEFECT FIX RUN. Byte-for-byte 0104's front matter with ONE knob added:
# dag_collapse. Everything else — corpus inputs, windowing, vocab, head params, seed,
# spark_conf — is copied verbatim so the two runs differ in the LABEL DAG and nothing
# else, and 0104's recorded numbers are a legitimate control.
#
# What the knob does: after the Mondo hierarchy is powered and reduced, repeatedly
# SPLICE OUT every class node with exactly one kept child (wiring its parents straight
# to that child) and DROP every class node left with none, to fixpoint. Terminals (the
# 2,513 powered anchors) and the root are never removed, so the label CONTENT is
# unchanged; only the abstract scaffolding between anchors shrinks.
#
# Why it should matter: 0104's readout banner reads "3,057 fittable nodes, 763
# degenerate (constant fallback)" at C=3,820, and the degenerate set is exactly
# {root} u {only-children}. Mondo is a multi-axis DAG; `reduce_to_anchor_hierarchy`
# gives each node its single NEAREST superset cover as parent, so overlapping covers
# "steal" terminals from one another and leave class nodes holding a single child.
# Under label_mask_mode: closure such a node is observed only on the rows its one child
# covers, so its observed train cell is single-class, its readout column is a constant
# fallback, and (0109 part B) that constant column pinned detection AUC at 0.5000.
# Predicted after the collapse: exactly ONE degenerate node, the root.
dag_collapse: true
readout_mode: distributed
readout_theta_topm: 256
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
dag_source: mondo
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

# 0109 — Whole-Mondo on a collapsed DAG: paying off the 763 degenerate nodes

**Why.** exp 0104 landed the first whole-Mondo numbers and, in the same banner, the
diagnosis of a structural defect it could not fix: **`3,057 fittable nodes, 763
degenerate (constant fallback)`** out of C=3,820. That is not a rare-node/small-cell
problem — it is one fifth of the label space carrying no information at all, and the
set has now been characterized exactly:

> **the degenerate set is `{root} ∪ {class nodes with exactly one kept child}`.**

The mechanism is construction, not data. Mondo is a multi-axis is-a DAG, and
`anchor_hierarchy.reduce_to_anchor_hierarchy` assigns each kept node its single
*nearest* superset cover as parent. When two class nodes have overlapping (not nested)
terminal covers — A={t1,t2,t3}, B={t1,t2}, C={t2,t3} — the shared terminals are STOLEN
by whichever cover sorts first, and C is left holding only {t3}. The flattening turns
the DAG into a tree and strews it with class nodes that have exactly one kept child.
Under `label_mask_mode: closure` such a node is observed only on the rows its one child
covers, so within its own observed train set it is constant-1: the per-node readout
cell is single-class, and the head emits a constant fallback column.

Those 763 nodes cost three things: K topic-blocks that learn nothing, C readout rows
that are fit and reported as degenerate, and — via the per-doc max — a **detection AUC
pinned at exactly 0.5000** in every 0104 readout.

**What changes.** Two independent fixes, one opt-in and one always-on.

- **A. `dag_collapse: true` (this run's only knob).** After powering and reduction,
  repeatedly splice out every class node with exactly one kept child (its parents wire
  straight to that child) and drop every class node left with none, iterating to
  fixpoint so a whole chain collapses. Terminals — the powered anchors patients
  actually attest — are never spliced, so **no label content is lost**; the run's
  positives per surviving node are identical. The root stays (one node, structural).
  The DAG build prints its own receipt: how many nodes were spliced, how many childless
  ones dropped, and the **predicted residual degenerate count, which should be 1**.
- **B. constant columns excluded from the detection pool (always on, both DAG paths).**
  A pre-existing EVAL bug, not a modeling change: the detection score is a per-doc max
  over node columns, so a single constant column above the informative ones makes the
  max constant and the AUC exactly chance. A constant column carries no per-document
  information by construction, so excluding it cannot lose signal. **Ranking and
  per-node metrics are untouched** — they score each column against its own labels and
  already report a degenerate node as skipped.

## Comparison protocol (this is the point of the run)

Same corpus inputs, same fit knobs, same seed, same `spark_conf` — a collapsed label
DAG and nothing else. 0104's recorded numbers are the control. Read, in order:

1. **The DAG-build receipt.** `[mondo]   dag-collapse (splice-fixpoint-v1): spliced N
   only-child class node(s), dropped M childless, in P pass(es); nodes 3820 -> ...;
   predicted residual degenerate = 1`. Expect N+M ≈ 762 and a predicted residual of 1.
2. **The readout banner: 763 → ~1.** The falsifiable claim. If the fittable count does
   NOT rise to ≈ the new C−1, the "only-children are the degenerate set" account is
   wrong and the remaining degenerate nodes must be enumerated before anything else in
   this doc is believed.
3. **Detection is no longer chance.** 0104: `detection (case vs bg) AUC=0.5000`. Here it
   must be a real number, and the line now names how many constant columns it excluded
   (post-collapse that count should be ~0, which is its own confirmation of item 2).
4. **Macro AUC/AP ON SHARED NODES vs 0104.** Do NOT compare the headline macros
   directly: 0104 macro'd over 2,106 scored nodes of a 3,820-node DAG and this run
   macro's over a different node set, so a difference could be pure composition. Join
   `results_readout.json`'s `per_node` rows on concept id, restrict to nodes present in
   both, and compare macro AUC/AP on that intersection. Expectation: **flat to slightly
   up**. The spliced nodes were carrying no information, but they were consuming topic
   blocks (K shrinks by ~762 × tpn) and closure mask area, so the surviving nodes get a
   slightly less diluted θ and a cleaner sibling contrast. A material DROP means the
   scaffolding was doing latent work (plausible: a class node's topic block can act as a
   shared basis for its subtree even when its own label is degenerate) — that would be a
   real finding and an argument for splicing the LABEL row while keeping the topic
   block, which this reduction deliberately does not attempt.
5. **Conditional (`P(child | parent)`) metrics.** The collapse REMOVES DAG edges, so the
   edge set the conditional readout scores changes: chains of one-child parents are
   gone, and the surviving parents have their real children. Expect fewer edges, each
   more meaningful; a rise in `cond_auc` here is the sharpening story the collapse is
   supposed to help.
6. **Cost.** K drops with C, so per-iteration fit cost and the readout's C·K L-BFGS state
   both shrink ~20%. Any wall-clock number that does NOT improve is worth a look.

## Cache / reproducibility notes

`dag_collapse` is an **opt-in, versioned** corpus input. It is folded into the bundle
cache key (with `dag_collapse_version` and the reduction module's source hash) **only
when it is on**, so every collapse-OFF key — including exp 0104's cached record bundle —
is byte-identical to what it was before this existed, and 0104 reproduces unchanged.
This run therefore takes a cache MISS on its first launch and assembles its own bundle
(~20 min of BigQuery), which is correct: it is a different corpus.

The reduction lives in `analysis/cloud/mondo_collapse.py` rather than inside
`mondo_dag.py` for exactly that reason — `compute_bundle_cache_key` folds
`_module_source_hash(mondo_dag)`, so any edit to that file (a comment included) would
move every Mondo key and orphan every cached bundle in every bucket.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/gated-conditional-voi && \
  CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=109  # smoke first
CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=109
make -C analysis/cloud report ID=109

# Recovery re-readout on a SAVED fit. Fits from now on record every Mondo build input
# INCLUDING dag_collapse, so no --dag-collapse flag is needed; it is here as the
# override for a manifest that ever loses it (a wrong value MISSES the cache rather
# than mis-scoring, and the drift gate catches a wrong rebuild).
make -C analysis/cloud gated-pc-readout ID=109 GPR_ARGS="--dag-collapse on"
```

The front matter's `spark_conf:` is injected into both commands automatically — see
0104's note; it is copied here verbatim and must not be retyped into
`CHARM_SPARK_CONF` / `GPR_SPARK_CONF`.

## Run log

**2026-08-30/31 — smoke attempt 1: the collapse RAN and its prediction was REFUTED —
the structural theory explains 143 of the 763 degenerates, not 762.** The fit
completed (30 dev iterations, early-saved) and the banner read: **C=3,820 → 3,677
(143 nodes removed), fittable 3,057 (unchanged), degenerate 763 → 620** — against the
diagnostic's prediction of 1. Two facts fall out immediately:

1. Every node the splice removed was degenerate (fittable count identical), so the
   reduction did exactly what it says on nodes that ARE structural only-children /
   childless classes — there were just far fewer of them than the "{root} ∪
   only-children exactly" characterization claimed.
2. The remaining **619 non-root degenerates are OBSERVATIONAL, not structural**: the
   leading hypothesis is a multi-child class node whose siblings contribute no
   observed rows — an only-child *in the data* even though not in the graph (under
   closure masking its observed cell is then single-class exactly as if the siblings
   did not exist). Sibling-support analysis of the 620 (from the saved fit's manifest
   + bundle) is the next diagnostic; a data-aware collapse (splice when the sibling
   OBSERVED support is empty, not just when the sibling set is) is the candidate v2
   of the reduction if it holds.

**2026-08-31 — RESIDUAL DEGENERACY SOLVED: subsumed category-anchors.** Two rounds of
the sibling-support diagnostic (v1 by child support, v2 by sibling support — each
refuted its predecessor with perfect bookkeeping: 620 = 1 root + 619 all-positive,
`n_pos == n_obs` EXACTLY on every one, siblings fully supported) forced a read of the
actual mechanisms, which chain as:

1. `powered_anchor_climb` attests EVERY powered ancestor of a coded concept
   (`concept_ancestor` unrestricted by depth) — a "Graves disease" doc attests both
   the Graves anchor and the "Disorder of thyroid gland" anchor above it.
2. `reduce_to_anchor_hierarchy` nests anchors under CLASS covers only, so a powered
   category-anchor and its own powered specific descendants land as SIBLINGS under
   the same class node.
3. `label_mask_mode: closure` observes a node only where it is positive or a SIBLING
   of a positive — so a category-anchor is co-attested (fact 1) on every doc that
   activates any of its "siblings" (fact 2) and is NEVER observed as a negative:
   all-positive cell, constant fallback.

Full decomposition of 0104's 763: 1 root + 143 structural only-children (this exp's
splice removed exactly those) + 619 subsumed category-anchors. The remedy for the 619
is a DAG-BUILD change, not a splice: nest anchor-under-anchor (make specific anchors
children of their category anchors per the same subsumption the climb already uses),
so a category-anchor's siblings become OTHER categories — which do fire without it —
and its cell gets real negatives. Payoff is not cosmetic: those 619 are common,
clinically meaningful category labels ("neoplasm of breast", "urinary tract
infectious disease", "disorder of thyroid gland") that currently return constants;
nesting makes them genuinely scoreable heads. Design note for v3: this also restores
real DAG depth that the class-cover-only nesting flattened away, so K, the gate, and
the conditional readout all see a better hierarchy — it is the principled fix the
splice was the down payment on.

The mechanism and fix, in one figure (anchors ● are coded+powered, class covers ▢
are abstract):

```
 TODAY (anchors nest under class covers only):        THE FIX (nest anchor under anchor):

        ▢ endocrine disorder                              ▢ endocrine disorder
   ┌───────┬────────┬───────────┐                     ┌─────────┴──────────┐
 ●thyroid ●Graves ●hypothyr. ●parathyr.        ●thyroid disorder    ●parathyr.
   ╰── category sits BESIDE its ──╯                ┌─────┴──────┐          ⇧ real
       own specifics (siblings)                 ●Graves ●hypothyr.      negatives

 climb: a code attests EVERY powered ancestor → firing any "sibling" of the
 category auto-fires the category → closure masking (negatives come only from
 siblings) can NEVER observe it as a negative → n_obs == n_pos → constant head.
 Nesting makes the category's siblings OTHER categories (which do fire without
 it) and the specifics' siblings their co-specifics (the conditional contrast).
```

The run then died at readout solve iteration ~55-60 (961 passes, 441/3,057 converged,
max|grad| 281) on the SAME driver-disk ENOSPC from `sc.broadcast` — with both destroy
fixes verifiably active (no fallback warnings in 961 passes, and pyspark 3.5's
`destroy()` confirmed to unlink the temp file). ADR 0047's mechanism is therefore
incomplete: something else consumes ~100 GB of master disk across a ~19k-second
fit+readout. The next recovery runs with a per-minute disk watcher (df + du over the
Spark temp dir) to catch the growth live instead of post-hoc; the fit is early-saved
and the solve checkpointed at iteration ~50 (v2 fingerprint), so the recovery resumes
rather than re-paying. Notable solver observation for the record: the collapsed DAG's
line search works measurably harder (961 passes by iter ~55 vs 0104's ~500 by 60).

**2026-09-01 — recovery readout died of the SAME driver-disk ENOSPC, and the second
autopsy was lost too. Leak #2 is recurrent, and the watcher moves in-band.** The
recovery re-readout reached solver iteration ~55 (~14.2 ks elapsed, 441/3,057
converged) and took the ENOSPC from `sc.broadcast` again, with every ADR 0047 destroy
fix active — the same signature as 08-31, so leak #2 is reproducible and not a
one-off. The cluster then timed out overnight, and the `nohup diskwatch` loop that was
supposed to catch the growth wrote to the master's local `~`: it died with the
machine, unread. That is the SECOND lost autopsy by the same mechanism, and it is why
the disk watcher is no longer a shell loop.

The watcher is now IN-BAND (`analysis/cloud/disk_telemetry.py`, started by both
`gated_pc_cloud` and `gated_pc_readout` right after the SparkSession exists): one
`disk_telemetry:` line to driver stdout every 120 s, carrying per-filesystem
used/avail and the six biggest top-level entries of each watched dir, which is what
separates JVM block-manager state (`blockmgr-*`) from Python broadcast temp files
(`spark-*/pyspark-*`) from log growth. Dataproc persists job driver stdout to the
staging bucket, so it rides the run log automatically and cannot die with the cluster.
Shipping alongside it, from the JVM-side hypothesis: `spark.cleaner.periodicGC.interval:
"5min"` in the front matter (ContextCleaner frees torrent-broadcast pieces and shuffle
metadata only on driver GC, which a large idle heap never triggers).

The run dir is on the bucket-mounted `RUNS_DIR` and is expected to still hold the
iteration-50 solver checkpoint, so `make -C analysis/cloud gated-pc-readout ID=109`
resumes from it rather than re-paying the first 50 iterations — now with the telemetry
and the periodic GC in force. What to read off the next attempt: whether the
`disk_telemetry:` used-G series climbs monotonically, and which top-level entry the
growth is under.

**2026-09-01 — the smoke numbers existed all along: a 60-iteration arm COMPLETED on
08-31 at 20:19 and dumped `results_readout.json` before the higher-budget
continuation started (and died).** The run dir's timestamps reconstruct it: arm
completes → results written 20:19, checkpoint deleted on completion → continuation
relaunches, writes a fresh iteration-50 checkpoint at 01:09, ENOSPCs at ~55. The
completed arm is the dev-smoke readout, budget-matched to 0104's 60-iteration smoke:

| 60-iter solve budget | macro AUC | macro AP | scored | skipped |
|---|---|---|---|---|
| 0104 smoke (C=3,820) | 0.6891 | 0.4745 | 2,106 | 1,714 |
| **0109 smoke (C=3,677, spliced)** | **0.6894** | **0.4747** | **2,106** | **1,571** |

A statistical tie on an IDENTICAL scored population — the splice's report card, and
the intended one: it removed only structurally-degenerate nodes (never scoreable),
so surviving-node metrics are unchanged while the DAG drops 143 nodes. The
arithmetic self-checks close: skipped 1,571 = 3,677 − 2,106, and detection reports
`n_constant_nodes: 619` — exactly the subsumed-category-anchor count from the
decomposition above, counted independently by the constant-column exclusion.

Detection is also the FIRST REAL detection number for the whole-Mondo corpus (0104
recorded the pre-fix 0.5000 artifact): **AUC 0.5622 / AP 0.9658 at prevalence
0.9609**, pooled over the 3,058 non-constant nodes. Read it with the prevalence in
view — 96% of docs attest something in a 3,677-node ontology, so AP is ceiling-high
trivially and AUC barely clears chance; per-doc "any label" detection is a nearly
degenerate task at this scale. The per-node ranking metrics are the meaningful axis;
detection stays a diagnostics line.

The record-budget continuation (readout_max_iter 200) is back in flight from the
iteration-50 checkpoint with telemetry and periodic GC active; its numbers, against
0104's record 0.6978/0.4845, close this experiment.

**2026-09-01 (final) — the continuation was STOPPED DELIBERATELY; the 60-iter smoke
tie above is this experiment's final result.** Rationale: the smoke-vs-smoke tie on
an identical 2,106-node scored population already delivers the experiment's entire
claim (the splice is metrics-neutral on surviving nodes and removes only
never-scoreable ones), and the structural numbers (763 → 620, splice removed exactly
the 143 structural only-children) are the trajectory input the native-label program
needs. A deeper-budget rerun of the same tie decides nothing: exp 0110's acceptance
protocol takes 0104's RECORD (0.6978/0.4845) as the deep-budget control, and this
experiment's role in that protocol is structural, not metric. Cluster time goes to
0110 instead.

Two operational findings from the final restart, recorded here because they came out
of this run: (1) checkpoint resume across a deliberate kill + relaunch worked again
(483/3,057 converged at resumed iteration 1); (2) scaling the fixed-instance geometry
from 8 to 20 executors (`GPR_SPARK_CONF="spark.executor.instances=20"` overriding the
front matter's 12) dropped the solve from ~60 s/pass to ~15 s/pass — the map stage
dominates and scales nearly linearly; note the front matter's `executor.instances`
caps utilization regardless of cluster size, so match instances to workers when
resizing. And the run's in-band disk telemetry is what localized driver-disk leak #2
(pyspark auto-broadcast of the dense treeAggregate zero — one ~100 MB closure file
per pass in `pyspark-*`, `blockmgr` flat), closed in ADR 0047's 2026-09-01 addendum;
runs from e6209c7 onward carry the sentinel-zero fix and should show a FLAT
`disk_telemetry:` line.
