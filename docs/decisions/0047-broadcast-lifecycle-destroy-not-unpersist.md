# ADR 0047 — Per-iteration Spark broadcasts are DESTROYED, not unpersisted

**Date:** 2026-08-30
**Status:** Accepted
**Context:** exp 0104's whole-Mondo scale-up (run log, 08-26 → 08-30) killed the same
class of run three separate ways before the mechanism was isolated: two driver-side
`[Errno 28] No space left on device` deaths during fits (08-26/08-27, then attributed
to gcsfuse append behavior), one at solve iteration ~50 of the recovery readout
(08-29), and one at solve iteration ~180 of the record run's readout after its
100-iteration fit (08-30). The last two were unambiguous: the ENOSPC was raised by
`sc.broadcast` itself, from the tempfile write inside `pyspark/broadcast.py::dump`,
on a master whose disk had been ~100 GB free at launch. Commits 092f553 (readout
seam), a265c46 (spark-vi runner seam), and 8346613 (repo-wide audit) are the
implementation this ADR records.

## Context (the mechanism, since it is not in pyspark's docs where you'd want it)

Every `sc.broadcast(value)` in pyspark materializes THREE things:

1. a **driver-local pickled temp file** under the Spark temp dir (`broadcast.py`
   writes the pickle to disk before the JVM ever sees it);
2. a **driver block-manager copy** (TorrentBroadcast keeps the value in the driver's
   own block store, spilling to the same local disk once memory is short);
3. the **executor copies** fetched on first use.

`unpersist()` frees class 3 ONLY. Classes 1 and 2 survive until `destroy()` — or
until the SparkContext shuts down, whose cleanup hook is also why the leak evaporates
from `df -h` the moment a crashed run exits, and why it went undiagnosed for a week.
"Unpersist the previous broadcast so we don't leak" (the runner's original comment,
and RISKS_AND_MITIGATIONS.md's original prescription) was therefore only a third of a
cleanup, and the missing two thirds scale with iteration count:

- **fit seam** (`spark_vi/core/runner.py`): one λ-shaped globals broadcast per SVI
  iteration ≈ **355 MB × 2 driver-side copies × 100 iterations ≈ 70 GB** at
  whole-Mondo (K≈3,827, V≈11.6k);
- **readout seam** (`analysis/cloud/distributed_readout.py`): one (V, b, mask)
  parameter broadcast per L-BFGS line-search pass ≈ **117 MB × 2 × ~450 passes ≈
  105 GB** for a 200-iteration budget.

Either seam alone fills a 123 GB master. The arithmetic closed exactly on 08-29:
19 GB baseline + 2 × 117 MB × 449 passes ≈ the full disk, at the iteration the run
died. The failure is also self-masking twice over: the crash's shutdown hook deletes
the evidence, and the *proximate* error surfaces wherever the disk-full lands next
(a gcsfuse append, the NameNode's resource monitor latching safe mode, the AM's
staging write) — all of which were chased as root causes first.

## Decision

### 1. Every per-iteration / per-call broadcast is released with `destroy()`, via a never-raising helper

The pattern, replicated as a module-local `_destroy_broadcast(bcast)` at each seam
(readout kernels, VI runner, STM, predictive_gain, coherence, spectral init):
`destroy()` → on any exception, fall back to `unpersist()` → log; **never raise** —
these run in `finally` blocks on the error path, where a raised cleanup exception
would REPLACE the real one mid-propagation (observed: an unpersist NPE on a
YARN-killed application masking the actual failure). Destroy is safe at these sites
by construction: each broadcast's consuming action (treeAggregate / treeReduce /
collect) has RETURNED before the destroy runs, so nothing can reference it again; a
retry (ADR-adjacent: the readout's `_retry_spark_action`) builds a FRESH broadcast
inside the retried closure rather than reusing one whose blocks died with the
executors. A minimal probe (create b_t, run job, destroy b_{t-1}, repeat) confirms
per-iteration destroy of the prior broadcast is safe in pyspark 3.5.

### 2. Lineage-held broadcasts are the one exempt class, and each site says so

A broadcast captured by the closure of a RETURNED lazy RDD/DataFrame (the runner's
`transform`, the model `_transform` UDFs in lda/hdp/gated_lda/pc, `score_cells_df`,
the covariates spec) must NOT be destroyed — the caller's later actions still need
it, and a destroyed broadcast fails them with `assertValid`. These are bounded (one
per call, not one per iteration), their lifetime is delegated to Spark's
ContextCleaner on GC of the returned object, and every such site now carries a
comment naming this contract so the audit's distinction survives the next editor.
`F.broadcast(df)` join hints are not broadcast objects and are out of scope.

### 3. Lifecycle tests assert destroy counts, and their proxies track ids, not objects

`spark-vi/tests/test_broadcast_lifecycle.py` pins exact DESTROY counts on both
`fit()` terminal branches (max-iterations and convergence) and asserts the unpersist
fallback did not fire — the pre-audit version only counted unpersists, so the
runner's destroy was silently exercising its fallback in the test and the primary
path was unpinned. The harness rule learned doing this: a locally-defined proxy
class is cloudpickled INTO task closures together with its closure cells, so a
tracking list that holds real Broadcast objects re-registers every one of them for
shipment with each later task — survivable when they are merely unpersisted
(lazily re-served), fatal once they are destroyed. Proxies record JVM broadcast
IDS. This is a harness artifact, not a production hazard (production code keeps no
such lists), but it is exactly the kind that fabricates a "destroy is unsafe"
conclusion, so it is recorded here.

## Alternatives considered

- **Keep unpersist, add a temp-dir sweeper** (cron `rm` under the Spark temp dir):
  rejected — it races the files' legitimate lifetime (a broadcast's temp file is
  live until its last task fetch), fixes only class 1 of the two leaked copies, and
  treats the symptom on one machine instead of the lifecycle everywhere.
- **Fewer, longer-lived broadcasts** (reuse one broadcast slot, mutate contents):
  impossible — broadcasts are immutable by design; "re-broadcast per iteration" IS
  the correct pattern, it just needs the correct release.
- **Bigger master disk**: postpones the death by a constant factor on a leak that is
  linear in iterations, and the record budget (100-iter fit + 200-iter readout +
  calibration re-solves) already outruns any reasonable disk.

## Consequences

- Driver disk usage is FLAT over a solve/fit (verifiable live: `watch df -h /` no
  longer climbs ~14 GB per 10 readout iterations).
- The 08-26/08-27 fit-phase ENOSPC deaths, previously attributed wholly to gcsfuse
  append behavior, are re-attributed as dual-cause: the gcsfuse batching fix was
  real and stays, but the disk pressure that made appends fail had this leak as its
  larger source. `RISKS_AND_MITIGATIONS.md` §Broadcast lifecycle now prescribes
  destroy and cites this failure.
- The retired in-memory PC path and every other per-call site were converted in the
  same audit (8346613), so no dormant seam re-imports the bug when revived.
- Known limitation: the scalable spectral-init path (`gated_init.py` →
  `spectral_init_scalable.precompute_projection_rows`) is dead at HEAD from an
  unrelated missing symbol (pre-existing, likely a main-branch fork casualty), so
  its converted site cannot be exercised until the branch reconciliation restores
  it — flagged there, not fixed here.

## Addendum (2026-09-01): leak #2 — pyspark AUTO-broadcast of large task closures

The destroy discipline above closed leak #1 and runs still died of driver-disk
ENOSPC (exp 0109, 08-31 and 09-01, ~100 GB over a multi-hour readout). In-band
disk telemetry on the 09-01 recovery localized the growth: all of it inside the
app's `spark-*/pyspark-*` temp dir, ~one ~100 MB file per L-BFGS data pass, with
`blockmgr-*` flat and our explicit-broadcast count at zero — so NOT a broadcast
we created, and NOT JVM state (`spark.cleaner.periodicGC.interval=5min` was
active and doing its job).

**Mechanism.** pyspark pickles each job's task closure; any closure over ~1 MB
(`PythonUtils.getBroadcastThreshold`) is silently wrapped in an INTERNAL
`sc.broadcast` of the pickled command. That broadcast is invisible to caller
code: no handle is returned, so no `destroy()` of ours can ever reach it, and
its driver-side pickle file under `pyspark-*` is unlinked only at context
shutdown. The offender was the `treeAggregate` ZERO VALUE in the per-pass stats
seam (`distributed_readout.SparkStatsFn.__call__`): a dense
`(np.zeros(C), np.zeros((C,K)), np.zeros(C))` tuple ≈ 108 MB pickled into every
pass's closure. The explicit parameter broadcast right next to it was destroyed
per pass, exactly per this ADR — the leak was its closure-borne twin.

**Fix (07d9d47 follow-up).** treeAggregate zeros are now a `None` sentinel with
identity handling in the combiner; partials are allocated executor-side by the
partition kernels, and the driver substitutes zeros only in the (unreachable in
practice) empty-corpus case. Applied to the per-pass stats seam and the one-time
`masked_moments` zero (~217 MB). Small zeros (coverage, diagnostics) stay as-is
— they sit under the auto-broadcast threshold.

**Doctrine extension.** The lifecycle rule now has a closure clause: NOTHING
ARRAY-SHAPED RIDES A TASK CLOSURE. Ship parameters via an explicit broadcast
(destroyed per this ADR); ship reduction identities as sentinels; keep every
pickled closure under the 1 MB auto-broadcast threshold. A `disk_telemetry:`
line whose `pyspark-*` dir grows linearly in passes is the signature of a
violation.
