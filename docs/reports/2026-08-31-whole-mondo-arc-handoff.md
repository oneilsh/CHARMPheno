# Whole-Mondo arc handoff — 2026-08-31

**Branch:** `claude/gated-conditional-voi` (all work pushed through `7ed9871`).
**Successor to:** the PC-arc closeout (2026-08-20). Everything below happened since.
**Purpose:** session-compaction handoff — state, results, open threads, next actions.
Primary sources are the exp docs; this report is the map, not the territory.

## 1. Headline results (the numbers that matter)

- **Exp 0104 — whole-Mondo record (the first): macro AUC 0.6978 / AP 0.4845 over
  2,106 nodes** (unsup gated LDA + distributed readout; full 100-iter fit, ~220
  cumulative solve iterations across checkpoint resumes vs the 200 budget). Dev
  smoke was 0.6891/0.4745 — the smoke→record gap matched 0103's. Run log has the
  full crash-and-resume provenance. This is the PRE-COLLAPSE BASELINE.
- **Exp 0103 A/B gate PASSED** (earlier): distributed readout ≡ driver readout,
  macro |Δ| ≤ 1.1e-4 both arms. Reference cardiovascular unsup: 0.7584/0.5428.
- **Exp 0109 — collapsed-DAG smoke: fit done, readout IN FLIGHT at handoff time**
  (a resumed `gated-pc-readout ID=109`; numbers land in its `results_readout.json`).
  The splice removed exactly the 143 structural only-children (fittable unchanged
  at 3,057; degenerate 763 → 620). Detection-metric fix (constant columns
  excluded) is always-on in any readout run from current code.
- **PC is still dead** (closeout §5 stands; nothing here reopened it).

## 2. The degeneracy science (the week's biggest finding)

0104's 763 degenerate label heads are now FULLY decomposed — see
`docs/experiments/0109-whole-mondo-collapsed-dag.md` (run log + ASCII figure) and
`analysis/cloud/diag_sibling_support.py` (the diagnostic that did it, two
refutations deep):

- **1 root** — structural, by design.
- **143 structural only-child class nodes** — single-nearest-cover flattening
  ("terminal stealing"); removed by 0109's `mondo_collapse` splice-to-fixpoint.
- **619 SUBSUMED CATEGORY-ANCHORS** — the deep mechanism: the climb attests every
  powered ancestor (`concept_ancestor`, unrestricted depth) + the anchor hierarchy
  nests anchors only under class covers (category-anchors sit as SIBLINGS of their
  own specifics) + closure masking gives negatives only via siblings ⇒ a category
  is co-attested on every doc that fires a "sibling" and is never observed
  negative. `n_obs == n_pos` exactly, all 619. These are common clinical category
  labels (thyroid disorder, breast neoplasm, UTI...) currently wasted as constants.

**Resolution (direction APPROVED by Shawn):** stop patching the anchor
construction; adopt main's map-and-roll as the label front-end — see §4.

## 3. Infrastructure: every failure mode root-caused and fixed

A week of run deaths, each initially misattributed, all closed. ADR 0047 +
0104's run log carry the full forensics; the fixes in force:

| mechanism | fix | where |
|---|---|---|
| Executor JVM heap OOM self-masked by Dataproc's `OnOutOfMemoryError='kill %p'` (the week of "kill swarms") | memory geometry `cores=2/8g/3g` (fits the 13,544 MB YARN max) | 0104/0109 `spark_conf` front matter |
| Driver-disk ENOSPC: per-iteration broadcasts leak driver-local pickle + block-manager copy under `unpersist` | `destroy()` everywhere via `_destroy_broadcast` (readout, VI runner, repo-wide audit) | ADR 0047; commits 092f553/a265c46/8346613 |
| Job aborts under executor churn (excludeOnFailure starvation) | retry-with-backoff on every driver-blocking action; `excludeOnFailure.timeout: 10m`; fail-fast when the SparkContext itself died | `distributed_readout._retry_spark_action` |
| Lost multi-hour solves | per-arm solver checkpoint every 10 iters + fingerprint-v2 warm resume (counts-based — v1's float-byte fingerprints never matched across runs; proven cross-cluster) | `gated_pc_cloud` ckpt seam |
| Dynamic-allocation churn | DA off + fixed `executor.instances` | `spark_conf` |
| Ops retyping | `spark_conf:` front-matter key read by BOTH `make exp` and `make gated-pc-readout`; manifest self-sufficiency (records `readout_max_iter`, `dag_source`, `billing`...); billing/cdr env fallback; MONDO-witness guard against wrong-corpus rebuilds; `GPR_DRIVER_MEMORY` knob (16 GB masters) | run_experiment / Makefile / gated_pc_readout |

**One infra mystery OPEN:** the 08-31 0109 smoke died of driver-disk ENOSPC WITH
all destroy fixes verifiably active (pyspark's `destroy()` confirmed to unlink the
temp file). ~100 GB consumed over ~19 ks by something ADR 0047 doesn't cover, and
the 09-01 recovery readout died the same way at solver iter ~55 — recurrent, not a
one-off. The watcher is now IN-BAND: `analysis/cloud/disk_telemetry.py`, started by
both gated_pc drivers right after the SparkSession exists, prints one
`disk_telemetry:` line (per-filesystem used/avail + the six biggest top-level
entries per watched dir + a `broadcast*` file count) to driver stdout every 120 s.
The `nohup diskwatch` loop it replaces is superseded because local logs die with the
cluster — twice now the master was torn down before its `~/diskwatch.log` was read —
whereas Dataproc persists job driver stdout to the staging bucket. Mitigation
shipping alongside, on the JVM-side hypothesis (ContextCleaner frees task-binary
broadcast pieces and shuffle metadata only on driver GC, which a large idle heap
never triggers): `spark.cleaner.periodicGC.interval: "5min"` in 0104/0109 front
matter.

## 4. The approved plan: native Mondo label space

`docs/superpowers/plans/2026-08-31-native-mondo-label-space-plan.md` — read it in
full before building; the short form:

- **Scope decided by Shawn:** LABELS adopt main's `source_climb` map-and-roll
  front-end; source-vocab FEATURES explicitly deferred; publishing/egress
  untouched; HPO later.
- **Design:** attribution frame → frontier; producer-side closure-support powering
  at `min_positives=100` (main's `rollup:false` means this piece is new); label
  DAG = `nearest_mapped_parents` induced multi-parent Hasse + the 0109 splice as
  thin-chain post-pass (subsumed-sibling trap impossible by construction);
  all-Mondo-id engine space (retires the positive-OMOP-cid terminal convention);
  multi-parent verification checklist in the plan.
- **Key survey facts (2026-08-31, origin/main):** the branches share NO merge base
  (port, not merge — 1,410 vs 97 commits); `mondo_to_omop_mapping.py` is
  byte-identical on both (the shared seam); main's pure core
  (`nearest_mapped_parents`, `meaningful_skeleton`, `reduce_tie_map`, 33 tests in
  `analysis/cloud/tests/test_mondo_usage.py` on main) independently reinvented
  this branch's two DAG patches — the convergence that motivated the move.
- **Numbering:** main and this branch BOTH have an exp 0109 (dual-axis vs
  collapsed-DAG). The native-label experiment takes **0110**; resolve the dual
  0109 at unification.
- **Acceptance:** run `diag-sibling-support` on the new bundle BEFORE any fit
  (degeneracy is a corpus property); expect `1 + small thin-chain residue`
  (trajectory claim: 763 → 620 → ~1). Macro reported both on shared nodes
  (comparability) and the full new space (deliverable).

## 5. Open threads, in priority order

1. **0109 smoke readout in flight** — collect its macro + detection lines and the
   `disk_telemetry:` tail; record in 0109's run log; that closes the splice
   report card.
2. **0110 port + experiment** (plan §4/§7) — agent-buildable; waiting on nothing
   but a "go" (direction already approved; the port itself was not explicitly
   green-lit as tonight's work).
3. **Driver-disk leak #2** — catch with the in-band telemetry (see §3); fix; ADR
   0047 addendum.
4. **Top-m pricing test** — `gated-pc-readout ID=103 --readout-theta-topm 256` vs
   0103's recorded full-K numbers; decides whether `readout_theta_topm: 256`
   stays in record configs (ΔAUC ≲ 1e-3 keeps it).
5. **Cardiovascular-subset comparison** — 0104's per-node rows vs 0103's (isolates
   scale-up effects on known nodes).
6. **Dead code at HEAD:** `gated_init.py` imports `precompute_projection_rows`
   from `spectral_init_scalable` which does not exist (9 pre-existing test
   failures; scalable spectral-init path dead; likely a fork casualty) — fix at
   unification, not before.
7. **Unification decision** (which trunk absorbs which) — deferred until 0110
   numbers exist.
8. Queued small items: solver line-search backtrack cap (bounds the ~25
   passes/iter deep phase — offered, never requested); ship-only-searching-rows
   partials (the durable memory+wall-clock fix); branch's 0104 doc still says
   "status: pending" (flip to done when 0109 closes the pair).

## 6. Operational cheat-sheet

- **Idempotent preamble** (fresh cluster / branch swaps):
  `cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi &&
  git checkout claude/gated-conditional-voi && git pull origin
  claude/gated-conditional-voi`
- **Safe-mode latch after any disk-full:** `hdfs dfsadmin -safemode get` →
  `leave` (it never auto-releases).
- **Recovery readout:** `make -C analysis/cloud gated-pc-readout ID=N` — manifest
  self-sufficient for fits from current code; resumes from `readout_ckpt_*.npz`
  automatically; rebuilds the bundle on a cold cache (write-through).
  `GPR_DRIVER_MEMORY=4g` on 16 GB masters.
- **Degeneracy diagnostic:** `make -C analysis/cloud diag-sibling-support ID=N`
  (needs a cache HIT; never rebuilds).
- **Diskwatch:** nothing to start — it is in-band and always on. Both gated_pc
  drivers run `analysis/cloud/disk_telemetry.py`, one line per 120 s to driver
  stdout; read it with `grep disk_telemetry: <run>/driver_log.md` (or the
  persisted job log). The old `nohup ... > ~/diskwatch.log` loop is retired: a
  local log dies with the cluster, and twice the master was torn down before
  anyone read it.
- **Cluster shape that works:** 8+ non-preemptible n2-standard-4 workers,
  n2-standard-8 master (16 GB masters OOM-kill the driver during Mondo downloads
  at the default 8g JVM), no spots needed. `spark_conf` in front matter handles
  the rest.

## 7. Standing constraints (unchanged)

No PC runs. No GCS bundle cache (HDFS per-cluster, rebuild-on-fresh accepted).
Never commit patient-level data. Work only on `claude/gated-conditional-voi`.
Subagents (Opus) for substantial builds. Exp docs are the record: dev smokes and
record runs share one doc; run logs carry the forensics.
