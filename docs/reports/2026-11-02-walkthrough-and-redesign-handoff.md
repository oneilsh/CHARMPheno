# Handoff: the spark-vi walkthrough (2026-10-29 .. 11-02) and the redesign in flight

**Branch:** `claude/dismech-ribbon` (dev branch; cluster commands use its preamble).
**Plan of record:** [plans/2026-11-02-dag-gated-user-api-plan.md](../superpowers/plans/2026-11-02-dag-gated-user-api-plan.md)
(WP 1-14 with dispositions) and its spec
[specs/2026-10-30-dag-gated-user-api-design.md](../superpowers/specs/2026-10-30-dag-gated-user-api-design.md).
This file holds what those two do not.

## Where things stand

- Done this session: the post-0167 cleanup (`ee72d88b`), a maintainer-led walkthrough of
  the generic code (core, dag, the DAG-gated model and shim, readout, eval.metrics,
  DagHeads) with fixes landed as it went, then plan WPs 1, 2, 13.
- **WP2 (`566db2b6`) was committed before its spark-vi slow tier finished** (the stop
  hook forbids idling with uncommitted work). Its log:
  `/tmp/claude-0/-home-user-CHARMPheno/7ca57e7e-dd57-5d96-b2f7-642d51809940/scratchpad/wp2_slow.log`
  (if the container was recycled, rerun `PYTHONPATH=.:spark-vi:charmpheno poetry run
  pytest spark-vi/tests -p no:randomly -q -m "not cluster"`, ~70 min, background).
  Verify it before building WP3 on the same E-step.
- Next: WP3 (sparse partition statistics), then WP4 onward. Pooled multi-domain
  normalization (old item 15) is OUT (a modeling change, its own experiment).

## The REVIEW_LOG entry (NOT yet written; maintainer asked to wait)

After WP14, the maintainer walks through the redesign's changes with the agent (a
second walkthrough). BOTH walkthroughs go in ONE dated REVIEW_LOG entry, written at the
end of the second. The first walkthrough's part (keep it; commit hashes as on the
remote after the 2026-11-02 history scrub may differ from the ones below, so cite by
subject if a hash does not resolve):
- **Areas reviewed:** spark_vi.core (model, loop, config, result, runner, local_runner,
  retry, broadcast), dag (layout, ops), models/topic/dag_gated_lda, the DAG-gated MLlib
  shim, readout (solver, kernels, heads, spark, local, stacked, evaluate), eval.metrics,
  mllib/readout (DagHeads).
- **Shipped in passing** (commits `d62b280d` .. `cd6be3ff`, list above in `git log`):
  resume continues the seed stream and diagnostics; empty minibatch records NaN; E-step
  retry; atomic checkpoints; DAG-gated resume (`fit_ckpt`, identity file,
  `--fresh-fit`); ancestor-walk `frontier_from_coded`; dead `allowed(v)`;
  `children_and_min_depth`; E-step submatrix slice (~4x per doc); sparse closure
  operator (3.7x, 58 MB -> 0.4 MB broadcast); top-m as a plain width (default 100) with
  mass selection and coverage removed; distributed unit-pooled detection;
  `merge_all`; domain-neutral wording across spark_vi; scikit-learn a dev dependency.
- **Pre-existing bugs caught:** gated alpha step used full sub-simplices during
  leaves-first warm rounds (`dac75728`); `infer_local` drew from the global RNG,
  breaking VIModel purity (`9f7225ba`); readout packed projection built outside retry
  (`96a0f4cf`); unbounded per-unit collect in detection (`15bbe7c7`); O(P n log n)
  tally merges (`cd6be3ff`); `save_result` not atomic (`47f7e941`).
- **Threads parked:** pooled multi-domain normalization and token weighting; a held-out
  ELBO for mini-batch early stop (declined: current behaviour is fine); optimize_eta
  (declined); the readout's home (spark-vi vs charmpheno, still to decide; lean:
  spark-vi); the remaining spark-vi files outside the walkthrough (STM, LDA/HDP shims,
  eval.topic, spectral init) were swept for wording only, not reviewed.

## Decisions taken in the walkthrough (beyond the spec)

- No train/test notion in the shims; callers split (MLlib convention).
- Readout top-m: `DEFAULT_READOUT_THETA_TOPM = 100` in `dag_gated_readout`; library
  default 0. Old manifests with only a mass target re-read at 100 and TAG outputs
  `topm100` (never overwrite the record).
- `readout.evaluate.detection(columns="nonroot", unit_name=...)`; the clinical driver
  passes `unit_name="person"` so results keep `grain: person`.
- Persistence R13 in two levels (URI I/O done in WP1; MLWritable in WP10).
- The DAG-gated fit writes only the resumable `fit_ckpt` per interval (mid-fit npz
  snapshot removed); `fit_ckpt` is deleted after the finished fit is saved.
- Retry is generic Spark; `_is_bad_node_failure` is YARN wording (no-op elsewhere).

## Gotchas learned (do not relearn)

- **Any edit under `spark-vi/spark_vi/dag/` (even a docstring) moves the native-Mondo
  key** (`layout.py` moves every bundle key). Run
  `tests/scripts/test_case_finding_cache_mondo.py` before pushing; re-pin and label the
  commit "CACHE DROP". Current pins (after WP4): SNOMED 7ae90c671efc1b0a / 867b7b9b4c7373e0 /
  dcfbbc2052b41078, native Mondo b7e1f8c795568d57.
- **Do not edit spark-vi while its slow tier runs**: Spark workers re-import modules.
- Never wrap a URI in `pathlib.Path` (collapses `gs://b` to `gs:/b`); use `spark_vi.io.fs.join`.
- Monkeypatching `time.sleep` globally in a Spark test catches pyspark's own sleeps;
  record only backoff-sized sleeps (see `test_runner.py` retry test).
- `poetry lock` (poetry 2) has no `--no-update`; plain `poetry lock` kept existing pins.
- Test commands: spark-vi and charmpheno suites in separate invocations; ignore
  `scripts/tests/test_label_phenotypes.py`; `test_build_dashboard_smoke` is a
  pre-existing failure. Unit baselines at handoff: spark-vi 601, drivers 551,
  charmpheno 200.

## For the cluster, when the redesign reaches WP14

- The framings default to reading `stacked_mass99` heads/results (0163's names); a new
  readout writes `stacked` / `stacked_topm100`: pass the new heads tag explicitly.
- The conversion and workup sidecar keys moved this session (conversion_sidecar.py
  edits); both rebuild once on the next cluster.
- The Mondo-usage dashboard (gh-pages `/mondo-aou-usage/`) is being worked on from
  `main` in another session; nothing here touches it.
