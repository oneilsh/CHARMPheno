# Handoff: the spark-vi walkthrough (2026-10-29 .. 11-02) and the redesign in flight

**Branch:** `claude/dismech-ribbon` (dev branch; cluster commands use its preamble).
**Plan of record:** [plans/2026-11-02-dag-gated-user-api-plan.md](../superpowers/plans/2026-11-02-dag-gated-user-api-plan.md)
(WP 1-14 with dispositions) and its spec
[specs/2026-10-30-dag-gated-user-api-design.md](../superpowers/specs/2026-10-30-dag-gated-user-api-design.md).
This file holds what those two do not.

## Where things stand

- Done this session: the post-0167 cleanup (`57026136`), a maintainer-led walkthrough of
  the generic code (core, dag, the DAG-gated model and shim, readout, eval.metrics,
  DagHeads) with fixes landed as it went, then plan WPs 1, 2, 13.
- **WP2 (`bbb5a5d7`) was committed before its spark-vi slow tier finished** (the stop
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
end of the second. The first walkthrough's part (keep it; hashes are the
remote's after the 2026-11-02 history scrub; if one stops resolving, find it by
subject):
- **Areas reviewed:** spark_vi.core (model, loop, config, result, runner, local_runner,
  retry, broadcast), dag (layout, ops), models/topic/dag_gated_lda, the DAG-gated MLlib
  shim, readout (solver, kernels, heads, spark, local, stacked, evaluate), eval.metrics,
  mllib/readout (DagHeads).
- **Shipped in passing** (commits `fb6d0fcb` .. `57f9c9f5`, list above in `git log`):
  resume continues the seed stream and diagnostics; empty minibatch records NaN; E-step
  retry; atomic checkpoints; DAG-gated resume (`fit_ckpt`, identity file,
  `--fresh-fit`); ancestor-walk `frontier_from_coded`; dead `allowed(v)`;
  `children_and_min_depth`; E-step submatrix slice (~4x per doc); sparse closure
  operator (3.7x, 58 MB -> 0.4 MB broadcast); top-m as a plain width (default 100) with
  mass selection and coverage removed; distributed unit-pooled detection;
  `merge_all`; domain-neutral wording across spark_vi; scikit-learn a dev dependency.
- **Pre-existing bugs caught:** gated alpha step used full sub-simplices during
  leaves-first warm rounds (`e8ad2516`); `infer_local` drew from the global RNG,
  breaking VIModel purity (`74bf66bf`); readout packed projection built outside retry
  (`9800a844`); unbounded per-unit collect in detection (`f1b171a8`); O(P n log n)
  tally merges (`57f9c9f5`); `save_result` not atomic (`22034230`).
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

- **An edit to `spark-vi/spark_vi/dag/ops.py` (even a docstring) moves the native-Mondo
  key**; since WP11 `layout.py` is no longer folded (the bundle stopped depending on the
  engine layout), and `nodeset.py` never was. Run
  `tests/scripts/test_case_finding_cache_mondo.py` before pushing; re-pin and label the
  commit "CACHE DROP". Current pins (after WP11): SNOMED 94e21c0c95f5e9d4 /
  e872acd4ebe84669 (the second pin is now the `gate_attestation=ever` key) /
  4637ca2477bc417b, native Mondo ae054f1ef347c379.
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

## After the redesign (2026-11-03 .. 11-11), for the next session

- **Experiments 0168-0173 are done:** 0168 validated the redesign on 0163's
  configuration; 0169 was the first long fit; 0170 tried the local engine (correct,
  compute-bound, so Spark stays the default at 8 cores); 0171-0172 settled why node
  blocks starve (seed decay, about 75 iterations after seeding); 0173 is the first
  cleaned model of record (150 unsupported nodes dropped, accuracy matches 0168).
- **Spec R14 is built (WP15-16):** `nodeStatistics` by default, `dropUnsupportedNodes`
  on request. The walkthrough should cover it along with the redesign.
- **Export:** spec `specs/2026-11-11-dag-gated-dashboard-export.md` (`dag-gated-export/1`,
  25 + 25 terms per domain, about 1.3 MB gzip at full scale). The exporter is
  `charmpheno/export/dag_gated.py` with `analysis/cloud/export_dag_gated.py` and
  `make export-dag-gated ID=N RESOLVE_NAMES=1`. The browser is `dag-dashboard/index.html`,
  adapted from a copy of the Mondo usage dashboard's DAG engine. It is tested on the
  exporter's fixture and on a full-scale synthetic export (0.8 s load, 19 MB heap) but has
  not yet run on a real export. Next: run the exporter on 0173, check the sizes, view it,
  then decide on the gh-pages deployment.
- **The topic survey of 0173** (well-known and rarer diseases, the digest, a sample of
  the dropped nodes) was handed over as a command; its output had not come back yet.
