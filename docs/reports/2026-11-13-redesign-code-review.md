# Code review of the DAG-gated redesign (`8dab582..HEAD`), before the walkthrough

**Date:** 2026-11-13. Three independent reviewers (spark-vi engine + DAG; spark-vi MLlib
stages + readout; charmpheno + drivers + exporter), each verifying findings by script or
by reading. Scope: the spec 2026-10-30 work (WP1-16, R14), exps 0168-0173 tooling, and the
2026-11-11 export. This report is the triage; fixes land in the commits that cite it.

## Verified sound (by script)

- Node-set logic: 3,000 random DAGs (mixed `1` / `"1"` ids, multi-parent, `min_docs`
  0-4): gate equivalence after `fit_node_set` and after `drop_nodes`, chained drops equal
  one combined drop, JSON round trips. Row-sparse statistics add the same floats as the
  dense path.
- `dropUnsupportedNodes` with renumbering (multi-parent, 2 domains, `gateTpn=2`): every
  surviving row of λ, α, `gate_open_at` and the α trace identical; K and the kept DAG right.
- Persistence: Params with mixed ids, model with node set and `unsupportedNodes`,
  `Pipeline(DagGatedLDA, DagHeads)` on a local path, heads with a nested cleaned topic
  model. Seed extras: cross-domain offsets, qualified terms, ambiguity errors.
- Charmpheno / drivers: bundle key round trip (fit spec -> manifest -> readout spec, same
  key), label rules (person split first, gate labels train only, heads on label-year), the
  integer bridge after `--drop-unsupported`, exporter statuses / supports / codecs.

## Findings and dispositions

| # | where | finding | disposition |
|---|---|---|---|
| A1 | `export/dag_gated.py`, `topic_npmi.py` | **Egress:** exported NPMI averages pairs with joint count >= `min_pair_count` = 3, not >= 20 as the spec and docstring claimed (the `floor=20` in `score` is the gated reference size, not the pair cells). | fix: the exporter drops the NPMI block (with a message) unless `min_pair_count >= EGRESS_FLOOR`; readout `--npmi-min-pair-count`; spec and docstring corrected. 0173's existing export stays workspace-internal until re-made. |
| A2 | exporter / fit driver | The disclosure argument needs `minDocsPerNode >= 20` and the vocabulary's `min_patient_count >= 20`; nothing checks them. | fix: `build_export` refuses below the floor. |
| B1 | `io/export.py` `save_result` (non-local) | On `hdfs://` / `gs://` a crash mid-checkpoint mixes iterations (params overwritten in place, manifest last); on HDFS a file can be truncated. | fix: remote checkpoints write a fresh `params` directory per save and the manifest (written last) names it; the previous one is removed after. |
| B2 | `core/local_runner.py` | `processes > 1` with an unseeded model (`random_seed=None`) is neither identical to `processes = 1` nor reproducible (workers reseed from OS entropy). | fix: the parent draws one seed per chunk from its RNG; docstring states bit-identity for content-seeded models only. |
| B3 | `local_runner.py` | Forks on macOS (fork is listed there) although documented as serial. | fix: fork on Linux only. |
| B4 | `local_runner.py` | `_FORK_STATE` global is unsafe for two runners in threads. | fix: a module lock around set/fork/clear. |
| B5 | `io/fs.py` | `hdfs:/x` (single slash) classed local; `#` / `?` truncate `file:` paths. | fix: scheme regex; strip `file:` by hand. |
| B6 | `dag/nodeset.py` `count_support_spark` | Broadcast unpersisted, not destroyed. | fix. |
| C1 | `mllib/_common.py`, `featurizer.py` | Driver-side Params writers overwrite an existing path silently (MLlib refuses). | fix: refuse unless `overwrite()`. |
| C2 | docs (R13, module docstrings) | `Pipeline.save` / `PipelineModel.load` still use MLlib's executor-side metadata I/O, so a driver-local path fails on a multi-node cluster. | fix the docs: Pipeline persistence needs a shared filesystem (HDFS / GCS); single stages work anywhere. |
| C3 | `setNodeTerms` / `setGateSeedProfile` | A bare string term list is split into characters. | fix: refuse str. |
| C4 | `_term_names` | Checks total vocabulary width, not per domain. | fix. |
| C5 | `readout/heads.py` `read_heads` | Loop variable shadows the file name in log messages. | fix. |
| C6 | `recommended(docConcentration=None)` | Cannot restore the unset default. | fix: None clears. |
| C7 | `seed` help text | Says "Default unset"; `getSeed()` returns pyspark's hash default (fit reads seed only when set). | fix the text. |
| D1 | `loop.py` `on_iteration(step + 1)` / `--trace-flat-every` | Trace iteration numbers restart after a resume (0169-0173 were not resumed). | fix in the driver (offset by the resumed count). |
| D2 | fit driver manifest | After a resume `estimator_params.maxIter` records the remainder. | fix. |
| D3 | `run_experiment.py` | Front-matter `gate_open_seed` silently ignored. | fix: emit it. |
| D4 | `dag_gated_fit.py` | Pre-existing: driver functions that shape the cached bundle on a miss (`_gate_ever_attested`, the code ladder) are not folded into the bundle key. | **walkthrough decision**: moving them into a hashed module is a cache drop. |
| D5 | `experiments/defaults/_base.yaml` | Gated cohorts that do not set a knob get `_base.yaml`'s (e.g. `optimize_doc_concentration: true`, `n_bg: 20`), not the preset, contrary to `build_dag_gated_args`' docstring (0168-0173 set every knob). | **walkthrough decision**: move gated keys out of `_base.yaml`, or fix the docstring. |

Non-blocking cleanups (walkthrough agenda, not fixed now): duplicated label landing
(`FrontierFn`, `NodeSet.land`, `NodeLabeler`), duplicated exp(E[log β]) helpers, duplicated
setDag/getDag on two Params classes and DagHeads lacking setters, `_scan_labels_spark`
repeating `count_support_spark`, layout arithmetic outside `DagLayout.block`, `W_std` lost
on heads reload, no width check on the single-domain transform path, `gateSeedMode`
validated late, a log line when seed extras are set but `gateWarm = 0`, `extra_metadata`
able to overwrite core keys, and the exporter's copies of `mondo_cid` / `dag_from_edges`
/ `_longest_depth`.
