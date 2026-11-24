# 0054 — The readout, DagHeads and the ranking metrics move from spark-vi to charmpheno

**Status:** Accepted
**Date:** 2026-11-24

## Context

ADR 0051 §1 placed the per-node readout (`spark_vi.readout`), its MLlib stages
(`spark_vi.mllib.readout`: `DagHeadsEstimator` / `DagHeadsModel`) and the exact ranking
metrics (`spark_vi.eval.metrics`) in spark-vi as generic pieces. In practice they are
supervised per-node logistic regression and classification metrics serving the clinical
case-finding evaluation; no other spark-vi model has a readout, and none of it is
variational inference. The maintainer plans to publish spark-vi on its own
(`docs/reports/2026-11-24-spark-vi-split-scoping.md`), and decided (2026-11-24) to move
these pieces, and the simulator-study helpers, into charmpheno.

## Decision

1. `spark_vi/readout/*` -> `charmpheno/readout/*`; `spark_vi/mllib/readout.py` ->
   `charmpheno/readout/mllib.py` (the stages sit with the code they wrap, as spark-vi's
   shims do in `spark_vi.mllib`).
2. `spark_vi/eval/metrics/*` -> `charmpheno/evaluate/metrics/*`, and
   `eval/topic/{seed_panel,concentration_recovery}.py` -> `charmpheno/evaluate/`:
   `charmpheno.evaluate` already holds the evaluation tools (topic alignment, LDA
   comparison).
3. The held-out predictive likelihood that STM and `predictive_gain` use
   (`heldout_split`, `_predictive_loglik`) stays in spark-vi as
   `spark_vi/eval/topic/heldout.py`.
4. spark-vi drops the re-exports and never imports charmpheno; the readout keeps using
   spark-vi (node sets, persistence helpers, broadcast, retry).

## Alternatives considered

- A third package for the readout: worth it only if another project used it.

## Consequences

- Supersedes ADR 0051 §1's placement of `readout/`, `eval/metrics/` and
  `mllib/readout.py`; its other decisions stand.
- A `DagHeadsModel` / `DagHeadsEstimator` saved before the move records its class as
  `spark_vi.mllib.readout.*` and no longer loads (ADR 0052 §7: no backward
  compatibility). The heads npz the drivers write carries no class path and is
  unaffected; no in-repo fixture holds the old path.
- Executor-side readout code now pickles under `charmpheno.readout`; every submit path
  that runs a readout already ships the charmpheno zip.
- `charmpheno/omop/case_finding_assembly.py` (source-hashed) named
  `spark_vi.readout.labels`; corrected in its own announced cache-drop commit, which
  re-pinned all four tripwire hashes (docstring only; no corpus change).
- charmpheno declares scipy, and scikit-learn as a test oracle; spark-vi drops the
  latter.
