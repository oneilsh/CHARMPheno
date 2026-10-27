# ADR 0051 — spark-vi layout, two engines (Spark and local), and a model-agnostic readout

**Date:** 2026-10-28
**Status:** Proposed (maintainer review)

## Context

After ADR 0050, spark-vi holds the variational engines and the DAG-gated LDA. The
readout (per-node heads over θ, the closure product, the metrics), gated NPMI and the
block census live in `analysis/cloud` driver scripts beside clinical rules, and the
fit-side DAG helpers sit in `models/topic/dag_placement.py`. Two costs drive this
ADR: (1) every fit and readout needs a cluster, though one patient corpus fits in a
large machine's memory, and a readout pass is a scan of the whole θ frame (exp 0163:
22–65 s per pass, 700–900 passes); (2) the code that is generic (any model, any label
DAG) is entangled with code that is clinical. Decisions reached in discussion,
2026-10-26/27 (audit report §1, Q6): option A (generic pieces in spark-vi), the name
`readout`, `dag/` strictly generic, framings stay in `analysis/`, MLlib-compatible
flows including a heads Estimator, and exact collect-free metrics by a distributed sort.

## Decision

**1. Layout.**

```
spark_vi/
  core/          VIModel, VIConfig, VIResult; VIRunner (Spark) + LocalRunner; shared
                 broadcast/reduce utilities
  dag/           generic label-DAG mechanics: closure, descendants, heights,
                 transitive reduction, DagLayout (topic blocks over a DAG)
  models/topic/  lda, hdp, stm, dag_gated_lda; spectral init (flat, STM); gate seeds
  readout/       heads over ANY vector column: the batched L-BFGS solver, closure-
                 product stacking, the stats functions (Spark and local backends)
  eval/topic/    intrinsic: coherence incl. gated NPMI, block-topic census,
                 concentration
  eval/metrics/  model-agnostic: AUC, precision@recall, recall@FDR, lift, ROC/PR
                 curves (exact; Spark by distributed sort, local by numpy)
  mllib/         Estimator/Model shims: DagGatedLDA, LDA, HDP, STM, DagHeads
```

`readout` never imports `models`: its input is a vector column (θ, raw codes, any
representation), a multi-label target and a DAG. `dag` knows nothing of Mondo.

**2. Two engines, one model contract.** `LocalRunner` runs the same `VIModel`
(`initialize_global` / `local_update` / `update_global`) over an in-memory or
Arrow-chunked corpus. Fidelity target: each iteration's minibatch is a Bernoulli
sample at the configured rate, the Robbins–Monro schedule and the gate schedule are
identical, so a local fit is the Spark fit in distribution (not bit for bit: partition
order and RNG streams differ). The readout's solver is already backend-agnostic (a
stats function per pass); the local backend computes it with numpy over θ held once.
Spark stays the default for corpora that do not fit; the engines are tested against
each other on the same corpus.

**3. Collect-free metrics by a distributed sort.** Per node, ranks come from a Spark
sort of the (node, score, label) cells; AUC, precision@recall, recall@FDR and top-k
lift follow exactly, with no driver collect. Curves (ROC/PR) are emitted for a node
sample, thinned to a few hundred points. The local backend computes the same metrics
with numpy. The incident (new-diagnosis) rule is a per-patient filter applied before
the sort.

**4. MLlib.** `DagHeadsEstimator` fits the heads on a vector column + labels + the DAG
and returns a `DagHeadsModel` whose transform appends per-node (and stacked)
probabilities, so `Pipeline([DagGatedLDA, DagHeads])` is the end-to-end case-finding
model. The core API stays NumPy/RDD-level; the shims are thin.

**5. charmpheno** holds the clinical definitions: the native Mondo label-DAG builder
and code map, the first-attestation sidecar, gate-ever attestation, the prevalence
(new-diagnosis) rule, the egress floor (one constant, seven copies today), concept
naming, the ribbon reader. **`analysis/cloud`** keeps thin drivers (fit, readout,
inspect), renamed from `gated_pc_*` to `dag_gated_*` with the model class.

## Alternatives considered

- A third package for the readout: the cleanest boundary, but another artifact to
  version and ship on `--py-files`; `spark_vi.readout` can be split out later because
  it imports nothing from the models.
- Readout in charmpheno: simplest move, but buries generic code next to clinical rules
  and gives up spark-vi's domain-generality.
- A local engine that only approximates (full-batch, no minibatches): faster to
  build, but not the same algorithm; rejected for fidelity.

## Consequences

Order of work, each step test-gated and committed alone: (1) `spark_vi.dag` with
`DagLayout` moved out of `dag_placement.py` and the Mondo DAG code into charmpheno: one
announced cache drop, tripwires re-pinned; (2) `spark_vi.readout` extracted from
`gated_pc_cloud.py` / `distributed_readout.py` with the Spark backend, behaviour
unchanged; (3) `eval/metrics` with the distributed sort, replacing the driver collect;
(4) `LocalRunner` and the local readout backend, validated against Spark on one
corpus; (5) the `DagHeads` shim; (6) driver and model-class renames. Updates
`docs/architecture/SPARK_VI_FRAMEWORK.md` (layout, runners) in the commits that make
each change. The walkthrough follows step 6.
