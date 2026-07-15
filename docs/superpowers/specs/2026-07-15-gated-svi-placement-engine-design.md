# Gated SVI Placement Engine + MLlib Shim — Design

**Date:** 2026-07-15
**Status:** Design approved (brainstorm), pre-implementation
**Relates to:** the collapsed-Gibbs placement engine `spark_vi/models/topic/dag_placement.py` (the offline gold-standard validator); the OnlineLDA SVI stack (`models/topic/lda.py` + `core/runner.py` VIRunner + `mllib/topic/lda.py` shim, ADR 0009).

## Goal

Build the **production** half of the two-engine placement design: a distributed **stochastic
variational (SVI)** gated topic model that recovers the same node-tied topics as the validated
collapsed-Gibbs engine, runs on Spark through `VIRunner`, and is exposed through an MLlib
Estimator/Model shim so the OMOP layer calls it exactly like LDA/HDP. The collapsed-Gibbs engine
stays as the offline gold-standard oracle that this engine is validated against.

## Decisions (brainstorm)

- **Hard gate** (not DMR soft prior): each training doc's variational E-step is restricted to
  `allowed_set(frontier)` topics — the exact variational analogue of the Gibbs gate, so an
  SVI-vs-Gibbs equivalence check is meaningful. DMR (alpha_d = exp(Lambda x_d), Mimno & McCallum
  2008) is a documented v2, not v1.
- **Label is semantic:** the shim's `labelCol` is the **frontier** (set of node ids = the clinical
  truth), and the model holds the `DagLayout`; `allowed_set` is computed inside the model. The DAG is
  a tiny parent-map broadcast (kilobytes even at thousands of nodes).
- **Estimator/Model = fit/deploy:** `fit` trains gated (needs the frontier); `transform` folds a
  held-out patient in **ungated** (full-K CAVI) → the affinity profile. Finding the uncoded patient.

## Architecture

Reuses the entire SVI stack; the gate is one overridden method.

- **`GatedBOWDocument`** (add to `spark_vi/models/topic/types.py`, mirroring `STMDocument.groups`):
  `BOWDocument` fields + `frontier: frozenset[int]` (the doc's frontier node ids; empty = ungated).
- **`GatedOnlineLDA(OnlineLDA)`** (new `spark_vi/models/topic/gated_lda.py`): constructed from a
  `DagLayout` (supplies `K = dag.K` and `allowed_set`) plus the inherited LDA hyperparameters.
  - Overrides `local_update`: for each `GatedBOWDocument`, compute `allowed = dag.allowed_set(frontier)`
    (or all K when `frontier` is empty), run `_cavi_doc_inference` over the **sub-matrix**
    `expElogbeta[allowed]` with `alpha[allowed]`, and scatter the sufficient stats back to
    `lambda_stats[allowed, indices]`. Disallowed topics receive zero contribution from that doc — the
    variational twin of the Gibbs gate; this is what welds each node's topic to its subtree.
  - Inherits `initialize_global`, `update_global` (SVI natural-gradient beta step), `compute_elbo`,
    `combine_stats`, and `VIRunner` integration unchanged.
  - Note (scaling): the gated E-step costs O(|allowed|) per token, and |allowed| is the patient's
    frontier closure (bounded by DAG depth), NOT K. So K = thousands of nodes does not blow up
    per-doc training; the large-K cost is the global beta (K x V) broadcast + M-step, standard
    large-K LDA territory handled by VIRunner (sparse/blocked beta is a later option).
- **Deployment / profile:** `transform` (Model) folds each held-out doc in with the inherited,
  ungated full-K CAVI → `gamma` over K → aggregate to per-node affinity by summing each node's `tpn`
  topic block (the SVI analogue of the Gibbs `profile`). Output column `nodeAffinity` (Vector over
  the DAG nodes).
- **MLlib shim** (`spark_vi/mllib/topic/gated_lda.py`, mirroring `mllib/topic/lda.py` / ADR 0009):
  - `GatedLDAEstimator` — Params: `featuresCol` (BOW SparseVector), `labelCol` (frontier: an
    array/set of node ids per row), `dag` (the DagLayout / its parent map), plus the OnlineLDA
    hyperparameters. `fit` builds `GatedBOWDocument`s (features + frontier) and runs `VIRunner` with
    `GatedOnlineLDA`.
  - `GatedLDAModel` — holds learned beta + the DagLayout (+ int2cid for interpretation). `transform`
    adds `nodeAffinity`. Persistable like the LDA shim.

## Validation — the equivalence gate

The collapsed-Gibbs engine is the oracle. On a planted corpus (reuse
`tests/_stm_synth.dag_placement_corpus` / `dag_placement_corpus_multi`):

1. Fit both `fit_gated` (Gibbs) and `GatedOnlineLDA` (SVI, run locally without Spark via the model's
   `local_update`/`update_global` directly, or through a small in-memory driver) on the same corpus.
2. Assert the recovered node topics agree — per-node beta cosine high (e.g. >= 0.9) — and that the
   downstream placement metrics agree — family/subtype AUC and MRR within Monte-Carlo tolerance of
   the Gibbs numbers (family ~0.99 / subtype ~0.97 on the single-parent plant).
3. A separate lightweight Spark test confirms the shim end-to-end (tiny local SparkSession): fit on a
   few planted docs, `transform` produces a `nodeAffinity` column of the right width.

If the SVI engine cannot match the Gibbs oracle on the plant, that is a real inference bug to fix,
not a tolerance to loosen.

## Interfaces (boundaries)

- **In (fit):** a DataFrame with `featuresCol` (BOW) + `labelCol` (frontier node-id set) + a `dag`
  param. The OMOP layer (pieces 2/3) produces exactly these — the frontier from `frontier_from_coded`
  and the DAG from `ConditionDag.to_engine()`.
- **Out (transform):** a `nodeAffinity` Vector column (the affinity profile) — the case-finding
  readout; `render_profile` (already built) can visualize a single row.
- The engine core is domain-agnostic (integer ids); the DAG/frontier come from the OMOP layer.

## Testing

- Unit (synthetic, no Spark): `GatedOnlineLDA.local_update` gates correctly — a doc with frontier F
  contributes sufficient stats to `allowed_set(F)` rows only (zero elsewhere); a doc with empty
  frontier behaves like ungated OnlineLDA; `nodeAffinity` aggregation sums the right blocks.
- Behavioral (synthetic, no Spark): the SVI-vs-Gibbs equivalence gate above.
- Shim (tiny local SparkSession): fit/transform smoke — `nodeAffinity` column width == #nodes.

## Scope / deferred

- **In scope:** `GatedBOWDocument`, `GatedOnlineLDA` (gated E-step), the ungated `nodeAffinity`
  fold-in, the mllib shim, the SVI-vs-Gibbs equivalence validation.
- **Deferred:** DMR soft-prior variant (v2); sparse/blocked beta for very large K; the OMOP piece 2/3
  wiring (which then targets this shim); optimize_alpha/eta interactions under gating (start with the
  OnlineLDA defaults).

## References

- Hoffman, Blei, Bach (2010) — Online Learning for LDA (the SVI/online-VB this extends).
- Griffiths & Steyvers (2004) — collapsed Gibbs LDA (the oracle engine).
- Mimno & McCallum (2008) — Dirichlet-Multinomial Regression (the deferred DMR v2).
