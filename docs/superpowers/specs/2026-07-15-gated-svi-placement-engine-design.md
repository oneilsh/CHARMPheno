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

## Decisions (brainstorm + prototype validation)

> **Prototype update (2026-07-15).** The design below was validated locally (no Spark) against the
> collapsed-Gibbs oracle before planning (`scratchpad/gated_svi_proto.py`, `dag_svi_soft.py`). Two
> decisions changed as a result and are folded in below: (1) the equivalence gate is **placement-based
> and depth-weighted**, not per-node beta cosine; (2) **initialization is a pluggable strategy** with
> **random init as the validated default**. See "Prototype findings" below for the evidence.

- **Hard gate** (not DMR soft prior): each training doc's variational E-step is restricted to
  `allowed_set(frontier)` topics — the exact variational analogue of the Gibbs gate. DMR
  (alpha_d = exp(Lambda x_d), Mimno & McCallum 2008) is a documented v2, not v1.
- **Initialization is a pluggable strategy** (new): `GatedOnlineLDA` takes an `init` strategy.
  **`"random"` is the default** (Gamma(gamma_shape, ·) lambda, inherited from OnlineLDA) — the
  prototype showed it already matches the Gibbs oracle on placement at every depth, because the DAG
  gate itself supplies the topic-to-node identifiability that spectral init exists to provide in
  *ungated* LDA. A second built-in strategy, **`"spectral"`** (block-aligned, forward-topological /
  ancestors-first — see Architecture), is provided as a validated option for the real-DAG harness,
  **not the default** (on the synthetic plants it did not help the fit and could regress shallow
  nodes when the recovered seed row was imperfect). The strategy is an extension point: future
  strategies (e.g. seeding node topics from established phenotype profiles) plug in here without
  touching the E-step. The strategy produces the initial `lambda`; everything downstream is unchanged.
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
  - Inherits `update_global` (SVI natural-gradient beta step), `compute_elbo`, `combine_stats`, and
    `VIRunner` integration unchanged.
  - Overrides `initialize_global` to dispatch on the `init` strategy for the initial `lambda`
    (random default; see below). Every strategy returns a `(K, V)` lambda; the rest of the global
    param dict (alpha, eta) is unchanged.
- **Init strategies** (`spark_vi/models/topic/gated_init.py`, or alongside `gated_lda.py`):
  - `"random"` (default): the inherited OnlineLDA Gamma init. Validated best on the plants — the gate
    already welds topics to nodes, so no symmetry-breaking seed is needed.
  - `"spectral"` (optional): **block-aligned, forward-topological** init generalizing OnlineLDA's
    `spectral_init_beta` (background→foreground) to the DAG. Process nodes ancestors-first (sorted by
    `lay.depth`); for each node u, build the within-node co-occurrence `Q_u` over the docs that train
    u (those with u in the union of their frontier closures), find `tpn` anchors deflated against the
    background anchors **and u's already-recovered proper-ancestor anchors** (`find_anchors(seed_rows=…)`
    plus include-then-drop in `recover_beta`), and recover u's block. Forward order is required because
    a node can only be deflated against ancestors already recovered. Prototype: `scratchpad/dag_spectral_init.py`.
  - Extension point: a strategy is a callable `(train_docs, train_labels, lay, V) -> (K, V) lambda`;
    future strategies (phenotype-profile seeding) register here. `"spectral"` and any profile strategy
    only affect `initialize_global`; the gated E-step and equivalence gate are strategy-agnostic.
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

## Validation — the equivalence gate (placement-based, depth-weighted)

The collapsed-Gibbs engine is the oracle. On a planted corpus (reuse
`tests/_stm_synth.dag_placement_corpus` / `dag_placement_corpus_multi`):

1. Fit both `fit_gated` (Gibbs) and `GatedOnlineLDA` (SVI, run locally without Spark via a small
   in-memory driver over `local_update`/`update_global`) on the same corpus.
2. Assert the **placement metrics agree** — per-node AUC by depth, MRR, and top2 within Monte-Carlo
   tolerance of the Gibbs numbers (prototype: aucD1 = aucD2 = 1.000, mrr 0.914 vs Gibbs 0.905 on the
   single-parent plant). The gate is **depth-weighted**: deeper-node AUC is the metric we hold the
   line on (per user preference — deep-level accuracy matters more than shallow), and it must not
   regress below the Gibbs oracle within tolerance.
3. Assert **ground-truth own-block recovery**: each node's SVI topic peaks (argmax over the signature
   region) on that node's own planted block — the direct "gate welds topic to node" check, evaluated
   per-engine against the plant (not engine-to-engine).
4. A separate lightweight Spark test confirms the shim end-to-end (tiny local SparkSession): fit on a
   few planted docs, `transform` produces a `nodeAffinity` column of the right width.

**Not a gate: per-node beta cosine between the two engines.** The prototype refuted it — `fit_gated`
is a supervised, theta-free count estimator and `GatedOnlineLDA` is full VB-LDA with a per-doc theta
factor, so their betas differ by construction (cosine 0.44–0.84, dominated by common-pool allocation)
while placing identically. SVI's per-node beta is also *muddier* than Gibbs' at collinear parent/child
depth-2 nodes (an even-split ~0.5 own-mass) — benign, orthogonal to placement; the Gibbs engine stays
the offline gold-standard for deep-beta interpretation. That two-engine division of labor is the point.

If the SVI engine cannot match the Gibbs oracle on the **placement** metrics, that is a real inference
bug to fix, not a tolerance to loosen.

## Prototype findings (2026-07-15, evidence for the amended decisions)

- **Engine correct, placement equivalent.** The one-method `local_update` override (CAVI over
  `expElogbeta[allowed]`, sstats scattered to `lambda_stats[np.ix_(allowed, indices)]`) places =
  Gibbs at every depth on single- and multi-parent plants.
- **The gate supplies identifiability; spectral init is redundant with it and can hurt.** Random init
  matches the oracle; feeding a block-aligned spectral seed as a lambda pseudo-count (hard or soft,
  scales 20/60/150) biases the gate-restricted E-step toward the seed's imperfect rows and regressed
  shallow nodes deterministically. Hence random default, spectral optional.
- **Forward-topological is the right order *for the spectral strategy itself*** (ancestors-first):
  a node can only be deflated against ancestors already recovered. Reverse-topological is refuted for
  this deflation mechanism. (The ordering question raised earlier is thus resolved.)
- **Deep-level accuracy lever is iterations, not init:** random-init aucD2 rose 0.862 → 1.000 going
  from 150 to 250 local passes.

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
- Unit (synthetic, no Spark): init strategies — `"random"` yields the inherited Gamma lambda;
  `"spectral"` yields a block-aligned lambda whose each node block peaks on that node's own planted
  signature (forward-topological ancestors-first recovery); an unknown strategy name raises.
- Behavioral (synthetic, no Spark): the SVI-vs-Gibbs placement equivalence gate above (depth-weighted
  AUC/MRR within tolerance + ground-truth own-block argmax), run with the default `"random"` init.
- Shim (tiny local SparkSession): fit/transform smoke — `nodeAffinity` column width == #nodes.

## Scope / deferred

- **In scope:** `GatedBOWDocument`, `GatedOnlineLDA` (gated E-step), the pluggable init strategy
  (`"random"` default + `"spectral"` block-aligned forward-topological), the ungated `nodeAffinity`
  fold-in, the mllib shim, the placement-based SVI-vs-Gibbs equivalence validation.
- **Deferred:** DMR soft-prior variant (v2); phenotype-profile init strategy (future, plugs into the
  same init extension point); sparse/blocked beta for very large K; the OMOP piece 2/3 wiring (which
  then targets this shim); optimize_alpha/eta interactions under gating (start with the OnlineLDA
  defaults).

## References

- Hoffman, Blei, Bach (2010) — Online Learning for LDA (the SVI/online-VB this extends).
- Griffiths & Steyvers (2004) — collapsed Gibbs LDA (the oracle engine).
- Arora et al. (2013) — anchor-word spectral recovery (the `"spectral"` init strategy's basis).
- Mimno & McCallum (2008) — Dirichlet-Multinomial Regression (the deferred DMR v2).
