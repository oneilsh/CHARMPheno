# Gated-PC real-run readiness — pointing the composition at AoU/Mondo

**Date:** 2026-08-13
**Status:** Runbook. The composition + shim are built and validated on synthetic and
realistic-β data; this is the checklist to run it on real All-of-Us OMOP data against a
Mondo disease DAG for rare-disease case-finding.

## What is built (and where)

- **Composition** — `OnlinePCLDA(topic_engine=GatedOnlineLDA(lay, ...), head=DagClosureHead(closure_parents))`.
  Topic-side DAG gate welds each Mondo node's topic block to its subtree's patients; the
  label-side head predicts closure membership from the ungated θ. ADR 0042; ADR 0041 (the
  absolute `head_l2 = lambda_w` calibration) is load-bearing — do not set `head_l2=0`.
- **MLlib entry point** — `OnlinePCLDAEstimator` with `gateParent` / `gateNBg` / `gateTpn` /
  `frontierCol` / `closureParents` / `headL2` / `weightY` / `headOptimizer="newton"`.
- **Docs/data** — `GatedPCDocument` (features + frontier + label + labelMask).

## Inputs a real run must assemble (Spark DataFrame, one row per patient)

| column | type | meaning |
|---|---|---|
| `features` | `Vector` (V = vocab) | OMOP concept counts per patient (bag-of-codes; the same vocab the β bundle was built from — `scripts/build_sim_beta_npz.py` / HF `oneilsh/lda_pasc`). |
| `frontier` | `array<int>` | the patient's **most-specific attested Mondo nodes** (node ids in the layout's space). Gates topic training. Empty = background patient. |
| `label` | `array<double>` (C,) | closure membership: 1 for every node in the closure of the attested frontier, else 0. |
| `labelMask` | `array<double>` (C,) | 1 where a node's label is **observed/trusted** for this patient (semi-supervised: chart-reviewed or high-confidence coded), 0 elsewhere. |

## Steps

1. **Fix the Mondo sub-DAG.** Choose the target rare-disease nodes + their ancestors up to a
   practical root. Produce (a) `gateParent` = `{child_node: parent_node|[parents]}` (topic
   side) and (b) `closureParents` = length-C list of parent-index lists (label side). Same
   DAG for both is the natural choice; node ids must be the same `[0, C)` space as `label`.
   Mondo is a DAG (multi-parent) — both structures are diamond-safe.
2. **Map patients → frontier.** From each patient's coded conditions, attest Mondo nodes
   (concept→Mondo crosswalk), then reduce to the most-specific set with
   `dag_placement.frontier_from_coded`. This is the case-ascertainment step and the main
   source of label noise; treat coded labels as `labelMask=1` only where trusted.
3. **Size the layout.** `gateNBg` = background comorbidity topics (≈ the K you'd use for an
   unsupervised background model, e.g. 150–300 for real AoU vocab), `gateTpn` = topics per
   node (start at 1). K = `gateNBg + n_nodes*gateTpn` is derived — `k` is ignored when gated.
4. **Fit.**
   ```python
   est = (OnlinePCLDAEstimator(
              numLabels=C, labelCol="label", labelMaskCol="labelMask",
              frontierCol="frontier", weightY=<~tokens/doc; tune>,
              headOptimizer="newton", headLr=0.7, headL2=1e-3,
              weightYWarmupIters=10, subsamplingRate=<0.01–0.1 for scale>,
              maxIter=<100+>, seed=0)
          .setGateParent(gate_parent).setClosureParents(closure_parents))
   est._set(gateNBg=<n_bg>, gateTpn=1)
   model = est.fit(df)
   ```
5. **Score.** `model.transform(df_score)` appends `topicDistribution` (θ) and `probability`
   (per-node P(node) — the closure product for the DAG head). Deployment is **ungated**
   (label unknown at score time), by design. For placement you can also read
   `node_affinity` (topic-block mass) — the gate's native readout.
6. **Evaluate.** Held-out per-node AUC for ranking; for *discovery* use the Efron two-groups
   empirical-null per-node FDR (insight 0064) — ranking AUC ≠ FDR-controlled discovery, and
   on buried signal the binding constraint is information, not the lens.

## Known caveats / decisions to carry in

- **`head_l2` must be > 0** (default 1e-3 = Hughes `lambda_w`); 0 blows up on the separable
  topics PC creates (ADR 0041). Good basin ≈ 1e-4…1e-2.
- **Inject the hierarchy ONCE — do not stack the gate and the closure head.** The topic-side
  gate and the label-side DAG-closure head are two ways to encode the SAME Mondo hierarchy.
  On realistic β (`manual_gated_pc_realistic`): gated + FLAT head = 0.745 (best, beats ungated
  0.661), but gated + DAG-closure head = 0.495 (**collapses to chance**) — the closure product
  over already-hierarchy-structured gated topics degenerates. Same ordering on clean synthetic
  (gated+flat 1.00 > gated+DAG 0.84). **Recommended defaults:** (i) gate + flat head, or
  (ii) ungated + DAG-closure head. Run both and pick per the held-out metric; do NOT run
  gate + closure head together.
- **`weight_y` scale** ≈ tokens per document, "possibly much larger" (Hughes); tune on
  validation. `weightYWarmupIters` ramps it so early aggressive ρ does not shove the head.
- **Save/load does NOT round-trip Params** (deferred, ADRs 0009/0012). A reloaded DAG-head
  model scores with the FLAT head unless `closureParents` is re-set on the loaded model. For
  a save→score-later pipeline, re-apply `closureParents` (and `weightY>0`) after `load`, or
  score in the same session. The gate itself is not needed at score time (ungated), so
  `gateParent` non-persistence is harmless. **If a production save→reload scoring path is
  required, add Param persistence (a separate, general task).**
- **α optimization:** the gated engine owns a per-node α Newton step; leave PC's α to the
  delegate (`optimizeDocConcentration` flows to the engine). Do not double-optimize α.

## Evidence so far

- Clean synthetic (planted node topics): the gate lifts head-AUC 0.72 → 1.00
  (`manual_gated_pc_case_finding`).
- Realistic β (Mondo archetype on real cross-site LDA topics, `manual_gated_pc_realistic`,
  the closest proxy to a real AoU run without patient data): head-AUC mean —
  ungated+DAGhead **0.661**, gated+DAGhead **0.495** (collapse), gated+flathead **0.745**.
  The gate helps *with a flat head* and hurts *with the closure head*; see the "inject once"
  caveat. This is the load-bearing result for choosing the real-run configuration.
- The real AoU run is the only test that proves transfer; synthetic/realistic-β de-risk the
  mechanism, not the data.
