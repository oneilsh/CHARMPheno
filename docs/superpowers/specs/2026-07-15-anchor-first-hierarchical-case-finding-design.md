# Anchor-First Hierarchical Case-Finding — Design

**Date:** 2026-07-15
**Status:** Design converged (brainstorm), pre-implementation
**Prototype:** `scratchpad/dag_placement.py` (validated on sim; this spec promotes it to a tested module)

## Goal

Given patients with a known ontology diagnosis, hide that diagnosis and **place** them in the
ontology from their phenotype codes alone, scored by how close the placement lands (DAG-distance,
per-node precision/recall under heavy class imbalance). The phenotype profiles are **learned** from
data — no external knowledge-graph mapping (e.g. SNOMED→MONDO/HPO) is required. The real payload is
finding **uncoded** patients; the held-out-diagnosis experiment is how we measure accuracy.

Success is **recovery/ranking**, not calibrated coverage (a prior direction established that
calibrated prevalence coverage is a regime-hard problem largely orthogonal to case-finding; see
insights 0057/0058). We measure discriminative placement quality, not interval calibration.

## Scope

- **In scope (this spec):** the domain-agnostic placement engine + evaluation — promote the
  validated prototype to a proper, tested module operating on integer ids.
- **Out of scope (separate, environment-specific):** OMOP cohort assembly (patients → documents +
  labels + DAG). Specified here only as the interface the engine consumes and the requirements the
  assembly must satisfy (leakage, labeling, windowing). Built on the cluster.

## Architecture

Data flow:

```
anchors → build DAG → assemble patient-year docs + labels → gated-train topics
        → fold in held-out patients (unmasked) → affinity profile over the DAG → score
```

The engine is a small set of focused units over integer ids:

- `DagLayout` — DAG structure + topic-block layout (background block + `tpn` topics per node);
  computes `closure(v)`, `subtree(u)`, `allowed(v)`, `depth(v)`.
- `fit_gated(train_docs, train_labels, layout, V, ...)` — gated collapsed-Gibbs training.
- `profile(doc, beta_hat, layout, ...)` — unmasked fold-in → per-node affinity profile.
- `evaluate(profiles, test_labels, layout)` — per-node AUC, DAG-distance, MRR/top-k.
- `identifiability_annotation(...)` — post-fit diagnostic flagging inseparable node-pairs.
- `render_profile(...)` — text (DAG tree + unicode bars) for spot-checks.

## 1. DAG construction (anchor-first, identifiable)

- Pick anchor concept(s); pull the ontology subtree(s) beneath them **at whatever depth the ontology
  places them** (irregular: deep in some branches, shallow in others). Multi-anchor forms a forest
  under a shared background root; **single anchor for v1**.
- Keep every node with **permissive attestation** — drop only near-empty nodes (a topic cannot be
  learned for a node no patient populates). The attestation threshold is a low knob.
- Collapse only **single-child pass-through chains** (structural triviality — no branching means no
  distinguishable levels by construction). Keep everything else.
- **No separability collapse is baked in.** Rationale: the confusion between inseparable nodes is
  itself the diagnostic signal; collapsing early throws away the observation that would justify it,
  is irreversible, and hides errors. Because the output is a graded profile (not a hard placement),
  split affinity mass *is* the honest answer, so collapse is unnecessary to avoid a wrong call.
- The **identifiability compiler is repurposed as a post-fit diagnostic annotation**: it flags which
  node-pairs the design genuinely cannot separate (null-space / collinearity of the design), so split
  affinity mass reads as "real clinical ambiguity" vs "we lack the contrasts." It is the tool that
  would *inform* a later, data-driven collapse if ever wanted.
- Any future merges act **only within the structure** — chain (parent↔child) or sibling (same
  parent) — **never cross-branch.** Cross-branch phenotype similarity is a *reporting* fact (the
  profile lights up both), not a structural merge; merging across branches would destroy the
  clinically meaningful ontology distinctions that are the method's whole point.

## 2. Documents & labels

- **One patient-year per patient**, diagnosis-anchored and non-empty; **conditions only** (drugs
  deferred).
- **Label = the most-specific node coded in the selected window.**
  - Multi-level same-path (e.g. T2D + T2D-with-complications) → the **deepest** node. This is native
    to the closure (the gated mask informs every ancestor block), **not** dropped — it is the
    method's home turf, not the flat-model comorbidity conflict of the GLP1 work.
  - Sibling-level ambiguity within the anchor (e.g. renal + ophthalmic, neither an ancestor of the
    other) → the **lowest common ancestor (LCA)** label. The inability to pick a sibling *is* the
    answer: place at the granularity the evidence supports.
  - Cross-anchor multimorbidity (entirely different anchor diseases) → **dropped for v1** (defers true
    many-to-many).
- The label is computed **from in-window diagnoses only**, keeping label and document temporally
  consistent: diagnoses in different years yield cleanly single-branch labels per year; LCA fires
  only when siblings are genuinely concurrent in one window.
- **Instrumentation (required):** report the **LCA-collapse rate** and **per-node training counts**,
  so subtype-node starvation from sibling ambiguity is observable in the metrics, not assumed away.
- Progression/trajectory modeling is explicitly **parked** — v1 is a static one-year snapshot.

## 3. Leakage control

- **Fitting: documents are left intact.** Anchor codes help anchor the topics; the real uncoded
  targets lack them anyway; and fit-with / score-without mirrors deployment. (A "strip anchor codes
  from fits too" variant is a **later comparison test**, not a v1 branch.)
- **Evaluation: strip every code matching any DAG node** from test documents before scoring, so
  placement cannot read back the label.

## 4. Model — gated-train engine

- Topic layout: a shared **background block** + **`tpn` topic blocks per DAG node** (`tpn` a knob,
  **default 1**).
- **Training** on labeled patients uses a **closure mask**: a patient at node `v` may write only to
  `background ∪ (blocks of u for u in closure(v))`. This ties topics to nodes **structurally**,
  eliminating the post-hoc alignment step that was the prototype's cross-family-noise source.
- **Spectral (greedy-anchor) initialization** seeds `beta` (reuse `spectral_init.find_anchors` /
  `recover_beta`; the scalable variant exists for cluster scale). **No KG seeding.**
- `tpn > 1` is a real thread, parked: it lets the model discover phenotype-*presentation* clusters
  within a diagnosis (orthogonal to ontology substructure) and, across nodes, recurring presentation
  patterns — a discovery capability, not just a fit knob.

## 5. Output — affinity profile

- Each held-out patient is folded in **unmasked** → a loading on every topic → a **node-affinity
  profile over the whole DAG** (mass on each node's block), **not** a single hard node.
- Depth is **graded per patient**: the profile concentrates as deep as the evidence supports and
  stays shallow when it doesn't. Text rendering (DAG tree + bars) for spot-checks; richer
  visualization deferred.

## 6. Evaluation

- **Per-node case-finding AUC** — does affinity on node `u`'s block rank `u`'s subtree members above
  non-members.
- **DAG-distance** — ancestor-hit (predicted node on the true root→node path), hops-to-truth,
  per-depth breakdown.
- **True-node MRR / top-k.**
- **Imbalance-aware** — average precision / precision-recall for rare nodes.
- All computed on **leakage-stripped** test documents.

## 7. Interface (domain-agnostic core)

The engine consumes three integer-id objects, keeping it domain-neutral and independent of the
cohort assembly:

- `docs` — list of 1-D int arrays (a patient's phenotype-code ids; leakage-stripped upstream for
  test docs).
- `labels` — int array, each patient's most-specific (or LCA) DAG node.
- `dag` — `{child: parent}` parent map (root has no entry).

## Validation status

Sim (model-matched, `hierarchical_placement.gen_population`): gated-train **family AUC 0.99 /
subtype AUC 0.97**, **MRR 0.885 ± 0.008**, **top-2 0.984 ± 0.002** across 5 seeds. The gated-train
design fixed the prototype's cross-family alignment noise (subtype AUC 0.68 → 0.97). Caveats: sim is
model-matched; real data is harder (noisier topic recovery and labels), and the leakage line is
where a real result lives or dies. First real cohort: **diabetes, L1 (type: T1/T2/gestational)
first**, adding the complication layer as a follow-up.

## Testing approach

- Unit: `DagLayout` closure/subtree/allowed/depth on a fixed small DAG (including irregular depth and
  a single-child chain); leakage strip removes exactly the DAG-node codes; LCA labeling on a patient
  with two sibling diagnoses; deepest-node labeling on a same-path pair.
- Behavioral (sim, plant-and-recover): the `gen_population` plant → gated-train → evaluate reproduces
  family/subtype AUC in the validated range; the identifiability annotation flags a deliberately
  non-separable sibling pair; per-node training counts and LCA-collapse rate are reported.
- Domain-agnostic discipline: engine tests use integer ids only; no clinical vocabulary in the
  engine or its tests.

## Deferred / parked (explicit)

Progression/trajectory modeling; cross-anchor many-to-many multimorbidity; `tpn > 1` exploration;
KG seeding (optional enhancement, not required); data-driven node collapse (later, informed by the
identifiability annotation); strip-anchor-codes-from-fit comparison; richer visualization; cluster
scaling of the fold-in.
