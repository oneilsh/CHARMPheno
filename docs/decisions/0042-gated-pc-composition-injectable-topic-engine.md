# 0042 — Gated-PC composes via an INJECTABLE topic engine, not a subclass chain

**Status:** Accepted
**Date:** 2026-08-13

> **Numbering note.** Shared-counter caveat as ADR 0038–0041; renumber if a sibling
> claimed 0042 first.

## Context

The supervised-head seam spec (`docs/superpowers/specs/2026-08-12-pc-supervised-head-seam-design.md`)
frames PC as a 2×2 of independent seams: a **topic-side** seam (the E/M step — plain
`OnlineLDA` vs the DAG-gated `GatedOnlineLDA`, which welds each node's topic block to its
subtree's documents) × a **label-side** seam (the head — `FlatLogisticHead` vs
`DagClosureHead`). Step 3, "Gated-PC composition" (task #14), is the DAG-gated × head
quadrant, and the spec explicitly parked a **plumbing fork**: `GatedOnlineLDA` *is-a*
`OnlineLDA` (subclass, overrides `local_update`/`initialize_global`), whereas `OnlinePCLDA`
*has-a* `OnlineLDA` (delegates to an instance it hard-constructs). Composing them could go
two ways:

- **Subclass chain:** a `GatedOnlinePCLDA(GatedOnlineLDA)` that mixes the head logic in.
- **Injectable delegate:** let `OnlinePCLDA` accept a pre-built topic engine.

## Decision

**Injectable delegate.** `OnlinePCLDA.__init__` gains `topic_engine: OnlineLDA | None =
None`; when provided it becomes `self._lda` (K/V taken from the engine, the LDA-building
kwargs are the engine's own), else the prior `OnlineLDA` is constructed unchanged. A caller
composes by passing `topic_engine=GatedOnlineLDA(lay, vocab_size=V, ...)`.

This works with **zero changes to the head machinery** because the head is already
topic-engine-agnostic: it reads only `global_params["lambda"]` (to form `expElogbeta`),
`alpha`, and `K` — never any `OnlineLDA` internal. Since `GatedOnlineLDA` IS-A `OnlineLDA`,
every delegate call `OnlinePCLDA` already makes (`initialize_global`, `local_update`,
`update_global`, `combine_stats`, `compute_elbo`, `infer_local`, property passthroughs)
dispatches to the gated overrides for free. A new `GatedPCDocument` (= `PCDocument` ∪
`GatedBOWDocument`) carries `.frontier` (gates topic *training*) and `.y`/`.label_mask`
(supervises *prediction*), duck-compatible with both consumers.

**The head stays on the ungated, label-free full-K θ.** Training gates the *unsupervised*
E-step (the delegate's gated `lambda_stats`), but the head's supervised θ — the one it
shapes and predicts on — is the ungated full-K label-free CAVI mean. That preserves
Hughes's train/deploy invariant (the head's θ is identical at train and score time) and
matches the gate's own deployment path (held-out docs fold in ungated, label unknown). The
gate and the head thus act through different θ pathways but the *head's* θ is consistent.

## Alternatives considered

- **Subclass chain (`GatedOnlinePCLDA`).** Rejected: it would duplicate the ~200 lines of
  head-stat emission/combination/M-step that live in `OnlinePCLDA`, and force a re-mix for
  every future topic-engine variant. Injection reuses all of it for any `OnlineLDA` subclass.
- **Gate the head's training θ too** (differentiate through the gated per-doc sub-simplex).
  More faithful to the *gated* training π, but it would (a) reintroduce a train/deploy θ
  mismatch the head is designed to avoid, since scoring is ungated, and (b) need the
  `grad_topics_stat` scatter to respect each doc's `allowed` set. Deferred as a possible
  refinement; the ungated-head choice is simpler and deploy-consistent.

## Consequences

- **Positive.** Task #14 lands as a small, backward-compatible change (default
  `topic_engine=None` is byte-for-byte the prior behavior). All four 2×2 quadrants are now
  runnable through one class. 29 PC+gated pyspark tests pass.
- **Validated end-to-end** (`manual_gated_pc_case_finding`, planted node topics on a 6-node
  DAG): the topic-side gate transforms case-finding — head-AUC 0.72 (ungated+DAG head) →
  **1.00** (gated+flat) / 0.84 (gated+DAG head), node_affinity ≈ 0.84. The gate welds node
  topics to their subtree's docs, so a downstream head reads them almost trivially.
- **Finding on the head choice.** On cleanly, *independently* separable synthetic nodes the
  flat head is optimal and the `DagClosureHead`'s monotone `P(child) ≤ P(parent)` coupling
  can *hurt* (0.84 < 1.00). The closure head earns its keep where the hierarchy regularizes
  a weak/rare/noisy signal (the realistic EHR benchmark), not on clean synthetic bars —
  choose the label-side seam to match the data, independently of the topic-side gate.
- **Open.** α-optimization: `GatedOnlineLDA` owns a gated per-node α Newton step; when both
  it and any future PC-side α move are active they must be reconciled in one `update_global`
  (today PC leaves α entirely to the delegate, so no collision). The mllib shim does not yet
  expose `topic_engine`; the composition is Python-API-only for now.
