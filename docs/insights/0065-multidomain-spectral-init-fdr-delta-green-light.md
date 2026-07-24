# 0065 — Multi-domain spectral init recovers domain-anchored nodes; FDR-delta green light on synthetic (with a control caveat)

**Date:** 2026-07-24
**Context:** SP1 of the multi-domain (MixEHR-style) gated DAG LDA arc (branch `multidomain-spectral-init`). Motivation: conditions + drugs, where a corroborating domain sharpens node-vs-background contrast. All results are on planted, domain-agnostic synthetic corpora (two integer-id token domains); no real cohort.

## Findings

1. **The joint-Q anchor recipe recovers a node anchored from the second domain alone.** One co-occurrence Q over the concatenated `[domain-0 ; domain-1]` vocab (cross-block intact) → one greedy anchor search with a per-domain candidate floor → recover_beta → split_domains recovers a node's per-domain signatures even when its domain-0 signature is made ambiguous (identical to its parent's block) but its domain-1 signature is unique: support-overlap 0.916 vs a 0.4 gate. The single joint Q, not two aligned per-domain runs, is what carries the tie (Q_01 = (B_0)ᵀ A (B_1)).

2. **The per-domain candidate floor is load-bearing, not cosmetic.** With a pooled marginal floor the sparser domain's anchors fall below the joint mean and none are selected. Concretely, path-length dilution (a 2-hop leaf gets ~half the per-node token budget of a 1-hop node) pushed the ambiguous node's domain-1 block below the pooled floor; flooring within each domain rescues it. This is the anchor-word analogue of insight 0021 (universal-anchor mass concentration) but on the domain axis.

3. **FDR-delta green light (the specificity mechanism works on synthetic data).** Using the existing per-node empirical-null FDR readout: a node whose domain-0 signature is ambiguous vs its sibling gets **0 / 204** confident node-specific discoveries at q = 0.10 from domain 0 alone, but **204 / 204** once its unique domain-1 signature is present — a stable ~190–215-discovery gap across 5 seed pairs (domain-1-OUT always 0). A **ubiquitous** domain-1 token (present in every doc) gives **no** gain (control flat 218/218). Leakage is held fixed (the exact per-node marker codes are stripped in both arms), so the lift is corroboration, not label smuggling. Both arms drop domain-1 columns from the *same* documents (never a different refit).

## Caveat (why this is a mechanism demonstration, not the specificity claim)

The negative control changes **two** variables at once — it removes the domain-0 ambiguity *and* makes domain-1 ubiquitous-only — because the current generator cannot express "ambiguous node whose domain-1 signal is ubiquitous-only" (a `b_only_node` always grants that node a unique domain-1 block). The control still refutes the two rival explanations that matter (the gain is not "just more data / extra anchor candidates," and a non-node-specific domain-1 token yields nothing), but it does not isolate node-specificity as a single variable. The tighter control needs a generator knob (draw the ambiguous node's domain-1 contribution from the shared common pool) and is deferred to SP2's richer multi-domain validation.

**Bottom line:** the init recipe and the FDR readout together demonstrate the corroboration *mechanism* on planted data — a node-specific second domain rescues an otherwise-unplaceable node. Whether *real* drugs corroborate real diagnoses is the SP4 real-cohort ablation; this is the synthetic green light that precedes it.
