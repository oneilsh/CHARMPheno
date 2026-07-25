# 0066 — Multi-domain gated LDA under random init suffers per-seed topic death; the spectral seed is the fix

**Date:** 2026-07-24
**Premise questioned by [0067](0067-background-starved-plants-frame-a-correct-spectral-seed-as-broken.md)** — every measurement below was taken on a corpus with **zero background documents**, which independently causes the background block to absorb foreground signatures (the absorption this entry describes). Whether random multi-domain init still suffers topic death with a background pool present is untested. Read the conclusion as established for background-starved corpora only.
**Context:** SP2 of the multi-domain gated DAG LDA arc (branch `multidomain-spectral-init`). The dict-λ multi-domain `GatedOnlineLDA` (per-domain topic-word matrices, shared gated θ). Findings on planted, domain-agnostic two-domain synthetic corpora.

## Finding

Fitting the multi-domain gated model from **random init** is **per-seed fragile**: a node's foreground block topic can suffer **topic-death** — it never gains initial traction, and its node-specific signal is absorbed by the background block instead. On a flat 3-node DAG (each node a direct child of root, clean unique per-domain signatures), worst-node recovery across `random_seed` 0–4 was `[0.13, 0.51, 0.13, 0.50, 0.50]`: **2 of 5 seeds** left a node's block topic stuck at the uniform prior (recovery ≈ len(support)/V_m ≈ 0.15/0.125), while 3 recovered every node at ~0.5. The starvation is visible after a single E-step — the dead topic's `lambda_stats` row-sum is ~10³× smaller than its healthy siblings' despite the node having a normal share of documents and a correctly-gated allowed set.

This is the classic LDA local-optimum / topic-death phenomenon, not a multi-domain-specific bug. The multi-domain E/M-step itself is **correct**: seeded near the planted signatures (or even with a gentle 0.3 traction bump on each block topic's own support), the same fit recovers **every** node/domain to **0.97–0.999** (oracle seed) or the ~0.5 structural ceiling (gentle seed) deterministically. The half-mass ceiling is expected — half of each document's per-domain tokens are the shared common pool (absorbed by the background block) and half are the node signature.

## Why it matters

1. **The SP1 spectral seed is load-bearing, not optional.** SP1 built `split_domains` explicitly as the per-domain λ seed; SP2 Task 1 deferred wiring it (random init was assumed sufficient, as it is for the single-domain gated model). This finding shows random init is unreliable for the multi-domain fit, so the spectral seed (which gives every block topic traction on its domain-anchored signature) must be wired before the production/recovery/SVI≈Gibbs paths are trusted.
2. **It exposed a hidden-tautology test.** The original SP2 Task-2 "recovery" test asserted only `beta.sum(1) == 1` (true for any λ) and discarded the planted ground truth — it passed on a seed where a node topic was dead. Replaced with a genuine per-domain recovery assertion (gentle-traction seed → refine → assert >0.4 mass on each node's planted support in both domains), which isolates E/M correctness from init fragility.

**Bottom line:** the multi-domain dict-λ E/M-step is correct; random init is the weak link. Wire the SP1 spectral seed (deferred in SP2 Task 1) as the production init before relying on multi-domain recovery, and validate recovery with a traction-seeded (or spectral-seeded) fit, not bare random init.
