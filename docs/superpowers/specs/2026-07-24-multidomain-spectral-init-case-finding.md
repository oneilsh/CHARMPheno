# Multi-domain spectral init for case-finding — design brief

**Date:** 2026-07-24
**Target branch:** `case-finding` (engine: `spark-vi/spark_vi/models/topic/`; consumer: `dag_placement.py`)
**Status:** Design brief for a future implementation session. NOT a task-by-task plan — expand into one with `superpowers:writing-plans`.

## Goal

Extend the anchor-word spectral init to **multiple token domains** (v1: two — conditions + drugs) so a DAG node's topic can be anchored by whichever domain is purest for it, and each topic recovers a proper per-domain distribution. Motivation: raise case-finding **specificity** (lower per-node FDR) by letting a corroborating domain (drug attests diagnosis) sharpen the node-block-vs-background contrast where conditions alone are ambiguous.

## Why the naive approach fails, and the fix

Running Arora greedy independently per domain gives two **unaligned** anchor bases — topic-k-in-conditions and topic-k-in-drugs have no correspondence, and there's no canonical post-hoc alignment. Do **not** try to align two runs.

The tie already lives in the data. In the shared-topic generative model a drug and a condition drawn from the same document share the same θ, so the cross-domain co-occurrence is exactly the shared-topic signal:

    Q_CD[i,j] = P(cond=i, drug=j) = Σ_k A_k · β^C_ki · β^D_kj = (B_C)ᵀ A (B_D)

So build ONE joint co-occurrence over the concatenated vocabulary `[conditions ; drugs]` with the cross-block intact, and run ONE greedy on it. A pure condition anchor's Q̄ row already carries that topic's drug profile via the cross-block, so a single anchor (from either domain) defines the topic across both domains.

## Algorithm (reuses existing `spectral_init.py` almost verbatim)

1. **Concatenate vocab**, drugs offset after conditions into one index space. Document `indices` must span both domains.
2. **Joint Q** via existing `word_cooccurrence(docs, V_C+V_D)` — the `outer(n,n)` block over a doc's combined tokens produces `Q_CC`, `Q_DD`, and the cross `Q_CD` automatically.
3. **`find_anchors` on the joint Q̄** — one greedy, K anchors from the pooled hull; anchors may be conditions or drugs.
4. **`recover_beta` on joint Q** → K×(V_C+V_D) joint β.
5. **Split + per-domain renormalize** each topic row into its condition-slice and drug-slice, each renormalized to sum 1 → `β^C` (K×V_C), `β^D` (K×V_D): the two MixEHR-style basis matrices, sharing topic identity.

### The two required code changes

- **Per-domain candidate floor** in `find_anchors`. The current floor `thr = min_marginal_frac * marginal[pos].mean()` is computed on the *pooled* marginal — if one domain is sparser, none of its words clear the joint-mean bar and no anchor ever comes from it. Fix: pass domain boundaries; compute the floor *within each domain*. (Row-normalization already makes anchor *selection* scale-free, so per-domain row scaling is NOT needed; the split step in (5) handles the Bayes-flip magnitude difference.)
- **Split/renormalize** as a post-recovery step producing `(β^C, β^D)`.

Everything else (`word_cooccurrence`, `find_anchors` greedy geometry, `recover_beta` NNLS) carries over unchanged.

## Load-bearing prerequisite

The entire tie is `Q_CD`, which comes from **within-document** cross-domain co-occurrence. If drugs and conditions live in *separate* documents, `Q_CD = 0` and you are back to two disconnected hulls. The doc unit (decision 0018 seam) MUST bundle the co-occurring drugs and conditions in one document (e.g. a condition_era/visit window). State this constraint in the plan.

## Composition with existing structure

- **Gating (background/foreground):** orthogonal. In `spectral_init_beta` step 2, build the joint multi-domain Q *per group*. A rare arm's foreground anchor may be an arm-specific drug, recovered on the within-group joint `Q_g`, deflated against the joint background anchors.
- **`dag_placement.py`:** it already reuses `spectral_init`. The multi-domain init drops in; a leaf/subtype node where conditions are ambiguous can anchor on a specific drug.

## Validation

- **Synthetic (engine, domain-agnostic):** extend `_stm_synth.py` with a two-domain generator — K topics, each with a coupled (condition-signature, drug-signature). Assert both `β^C` and `β^D` recover the planted phenotypes, including topics anchored from *either* domain, and that a topic with no pure condition but a pure drug is still recovered.
- **FDR-delta ablation (the specificity claim):** using the existing per-node FDR readout (`dag_placement.evaluate` fdr block, plan `2026-07-21-case-finding-fdr-readout.md`), refit with drugs in vs. out and report per-node FDR shift at fixed sensitivity. Expected: gains concentrated at leaf/subtype nodes. This is the empirical green light before real cohorts.

## Risks to call out in the plan

1. **Only node-specific data helps.** A generic co-prescribed drug (PPI, analgesic) behaves like the universal anchors of insight 0021 — loads onto background, does nothing for a node's p-value, may dilute. Value is measured per-domain by the FDR-delta ablation, not assumed.
2. **Attestation vs leakage is one axis.** The most specificity-boosting drug is the most leakage-prone. Use the branch's existing leakage-strip / identifiability utilities; the honest ablation holds leakage fixed and shows FDR dropping from *corroboration*, not from smuggling the label into features.
3. **Domain imbalance** is the most likely thing to silently break init — verify the per-domain floor actually lets drug anchors through on real token-count ratios.

## Constraints (from branch conventions)

- Engine stays **domain-agnostic**: integer token/domain-boundary ids only, no clinical vocabulary in `spark_vi` or its tests. The domain edge lives in `analysis/cloud`.
- Cite methods in docstrings: anchor-word = Arora et al. 2013; corroboration/attestation for phenotyping = Halpern & Sontag (anchor-and-learn), JAMIA 2016.
- No LaTeX; Unicode Greek only. TDD (`superpowers:test-driven-development`). `case-finding` does not auto-push — push only when asked.

## References

- `spark-vi/spark_vi/models/topic/spectral_init.py` — `word_cooccurrence`, `find_anchors`, `recover_beta`, `spectral_init_beta` (block-aware).
- `spark-vi/spark_vi/models/topic/dag_placement.py` — placement engine (consumer).
- Plans: `2026-07-15-anchor-first-hierarchical-case-finding.md`, `2026-07-21-case-finding-fdr-readout.md`.
- Insights: 0041 (drug-anchor track = drop-in sibling), 0021 (universal-anchor mass concentration).
- Decision 0018 (doc-unit abstraction — the co-occurrence prerequisite).
