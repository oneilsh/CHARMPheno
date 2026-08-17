# 0069 — The ridge-bounded co-fit head is itself well-calibrated at 41-anchor scale: the UNIFIED P(child|parent) model works without post-hoc calibration or Firth

**Date:** 2026-08-17
**Topic:** svi, pc, case-finding, calibration, prediction-constrained

**Status:** Confirmed on exp 0082 (41-anchor, 3-domain, closure-mask, `head_l2=0.01`,
`head_penalty=none`)

The conditional-diagnosis readout was built two ways: a **two-stage** path (a fresh
head-independent LogisticRegression on θ — `pc_topics_lr` — optionally isotonic-calibrated)
and the **unified** co-fit head itself (`sigmoid(w_CK·θ)`, a single model emitting
`P(child|parent)` with no post-hoc fit). The open worry was that the co-fit head, which
runs away to `|w|→∞` on PC-separated topics, could never emit *calibrated* conditional
posteriors — motivating Firth (insight 0067, exp 0080). Exp 0082 ran both heads on the same
41-anchor 3-domain closure fit and printed both conditional readouts. The unified head wins
the worry.

**The ridge-bounded co-fit head is at least as calibrated as the two-stage readout, with
only a small discrimination tax.** Head-to-head (co-fit head vs `pc_topics_lr`, same
cohorts, `head_penalty=none`, `|w|max=2126`):

| metric | two-stage readout LR | unified co-fit head | Δ |
|---|---|---|---|
| pooled conditional ECE | 0.0119 | **0.0098** | −0.0021 (unified better) |
| macro AUC (49 nodes) | 0.7242 | 0.7064 | −0.018 |
| cond_AUC depth 0 / 1 / 2 | 0.716 / 0.666 / 0.721 | 0.706 / 0.657 / 0.715 | −0.011 / −0.009 / −0.006 |

So a single model, no post-hoc calibration, emits `P(child|parent)` that is *better*
calibrated (pooled) than the two-stage readout and within ~0.01–0.02 AUC of it.

**The discrimination tax concentrates in the smallest cohorts — the leverage story in
reverse.** Per-parent balanced accuracy: where n is large the unified head matches or
edges the readout (Congenital heart disease n=745: 0.299 ≥ 0.281; 29-way forest root
n=5424: 0.096 vs 0.103 ≈ tie; Congenital-anomaly d2 n=254: 0.240 ≥ 0.226). The one clear
loss is the smallest cohort (Amyloidosis n=66, 2 children: readout bal_acc 0.644 / top1
0.864 vs unified 0.534 / 0.636) — a clean 66-doc split the fresh LR fits better than the
jointly-trained co-fit head. SLE / Sarcoidosis are trivial (one child dominates), correctly
flagged by both.

**Why the ridge suffices where Firth failed.** A fixed absolute L2 (`head_l2`) is
leverage-independent, so it bounds `|w|` regardless of n; Firth's `+½·log det I` pull ∝
leverage ≈ `p/n`, which vanishes on the large-n well-powered nodes and is zeroed by the
`pinv` truncation on the rank-deficient rare leaves (exp 0080). At 41-anchor scale
`|w|max` sat at ~2126 (the max is driven by an *unscored* rare leaf; the well-powered nodes
that enter the ECE pool carry modest weights, which is how `|w|~2126` and ECE~0.01
coexist).

**The pessimism was mis-anchored on rare6.** The evidence that made the unified head look
hopeless — `|w|` 1.3e4, garbage calibration (exp 0079) — all came from rare6, whose
per-node cohorts are tens of docs (the small-sample regime). Scaling to 41 anchors *tamed*
the head (`|w|` → ~2100) and calibrated it. Scale moved us toward the good regime, not
away — the opposite of the usual small-data worry. The residual bad regime is the small-n
pocket (Amyloidosis n=66), the last vestige of rare6.

**Consequences.**
1. Firth is unnecessary for the unified conditional-probability model; the ridge delivers
   it. Recorded as **ADR 0043** (Firth + inner-loop Path B removed).
2. The two-stage readout LR remains the *reference* (head-independent representation-quality
   metric) and a *fallback* calibration route, not the only way to a VOI-ready posterior.
3. **Caveat — pooled vs per-node.** ECE 0.0098 is *pooled* over parent cohorts; it can
   average out an overconfident node against an underconfident one. A per-node reliability
   readout is the outstanding confirmation before blessing the unified head in a writeup —
   staged as the next cheap (~15 min) run.
4. The residual small-cohort discrimination tax, if worth closing, points at **hierarchical
   shrinkage** (child head → parent, data-estimated strength — parameter-free), not Firth
   and not a `head_l2` sweep.
