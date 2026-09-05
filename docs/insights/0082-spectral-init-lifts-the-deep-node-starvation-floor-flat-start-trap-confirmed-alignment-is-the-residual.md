# 0082 — Spectral init lifts the deep-node starvation floor: the flat-start trap was the lever (not budget); anchor ALIGNMENT is the residual, and it's a token-mass bias

**Date:** 2026-09-05
**Topic:** lda, init, spectral, gated-pc, case-finding
**Status:** Confirmed on exp 0114 (CV branch, MONDO:0004995, tpn=5, fit-only, spectral
scalable seed) vs exp 0113 (same, random init). The A/B is clean — only the seed differs.
**UPDATE (exp 0115):** the alignment residual's burst-bias vs objective-misalignment fork
is RESOLVED toward OBJECTIVE — `count_transform: binary` (per-doc presence) did NOT
dislodge the pregnancy anchoring on the cardiomyopathy family (floor held, redundancy
even cleaner, but intrinsic/dilated CM stayed pregnancy/genetic-anchored). So the
misalignment is genuine — at presence level, the most-separable direction for these
heterogeneous nodes IS the demographic/etiologic stratum, not a counting artifact. The
lever is a label-aware objective (supervision / nuisance deflation), gated on whether it
actually costs DETECTION (a `gated-pc-readout` question), not counts.

**Relates to:** 0079 (the depth starvation this resolves — its "MECHANISM UNRESOLVED /
init untested at depth" is now settled: init was it), 0080/0081 (the shallow/background
structure), 0077 (measurement burstiness → binary presence, the fix this extends), 0066
(PC shaping marginal on AoU — but the alignment residual is arguably PC's payoff regime).

## Observation

Flipping ONLY `init: random → spectral` on the CV-branch `tpn=5` fit collapses deep-node
starvation:

| | 0113 random | 0114 spectral |
|---|--:|--:|
| node topics starved (eff-support frac > 0.5) | **72%** | **1%** |
| depth-4 median evidence / frac | 62.5 / 0.99 | **1000 / 0.01** |
| depth-5 / -7 median evidence | 61.8 / ~61 (prior floor) | 719 / 236 |
| min evidence over all topics | ~4 | 10.5 |

Every depth is now sharp with real posterior mass; the ~61 prior-floor plateau 0113 sat on
past depth 4 is gone. Sibling redundancy stays clean (0/81 parents collapsed, worst
fed-cosine 0.36).

## Interpretation — two distinct findings

**1. The flat-start / deflation trap was the binding constraint on deep-node topic
learning.** 0113 held budget at `tpn=5` and still starved 72%; 0114 changes only the seed
and starves 1%. So budget was not the lever (0113 falsified it), and neither strip-scope
nor rarity is needed to explain the starvation — a uniform leaf simply cannot bootstrap
its distinctive tokens against sharp ancestors from a flat start (0079's leading
hypothesis), and a sharp anchor-word seed hands it the concentration to do so. This
settles 0079. NOTE it does not require `tpn>1` in principle; the seed, not the budget, is
what forms the block. (The gate-already-breaks-symmetry null on synthetic plants, insight
0067, was a DIFFERENT symmetry — unrelated blocks, not the node-vs-ancestor deflation that
bites with depth.)

**2. Sharp ≠ the core phenotype — and the misalignment is a TOKEN-MASS bias, not a
spectral-criterion flaw.** Most fed deep topics are clinically coherent (AF, systolic/
diastolic HF, valve). But a minority are sharp on the WRONG signal: intrinsic/dilated
cardiomyopathy topics dominated by PREGNANCY codes (trimesters, gestation weeks), a
cardiomyopathy topic by generic primary-care symptoms. The anchor search maximizes
co-occurrence residual norm and the topic evidence maximizes token mass; BOTH are
dominated by whatever REPEATS most per document. Conditions and drugs are RAW per-visit
occurrence counts (`multi_domain.py:456`: only the measurement domain is binarized,
insight 0077), so a gestation-week code recorded every prenatal visit gives one patient
~20+ pregnancy tokens vs one diagnosis token. Pregnancy won the intrinsic-CM anchor by
VOLUME, not meaning — the same bias behind 0113's "fed but generic lab panel" topics. So
spectral's criterion is not misaligned with node meaning per se; it is aligned with token
mass, and raw-count BOWs make token mass a proxy for utilization, not phenotype. This is
also, crucially, an OBJECTIVE property, not a seed property: the unsupervised deflated
CAVI has no term preferring the cardiac footprint over the demographic one, so a different
seed would just land in a different arbitrary basin — the seed determines which basin, the
objective determines that "most-separable" ≠ "phenotype."

## Implications

- **Spectral init is the deep-node starvation fix** and should be the default for any deep
  gated fit; starvation was never a fundamental limit.
- **The alignment residual decomposes** into (a) burst / utilization-volume bias — TESTED
  and FALSIFIED for the cardiomyopathy family by exp 0115 (`count_transform: binary` left
  the pregnancy anchoring in place; the floor held and redundancy even improved, but the
  CMs stayed pregnancy/genetic-anchored — so at presence level the demographic stratum is
  genuinely the most-separable direction, not a counting artifact); (b) genuine objective misalignment (separability ≠ phenotype) — only a
  label-aware objective (supervision / PC) aligns the criterion with node meaning, and this
  hidden-signal-under-a-dominant-one regime is exactly where insight 0066 says PC pays,
  unlike the AoU antidepressant task; (c) genuine label heterogeneity (peripartum CM IS
  pregnancy-associated — some "misaligned" topics are findings about the label, not bugs).
- **For case-finding specifically**, a demographic-anchored topic is a spurious predictor
  (flag pregnant patients as CM); whether that actually bites detection vs is absorbed by
  the localized head is an open `gated-pc-readout` question on 0114.
- **Anchor quality** (which basin a node's block falls into) is now the frontier, replacing
  starvation. `count_transform` (0115) tests the cheap half; supervision is the expensive
  half, gated on 0115's residual.

## Method note

Read off the saved fit-only λ with `inspect_topics.py --digest` (off-cluster), node-for-node
vs 0113. `spectral_d: 768` (below the dense K-floor, which the scalable per-node recovery
does not need) and the memory-safe batched seed (auto-sized to `spark.driver.maxResultSize`)
made the whole-branch spectral seed tractable on a lean cluster — see 0114's cost note.
