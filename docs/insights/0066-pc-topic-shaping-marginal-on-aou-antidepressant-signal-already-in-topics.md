# 0066 — Prediction-constrained topic-shaping is marginal on the AoU antidepressant task: the signal is already in the unsupervised topics

**Date:** 2026-08-11
**Topic:** pc
**Status:** Confirmed

The point of a prediction-constrained (PC) topic model over a two-stage "fit topics,
then a classifier" pipeline is that the label *reshapes the topics* to be more
predictive. On the AoU `mdd_stable_treatment` task (predict which of 10 antidepressants
a patient stably takes, from their all-history OMOP BOW), that reshaping buys almost
nothing.

Reading `pc_topics_lr` — a fresh, fully-converged LogisticRegression on the *final*
topic proportions, which isolates topic quality from the co-fit head's own convergence:

| representation | macro AUC |
|---|---|
| PC-supervised topics + LR (`pc_topics_lr`) | **0.619** |
| unsupervised two-stage topics + LR | 0.614 |
| LR on raw 5000-dim codes | 0.613 |

Supervision edges unsupervised by ~0.005 — real but tiny, and this holds *after* the
co-fit head was fixed to actually converge (insight 0065), so it is not an artifact of a
broken head signal driving the topic correction. The three numbers agreeing (~0.61)
means a 50-topic representation already captures essentially all the raw-code predictive
signal, and the labels have little *additional* structure to impart. This is a **data**
finding, and it is consistent with Hughes et al.'s own choice to report a two-stage
model (topics → LR/extra-trees) in the JAMA Network Open 2020 clinical paper rather than
the co-fit head that is the AISTATS 2018 methods contribution.

**Why it does not generalize to "PC never helps".** The regime where PC *should* help is
a **hidden low-mass signal**: a predictive topic that is rare enough that an
unsupervised fit spends its K topics on the dominant structure and misses it, so the
label is needed to surface it. The vendored `toy_bars_3x3` synthetic (K_fit < K_dom,
one low-mass predictive topic) shows exactly that — PC 0.56 → 0.91 where two-stage
stays near chance. The AoU antidepressant task is the opposite regime: the predictive
structure is high-mass and already recovered unsupervised. The forward-looking test is
the Mondo-mapping-based rare-disease cohorts (still AoU data), where a rare phenotype is
the label — the hidden-low-mass regime where topic-shaping should produce a real
`pc_topics_lr` gain.

**Implications.** Use `pc_topics_lr` (not the co-fit head's AUC) as the measure of
whether supervision improved the *representation* — it is convergence-robust and directly
comparable to the two-stage baseline. On a task where LR-on-codes ≈ two-stage ≈
PC-topics, expect no PC benefit and prefer the simpler two-stage; the co-fit head's value
is entirely conditional on the label carrying representation-level signal the
unsupervised fit misses.

**Setting context.** AoU OMOP `mdd_stable_treatment`, N=45,991 (34,515 train / 11,476
test), 10 Hughes drugs fully observed, all-history fused-vocab BOW (5000 terms, avg 322
tokens/doc), K=50, weight_y=1000, distributed VI-native PC (`OnlinePCLDA`), Newton head.
Experiments 0072–0075. Two-stage 0.614 is from 0072; `pc_topics_lr` 0.619 is stable
across the two Newton runs (0074/0075).
