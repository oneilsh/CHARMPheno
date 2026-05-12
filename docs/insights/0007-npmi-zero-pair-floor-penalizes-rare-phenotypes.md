# 0007 — NPMI floors at −1 for zero-pair counts, penalizing rare phenotypes
**Date:** 2026-05-12
**Topic:** npmi | diagnostics
**Status:** Observed
**Resolved by:** [ADR 0017 revision 2026-05-12](../decisions/0017-topic-coherence-evaluation.md#revisions)

NPMI(w_i, w_j) = log[P(i,j) / P(i)P(j)] / −log P(i,j). When the pair
(w_i, w_j) never co-occurs in the reference corpus, P(i,j) → 0 and
NPMI → −1 (the metric's floor).

A rare-phenotype topic — say, sickle cell with concepts that genuinely
co-occur in patients with the disease but appear together in only a
handful of corpus docs — will have several top-word pairs whose joint
counts round to zero in the held-out reference. Those pairs contribute
−1 each to the topic's mean NPMI, dragging it down for sparsity reasons
rather than for incoherence.

Conversely, common-comorbidity catch-all topics (HTN + HLD + T2DM)
score high NPMI not because they're more *coherent* but because their
top-word pairs always have non-zero joint counts in any reasonably-sized
held-out set.

**Implications.** Corpus-mean NPMI rewards prevalence over discovery.
For phenotype evaluation, look at the **per-topic NPMI distribution**,
not just the mean, and **inspect the topics that score low NPMI** —
some of them will be exactly the rare-phenotype topics we care most
about. Mitigation options to explore: Laplace smoothing of joint counts,
or restricting NPMI evaluation to top-word pairs that meet a minimum
joint-count threshold (and reporting how many topics fall below that
threshold).

**Setting context.** Observed when comparing K=25 LDA NPMI distributions
across patient-lifetime vs patient-year runs. The "rare but clean"
topics (sickle cell, kidney transplant, SLE, CLL) tended to score lower
NPMI than the chronic-comorbidity catch-all despite being more
interpretable as phenotypes. Eval driver settings as in
[ADR 0017](../decisions/0017-topic-coherence-evaluation.md).

## Mitigations adopted

ADR 0017 revisions on 2026-05-12 adopted both mitigation paths that
were proposed here, in combination:

1. **Default NPMI reference switched from holdout-only to full BOW
   (train ∪ holdout).** 5× more documents, dramatically fewer
   zero-joint-count pairs for rare phenotypes. Methodologically
   defensible: NPMI on a fixed (post-fit) topic-word distribution
   has no overfitting concern. CLI flag: `--npmi-reference {full,
   holdout}`, default `full`.
2. **Min-pair-count threshold replaces the −1 floor in the default
   path** (`--npmi-min-pair-count`, default 3). Pairs with joint
   count below threshold are *skipped*, not floored. Topics report
   coverage (fraction of pairs that cleared the threshold) alongside
   NPMI. A rare-phenotype topic with several below-threshold pairs
   now reads "NPMI=+0.4, cov=55%" instead of being dragged to a
   sparsity-penalty negative score. Topics with zero coverage report
   NPMI=NaN and are excluded from summary stats.

The Laplace-smoothing alternative (option C in the original brainstorm)
was rejected as muddying the metric's discrimination for marginal
benefit.

This insight is now historical context: the failure mode it describes
no longer applies under the new defaults. The mechanism remains
worth knowing for two reasons — (a) someone reproducing pre-2026-05-12
NPMI numbers via `--npmi-reference holdout --npmi-min-pair-count 1`
will see the original behavior, and (b) the per-topic coverage
metric exists *because* this failure mode does, so understanding it
explains why coverage matters.
