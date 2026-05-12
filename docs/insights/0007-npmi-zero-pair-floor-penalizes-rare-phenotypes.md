# 0007 — NPMI floors at −1 for zero-pair counts, penalizing rare phenotypes
**Date:** 2026-05-12
**Topic:** npmi | diagnostics
**Status:** Observed

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
