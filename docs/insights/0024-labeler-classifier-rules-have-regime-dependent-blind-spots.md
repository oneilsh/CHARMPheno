# 0024 — LLM-classifier rubrics have regime-dependent blind spots: rules that look robust on the development corpus can mis-fire when the feature distribution shifts
**Date:** 2026-05-22
**Topic:** ops | diagnostics | labeling

The phenotype labeler in [`scripts/label_phenotypes.py`](../../scripts/label_phenotypes.py)
uses an LLM (gpt-5) with a multi-step rubric to classify each topic into
one of `{phenotype, background, anchor, mixed, dead}`. The rubric exposes
several diagnostic signals to the model — per-topic α, KL(β‖corpus), NPMI,
α distribution histogram — and prescribes a decision order in plain text.

One step of that decision order said:

> **Check KL against the dead threshold.** KL ≤ threshold → `dead`, stop.
> The topic is the corpus baseline regardless of how the top-N reads.

This rule worked well in the **full-corpus regime** where the labeler had
been developed: by [insight 0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md),
full-corpus LDA decomposes the baseline coding load into **three** medium-α
catch-alls (acute-presentation 29% / generic-chronic 20% / cardiometabolic
13%), each with α in the 1.5–3× median range. None of them had extreme α
relative to the rest of the fit, so "low KL → dead" rarely mis-classified
the high-α catch-alls — they had enough KL above threshold to pass the dead
check anyway.

On the **cohort regime** ([insight 0021](0021-cohort-corpora-two-anchor-mass-concentration.md)),
the picture changes. Cohort filtering compresses the phenotype variance,
which concentrates the universal-symptom baseline into **one or two** topics
with α 10–20× the rest of the fit. On the dementia first-year cohort
(K=40), the anchor topic t18 had α=0.227 (14× the median 0.0159) and
absorbed 66% of corpus mass; t13 had α=0.201 (12× median) and absorbed 22%.
Both also had top-N word distributions matching the corpus marginal closely
(low KL, by design — that's what universal-symptom catch-alls do). The
labeler's rule fired on KL alone, classified both as `dead`, and the
dashboard ended up labeling 88% of corpus mass as "unused / low-signal
topic."

The user caught this immediately: "calling something dead with such a weight
would probably look shady to someone not closely familiar."

## What the rule was missing

The rubric had **already exposed α magnitude** to the model — the per-topic
α value was in the per-topic stats, and the fit-level α distribution
(min/median/max) was in the decision-context preamble. The text rule just
didn't *use* that signal at step 1. It said "low KL → dead regardless of
α" because the design intent was robustness against α-floor topics (which
have meaningless α and shouldn't classify on α alone). But "α near floor"
and "α at the fit's max" are dramatically different signals, and the rule
collapsed them.

The fix added an α-magnitude disambiguator at step 1:

- KL ≤ threshold + α well above median (3×+ median) → `background`
  (the high-mass baseline absorber — corpus-marginal vocabulary BUT
  absorbs real mass).
- KL ≤ threshold + α near floor → `dead` (the low-mass noise-floor
  topic whose top-N is just η-smoothing).

After the fix, the dementia labeler correctly classified t18 as
`[background] Acute symptom catch-all` and t13 as `[background] Chronic
comorbidity catch-all`. The other ~26 phenotype topics were already
labeled correctly; only the two extreme-α-low-KL topics changed quality.

## Generalizable lesson

LLM-classifier rubrics carry **implicit assumptions about feature
distributions** even when they appear distribution-free in prose. The
"regardless of α" phrasing *looked* like a robustness clause — "don't be
misled by a single noisy signal" — but functionally was a brittleness:
the rule refused to use a strong disambiguating signal when it was
available, because the rubric author had only seen the development-time
regime where that signal didn't cleanly disambiguate.

Three properties of the failure mode worth naming:

1. **The rule was right on the development data.** Full-corpus runs
   never had a topic with α 14× the median, so "low KL → dead" was
   correct in every instance the rubric author tested against.
2. **The mis-classification was *consistent* with the rubric text.** The
   labeler wasn't hallucinating; it was reading the rule literally. The
   rule itself was wrong for the new regime.
3. **The exposed signal that would have caught it was sitting right
   there in the prompt.** Per-topic α was in the stats blob; fit-level
   α was in the preamble. The rule just declined to consult them.

The structural fix is the same as for any classifier with a hidden
regime-dependent assumption: write rules that consult the *combination*
of signals when feasible, not single signals with "regardless of"
override clauses. "Regardless of X" should be a red flag in any rubric
— it's the verbal form of `if cond: return; # ignore everything else`,
which is exactly where regime-dependent bugs live.

## Relationship to [0023](0023-producer-consumer-unit-mismatches-invisible-until-small-scale.md)

[0023](0023-producer-consumer-unit-mismatches-invisible-until-small-scale.md)
documented a different failure mode of the same shape: a dimensional unit
mismatch (token-frequency × corpus-size-docs treated as doc count) was
invisible at large corpus scale because the wrong quantity correlated with
the right one closely enough to "qualitatively work." Both bugs:

- Originated in development-time code paths that were tested only on
  large-corpus inputs
- Behaved correctly in that regime by accident, not by correctness
- Failed loudly on the first small-cohort run where the convenient
  correlation broke

Both bugs were caught by **the same user observation**: "this output looks
wrong for the dementia cohort." The dementia run is small enough to expose
assumptions that the cancer and gen-pop runs had been silently masking.

The generalizable practice for future labeler / classifier / threshold
rubrics: **test the rubric on the smallest expected cohort before
shipping**, not just on the largest. Small cohorts compress feature
distributions in ways that expose assumption seams large corpora hide.

## Implications

1. **Audit the rest of the rubric for "regardless of" clauses.** Each one
   is a candidate for the same failure mode. As of the fix, the only
   remaining one is in the dead-case-a description, but periodic review
   is cheap.
2. **The cohort-corpora regime should be the rubric's
   development-and-test regime going forward,** not the full corpus.
   Small cohorts are where the rubric is most stressed; designing for
   the stressed regime first costs nothing on the easy regime.
3. **Consider an automated regression check on labeler output.** A
   sanity test that "no topic carrying >5% of corpus mass should be
   classified as `dead`" would have caught the bug pre-deployment. Trivial
   to add as a post-labeling assertion.

**Setting context:** Phenotype labeler classification pass on the
`first_dementia_year` cohort bundle (~9k docs, K=40, gpt-5 with reasoning,
KL-primary rubric with data-driven dead threshold). α distribution
min=0.0141, median=0.0159, max=0.2255 (14× median). Two topics with low
KL but extreme α were classified `dead`; user-flagged; three-line rubric
edit added α-magnitude disambiguator at decision step 1. Same fix improved
the cancer cohort labels (6 dead → background flips) and was a no-op on
gen-pop (which has α max only 3× median, never triggered the rule).
