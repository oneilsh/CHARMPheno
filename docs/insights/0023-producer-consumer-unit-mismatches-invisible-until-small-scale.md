# 0023 — Dimensional unit mismatches in producer/consumer pairs are invisible at large input scale; small cohorts expose them
**Date:** 2026-05-18
**Topic:** ops | diagnostics

The dashboard export pipeline had a silent bug in the small-cell
guard for ~2 months: it filtered the displayed vocabulary on a
quantity whose units didn't match what the docstring claimed. The
bug was invisible on the gen-pop (~100k docs) and cancer (~275k
docs) bundles because the corpus was large enough that the
mis-computed threshold still cleared the right *order of magnitude*
of vocabulary. The dementia cohort (~9k docs) was the first run
where it bit: V_displayed = 60 codes instead of the ~3,000 it should
have been, and the labeler — looking at the top-60 universal-symptom
projection of every topic — couldn't see any of the actual phenotype
content. Lewy body, HIV, scleroderma, breast cancer, all silently
suppressed.

## The bug

[`charmpheno/charmpheno/export/corpus_stats.py:44`](../../charmpheno/charmpheno/export/corpus_stats.py)
produced `code_marginals[i] = occurrences[i] / total_tokens` —
**token frequency**.

[`charmpheno/charmpheno/export/dashboard.py:45`](../../charmpheno/charmpheno/export/dashboard.py)
filtered with `marginals * corpus_size_docs >= min_doc_count`, with
a docstring claiming this is "empirical doc count." The expression
expands to `occurrences / total_tokens * n_docs = occurrences /
mean_codes_per_doc`. Not doc count; not occurrence count; not any
quantity with a clean semantic. Dimensionally wrong.

## Why it didn't surface earlier

The expression `occurrences / mean_codes_per_doc` is roughly an
**upper bound** on doc count for codes that mostly appear once per
doc (most condition codes), so the filter is more conservative than
the docstring promised but still cleared enough common codes on
large corpora to *look* like it worked. At full-corpus scale, the
top thousands of common codes were so frequent that even the
over-tight threshold kept the bundle usable:

| cohort     | docs   | mean codes/doc | bug-affected threshold (effective occurrences) | V_displayed actual |
|------------|--------|----------------|-----------------------------------------------|--------------------|
| gen-pop    | 101k   | 57             | ≥1,140 occurrences                            | 922 codes          |
| cancer     | 275k   | 253            | ≥5,060 occurrences                            | 1,963 codes        |
| dementia   | 9k     | 130            | ≥2,600 occurrences                            | **60 codes**       |

The threshold scales with `mean_codes_per_doc`, and `mean_codes_per_doc`
isn't well-controlled across cohorts. On dementia, ~6,900 trained
vocab codes were narrowed to 60 — the most-common 60 codes by token
frequency, which by Zipfian distribution were almost entirely the
universal-symptom + chronic-comorbidity baseline. Every phenotype-
specific code (HIV, REM behavior disorder, Diffuse Lewy body,
Systemic sclerosis, Bladder carcinoma, Epilepsy beyond generic
"Seizure", ...) failed the broken filter.

## Why no walkthrough caught it

The producer (token-frequency definition) and consumer (claimed-to-be
doc-count filter) live in adjacent files, are small (~50 LOC each),
and have docstrings claiming a contract — but the contract was
broken by the producer, not the consumer. A walkthrough that read
either file in isolation would have found the local logic
self-consistent. Catching this required reading both files together
and asking *"is the marginal here the same kind of quantity the
filter assumes?"* That kind of cross-file unit-check is what
walkthroughs are for, but it's also exactly what's easy to skip
when the code "looks reasonable" in each location.

## Generalizable lesson

**A bug whose effect scales with corpus dimensions can hide forever
behind corpus dimensions that are roughly fixed.** Three contributing
factors made this particular bug invisible:

1. **The wrong quantity correlates with the right quantity.**
   `occurrences / mean_codes_per_doc` rises with corpus size in
   roughly the same direction as `doc_count`, so the filter behaved
   "qualitatively right" on the corpora seen during development.
2. **The output (V_displayed) wasn't tracked as a regression metric.**
   No one notices a vocab going from 1,963 → 1,950 across runs. Even
   60 didn't trigger an alarm until labels came out bad.
3. **The downstream effect was upstream-obscure.** The labeler used
   the bundle as ground truth; bad labels traced back to bad bundle
   contents, but only by reading the bundle's `vocab.json` and the
   model's actual top words side-by-side.

## Suggested guards

- **Treat per-code unit semantics as part of the contract.** Adding
  a unit suffix to fields (e.g. `code_marginals_token_freq`,
  `code_doc_counts`) makes cross-module dimension errors fail at
  read time. Type aliases (`TokenFraction = float`, `DocCount = int`)
  would catch it at static-check time.
- **Log V_displayed prominently in every export.** A 30× change
  between cohorts (1,963 → 60) is the kind of regression that should
  be impossible to miss. Add to the build-dashboard summary print
  and treat it as a metric to watch across runs.
- **Producer-consumer pairs deserve a paired test.** A unit test
  feeding `compute_corpus_stats` output directly into
  `select_top_n_with_min_cell` with realistic corpus dimensions
  would have caught the unit mismatch the first time it was added.

## The fix

[`compute_corpus_stats`](../../charmpheno/charmpheno/export/corpus_stats.py)
now tracks per-code distinct-document counts as a separate field
alongside token-frequency marginals.
[`select_top_n_with_min_cell`](../../charmpheno/charmpheno/export/dashboard.py)
filters on doc counts and ranks on marginals — preserving both
semantics explicitly. The two quantities are now distinct in the
public API so substituting one for the other is a type-checked
mistake rather than a silent one. A regression test
(`test_min_doc_count_filter_independent_of_token_freq`) pins the
fix against the specific failure mode that hid the bug.

**Setting context:** Dashboard export pipeline, caught during the
dementia cohort bundle build (~9k docs, K=40). Bug originated in
the initial dashboard-export ADR/spec
(`docs/superpowers/specs/2026-05-13-dashboard-design.md`); affected
gen-pop and cancer bundles silently for ~2 months but masked by
their larger corpus sizes.
