# Insight 0053 — Mean-zero stick-breaking under-attests later background sticks; block-mass, not single-word argmax, is the sound recovery measure

**Date:** 2026-07-13
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | stick-breaking | diagnostics | test-design
**Status:** Observed
**Relates to:** the DAG background-only member support step (Task 3, flat background-only E-step recovery); the recovery-measure choices in the DAG/ontology PG-STM suite.

**Context:** Validating the flat (non-gated) background-only mean-field E-step (`_bg_estep_doc`)
by planting a pure background-only corpus (documents with no foreground group, tokens drawn from
a flat background stick-breaking `theta_bg = stick_to_simplex(psi_bg)`, `psi_bg ~ N(0,
sigma_true[bg,bg])`) and checking the fit recovers the planted background topics. The first
construction measured recovery by single-word argmax (does each planted background topic's
top word equal some fitted background topic's top word, up to label permutation) with a bar of
4-of-5. It failed: overlap 2/5 at the plant spec, 3/5 after both sanctioned escalations
(more background-only docs, more iterations).

**Finding — two compounding causes, neither an estimator bug:**

1. **Attestation skew from stick-breaking geometry.** A stick-breaking map with mean-0 sticks is
   not exchangeable across topics: `E[theta_1] > E[theta_2] > ... > E[theta_B]` (the first stick
   takes ~sigmoid(0)=0.5 of the mass in expectation, the next ~half the remainder, and so on).
   So a background-only corpus generated from mean-0 sticks systematically under-generates the
   LATER background topics. Measured planted-topic signature-block token fractions were
   ~[0.67, 0.74, 0.45, 0.27, 0.18] — the 5th background topic receives little corpus mass and is
   therefore weakly identified FROM THE DATA, independent of the estimator.

2. **Single-word argmax is the wrong recovery measure here.** Under overlapping topic-word
   signatures (`real_beta_from` uses `topic_overlap=0.6`), each topic's high-probability window
   overlaps its neighbors', so several words are near-tied for the max and `np.argmax` is
   tie-break-brittle. Lowering `topic_overlap` did NOT help — it made argmax recovery WORSE
   (probe: 1/5 at overlap 0.0/0.1) because sharpening the signatures shifts which single word
   wins without changing the attestation skew. Argmax conflates "is this topic recovered" with
   "which exact word won a near-tie," and with the attestation skew of cause (1).

The flat E-step itself recovers every ADEQUATELY-ATTESTED background topic. Under the
codebase-standard block-mass measure (`planted_recovery`: does some fitted background topic put
>= 0.5 mass on the planted topic's high-probability word block), recovery is a stable 4/5 at both
the plant spec and the escalated settings, with per-planted best-fitted block-mass
[0.97, 0.90, 0.86, 0.63, 0.42] — topics 0-3 clearly recovered, and the single miss is exactly the
least-attested 5th topic (0.42 mass, 0.18 corpus fraction).

**Consequences:**
- The Task-3 flat-path recovery guard uses `planted_recovery` (block-mass), the same
  attestation-robust measure every other DAG-suite recovery test uses. Single-word argmax is not
  a sound recovery criterion for overlapping-signature topics or stick-breaking-attested corpora.
- The 5th-topic miss is an information/attestation property of the plant (a manifestation of the
  information wall of [[project_dag_ontology_pg_stm]] / insight 0052), NOT an estimator limitation
  of the flat E-step. A plant that wanted all B background topics equally attested would need to
  balance the stick-breaking mean (or rotate a dominant topic across documents) rather than draw
  every doc from the same mean-0 sticks.
- This is consistent with the recurring theme that only well-attested / identified quantities are
  recoverable, and that the recovery MEASURE must be chosen to isolate the estimator from the
  identifiability of the planted signal (cf. insights 0044 / 0048 / 0050 / 0052).

**Does not claim:** anything about real-data recovery or transfer (synthetic proves
math-correctness only), nor that the flat background-only E-step recovers UNDER-attested topics
(it demonstrably cannot when the plant barely generates them — that is the information wall, and
climbing it needs data/pooling, not a better estimator).
