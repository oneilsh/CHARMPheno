# 0035 — A rare-disease gated foreground (EDS, ~1k docs) recovers clinically faithful sub-phenotypes against a full-population background

**Date:** 2026-07-03
**Topic:** stm | gating | cohorts | rare-disease
**Status:** Confirmed (real-cohort cluster run exp 0030)

Exp [0030](../experiments/0030-stm-population-eds-gated.md) is the first test of the
gated architecture in a genuinely *rare*-disease regime: a whole-population background
with an Ehlers-Danlos syndrome (EDS, OMOP ancestor 79145) foreground. It answers two open
questions — does the gate carve real structure when the foreground is a tiny fraction of
the corpus, and how much population do you need to fit under it.

## Finding 1 — ~1k foreground docs is enough for the gate to carve real sub-phenotypes

The full-population fit (`person_mod: 1`) resolved to 332,502 persons / 191,872 fit
documents, of which the **EDS arm was just 956 documents (≈0.5%)**. Despite that
imbalance, the 20 EDS foreground topics did NOT collapse into the background — they
recovered the recognizable EDS comorbidity clusters a clinician would name: POTS /
dysautonomia (tachycardia, orthostatic hypotension, autonomic failure), MCAS (urticaria,
anaphylaxis, systemic mast cell disease), joint instability (shoulder dislocation,
hypermobility syndrome, MVP), vascular EDS (aortic aneurysm/dissection, collagen disease),
and GI dysmotility (gastroparesis + fibromyalgia + POTS overlap). EDS-block NPMI mean
+0.156 (max +0.286, reference=956 docs); background mean +0.180. This is the mechanism
insight [0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md) predicted —
the hard block partition keeps a rare group's tokens from being diluted by the majority —
demonstrated at a far more extreme prevalence ratio (~0.5%) than the balanced
cancer/dementia validations.

## Finding 2 — for a rare foreground, sampling the background is the load-bearing knob

The EDS arm scales with the population you fit. At 956 docs on the full sample, the topics
are crisp; at `person_mod: 4` (25%) the same arm would have been ~240 docs, almost
certainly too thin to carve 20 stable topics. The practical rule for
`apply_population_disease_cohort` on a rare disease: **take the full population (or as much
as the cluster allows) so the foreground arm clears a few hundred documents** — the
background is cheap to subsample, the rare arm is not. Σ stayed bounded (all Σ_ii = 1,
block-wise unit-diagonal, ADR [0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md);
no runaway), so the larger, more heterogeneous corpus did not reintroduce the instability
of insight [0033](0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md).

## Takeaway

The gated STM generalizes cleanly from "two known diseases" to "one rare disease on a
population background." The cohort knob that matters for a rare foreground is background
sample size (set it high), not the foreground topic count. This makes population+disease a
reusable rare-disease phenotyping instrument, not a one-off.
