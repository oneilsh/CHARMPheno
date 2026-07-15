# 0040 — Light-coder (short-document) years are mostly routine/screening, with real structured pockets

**Date:** 2026-07-08
**Topic:** stm | gating | cohorts | short-documents | doc-length-floor
**Status:** Confirmed (real-cohort cluster run exp 0031)

Exp [0031](../experiments/0031-stm-population-sparse-gated.md) turns the gated
architecture on a question about the *data*, not a disease: what is actually in the
short documents we routinely floor out? The whole windowed population is split by
in-window **coding density** — dense years (≥20 codes) form the background, **light-coder
years (5–19 codes) get their own `sparse` foreground block** — with no disease anchor. If
the sparse foreground reads as wellness/screening/routine, the `doc_min_length` floor is
justified; if it reads as structured conditions, short docs carry signal we are discarding.

## Finding 1 — the answer is "mostly routine, but not empty"

The density split landed near 50/50 (31,269 light-coder docs vs ~31,300 dense; 62,565
total, vocab 6115). The 10 sparse foreground topics read predominantly as
wellness / screening / routine / acute-minor care: metabolic screening (HTN +
hyperlipidemia + prediabetes; T2DM), refraction (myopia / presbyopia / astigmatism),
audiology (sensorineural hearing loss / cerumen / tinnitus), vitamin-D and alcohol
screening, acute URI (pharyngitis / sinusitis / bronchitis), obesity + snoring, and
acute cardiorespiratory symptoms (cough / palpitations / dyspnea / COVID-19).

But two topics carry genuine structured signal: **MSK** (joint pain / knee osteoarthritis
/ carpal tunnel — the single most coherent sparse topic at NPMI **+0.220**) and the
metabolic-comorbidity pair. So the `doc_min_length: 5` floor is well-justified as a
routine-care cutoff, yet it is not lossless — a minority of short docs encode real
MSK and metabolic structure.

## Finding 2 — low block coherence is the signal, not a defect

The sparse block's mean NPMI is **+0.100**, roughly half the background's **+0.192**. This
gap is *diagnostic, not disappointing*. Two mechanisms, both intrinsic to short documents:

1. **Fewer within-doc pairs.** NPMI is a co-occurrence statistic; a 5–19-code document
   contributes far fewer word pairs than a 20+-code one, so per-topic NPMI is
   mechanically depressed even when the topic is clean.
2. **Routine care is diffuse.** Screening/wellness spans many unrelated codes (an eye
   exam, a vitamin-D check, a sore throat) that do not co-occur, so the *content itself*
   is low-coherence by construction.

The spread across the block traces the wellness-to-signal gradient directly: the diffuse
floor is topic 43 (headache / abdominal pain / acne / dysuria, +0.022); the coherent
pocket is topic 46 (MSK, +0.220). Reading a low-but-nonzero foreground-block NPMI as
"the gate found the diffuse routine mass, plus a few real clusters" is the correct
interpretation — not "the gate failed."

## Finding 3 — the gate stays healthy with no disease anchor

Removing the disease arm entirely (contrast the cancer/EDS foregrounds of exp 0028/0030)
does not destabilize the block-wise unit-diagonal Σ: all Σ_ii=1, eigenvalues 0.203–5.07,
no runaway (the eval's `runaway = topic 49 Σ_ii=1.000` is argmax over the constant unit
diagonal, not a blowup). The gate is a general density/covariate-split tool, not one that
needs a clinically coherent foreground to remain identifiable.

## Relationship to prior insights

Complements [0035](0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md)
(a *rare-disease* foreground carves crisp sub-phenotypes): here a *non-disease*,
density-defined foreground carves mostly-diffuse routine care plus a couple of real
pockets — the two together bracket what the gate does when the foreground is, versus is
not, a coherent clinical entity. Extends the NPMI-reading caution in
[0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md) /
[0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md): a foreground block's
mean NPMI must be read against its documents' code counts, not against the background's.
