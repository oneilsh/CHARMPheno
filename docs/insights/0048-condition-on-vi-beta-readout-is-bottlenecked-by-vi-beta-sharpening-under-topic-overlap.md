# Insight 0048 — The condition-on-VI-β Σ read-out is bottlenecked by VI's β-sharpening under realistic topic overlap; the Gibbs machinery is exact (oracle-β recovers the plant), the ceiling is VI's β

**Date:** 2026-07-12
**Branch:** pg-stm
**Relates to:** insight 0045 (conditioning on β pins labels — established on a *separable* corpus), 0047 (Gibbs recovers full-rank Σ, VI attenuates), 0044 (mean-field wrong correlation sign). Queue item (4): plant-and-recover pedigree of the condition-on-VI-β read-out, on a **realistic overlapping-topics** corpus.

## Setup

Plant a Σ correlation in stick space (identified), fit mean-field VI, then read Σ
out by conditioning on VI's β and Gibbs-sampling ψ,Σ (`pg_stm_gibbs(beta_fixed=β_VI)`).
Compare recovered correlations across four arms: planted truth, VI mean-field Σ,
cond-VI-β Gibbs (the read-out), oracle-β Gibbs (reference). **Realism:**
`topic_overlap=0.6` → adjacent-topic cosine 0.576, peak word prob 0.036 (real EHR
topics peak ~0.01–0.14). Corpus: K=12 (bg4, 2×fg4), V=400, D=2200, doc_len=70,
planted intra-block correlation 0.40.

## Result — the read-out is far better than mean-field but far short of oracle

Background block (best-identified), off-diagonal correlations (planted = 0.40):

| arm | mean |corr| error (MAE) | sign-agree |
|---|---|---|
| VI mean-field | 0.453 (inflates to ~0.99) | 1.00 |
| **cond-VI-β Gibbs** | **0.366** (inflates to ~0.90) | 1.00 |
| oracle-β Gibbs | **0.061** (recovers ~0.44) | 1.00 |

Group-A foreground block (planted 0.40):

| arm | MAE | sign-agree |
|---|---|---|
| VI mean-field | 0.596 | 0.33 (mostly wrong sign) |
| **cond-VI-β Gibbs** | **0.475** | 0.50 |
| oracle-β Gibbs | **0.053** | 1.00 |

Scale (mean diag Σ; planted 4.0): VI 1.67, cond-VI-β 2.66, oracle-β 3.83.

**The Gibbs machinery is exact** — oracle-β recovers the planted correlation to
MAE ≈ 0.05–0.06 and the scale to 3.83/4.0. **The bottleneck is VI's β.**

## Root cause — VI sharpens β, and the too-separated β corrupts the Gibbs Σ

VI's β is *less* overlapping than the truth: adj_cos(β_VI)=0.231 vs true 0.576, even
though `planted_recovery` scores it 12/12 ("topics recovered"). VI separates topics
that genuinely overlap; the Gibbs sampler conditioned on that too-separated β
mis-assigns the shared tokens, which distorts the per-doc ψ and **inflates** Σ (the
background correlations blow up to 0.90 instead of 0.40). So the read-out inherits
VI's β distortion. On the *separable* corpus of insight 0045, β_VI ≈ oracle, so
conditioning looked sufficient — realistic overlap is what exposes the gap.

Prime suspect for the sharpening: VI's β Dirichlet prior `beta_eta=0.1` (< 1) is a
*sparse* prior that actively pushes β toward peaky, separated topics.

## Fix verdict — the β-prior fix FAILS; freezing VI's β is not viable

Is VI's β-sharpening fixable by the β Dirichlet prior (`beta_eta`)? A VI-only sweep on
a small corpus suggested yes — `beta_eta=1.0` restored adj_cos(β_VI) from 0.28 (sparse
default 0.1) to 0.52 ≈ true 0.57. But that is necessary, not sufficient, and it does
not survive real identification:

- **Small corpus (D=400), cond-VI-β bg MAE:** oracle 0.079; `beta_eta=0.1` 0.587;
  `beta_eta=1.0` 0.462 — better, but nowhere near oracle. And oracle recovers on this
  exact corpus, so the failure is β accuracy, not weak identification.
- **Pedigree corpus (D=2200), cond-VI-β:** `beta_eta=1.0` makes it *worse* — bg MAE
  rises 0.366 → 0.450, fg sign drops to 0.33, and the **scale explodes to 10.0**
  (planted 4.0; oracle 3.83). At D=2200 the likelihood dominates the β prior, so
  `beta_eta=1.0` only lifts adj_cos to 0.395 (not the small corpus's 0.509) — mean-field
  sharpening wins as data grows — and conditioning on that overlap-mismatched β blows up
  the Gibbs ψ-scatter.

**Verdict: freezing VI's β is not viable for the Σ read-out under realistic topic
overlap.** Oracle-β recovers perfectly (MAE 0.061), so the target is reachable, but VI's
β — under any β-prior setting — is not accurate enough, and a naive prior tweak backfires.

## What this changes — item-4-as-specified is dead; pivot to co-sampled β (option B)

The condition-on-VI-β read-out (freeze β, Gibbs-sample ψ,Σ) does NOT survive realistic
topic overlap. The forced pivot: **do not freeze β — co-sample it in the Gibbs,
warm-started from VI's β.** The warm start puts β in the correct label-aligned basin
(avoiding the insight-0045 label-switching that afflicts random-init free-β Gibbs), and
the Gibbs β-draws let β correct *toward the truth* that oracle-β proves recovers Σ.
Needs a `beta_init` warm-start knob (distinct from `beta_fixed`) and a test that the
warm-started chain stays label-aligned while refining β. If warm-started co-sampling
does not stay aligned, the fallback is a structured/collapsed variational posterior that
does not attenuate β.

## Caveats

- One seed / one config; the mechanism (VI sharpening → conditioning corrupts) is
  clear but the magnitude is config-specific.
- The foreground block is more weakly identified than background even for oracle-β on
  other seeds; here oracle-β nails it (sign 1.00, MAE 0.05), so the cond-VI-β failure
  is specifically the β distortion, not weak identification.
