# Insight 0045 — The free-Gibbs Σ read-out is confounded by topic label-switching, not weak identification; conditioning on β pins labels and the scarce-group scale is wide-but-stationary (no runaway resurfaces)

**Date:** 2026-07-12
**Branch:** pg-stm
**Relates to:** insight 0033 (softmax+Laplace Σ runaway), insight 0044 (mean-field VI reads the wrong Σ correlation sign), ADR 0034 (unit-diagonal block-Σ pin), the at-scale null result (exp 0050/0051: PG/VI reformulation stable estimator-independently)

## Context — the probe

This is queue item (1) from the Fable exchange after the at-scale null result: the
**unpinned free-Gibbs-Σ probe**. Fable's framing: the softmax+Laplace runaway was a
point estimate feeding its own prior (a fixed-point iteration with no stationary
distribution). A proper Gibbs sampler has no such loop — Σ is drawn from its
conditional and a proper prior gives a proper joint, so the chain *cannot diverge to
infinity*. What it *can* do is mix into a very wide, heavy-tailed posterior on
weakly-identified (scarce-group) variances — the honest Bayesian encoding of "this
variance is unidentified by the data." So the probe is judged on **posterior
geometry, not trajectory divergence**, with two distinguishable outcomes:

- **(a) wide-but-stationary** — scarce σ²_k has a huge credible interval but the
  chain is stationary → the read-out is *working*: ship the well-identified
  correlations, report the scarce-block scales as unresolved. No export hole.
- **(b) non-stationary wandering** — under a too-weak / near-flat prior the posterior
  is effectively improper along the unidentified direction and the chain drifts
  without stabilizing → the export needs a genuinely informative scale prior.

Diagnostic: rank-normalized split-R̂ (Vehtari et al. 2021, `_mcmc_diag.improved_rhat`)
on log σ²_k across 4 independent chains, plus the pooled 5–95% credible-interval
width. Corpus: a **stick-native gated corpus** (`gated_ln_corpus_stick`, Σ identified
in stick space) with a deliberately **scarce group** (majority group A gets ~92% of
docs, scarce group B ~8%), so B's foreground-stick variances are the weakly-identified
directions the probe targets. `eta_scale=4.0` → true diagonal variance = 4.0.

## Finding 1 — free full-Gibbs Σ is confounded by label-switching, NOT weak identification

Running the free full-Gibbs sampler (`pg_stm_gibbs`, which draws β/Γ/Σ and all
latents) gave a pattern **inverted** from the identification story: the
*well-populated* sticks (background + majority-fg, ~823 docs) read the WORST R̂
(1.3–3.0) and inflated their variance medians to 8–14 against a true value of 4, while
several scarce sticks read R̂ ≈ 1.0. That inversion is the tell.

Direct check (matching each chain's recovered β rows to the planted per-topic
signature blocks): **the recovered topics permute across chains.** The background
block mapped to planted topics `[1,5,0]`, `[1,2,0]`, `[1,0,2]` in three chains — a
background stick position even absorbed a foreground topic. Because the nested
stick-breaking decomposition is order-dependent, "stick *i*" is a *different topic* in
each chain, so the per-stick Σ variances and their cross-chain R̂ are comparing
incommensurable quantities. This is the same mechanism behind the earlier
distributed-sampler wrong-sign drift.

**Consequence:** a free full-Gibbs Σ read-out cannot be interpreted per-stick. The
hazard is label non-identifiability, not weak statistical identification.

## Finding 2 — conditioning on β pins topic labels and removes the confound

Re-running with β held fixed at the (oracle) planted β (`pg_stm_gibbs(...,
beta_fixed=β)`; fixing β pins topic identity through the token-assignment step
`z ~ θ_k·β_{k,w}`) collapses the confound completely:

| stick category (docs) | free-β R̂ | free-β median (true 4.0) | fixed-β R̂ | fixed-β median |
|---|---|---|---|---|
| background / majority-fg (~823) | 1.3–3.0 | 6–14 (inflated) | **≈1.00** | ~4–5 |
| scarce-fg (~77) | 1.0–1.5 (uninterpretable) | 3–5 | **≈1.00** | 3–5 |

Every stick converges (R̂ ≈ 1.00–1.04) and the well-identified medians return to the
true scale. **This is the empirical case for the condition-on-VI-β read-out** (the
read-out fork Fable argued on alignment grounds): a fresh relearn-on-subsample fits
its own permuted/label-switched topics, whose R would describe correlations among
different topics than the shipped β — a referential mismatch. Fixing β is the fix.

## Finding 3 — once labels are pinned, the scarce-group scale is WIDE-BUT-STATIONARY (outcome a)

Under the fixed-β read-out, the scarce group (~77 docs) shows Fable's outcome (a):
R̂ ≈ 1.00 even under the near-flat **weak** IW prior, but visibly wider CIs (width
2.1–3.3 vs ~1.5 for the well-identified sticks); one scarce stick is explicitly
WIDE-STATIONARY. The **informative** IW prior (centered on the true scale) tightens the
scarce block toward width ~1.8 and everything to ≈4, but it is a *refinement, not a
rescue* — the weak prior is already stationary. So the read-out works: ship the
well-identified correlations, report scarce-block scales as less-resolved; no export
hole.

**Extreme-scarcity confirmation (16 scarce docs, D=700).** Pushing the scarce group to
~16 docs does NOT break stationarity: the weak-prior scarce sticks stay R̂ ≈ 1.00 and
just widen further (width up to 4.10, CI [2.1, 8.6] on the true value 4.0), while the
well-identified sticks stay tight (width ~1.2, R̂ ≈ 1.01). No wandering (outcome b)
appears anywhere in the tested scarcity range (77 → 16 docs) — the free Gibbs Σ is
robustly wide-but-stationary. The informative prior's tightening of the scarce block
*grows* with scarcity (width 4.10 → 2.24 at 16 docs) but is never required for
stationarity. Secondary observation: at the shorter corpus the weak prior mildly
inflates even the *well-identified* variances (medians 5.7–7.3 vs true 4.0 — the
small-df IW upward bias); the informative prior corrects those to ~4–5 too, so it
improves scale accuracy generally, not only for scarce blocks.

## Finding 4 — the runaway does NOT resurface in exact free Gibbs

Fable's sharpest hypothesis: if mean-field under-dispersion is the *only* stabilizer
(an "implicit trust region"), removing it — exact Gibbs Σ, which has no such damping —
should let the runaway resurface. **It does not.** Even the free-β sampler under a
near-flat prior plateaus at finite, stationary variances (medians ≤ 14, and those are
label-switching inflation, not divergence); the fixed-β sampler plateaus at ≈4–5. The
proper prior + proper posterior is stationary without needing mean-field's damping.
So "the cure is merely inference-level damping" is refuted in this synthetic,
identified setting: the reformulation is genuinely stable, consistent with the
at-scale null result (exp 0050/0051).

## Caveats (do not overclaim)

- **Synthetic, well-specified, identified corpus.** The generator is the model
  (stick-native, Σ identified). The insight-0033 real-data runaway was at K=50 on
  messier, model-misspecified data. This probe shows the *mechanism* is benign under a
  correct model; it does not prove the real corpus behaves identically.
- **Oracle β.** The fixed-β arm conditions on the planted β. The shipped read-out
  conditions on VI's converged β, which is close (insight 0044: β is what mean-field
  gets right) but not exact.
- **eta_scale = 4.0** was chosen as a plausible real-ish scale; the qualitative
  outcomes (label-switching confound, wide-but-stationary, no runaway) are not
  scale-specific but the exact widths are.

## What this changes

1. The Σ read-out **must** condition on a fixed (VI) β — this is now empirically
   forced, not just an alignment argument. Feeds queue item (4).
2. The free-Gibbs stationarity verdict is **outcome (a)**: no export hole; the
   informative scale prior (half-t on the triangulated band) is a refinement that
   tightens scarce-block scales, and its provenance is now empirically motivated.
3. A reusable rank-normalized split-R̂ diagnostic (`_mcmc_diag.py`) is now available
   for all downstream Gibbs work (items 4–5).
