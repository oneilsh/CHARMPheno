# Update: in-band global scale (A1) results, and the concentration gold-standard question

**For the reviewer.** Follow-up to the prior-family-vs-scale diagnostic. We built and ran the
two experiments you recommended (test A1 for stability; compare LDA vs the logistic-normal on
the same corpus). This reports what happened, a twist neither of us predicted, and a
methodological gap we'd like your read on: we have no gold standard for the true per-document
concentration, and we think held-out prediction is the way to get one.

## 0. Recap (self-contained)

Gated logistic-normal topic model: η_d ~ Normal(Γᵀx_d, Σ), θ_d = softmax(η_d), topics
partitioned into a shared background block plus per-group foreground blocks (a document
expresses background ∪ its own group only). Fit by distributed online variational inference;
the per-document E-step is a MAP/Laplace estimate of η. A free per-topic Σ variance runs away
at fit time in the gated setting (a document-scarce minority topic's variance blows up), so
the shipped fit pins Σ to a unit-diagonal correlation — stable, but it discards the variance
SCALE, and the generative tool (draw η ~ Normal(Γᵀx, Σ)) then produces over-diffuse documents.
The open problem: recover a bounded, non-unit GENERATIVE scale, data-driven, without
reintroducing the fitting runaway.

Two candidate scales existed before this update: a frozen-β post-fit pooled estimate (β and
the correlation frozen, a single scalar estimated by a law-of-total-variance EM), which
converged to **3.67**; and a synthetic plant-and-recover diagnostic suggesting STM inference is
unbiased at scale ≈ 5–7. You recommended testing **A1**: a single pooled scale τ² estimated
IN-BAND at fit time, Σ = τ²·R (R the unit-diagonal correlation), with a damped update to avoid
a τ–β sharpening ratchet.

## 1. A1 results — stability CONFIRMED, with a twist

**Stability: confirmed, exactly as you predicted.** A1 converged cleanly: τ² bounded, the Σ
diagonal uniform (so the per-iteration variance log reads min = max = τ²), objective improved,
no single topic self-amplified, and the damped update (multiplicative per-iteration cap on τ²)
held. The pooling argument transferred in-band: one document-scarce topic's noise is averaged
against every other topic's and cannot drag the scalar. This is a genuinely stable in-band
concentration parameter — the thing the field lacked for the logistic-normal.

**The twist: τ² converged BELOW the frozen-β post-fit estimate — 2.36 vs 3.67.** We expected
in-band estimation (β not frozen) to REDUCE the conservative bias of the frozen-β estimate.
The opposite happened. In-band, β co-adapts to whatever Σ is: a small Σ tightens the prior, the
per-document MAP mode is shrunk toward Γᵀx, the between-document residuals stay small, and τ²
settles at a LOW self-consistent fixed point. Freezing β (the post-fit estimate) removes that
downward pull, so it lands higher. The two data-driven estimators bracket the same
inference-induced compression from opposite sides, and neither reaches the synthetic-diagnostic
band.

## 2. Real-corpus prior-family reference (your recommended comparison)

LDA with document-concentration (α) optimization, on the SAME documents, non-gated:
α optimized down to ≈ 0.018, and the per-document concentration read:

| model | per-doc top_mass (median) | effective #topics (median) |
|---|---|---|
| STM, A1 in-band (scale 2.36) | 0.269 | 8.5 |
| LDA, α-optimized (α ≈ 0.018) | 0.513 | 2.8 |

Leading with top_mass (a max, so unaffected by the gating support-size difference): **LDA
documents read ~2× more peaked than the logistic-normal's.** Per your earlier point, α-optimization
on short documents reads HOT — mean-field VB under-estimates posterior spread, biasing α̂ down —
so 0.513 is an upper bound, not ground truth.

## 3. The honest gap — we have a bracket, not a point, and no gold standard

Here is where we want your read. The plant-and-recover diagnostic measures INFERENCE BIAS:
on synthetic documents with a KNOWN planted concentration, STM-MAP recovers it faithfully at
scale ≈ 5–7 and LDA-α over-sharpens. That is NOT a measurement of the true concentration of
REAL documents — it says "IF you generate at 5–7, STM inference is self-consistent," which is
weaker than "real documents are at 5–7." We had been treating the two as the same; they are not.

On real data we therefore have only a **bracket**: true per-document top_mass lies somewhere in
**[0.27, 0.51]** — STM at low scale is known-biased-diffuse (0.269 is a floor), LDA-α is
known-biased-peaky (0.513 is a ceiling). Neither endpoint is ground truth, and nothing we have
pins the value within the bracket. Choosing a generative scale of "5–7" would over-claim: it
rests on the synthetic diagnostic plus one bounded fit, not a real-data measurement.

## 4. Proposed gold standard — held-out within-document prediction

To pin the concentration without a known θ and without privileging either prior family: hold
out a random subset of each document's tokens, infer θ from the visible subset with β fixed,
and score predictive log-likelihood on the held-out tokens. Then SWEEP the concentration knob
(the logistic-normal's Σ scale; LDA's α). Too diffuse → flat predictive distribution → low
held-out likelihood; too peaky → over-commits to the visible tokens' topics and misses
held-out tokens from a secondary theme → also low. The interior maximum is the data's true
effective concentration — a prior-family-agnostic, ground-truth-free calibration objective.
This also directly answers "which prior fits document peakiness better": whichever attains the
higher held-out predictive likelihood at its own optimal concentration.

## 5. Questions for the reviewer

1. **Is held-out within-document token prediction the right external gold standard for
   concentration?** Or do you prefer a raw data-side statistic (your B2)? If B2, what specific
   token-diversity observable best proxies per-document concentration WITHOUT passing through
   either model's inference?
2. **A1 converges BELOW the frozen-β estimate, not above.** Does that change your "A1 primary,
   frozen-β estimate as backstop" ordering? A1 is the stable fit-time parameterization but the
   most conservative of the three fit-anchored scales (2.36 < 3.67 < synthetic 5–7).
3. **The co-adaptation ceiling.** A1's τ² is pinned low because β re-adapts to Σ every
   iteration. Is there a principled way to let the scale float free of β WITHIN a fit — a
   two-timescale update, or a β-frozen scale-only phase after burn-in — that wouldn't
   reintroduce the runaway? Or is the frozen-β post-fit estimate structurally the best that
   in-band estimation can do, and the real answer is external calibration (§4)?
4. **If the held-out optimum lands between the two model readings**, is matching the generative
   scale to that optimum — so generated documents pass back through inference at the calibrated
   concentration — the right target? This is your C1, but with held-out predictive likelihood as
   the calibration objective rather than a curated seed panel. Do you see a failure mode in
   using held-out-LL rather than a seed panel as the gate?

## Method notes

- All concentration numbers are per-document (top_mass = largest single-topic share;
  effective #topics = inverse-Simpson / Hill order 2, Hill 1973 / Jost 2006), medians over the
  full corpus (~48k documents, ~44 tokens each, ~6k-term vocabulary).
- The logistic-normal readout uses the gated MAP mode; the LDA readout uses the variational
  mean. The top_mass comparison is robust to that difference; the effective-#topics comparison
  is additionally confounded by the gated model's per-document allowed-set support, so we lead
  with top_mass.
- References: online variational inference (Hoffman, Blei & Bach 2010); logistic-normal /
  correlated topic model + Laplace posterior (Blei & Lafferty 2007); Dirichlet α-optimization
  (Blei, Ng & Jordan 2003; Minka 2000); mean-field VB under-dispersion vs collapsed inference
  (Teh, Newman & Welling 2007; Asuncion et al. 2009).
