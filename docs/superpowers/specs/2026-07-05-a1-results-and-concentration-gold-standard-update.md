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
documents read ~2× more peaked than the logistic-normal's.** We initially attributed this to
α-optimization reading HOT on short documents (mean-field VB under-dispersion). **A synthetic
plant-and-recover experiment (§4.5) refutes that as the cause** — with β fixed, LDA
α-optimization reads at or BELOW the true concentration, not above. So the ~2× gap is not an
α-inference artifact; it is driven by STM's fit scale being too low (§1) and by LDA co-fitting
a sharper, more document-specific β (which the fixed-β synthetic cannot reproduce). Whether
0.513 over- or under-states the truth is now genuinely open.

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

## 4.5. UPDATE — synthetic validation of the gold standard (done)

We built the §4 method and validated it on synthetic data where the true concentration is
KNOWN. Plant documents at a known per-document concentration over a SHARED-TERM topic matrix
(topics share a vocabulary pool, so inference must disambiguate), two planting mechanisms so
neither family is privileged (logistic-normal η~N(0,level·I) and Dirichlet θ~Dir(level·1)),
four concentration levels each. Recover with the logistic-normal MAP (sweep Σ scale c) and LDA
CAVI (sweep α), β FROZEN at truth to isolate concentration inference from topic learning. Run
at a clean regime (K=8, V=400, doc_len=60) and the REAL regime (K=60, V=5000, doc_len=44).

Three results:

1. **The gold standard works — both families, both regimes.** In all 8 (mechanism × level)
   cells, the held-out-LL argmax recovers a θ̂ whose median top_mass matches the planted median
   within a small tolerance (worst-case absolute error 0.068 at the real regime, 0.048 at the
   clean regime). The diffuse end never wins on a peaky corpus; the argmax is interior. So
   held-out within-document prediction is a trustworthy, prior-agnostic, ground-truth-free
   concentration calibration — and it holds at the real regime (K=60, 44-token documents), the
   regime we would apply it in.

2. **The logistic-normal MAP recovers concentration MORE faithfully than LDA at fixed β**
   (real-regime mean absolute error 0.019 vs 0.033), and on BOTH generative mechanisms (no
   "matched prior" advantage for LDA on its own Dirichlet-generated data).

3. **LDA α-optimization does NOT read hot — it reads COOL.** With β fixed at truth, LDA's own
   α-optimization recovers at or BELOW the planted concentration (real-regime mean top_mass
   Δ = −0.014; several cells badly under-concentrated). This refutes the mean-field-VB "reads
   hot" story as the explanation for the real-data ~2× gap. It also picks a more diffuse α than
   the held-out-LL optimum — its internal objective is not the held-out-predictive one.

Consequence: the real STM-vs-LDA gap is not an α-inference artifact. It is STM's fit scale
being too low plus LDA co-fitting a sharper β (untestable with β frozen). The open items are
(a) run held-out-LL on the REAL corpus to pin the scale, and (b) a synthetic run where each
family CO-FITS β (rather than freezing it at truth) to isolate the β-learning contribution.

## 5. Questions for the reviewer

1. **Held-out prediction now recovers the known concentration synthetically (§4.5) at the real
   regime.** Do you agree that is sufficient validation to run it on the REAL corpus as the
   calibration, or would you still want a raw data-side statistic (your B2) as an independent
   cross-check? If B2, what specific token-diversity observable best proxies per-document
   concentration WITHOUT passing through either model's inference?
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
5. **The real gap now looks like β-learning, not α-inference (§4.5, Finding 3).** With β frozen
   the two families recover the same known concentration; the real ~2× gap must then come from
   LDA co-fitting sharper topics (plus STM's low fit scale). Is a synthetic run where each family
   CO-FITS β the right way to isolate the topic-learning contribution — and if LDA's co-fit β is
   what makes its documents peaky, is that legitimate signal (real topics ARE that sharp) or
   overfitting we should not calibrate the generative scale to?

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
