# Update 4 — where the model landed: finalized scale estimator + the compression lesson made concrete

Follow-up to updates 2/3 and the review-3 exchange. Data shape unchanged from prior notes:
documents are user-day bag-of-words aggregates; a shared BACKGROUND topic block plus per-group
FOREGROUND blocks under hard masking (a document expresses background ∪ its own group); thin
covariates (coefficients ±1-3 nats against a ~7-nat intercept gap). All quantities below are on
the real corpus.

## 1. The scale arc is closed — the layered answer held

The original problem was "the fit discards the generative scale and nothing recovers it safely."
The answer that survived all the data:

- **unit-pinned fit** for stability (Σ = R, diagonal pinned to 1);
- a **held-out-predictive-LL-calibrated generation scale** applied at export (Σ_gen = c·R), honest
  and prior-family-agnostic;
- the pooled-EM scale estimator **retired** — it had positive feedback in the scale direction with
  no trust region (larger c → weaker (1/c)R⁻¹ prior → η̂ wanders → larger measured variance → larger
  c); pooling protected a single low-ess topic but never the global scale;
- co-fitting β at the calibrated scale left as an **optional, bounded, non-monotone** refinement
  (skippable); ship default is the unit fit + generation-only c.

## 2. The estimator, finalized: log-grid + quadratic smoothing (this was the open loose end)

The prior estimator was argmax over a coarse, roughly-LINEAR grid. Two problems, both of which you
had flagged in spirit: (a) c is a MULTIPLICATIVE scale, so a linear grid mis-resolves it; (b) the
held-out-LL curve is a broad, flat SHELF, so argmax over a coarse grid on a flat noisy curve is a
quantized, jittery point (this is what made the refit loop wander).

The replacement:

- **geometric grid** — 13 points, ratio ~1.41, over [0.5, 32]. Even resolution in log c.
- **local quadratic fit in log c** near the peak: LL(u) ≈ a + b·u + ½q·u², so c* = exp(−b/q). This
  gives a sub-grid, noise-averaged c*, AND the curvature q is the identifiability — q ≈ 0 (a flat
  shelf) yields a large, honest standard error rather than a false-precise point.

Real-corpus result (shipped holdout fraction 0.5):

- **smoothed c* = 4.58** (interior; grid argmax was the grid point 4.0 — smoothing recovered the
  off-grid vertex between 4.0 and 5.66);
- curvature q = −0.062 (shallow-concave — exactly the "broad shelf, not a sharp point" you
  predicted);
- robustness across holdout fractions {0.5: 4.58, 0.8: 3.78, 0.95: 3.65} — the shelf showing itself
  (the estimate drifts down as the visible token set shrinks). Read c* as "somewhere on a broad
  shelf around 4," not a sharp 4.58.

**Honest caveat on the SE.** The delta-method SE on the vertex is the classic ratio-of-Gaussians
(Fieller) pathology exactly when q is itself small and uncertain — precisely our regime. Across
seeds it can come out misleadingly small (we saw 0.03×–0.68× of c*). So the SE is INDICATIVE, not a
calibrated interval; the robust identifiability signals are the **interior flag and the sign of q**.
A genuinely calibrated band would need the document-bootstrap you queued (resample documents,
refit the quadratic) — deferred, not shipped.

## 3. The compression lesson, made concrete (and a little sharper than before)

Wiring the per-document θ̂ into a downstream display exposed the compression point in the flesh.
The per-document θ̂ from the UNIT-scale fit are the over-diffuse "not a measurement" outputs we kept
warning about: on the real corpus the MEDIAN document's θ̂ on a common shared-block topic sits ≈ the
uniform value 1/K. In other words, at the fit scale the typical document looks like it expresses
every common topic at chance level — the inference barely moved θ̂ off the diffuse prior mean.

Re-inferring the same documents at the CALIBRATED scale (c ≈ 4.58, the wider generative
concentration) pulls each document's θ̂ onto the topics it actually supports and BELOW uniform on the
rest — the median θ̂ on those common topics roughly halves. So the operational rule is now concrete:
**any per-document use of θ̂ — display, downstream feature, membership — must infer at the calibrated
scale, never the fit scale.** This is the "two roles of Σ" point in its sharpest form: the fit scale
regularizes β estimation; the calibrated scale is the honest per-document concentration. They are
different numbers and each is wrong for the other's job.

## 4. Where next — replacing the last hand-set knob

Surfacing "how many documents express topic k" still leaned on a threshold on θ̂_k, which is (a) an
arbitrary cutoff and (b) worse, blind to whether the document contains any of topic k's
characteristic tokens — a document with prior-inflated θ̂_k but none of k's words is counted anyway.
We think the model hands us a threshold-free replacement directly (per-token responsibilities +
their θ-posterior uncertainty). That is the subject of the companion proposal
(`2026-07-06-responsibility-presence-metric-proposal.md`) — the interesting open question we'd value
your read on.
