# Update 2: real-corpus scale calibrated (c* = 5.0), the in-band EM retired, and the fit-vs-export-step design question

Follows `2026-07-05-a1-results-and-concentration-gold-standard-update.md` (that report ended
with the held-out-LL gold standard validated synthetically, gated and non-gated, and the open
item "run it on the real corpus to pin the scale"). This is that run, plus a failure we hit and
a design question your priority-3 refit-loop idea raised for us.

Setup recap in one line: online-VI topic model with a logistic-normal document prior
η ~ Normal(Γᵀx, Σ), θ = softmax(η); topics are partitioned into a shared background block plus
per-group foreground blocks, and each document is hard-masked to background ∪ its own group.
The fit pins Σ to a unit-diagonal correlation R for stability, which discards the generative
concentration SCALE; we recover a single scalar c so the generative covariance is Σ_gen = c·R.
Corpus: ~48k documents, ~44 tokens each, ~6k-term vocabulary, K = 60, two groups.

---

## A. The real-corpus result — c* = 5.0, clean interior maximum

We swept the held-out-predictive-LL objective (§4 of the prior report) over the scale grid
c ∈ {1, 2, 3, 5, 8, 12, 20} on the real gated corpus, β and R frozen at the fit, holdout
fraction 0.5. Mean per-token held-out log-likelihood:

```
c:     1        2        3        5        8        12       20
LL:  -6.6982  -6.6360  -6.6184  -6.6147  -6.6228  -6.6372  -6.6619
                                 ↑ argmax (interior, not at a grid boundary)
```

A smooth concave curve peaking at **c* = 5.0**. Robustness across holdout fractions (how the
argmax moves as the visible token set shrinks toward the small-seed regime):

```
holdout_frac:  0.5    0.8    0.95
argmax c*:      5      3      3
```

A tight, stable 3–5 band. The argmax drifts DOWN as we hold out more tokens — expected: fewer
visible tokens make θ̂ noisier, which favors a slightly milder prior. Holdout 0.5 has the most
information to infer θ̂ from and is the one we ship.

**This confirms the synthetic prediction on real data.** Every fit-anchored estimate of the
scale is compressed below the truth, and the unbiased held-out answer is above all of them:

```
unit prior      1.00    (fit stability pin — discards the scale)
A1 in-band τ²   2.36    (co-adapts with β each iteration → biased low)
frozen-β EM     3.67    (Laplace-bias — see B, and it is also unstable)
held-out c*     5.00    ← unbiased, cross-validated
```

c* = 5.0 is ~2× the in-band A1 reading and lands in the ~5–7 band the synthetic study
predicted for this regime. The generative simulator now runs at Σ_gen = 5·R and concentrates
as intended (the previous "every sample peaks on a different topic" diffuseness is gone). This
is the number we are shipping.

---

## B. The in-band EM estimator is not just biased — it is unstable. We retired it.

Before adopting held-out-LL we had an intermediate estimator: an iterated pooled EM that, with
β and R frozen, re-estimates a single pooled scale c by the law of total variance (between-doc
variance of the posterior mode + mean Laplace posterior variance), pooled over the free topics.
We had argued it was runaway-safe because pooling to ONE scalar prevents any single low-support
topic from self-amplifying (a per-topic free diagonal had previously blown up on us).

On the real corpus it ran away:

```
[eta_scale] iter 0: c=1.6651
[eta_scale] iter 1: c=2.2766
...
[eta_scale] iter 7: c=3.6094      ← climbing smoothly toward ~3.6
[eta_scale] iter 8: c=1116.6869   ← 300x jump in one iteration
[eta_scale] iter 9: c=770918.12   ← gone
```

Root cause — a positive feedback loop in the SCALE direction with no trust region: larger c →
weaker prior precision (1/c)·R⁻¹ → posterior modes wander further from the mean → larger
measured between-doc variance → larger c. The likelihood term keeps the modes pinned up to
c ≈ 3.6; past that the prior is weak enough that some document's Laplace E-step destabilizes
(mode runs off, or the Hessian goes near-singular so its inverse blows up), that document's
inflated contribution yanks the pooled mean up, the prior collapses, and every document
de-pins.

The correction to our earlier reasoning: **pooling protects against a single low-support TOPIC
self-amplifying, but NOT against the global SCALE self-amplifying** — because c feeds back into
every document's prior uniformly. The bounded grid sweep has no such loop (it scores each fixed
c and takes an argmax; a shaky E-step at large c just scores a worse predictive LL and loses),
so it cannot run away. We removed the EM from the export path entirely; the held-out sweep is
now the sole scale estimator. Net for the "which estimator" question: the in-band / frozen-β
family isn't merely biased low, it can be actively unsafe, which we think further favors
external held-out calibration over any in-band variance-matching scheme.

---

## C. The design question you surfaced — calibrate IN the fit, or as a second step?

Your priority-3 "refit loop" idea (pin the scale, refit, recalibrate; "unit was never
load-bearing, PINNED was") put a finger on something that bothers us about the current shape,
and we want your read on it.

**The discomfort.** Right now c* is computed as a post-fit EXPORT step. The fit itself has a
clean property we value: it is online VI, checkpointed every iteration, resumable from any
point — stop and restart anywhere and you get the same model. Bolting scale calibration on
downstream breaks that cleanliness: the calibrated scale is not part of the checkpointed model
state, it is recomputed outside the loop. It is a separate step that "messes with stop-and-
restart-at-any-point." If someone resumes training from a checkpoint, the export-time c* they
computed earlier is now stale.

**The natural question: continued fits.** If we pin Σ_ii to the calibrated c* and CONTINUE
fitting, does the model get worse or better? We think **better**, and this is exactly your
refit-loop point: at the higher pinned scale, β re-sharpens toward the true concentration
(the prior no longer over-shrinks η), so a continued fit at c* should improve topic
definition, not degrade it. A continued fit at the OLD unit pin, by contrast, just relaxes back
toward the compressed scale. So "continued fitting" is not a threat to the calibrated scale —
it is the mechanism that bakes it in.

**The cadence question — every iteration? every 10?** If refitting at c* is better, why not
recalibrate continuously? We worked through this and think the answer is "a small number of
discrete outer rounds," for three reasons:

1. **Cheap-and-continuous gives the WRONG scale.** The only calibration cheap enough to run
   every iteration is the in-band one (A1), and that is precisely the biased estimator — it
   converges to 2.36 because β co-adapts to Σ every step. You cannot get both cheap and
   unbiased in-band; the bias IS the co-adaptation that per-iteration updating creates.
2. **The unbiased objective is expensive.** Held-out-LL is a grid × holdout sweep of E-step
   passes over the corpus (here, ~7 min for one holdout × 7 grid points). Running it every 10
   iterations would multiply fit cost by ~10–100×.
3. **You cannot calibrate on a not-yet-converged β.** Early-iteration β is garbage, so an
   early c* is garbage. Calibration is only meaningful after β burn-in — which already
   collapses "every iteration" down to "a couple of late rounds."

And empirically (your own observation) the (β, c*) fixed point is reached fast: the second
calibration moves much less than the first. So the sweet spot is **fit → calibrate c* → pin
Σ_ii ≡ c* and refit → recalibrate (barely moves) → done** — ~2 outer rounds, ~all the benefit
at ~2× calibration cost instead of 20–200×.

**The synthesis we think resolves the discomfort.** This hybrid actually reconciles the two
framings. Calibration is EXTRINSIC (held-out prediction, run between rounds — unbiased, no
co-adaptation), but its output is folded back as an INTRINSIC pinned model constant. After the
final refit, c* is not a downstream export artifact anymore — it is the pinned Σ diagonal in
the checkpointed model state. Resume-from-checkpoint is clean again, because the calibrated
scale travels WITH the model. The "separate step outside the loop" only exists during the 1–2
calibration rounds, not at inference or export time.

**What we'd need to build it** (not done — this is the proposal): an engine option to pin Σ_ii
to an arbitrary constant c* (we have the unit-pin; this generalizes it), so we can refit at the
calibrated scale rather than re-deriving it downstream every export.

**Questions for you:**

1. Do you agree the 1–2-round pin-and-refit is the right cadence, and that per-iteration
   recalibration is a false economy (it can only ever be the biased in-band update)?
2. Is folding c* back as a pinned intrinsic constant (so the model checkpoints at the
   calibrated scale) the right way to dissolve the "step outside the loop" problem — or do you
   see value in keeping calibration permanently external and just versioning it alongside the
   checkpoint?
3. Any failure mode in continued-fitting at the pinned c* that we are not seeing — e.g., β
   over-sharpening across successive refits, or the recalibration drifting rather than
   converging? Our expectation is a contraction to a fixed point; is there a regime where it
   is not?

---

## D. Status of the prior report's open questions

- **Q1 (run held-out-LL on the real corpus?)** — done, §A. Interior maximum, c* = 5.0, gated
  so the scale is well-posed (per §4.6 gating makes it identifiable). We did not run your B2
  raw-data cross-check; still open whether you want it as an independent gate given the
  synthetic + real agreement.
- **Q3 (co-adaptation ceiling / let the scale float within a fit?)** — §B and §C are our
  answer: in-band floating either co-adapts (biased) or, unpinned, runs away. The pin-and-refit
  outer loop is the "float free of β" mechanism that doesn't reintroduce the runaway, because
  the freeing happens BETWEEN rounds with β frozen, not within a round.
- **Q4 (held-out optimum as the generative target)** — this is what we shipped: Σ_gen = c*·R
  with c* the held-out argmax, so generated documents pass back through inference at the
  calibrated concentration. Still your C1 in spirit, held-out-LL as the gate instead of a seed
  panel.

## Method notes

- Held-out-LL is mean per-token: corpus-wide held-out log-likelihood ÷ corpus-wide held-out
  token count, so it is comparable across c and corpus size. Split is seeded per document
  independent of c, so the sweep is a controlled comparison on identical splits.
- Inference uses exp(E[log β]) (matching the fit's data term); scoring uses E[β] — conflating
  the two miscalibrates the recovered scale.
- References as in the prior report; the runaway/trust-region framing follows the standard
  observation that fixed-point variance updates need a step cap under weak-prior feedback.
