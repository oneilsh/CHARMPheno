# Update 3: the seed-panel gate, the refit-loop dynamics (no ratchet), and co-fit-beta on the real corpus (no benefit)

Follows `2026-07-05-real-corpus-scale-result-and-fit-integration-update.md` (Update 2), which shipped
the held-out-calibrated generative scale c* = 5.0 on the real corpus, retired the unstable in-band EM,
and posed the in-fit-vs-separate-step design question. This update closes three of your open items:
your §A over-commitment push, your Q3 (does the refit loop ratchet?), and your Q5 (does co-fitting
beta at the calibrated scale help?). Two of the three answers are clean negatives, and together they
simplify the design substantially.

One-line setup recap: gated logistic-normal topic model, eta ~ Normal(Gamma^T x, Sigma), theta =
softmax(eta); topics split into a shared background block plus per-group foreground blocks, each
document hard-masked to background union its own group; the fit pins Sigma to a unit-diagonal
correlation R for stability, and a scalar c recovers the generative scale via Sigma_gen = c*R.

---

## A. The seed-panel acceptance test — your over-commitment push, resolved (ship 5), plus a new failure it exposed

You asked us to settle empirically whether c = 5 over-commits on the tool's hard case, a 1-2 token
"seed" prefix (conditioned generation from almost no data), versus a milder c = 3. We built the
seed-panel test: seed each foreground topic with its own top signature code(s), run the gated
conditioned Laplace E-step at a swept c, and read the resulting theta's concentration + whether a
plausible secondary interest survives.

**Method validated on synthetic ground truth first.** On a planted corpus the test correctly DETECTS
over-commitment: as c climbs past the planted scale, median top_mass rises and effective-#topics
falls monotonically, with a clear secondary-mass collapse by c = 8. So a "safe" reading is
informative, not mere insensitivity.

**Your specific worry is refuted — ship 5.** On the real model, restricted to seeds that land on their
own topic (your exact scenario: "right topic but implausibly total mass, secondaries erased"), c = 3,
5, and even 8 all keep top_mass modest (~0.12-0.38) and effective-#topics in double digits (13.9 at
c=3 -> 16.6 at c=5), with no secondary collapse. The acceptable band extends past 8, so 5 sits
comfortably inside it. No re-export needed; the shipped bundle already carries c = 5.

**But the test exposed the OPPOSITE, larger failure — the conditioning-aware-scale problem you
foresaw.** At c in {3,4,5}, MOST rare foreground seeds do NOT recover their own topic: self-recovery
is 5-15% (single-token seed) / 25-65% (two-token seed). They get pulled to common background topics.
The mechanism is exactly your "the prior does the most work when tokens are fewest": with the covariate
vector at its population-mean default, Gamma^T x places the prior mass where the corpus lives, and rare
foreground topics have strongly negative population-mean intercepts (e.g. a rare foreground topic's
intercept -4.31 vs a common background topic's +2.95). A 1-2 token seed carries too little likelihood
to drag eta out of that basin at c = 5; it takes c ~ 20-50 to make rare seeds self-recover -- and THAT
c would over-commit normal-length documents. So no single global scale serves both the tiny-rare-seed
regime and the full-document regime. This is your "in the limit, a conditioning-aware scale," now
quantified: the global held-out optimum (5) is right for average prediction over ~44-token documents
but too stiff for tiny rare seeds; the seed regime wants a separate, seed-size-aware treatment. We are
treating this as its own open problem, not a knob on the global c.

---

## B. The refit-loop dynamics (your Q3) — no ratchet, robustly; but the synthetic testbed is confounded for the sub-question

We added an engine option to pin Sigma_ii to an arbitrary constant c (generalizing the unit pin;
off-diagonals still standardized to correlations and clipped -- only the diagonal target changes), so
we can fit AT a calibrated scale and iterate fit -> calibrate -> refit. Your Q3: does that loop ratchet
the scale upward (refit at higher c -> beta sharpens -> documents more identifiable -> next calibration
returns higher c -> sharper still), or contract to a fixed point?

**No ratchet -- robustly, across two synthetic regimes.** The recalibrated c* stays low and does not
climb: a disjoint-vocabulary plant gave c* = 3 -> settled; a realistic-overlap plant (topic-support
Jaccard 0.333, matching real corpora, with half of every topic's mass on a shared pool so beta cannot
trivially separate topics) gave c* = 3 -> 2, settled in one round with no monotonic rise. Your
prediction that the held-out objective bounds the outer loop (an over-sharpened beta pays on held-out
tokens) is borne out: the loop never runs away upward.

**A methodological correction we owe you.** Our first pass used a disjoint-vocabulary plant, where each
topic's signature terms let beta separate topics trivially and ABSORB the per-document concentration --
so the unit fit was already at the planted concentration and the loop had nothing to climb. That is not
a real test. We now default synthetic corpora to realistic shared-mass overlap. Even so, the synthetic
testbed turned out to be confounded for the *sub*-question "does refitting SHARPEN beta": (1) the
concentration readout at each round uses that round's own pinned Sigma, so a top_mass comparison across
rounds mixes "beta changed" with "readout prior changed"; (2) each round re-fits from scratch rather
than continuing training; (3) c is relative to the FITTED correlation R, not the planting basis, so
"recover the planted scale" is ill-posed (the recovered *concentration* matches; the recovered
*c-value* need not). And the held-out LL curves are flat near the top (differences ~0.001-0.01 nats
across c in 2-6) -- weak identifiability even with realistic overlap. Robust takeaway: no ratchet. The
"does refit sharpen beta" question is only well-posed on the real corpus, which motivated C.

---

## C. Co-fit-beta on the real corpus (your Q5) — no benefit; beta does not sharpen

We ran the clean real-corpus A/B: a fresh fit identical to the shipped model (same corpus, covariates,
gating, K, seed) except Sigma pinned to 5*R throughout (Sigma = 5*R every M-step, verified: diagonal
exactly 5, positive-definite on each document's allowed marginal, no runaway -- a fixed pin has no
feedback). We compare topic coherence (NPMI, Roeder et al. 2015 -- beta-only, no Sigma confound) block
by block against the unit fit:

```
                       unit fit (Sigma=R)     co-fit (Sigma=5R)     delta
background (40 topics)  mean NPMI 0.1816        0.1912               +0.0096
                        median    0.1719        0.1795               +0.0076
                        max       0.6045        0.6002               ~0
foreground (20 topics)  mean NPMI 0.2284        0.2298               +0.0014
                        median    0.2299        0.2354               +0.0055
                        max       0.3512        0.3338               -0.017
```

**Co-fitting at c = 5 did not sharpen the topics.** The background shift (+0.010 mean) is within one
standard error (per-topic NPMI stdev ~0.087 over 40 topics -> SE ~0.014); the foreground block is flat.
The top topics are nearly identical between the two fits (both peak at NPMI ~0.60 on the same
background topic). Beta barely moved. The concentration readout is directionally consistent (with the
Sigma-confound caveat): the co-fit reads top_mass 0.246 under Sigma = 5R, and a *weaker* prior should
concentrate theta MORE, yet it is at or slightly below the unit fit's ~0.27 -- so beta certainly did
not sharpen, if anything softened. So the co-fit-beta benefit we both expected does not materialize on
real data, consistent with (and now un-confounded relative to) B.

---

## D. What A-C jointly resolve — the design simplifies

The Update-2 §C tension was: calibrate the scale IN the fit (intrinsic, resume-clean, but biased by
co-adaptation) versus as a separate EXTERNAL step (unbiased, but "a step outside the loop" that breaks
stop-and-restart). C dissolves it: **since co-fitting buys nothing, there is no reason to fold c into
the fit at all.** The recommended pipeline is the simplest one:

- Fit once at the unit pin (stable, resume-clean, ADR-0034 machinery unchanged).
- Calibrate c at export by held-out prediction (unbiased; c = 5).
- Apply Sigma_gen = c*R at generation time only.

Consequences for your priority list:
- **Priority 3 (the refit loop): recommend DROP.** It was motivated by "refit at c* -> beta re-sharpens";
  C shows it does not, and B shows the loop is stable anyway. The self-describing-checkpoint machinery
  (pinning c back into model state) is unnecessary complexity for no measurable gain.
- **Q3 (ratchet): answered -- no.** And moot on real data: with no beta-sharpening there is no c<->beta
  feedback channel to drive a ratchet in the first place.
- **Q5 (co-fit-beta): answered -- no benefit.** c is a generation-time concentration knob, not a
  training improvement. The unit-fit beta is already as coherent as a beta co-fit at the calibrated
  scale.
- **The B2/PMI raw-data cross-check: we agree it can be dropped as a gate.** You said three independent
  legs would suffice; we now have them -- synthetic recovery of the known concentration (both prior
  families, both regimes, gated), a real-corpus interior held-out maximum agreeing with the synthetic
  band, and a stable (here, trivially stable, because beta does not move) refit fixed point.

The one thing that got HARDER, not simpler, is the seed regime (A): the global scale is settled and
generation-only, but conditioned completion from a tiny rare seed is dominated by the population-mean
prior and needs a seed-size- or conditioning-aware scale that the single global c cannot provide.

---

## E. Open questions for you

1. **Do you agree the refit loop should be dropped?** The co-fit-beta null (C) plus the no-ratchet
   result (B) seem to remove its entire rationale. Is there a benefit you were tracking that NPMI +
   held-out prediction would miss -- some downstream property of a beta trained under the wider prior
   that unit-fit beta lacks?
2. **The conditioning-aware scale (A).** For tiny-seed conditioned generation, the prior mean Gamma^T x
   dominates and the global c can't rescue rare-topic recovery without over-committing full documents.
   Is the right move a scale that grows as the observed-token count shrinks (weaken the prior when there
   is little data to condition on), a conditioning-time reweighting of Gamma^T x, or simply a product
   rule of "seed with >= 2 codes / non-population-mean covariates when showcasing a rare topic"? Do you
   see a principled seed-size-aware prior we should reach for?
3. **The flat held-out LL curves.** Even in the realistic-overlap regime the held-out objective is flat
   near its optimum (~0.001-0.01 nats across c in 2-6). The argmax is well-defined and interior, but the
   weak curvature means the scale is only loosely identified. For a generation-only knob that is fine;
   would you want a tighter objective (e.g. a concentration-matching target rather than raw predictive
   LL) if the scale ever needed to be pinned more precisely?

Net: the global generative scale is settled (c = 5, generation-only, calibrated not tuned), the refit
loop and the raw-data cross-check both look droppable, and the one genuinely open problem is the
seed-size/conditioning-aware scale for tiny-prefix completion.
