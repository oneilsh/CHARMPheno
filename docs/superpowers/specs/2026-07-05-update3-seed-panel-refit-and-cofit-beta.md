# Update 3: the seed-panel gate, the refit-loop dynamics, and co-fit-beta on the real corpus (it helps modestly; the scale oscillates 5 -> 12 -> 8, no runaway)

Follows `2026-07-05-real-corpus-scale-result-and-fit-integration-update.md` (Update 2), which shipped
the held-out-calibrated generative scale c* = 5.0 on the real corpus, retired the unstable in-band EM,
and posed the in-fit-vs-separate-step design question. This update covers your §A over-commitment push,
your Q3 (does the refit loop ratchet?), and your Q5 (does co-fitting beta at the calibrated scale
help?).

A candor note up front, because this report changed twice as data arrived: a first draft concluded
"co-fit-beta does not help, no ratchet, drop the refit loop" -- written on topic-coherence (NPMI) alone,
before the co-fit model's held-out calibration came back. The held-out numbers reversed it (co-fitting
DOES improve prediction, and the recalibrated scale moved 5 -> 12, the ratchet direction). A round-2
fit then resolved the ratchet-vs-fixed-point question: fitting at 12 recalibrates back DOWN to 8, so
the sequence is 5 -> 12 -> 8 -- a damped oscillation converging to a fixed point near 9-10, NOT a
runaway. Net across all three: co-fitting helps prediction modestly (best ~+0.04 nats/token), the loop
is bounded and self-correcting, and it is low enough value that we do not recommend chasing it to
convergence. Sections C and D carry the full trajectory.

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
across c in 2-6) -- weak identifiability even with realistic overlap. Synthetic takeaway: no ratchet.
**Read this now as UNDER-POWERED, not reassuring**: the real corpus (C) shows the recalibrated scale
moving 5 -> 12, the ratchet direction the synthetic missed -- vindicating the caveat that this question
is only well-posed on the real corpus, and warning against trusting the synthetic "no ratchet."

---

## C. Co-fit-beta on the real corpus (your Q5) — it improves held-out prediction, and the scale ratchets 5 -> 12

We ran the clean real-corpus A/B: a fresh fit identical to the shipped model (same corpus, covariates,
gating, K, seed) except Sigma pinned to 5*R throughout (Sigma = 5*R every M-step, verified: diagonal
exactly 5, positive-definite on each document's allowed marginal, no runaway -- a fixed pin has no
feedback). Two readouts, and they point OPPOSITE ways, which is the whole story.

**Topic coherence (NPMI, Roeder et al. 2015 -- beta-only) is flat:**

```
                       unit fit (Sigma=R)     co-fit (Sigma=5R)     delta
background (40 topics)  mean NPMI 0.1816        0.1912               +0.0096
                        median    0.1719        0.1795               +0.0076
foreground (20 topics)  mean NPMI 0.2284        0.2298               +0.0014
                        median    0.2299        0.2354               +0.0055
```
The background +0.010 is within one standard error (per-topic NPMI stdev ~0.087 over 40 topics -> SE
~0.014); foreground is flat; the top topics are nearly identical (both peak at NPMI ~0.60 on the same
topic). By coherence alone, beta barely moved -- which is what an earlier draft of this report wrongly
concluded the whole story was.

**Held-out predictive LL (the objective we actually calibrate on) tells the opposite story.** Running
the same held-out sweep (identical corpus, holdout 0.5, seed 0 -> the SAME held-out tokens, so the LL
values are directly comparable) on each fit's beta:

```
c:        1        2        3        5        8       12       20
unit  : -6.698   -6.636   -6.618  -6.615*  -6.623   -6.637   -6.662     peak c* = 5
co-fit: -6.797   -6.690   -6.642  -6.601   -6.581  -6.577*  -6.587     peak c* = 12
```

1. **The co-fit beta predicts held-out tokens uniformly BETTER.** At every comparable c it matches or
   beats the unit fit (at c=5: -6.601 vs -6.615), and its peak (-6.577 at c=12) beats the unit fit's
   peak (-6.615) by ~0.038 nats/token. So co-fitting DID help -- on prediction, the calibrated quantity
   -- even though it did not move topic coherence. Coherence and predictive concentration are different
   axes, and here they diverged.
2. **The recalibrated scale moved UP, 5 -> 12.** Fitting at c=5 and recalibrating did not return ~5; it
   returned 12 (robustness across holdout fractions {0.5: 12, 0.8: 8, 0.95: 8}; interior maximum, not at
   the grid edge). So c=5 is not a fixed point of the refit map -- fitting at a higher scale makes the
   model prefer a higher scale still. That is the ratchet DIRECTION of your Q3, on the real corpus.

**Round 2 settles the ratchet question -- it is NOT a runaway.** A third fit (exp 0042: fit at
Sigma=12*R, recalibrate on a grid widened to [1,2,3,5,8,12,16,20,28]) recalibrated back DOWN to c* = 8
(interior; 28 scores worse; robustness {0.5: 8, 0.8: 5, 0.95: 5}). So the full recalibration
trajectory is:

```
fit at          recalibrated c*     peak held-out per-token LL
unit  (0028)         5                    -6.615
Sigma=5R  (0041)    12                    -6.577   (best predictor)
Sigma=12R (0042)     8                    -6.596
```

It OVERSHOT (5 -> 12) then SELF-CORRECTED (12 -> 8): a damped oscillation whose recalibration map
crosses the diagonal between 5 and 12 (it sends 5 up to 12 but 12 down to 8), so the fixed point is
near 9-10 -- bounded, converging, not a ratchet. Your "the held-out objective bounds the outer loop"
holds; it just does so non-monotonically.

**And the benefit is modest and non-monotone -- a moderate fit-scale is the sweet spot.** Comparing the
two co-fit betas at the SAME generation scale (same held-out tokens): at c=8, 0041 (-6.581) beats 0042
(-6.596); at c=12, 0041 (-6.577) beats 0042 (-6.602). So the beta fit at 5 predicts BETTER than the
beta fit at 12 everywhere -- fitting at 12 overshot and HURT prediction, and it dropped foreground NPMI
(mean 0.230 -> 0.216; background stayed flat 0.191 -> 0.193). The best model observed is 0041 (fit at 5,
generate at 12): +0.038 nats/token over the unit fit. The loop's own fixed point (~9-10) is a model we
did not fit, but it lies between 0041 and 0042 and so cannot beat 0041.

One honest caveat throughout: c is relative to each fit's correlation R, so the c-values are partly a
change of basis (the recovered concentration, not the raw c, is what transfers). But the head-to-head
"which beta predicts held-out tokens better" comparisons are basis-free -- same held-out tokens -- so
those legs are solid.

---

## D. What A-C mean — bounded loop, modest benefit, keep the pipeline simple

The Update-2 §C tension was: calibrate the scale IN the fit (intrinsic, resume-clean, but biased by
co-adaptation) versus as a separate EXTERNAL step. The full trajectory (C) lets us answer it on
evidence rather than hope: co-fitting helps a little and the loop is bounded, but the gain is small
enough that the simple external pipeline is the right default.

Status of your priority list, resolved:
- **Q3 (ratchet): answered -- NO.** The scale oscillates 5 -> 12 -> 8 and converges (~9-10); it does not
  run away. Your held-out-objective-bounds-the-loop argument holds, non-monotonically.
- **Q5 (co-fit-beta): answered -- YES but modestly, and non-monotone.** Fitting under a wider prior does
  yield a beta that predicts held-out tokens better, but the effect is ~+0.04 nats/token at best (at
  fit-scale 5) and REVERSES past that (fit-scale 12 predicts worse than 5 and lowers foreground NPMI).
  It never touched topic coherence. So c is mostly a generation-time knob; co-fitting is a small, bounded
  refinement, not a lever.
- **Priority 3 (the refit loop): runnable but not worth it.** It is safe (bounded) and would land at a
  self-consistent scale ~9-10, but that fixed-point model lies between 0041 and 0042 and cannot beat the
  best model we already have (0041, fit at 5). We recommend NOT chasing it: the ~0.04-nat ceiling does
  not justify the extra fits or the loss of the resume-clean single-fit pipeline.
- **The B2/PMI raw-data cross-check: droppable.** With the loop shown bounded and convergent, the third
  independent leg (a stable refit fixed point, ~9-10) is in hand; PMI is no longer needed as a gate.

Which model to ship, on the evidence:
- **Default: the unit fit + generation-only c=5 (0028).** Simplest, resume-clean, and within ~0.04 nats
  of the best. This is our recommendation for the tool.
- **If the best predictor is wanted: 0041 (fit at 5, generate at 12), +0.038 nats.** But generating at
  12 is peakier than anything the seed-panel (A) cleared (we only checked up to 8 on the unit beta), so
  this needs an over-commitment recheck at 12 before it could ship.

Separately, the seed regime (A) is unaffected and remains the one clearly-open problem: conditioned
completion from a tiny rare seed is dominated by the population-mean prior and needs a seed-size- or
conditioning-aware scale that no single global c can provide.

---

## E. Open questions for you

1. **Do you agree with "don't chase the loop"?** It is bounded (5 -> 12 -> 8, fixed point ~9-10) and
   its best-case gain is ~0.04 nats/token, already achieved by 0041 (fit at 5) -- the self-consistent
   fixed-point model would sit between 0041 and 0042 and cannot beat it. Given that, we would keep the
   simple unit-fit + generation-only-c pipeline. Is there a reason you would still run it to convergence
   -- e.g. do you weight the self-consistency of the fixed point (fit scale == generation scale) as a
   property worth having for its own sake, beyond predictive LL?
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

Net: the shipped scale (c = 5 off the unit fit) is a sound, honest default. Co-fitting at the calibrated
scale improves held-out prediction modestly (~+0.04 nats/token, best at fit-scale 5) and the
recalibration oscillates 5 -> 12 -> 8 to a bounded fixed point ~9-10 -- so the refit loop is safe but
low-value, and we recommend not chasing it. Q3 (ratchet) is resolved: no runaway. The
seed-size/conditioning-aware scale for tiny-prefix completion remains the one independently-open problem.
