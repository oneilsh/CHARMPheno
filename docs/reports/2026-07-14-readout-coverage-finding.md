# Read-out engine built; the coverage gate returns your pre-registered negative

Setup: nodes form an is-a DAG; each carries an additive offset over its ancestral closure; a
document attests a most-specific node; "increment" = a node's coefficient over its parent. This
closes queue item (2): the read-out engine, and its coverage-plant acceptance gate.

## What's built and validated

The full two-phase pipeline runs end-to-end: VI warm-start (+ a fractional-z soft gate for
partial-label docs) → the identifiability compiler on the expected design moment → a warm-started
co-sampled Gibbs pass emitting offset-INCREMENT draws on the identified quotient → a
per-coordinate-class read-out. Three things are validated:

1. **Ordering/recovery works** — planted increments recover in correlation (~0.6), consistent with
   the design-wall probe.
2. **The quotient is transparent to the engine** — fitting the compiler's quotient gives the same
   identified-coordinate posterior as fitting the original DAG (corr 0.99): the Gram invariant lifted
   to the posterior.
3. **The design-wall / emit-but-flag contract holds — 12/12 replicates.** Gauge and unresolved
   coordinates never emit a point estimate; unresolved carries its attestation recipe, gauge its
   convention. The half of the engine that refuses to answer un-identified directions is exactly
   right.

## The coverage gate — your pre-registered negative

The gate: redraw the node offsets from the prior each replicate, run the whole engine, ask whether
each identified coordinate's 90% interval covers the planted value. **Identified coverage fails**,
for two distinct reasons I've now separated:

- **Granularity (fixed).** Each node's offset lives in the full stick space, but a node is
  identified only on the sticks its own documents activate (its own foreground block). The other
  blocks sit pinned at the prior and can never cover a freely-drawn truth, so an all-coordinates
  check is 0 by construction. I restricted the read-out to each node's identified sub-block (it now
  reports a number only there). This removes the artifact — and the gate still fails, which is the
  point.

- **Calibration (the blocker).** Even on the node's OWN foreground stick, coverage is ~0.1 at a
  short chain and ~0.40 at a properly converged one — better mixing helps but does not calibrate.
  Two pathologies persist that more samples cannot fix: (i) the intervals are **overconfident**
  (too tight — they miss even when the sign is right); and (ii) the posterior **mean is attenuated
  toward zero** (systematically undershoots the planted magnitude; occasionally sign-flips when the
  truth is small). Recovery-correlation stays positive (~0.6): the ordering is right, the absolute
  calibration is wrong. **This reproduces the pinned-interval overconfidence we saw under mean-field
  VI — now under the exact co-sampled Gibbs.** Moving VI → Gibbs did not cure it.

So this is precisely the branch you pre-registered: scarce/identified coverage failing under exact
Gibbs is where the informative (LKJ / half-t provenance) priors have to earn their place. The gate's
acceptance test is marked expected-fail with a pointer to the finding — the threshold was not
loosened; it flips back the moment coverage is earned.

## The decision I'd put to you

The overconfidence (i) is a width problem, and the half-t-on-the-triangulated-scale you proposed
addresses it directly. But the mean attenuation (ii) is a SEPARATE failure: a prior on the scale
won't move a biased point estimate. The attenuation points at the depth-scaled ridge itself
shrinking the offset toward zero — i.e. the prior needs to live on the OFFSET/increment, not only on
Σ, and its center/strength is what determines whether the point stops undershooting. So: do you want
the provenance prior placed on the offset increments (a hierarchical shrinkage-with-a-measured-center
on the node coefficients) in addition to the half-t on the scale — or do you read the attenuation as
something a Σ-side prior alone will fix once the widths open up? That fork decides where the next
build puts the prior.

## In practical terms

We can now say, with a calibrated *interval*, how a child community's day of tweets differs from its
parent's — but only once we fix the two calibration failures: right now the engine reports the
*direction* of that difference correctly and refuses to guess where the data can't (both good), yet
its confidence intervals are too narrow and its point estimates are pulled toward "no difference." A
sharper interval that's honestly wide where it should be is the next step, and it's the priors that
buy it.
