# Coverage decomposition: the residual is none of the three candidates — it's latent-state inference

Short version: I ran the decomposition you designed, then two follow-ups it forced. The residual
under-coverage is **not** prior scale, **not** Σ-estimation, and **not** coordinate frame. All three
of the lanes on your table are excluded by direct measurement. The failure lives one level upstream,
in how well the latent per-item state is inferred from the observations — and it splits into two
separate axes that need different fixes. This refutes the τ-per-tier + joint τ+Σ hierarchy as the fix
for this particular failure, so I've held that build.

## The decomposition (your R=50 run, both marginal and joint)

Well-specified plant throughout: the planted increments are drawn from the fit's **own** prior
N(0, penalty⁻¹) and the true residual covariance is the identity, so by construction nothing here can
be prior or Σ misspecification. Identified coordinate is a 2-dimensional block (so joint ≠ marginal is
a real test). Coverage of its 90% intervals:

| cell | marginal | joint |
|------|----------|-------|
| Σ **sampled** (current engine) | 0.06 | 0.08 |
| Σ **pinned at the truth** | 0.04 | 0.04 |

Your predictions were: marginal ~0.5–0.7 in the pinned cell if Σ-estimation were the middle chunk;
joint nominal while marginal low if it were coordinate frame. Neither happened. Pinning Σ at the truth
changed nothing, and joint is as broken as marginal. And the earlier "match the prior → 0.20" that
suggested prior misspecification was dominant turned out to be R=10 small-sample noise: at R=50 the
matched-prior coverage is 0.06, statistically indistinguishable from the 0.00 mismatched case. Matching
the increment prior barely moved coverage.

So I looked at *what* the estimate is doing, not just whether it covers.

## What the estimate does: right scale, zero information, six-times-too-tight

Matched plant, Σ pinned, on the identified block: the estimate has the right marginal scale (SD 1.24
vs truth SD 1.16) but is **uncorrelated** with the planted value — best 2×2 linear map from truth to
estimate has R² = 0.07, and the off-diagonal terms are ≈ 0, which rules out a within-block rotation
(a rotation would show up as large off-diagonal / high R²). The intervals are ~6× too narrow relative
to the actual miss. So it is not attenuation-toward-zero and not a frame rotation — it is full-scale
noise inside overconfident intervals.

## The decisive cut: feed the estimator the true latent state

The cleanest experiment, and it's instant (no sampler): hand the estimator's own ridge regression the
**true** latent-state field — the planted closure-sums plus the true residual noise — instead of the
state inferred from the observations. Same design matrix, same penalty, same everything. Result, R=200:

- correlation(estimate, truth) = **0.996** on both coordinates,
- best 2×2 map ≈ **identity**,
- ridge posterior SD = 0.100 = RMS miss 0.100 → **calibration ratio 1.01.**

So the identifiability compiler is right (the coordinate genuinely *is* identified), the ridge, the
matrix-normal draw, the per-coordinate schema, and the interval construction are all correct and
**perfectly calibrated** — *given the true latent state.* The entire failure is upstream, in inferring
the latent state from the observations. This also exonerates the depth-scaled ridge I blamed in the
prior note (same penalty, perfect recovery given the state).

## Two axes, two different fixes

Running the real engine while increasing the observation count per item (the thing that sharpens
latent-state inference):

| observations/item | recovery R² | coverage |
|-------------------|-------------|----------|
| 80 | −0.11 | 0/10 |
| 320 | +0.20 | 1/10 |
| 1280 | +0.51 | 0/10 |

- **Recovery** (the point estimate) is **information-limited**: R² climbs monotonically toward the
  true-state ceiling of 0.99 as each item carries more observations. Under the gated likelihood with
  few observations, each item's latent state on the relevant coordinates is dominated by its prior
  mean (the current coefficient) rather than by its observations, so evidence about the coefficient
  barely propagates into the chain. More observations fix this.
- **Calibration** (the interval) is a **separate, structural** failure: coverage stays ~0 even at the
  point where recovery is decent (R² 0.51). The interval is the residual-regression posterior SD,
  Σ/√N, computed **conditioning on the latent state as if it were observed**. It therefore captures
  only residual-regression uncertainty and *structurally omits the latent-state inference
  uncertainty.* Sharpening the state improves the point but never widens this interval to match — only
  in the true-state limit does Σ/√N become the correct interval. More observations do **not** fix this.

## What I think this means for the build order

The τ-per-tier hierarchy addresses prior scale, and the chain above shows prior scale is not the cause
of this failure — so I don't think the joint τ+Σ build moves this number, and I've held it pending your
read. The two levers that the evidence actually points at are:

1. **Recovery:** more observation-evidence per item, or a stronger-than-observation information source
   about the relevant coordinates, or pooling — anything that sharpens latent-state inference.
2. **Calibration:** the coefficient posterior has to **propagate the latent-state inference
   uncertainty**, not condition on the state as if observed. The co-sampled chain is supposed to do
   this by re-sampling the state each sweep, but because the state is prior-dominated it under-
   disperses, so the emitted interval collapses. This is the real open design question: how to make the
   coefficient interval integrate the state posterior's spread — a proper joint-uncertainty propagation
   rather than a plug-in-the-current-state regression.

The question I'd put back to you: does this reassign the lane from "prior hierarchy" to "latent-state
uncertainty propagation," or do you read the climbing-R² recovery axis as something a stronger prior
still buys us on the calibration side? I don't yet see the mechanism by which it would, given the
true-state cell is already perfectly calibrated with the current flat prior — but you called the design
wall before I did, so I'd rather ask than assume.

One methodological note, since it's a satisfying instrument: the true-state oracle is the cleanest cut
I've had in this whole thread. It converts "coverage fails" from a lament into a coordinate — everything
downstream of the latent state is provably correct, so the entire remaining budget is spent in one named
place. Worth keeping in the toolkit.
