# (a) refuted, (b) confirmed — but it's basin-trapping, not glacial mixing

You called both mechanisms. Here's the adjudication you asked for, and one wrinkle that I think bears
on whether the collapsed update suffices alone.

## (a) — the reporting bug — refuted

The emitted interval already *is* the chain's marginal: the read-out reports quantiles of the
coefficient draws pooled across sweeps, not a per-sweep plug-in of the residual SD. Confirmed two
ways — by reading the read-out (it's `quantile(pooled draws)`), and by the width check I ran two
rounds ago (reported width equals the pooled draw 5–95 spread to three digits). So the last-hiding-
place plug-in isn't the disease. But your instinct was half-right in a sharper way: the pooled draw
SD collapses almost to the **analytic residual-regression floor** (the per-sweep matrix-normal noise,
SD 0.100 for this coordinate). Pooled SD sits at 1.4–2.2× that floor — so the *between-sweep*,
state-driven component of the variance, the part that should carry state uncertainty, is small. The
interval is a genuine marginal; the marginal is just nearly the conditional, because the state barely
moves between sweeps.

## (b) — coupling — confirmed, and it's worse than glacial

I grew the chain 120 → 480 → 1440 sweeps on one fixed well-specified truth, Σ pinned, and measured
the coefficient coordinate directly (not global diagnostics). Truth = [−0.365, 1.517].

| chain length | pooled SD / floor | point bias | autocorr lag-1 |
|---|---|---|---|
| 120 | 1.91 | 0.97 | 0.41 |
| 480 | 1.44 | 1.10 | 0.29 |
| 1440 | 2.19 | 0.87 | 0.42 |

Between-seed spread at 1440 (two seeds): **0.31 on a coordinate whose within-chain SD is 0.12.**

Read it as three tells:

1. **Bias is flat across a 12× range of chain length** — 0.97 / 1.10 / 0.87, no trend toward truth.
   Longer chains do not converge to the right place. This rules out "glacial-but-eventually-correct":
   there is no slow approach, the bias is a fixed feature of where the chain settles.
2. **Lag-1 autocorrelation stays ~0.4 at 1440** — the draws are genuinely coupled, not decorrelating
   with length. The chain is not producing near-independent samples even at 1440.
3. **Between-seed spread (0.31) ≫ within-chain SD (0.12)** — a ~2.6× ratio, i.e. a badly failed R̂ on
   this coordinate. Different seeds settle into *different* biased, tight regions.

Taken together: the coupled (state, coefficient) system doesn't wander slowly toward the truth — it
**locks into a seed-dependent biased basin** and stays there, tight. Each basin is a self-consistent
fixed point (state sampled near the coefficient's implied mean; coefficient sampled from that state
confirms itself), and the warm-start seed selects which basin. The sampler isn't under-resolving the
true posterior; it isn't visiting it. That is consistent with your ridge-lockstep picture, taken to
its limit: the coupling is strong enough that the shared direction isn't merely slow, it's
partitioned into attractors.

## What I think this means for step (2)

The collapsed / marginalized coefficient update still attacks the exact mechanism — the lockstep is
what forms the basins, and integrating the state out of the coefficient draw removes the
self-confirmation. So I'd still build it, and now there's a concrete baseline to beat, per coordinate:
autocorr ~0.4, R̂-fail (between/within ~2.6), bias ~0.9, at 1440 sweeps. The collapsed sampler should
drive autocorr toward 0, between-seed spread toward the within-chain SD (R̂ → ~1), and — via the added
state-posterior term in the design moment — widen the interval to cover. Your true-state oracle
remains the zero-noise regression gate (must reproduce 1.01), and the observation-count ladder is the
pre-registered prediction (well-specified coverage nominal at *every* rung, not just the data-rich one).

The wrinkle I want your read on: the bias is present **within** a basin and doesn't shrink with
iterations, so it isn't only a between-basin averaging problem — each basin is individually biased.
The collapse removes the coefficient's dependence on the specific sampled state, but the augmentation
variables (the PG draws that linearize the likelihood) are still evaluated at a sampled state, so a
biased state can still enter through them. Does that leave a residual bias after the collapse — i.e.
do we also need an interweaving / ancillary-augmentation step (sample the state in two parameter-
izations per sweep) to break the basin selection, or do you expect marginalizing the state out of the
coefficient update to dissolve the attractors on its own? My guess is the collapse gets most of it
because it removes the dominant feedback edge, but the flat within-basin bias is the one thing in this
table I can't yet attribute cleanly, and it's cheap to be wrong about before rather than after the build.
