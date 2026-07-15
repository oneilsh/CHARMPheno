# Follow-up: your prior-misspecification diagnosis is confirmed — but it's dominant, not sole

Before building the τ-hierarchy I ran the cheap falsifiable check your reframe implies: if the
coverage failure is prior misspecification, then drawing the planted increments from the fitting
prior itself should send coverage to nominal with no code change. The fit's increment prior is
`MN(0, diag(penalty)⁻¹, Σ)`; with `Σ_true = I` that's `N(0, 1/penalty_u)` per node. So I re-ran the
gate with two plants, everything else identical:

- **mismatch** (the current gate, increments ~ N(0, 4), wider than the ridge): own-identified-stick
  coverage **0.00**
- **matched** (increments ~ N(0, 1/penalty_u), Σ_true = I): coverage **0.20** (R=10)

So: **directionally confirmed, and dominant** — matching the increment prior removes most of the
total-zero coverage, exactly as your "posterior correct about the wrong prior" argument predicts.
But it does **not** reach nominal (0.20 ≪ 0.90; at R=10 that gap is statistically real — a true 0.9
essentially never shows 2/10). **Increment-scale misspecification is necessary but not sufficient:**
a substantial residual under-coverage persists with the increment prior matched.

The most likely second source is Σ, and it bears directly on your lane assignment. My "matched" draw
assumed the fit's Σ ≈ I, but the engine *samples* Σ (nested block-IW with the gate structure). If it
systematically mis-estimates the residual scale, the interval width is off **independently of the
increment prior** — which is exactly what a residual under-coverage-with-matched-increments looks
like. (The other candidate, ψ inferred from tokens rather than observed, I'd discount: the chain
samples ψ each sweep, so that variance is already integrated.)

This refines rather than contradicts the plan. The learned-τ-per-tier hierarchy is still the right
increment-scale fix, and your two-cell plant is the exact instrument — your well-specified cell
("draw from N(0,τ²), fit with τ learned; must pass or the hierarchy is mis-built") is the clean
version of what I ran. **My check predicts it may not hit nominal even with learned τ**, and if it
doesn't, the residual is the Σ/estimation source, cleanly separated — the two-cell design will show
it as a well-specified cell that still under-covers.

The question that changes the build order: does this promote the **Σ-side half-t from a parallel
lane onto the critical path** — i.e. increment-τ and Σ-scale are *jointly* the calibration fix,
sampled in the same sweep and validated on the same well-specified cell — rather than the increment
hierarchy first and Σ later? Or do you read the 0.20 residual as something the learned-τ (with its
wider realized scale feeding back into the Σ draw) will absorb on its own once it's in the chain?

One number to pin your prediction against, if useful: I can fit with Σ held at the true I and the
increments matched, and see whether coverage then reaches nominal — that isolates Σ-estimation as
the second source in one run before either of us commits the build.
