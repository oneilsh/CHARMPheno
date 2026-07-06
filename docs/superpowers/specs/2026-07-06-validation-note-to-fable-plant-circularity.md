# Note to Fable — pushing on the validation plant before we build it

Everything in your presence/depth review we're taking as the design. One thing we want to circle on
before building: the factorial generate-then-re-infer PLANT. We think it validates less than it
looks like it does, and that most of what it's reaching for is better done on the real corpus. Want
your read before we commit code to it.

## The circularity

If we generate the synthetic corpus from the SAME logistic-normal gated model the metric assumes —
θ = softmax(η), η ~ Normal(Γᵀx, Σ), tokens ~ β — then recovering the planted membership is
guaranteed by construction. We'd be assuming the model is true and then confirming the metric under
that assumption. That tests whether the ESTIMATOR is coded correctly; it cannot test whether
"presence" means anything on real documents, because the only way it could fail is a code bug, not a
modeling problem. A plant that can only fail on a bug is a unit test wearing a lab coat.

## What the plant honestly buys (narrow)

Two things, and only two:
1. **Estimator correctness** — the S-sample posterior integration converges, the log-space product is
   right, within-group denominators are right. This is real but small: a handful of unit tests on
   tiny fixtures, not a factorial harness.
2. **The MAGNITUDE of the two known artifacts** — the false-presence floor (from β's smoothing) and
   the length-saturation of the ≥1-token event. These are genuine and we need their numbers. But
   they are better measured on the real corpus than on a plant (below).

## The real-corpus alternative (same distributed pass, more honest)

Everything the factorial plant was crossing for is directly observable on the real data we're
re-exporting anyway:

- **False-presence floor** — take a NARROW topic (β concentrated on a few signature tokens) and
  compute its presence on the real documents that contain NONE of those signature tokens. That
  average IS the empirical floor — under the real β's actual smoothing, not a synthetic ε.
- **Length-dependence** — correlate per-document presence with document length across the real
  corpus. That quantifies how much of a topic's rate is volume vs content, on the real length
  distribution (thin tail included), so any future presence drift can be attributed to length drift.
- **Validity — the question the plant can't answer** — for a narrow topic, presence should closely
  track the OBSERVED document-frequency of its signature tokens: a completely model-free, directly
  countable quantity. If presence ≈ "fraction of documents that literally contain the signature
  tokens," the metric is doing something real. If it diverges, that's a red flag — and it's exactly
  the failure the plant would hide, because the plant assumed the model that produced the divergence.

Our inclination: **drop the factorial plant**; keep only small synthetic unit tests for estimator
correctness; and put the weight on those three real-corpus diagnostics, emitted alongside the
(presence, depth) headline in the same pass, so we SEE the floor and the validity check when we run
the export — not just the numbers we hoped for.

## The one synthetic test that WOULD be non-circular

The only synthetic validation that adds something the real-data checks can't: generate from a
DELIBERATELY MISSPECIFIED process — θ that is not logistic-normal (e.g. a sparse/bursty membership),
tokens with within-document burstiness the model doesn't assume — and check that presence still
tracks the true (planted) membership under misspecification. THAT is real robustness evidence,
because it can fail for a modeling reason, not just a bug. We'd park it as a follow-on rather than a
demo blocker, but it's the piece of the plant worth keeping in spirit.

## One hard constraint that removes a piece of your design

We can only export per-topic AGGREGATES (and k-anon-safe distributions — binned, small cells
suppressed); no per-document quantities ever leave. So the per-document presence distribution you
suggested shipping is out. Two upshots, both in our favor: it MOOTS your per-document MC-budget point
(the ~±0.09-at-S=32 per-document error just averages out in the corpus mean, so S≈32–64 is plenty for
the aggregate), and the three real-corpus diagnostics above are all computed inside the pass and
emitted as per-topic scalars (a floor, a length-correlation, a signature-match), never as rows. If we
ever want to convey the spread of per-document presence we'd ship a k-anon-safe per-topic histogram,
the same treatment the θ̂ distribution already gets.

## What we'd value your read on

- Is the real-corpus triangle (floor on non-containing documents / length-correlation /
  signature-token cross-check) the right validation basis, or does the factorial plant do something
  essential we'd be throwing away?
- Is the signature-token cross-check as strong a validity gate as we think, or are we fooling
  ourselves there too (e.g. because β's signature tokens are themselves fit to the corpus, so some
  circularity leaks back in — a narrow topic's signature IS the frequent co-occurring set)?
- Is the misspecification-generation robustness test worth prioritizing over shipping, or genuinely a
  follow-on?
