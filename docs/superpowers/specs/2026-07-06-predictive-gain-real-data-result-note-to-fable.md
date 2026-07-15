# Predictive-gain on real data: the smoother runs, and what the numbers say

Follow-up to the smoothing-floor note. We deployed the background-unigram smoother (length-aware
back-off, ε = λ/(λ+n), λ=1, plus a small uniform floor on the marginal so no token has zero back-off
mass) and ran it end-to-end on the real corpus (gated logistic-normal topic model: shared background
block ∪ per-group foreground blocks; ~48k documents, 60 topics, scale pinned at c* ≈ 4.6). The bundle
now self-reports its provenance, so we can confirm from the artifact alone: smoothing was active. Here
is what we found, and where we want your read before we call it.

## 1. The smoother works, but it barely moved the aggregates

Confirmed active, and the per-topic AGGREGATES are essentially unchanged from the un-smoothed run
(presence median 0.341 vs 0.340; mean-gain median 5.64 vs 5.61). What DID move is the extreme tail: the
single largest per-document Δ and the worst-case downdate discrepancy. So the smoothing floor we were
worried about was a TAIL effect — it inflated a handful of pathological (long, or niche-signature)
documents — not the driver of the per-topic aggregates. Our earlier "contamination" alarm was really
about the tail and the reliability audit, not the headline numbers. Useful to know, and it means the
plug-in-vs-contrast framing was sound; the floor was a second-order tail artifact, now bounded.

## 2. The aggregates look like they measure something real — and it is NOT prevalence

The leave-one-topic-out held-out mean-gain per topic:
- correlates **+0.42 with topic coherence** (the NPMI coherence score) — better-formed topics carry
  more held-out predictive gain. Sanity check passes.
- correlates **−0.81 with topic prevalence** (corpus-mean topic mass) — mean-gain is HIGH for rare/niche
  topics and LOW for common ones.

That −0.81 is, we think, the intended "unique contribution" behaviour of a leave-one-out contrast: a
common topic whose tokens other topics (or the background) also explain ablates cheaply → low gain; a
niche topic nothing else explains hurts a lot when removed → high gain. It is the opposite of a
prevalence/size readout, which is exactly what we wanted depth to be. mean-gain also discriminates
cleanly (0 → 27 nats across topics).

## 3. The open problem the smoother did NOT fix: the one-step downdate is unreliable

The cold-vs-fast audit (exact re-inference vs the one-Newton-step Hessian downdate) has a worst-case
per-document discrepancy of ~**1337 nats** on the real corpus. The smoother did NOT reduce this — so it
is NOT the floor cliff we hypothesised; it is the one-step approximation genuinely failing on some
documents (your Q5, and our Task-4 caveat about the covering redistribution being large for high-mass
topics, materialising on real data). We currently export only the MAX discrepancy; we are adding the
MEAN so we can tell whether this is a handful of pathological documents (aggregates over 48k docs are
then fine) or a widespread bias (aggregates are suspect). That certification run is queued.

## What we would value your read on

1. **Presence below one-half.** presence (fraction of a topic's documents with Δ_k > 0) has a median of
   ~0.34 — i.e. the median topic adds POSITIVE held-out lift in only ~1/3 of the documents it is
   assigned to; in the other ~2/3, removing it slightly IMPROVES held-out prediction. Is that a real,
   interpretable finding (an over-complete gated model routinely assigns a topic to documents where it
   does not help held-out prediction), or is it a symptom of downdate noise flipping the sign of a
   near-zero Δ? How should we read presence sitting well below 0.5 — as signal, or as a warning that Δ
   is dominated by noise around zero?

2. **Is mean-gain just inverse frequency?** The −0.81 with prevalence is strong. Is that the honest
   unique-contribution signal, or is mean-gain over-dominated by rareness — a niche topic's signature
   tokens are unexplained-by-others, so its per-document gain is inflated simply because it is rare,
   making mean-gain effectively "1 / frequency" wearing a predictive-nats costume? If the latter, is a
   length- or frequency-normalised gain the right readout, or does that throw out the baby?

3. **The downdate, revisited.** With the divergence confirmed as NOT a floor artifact, which way do you
   lean: (a) accept the fast aggregates IF the mean discrepancy turns out small (rare bad docs), (b)
   fall back to the exact cold solve for the aggregates (expensive but tractable on a document sample —
   the aggregate is a mean), or (c) a better one-shot downdate (a few Newton steps / a trust region on
   the high-mass documents where the covering move is large)? We built the audit precisely so it could
   tell us this; we want your prior on the fix before we spend the compute.

4. **Held-out vs full-document attribution.** Now that there is a proper marginal back-off, is
   leave-one-topic-out on a HELD-OUT split still the right contrast for presence/depth, or should the
   per-topic aggregates use FULL-document attribution (no held-out split — score the contrast on all of
   a document's tokens), reserving the held-out split only for the scale calibration? The held-out split
   halves the signal per document and adds sampling noise to every Δ; if the back-off makes a
   full-document contrast well-behaved, dropping the split might sharpen presence/depth considerably.

Net: the estimator runs, the smoother is a real (if second-order) fix, and mean-gain looks like a
genuine unique-contribution signal. The two things we are unsure of are whether presence-below-half is
signal or noise, and whether the one-step downdate is trustworthy in aggregate — the second we will
answer empirically with the mean-discrepancy certification; the first we would value your read on.
