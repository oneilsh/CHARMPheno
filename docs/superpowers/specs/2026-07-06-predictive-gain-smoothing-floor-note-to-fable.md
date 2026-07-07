# Predictive-gain on real data: a smoothing-floor contamination, and a background-mixing fix

Follow-up to the presence / depth / prominence design you reviewed. We implemented the whole
leave-one-topic-out held-out predictive-gain estimator — the cold reference solver, the
Hessian-downdate one-Newton-step fast path, the distributed corpus aggregation, the permuted-topic
null band, and the cold-vs-fast discrepancy audit — and ran it end-to-end on the real corpus (a gated
logistic-normal topic model: a background topic block shared by all documents, plus per-group
foreground blocks, hard-masked; ~48k documents, 60 topics, scale calibrated at c* ≈ 4.68). The
estimator runs and the pipeline is sound, but the numbers came back contaminated in a way that is
diagnostic, and we want your read on the fix before we commit it.

## What we observed

- The cold-vs-fast **downdate audit blew up**: max per-topic |Δ_cold − Δ_fast| ≈ **1000 nats** over a
  50-document sample. On the synthetic fixtures this discrepancy was ~1e-6.
- Per-document Δ ranged to **~2900 nats**. For a held-out half of ~20 tokens, an honest per-document
  gain cannot exceed a couple hundred nats — so the tail is an artifact, not signal.
- Downstream, **presence and depth stopped discriminating**: presence clustered in a narrow band and
  depth sat at ≈ 1/K (uniform). Consistent with a few artifact spikes dominating every aggregate.

## Diagnosis: the metric is measuring distance to the log-floor

The held-out predictive score is LL(d | S) = Σ_w n_w · log(p_S(w)), where the predictive token
distribution over the allowed topic set S is p_S(w) = Σ_{k in S} θ_k^(S) · β_kw (θ inferred on the
visible tokens, renormalized over S), and we guard the log with a computational floor:
log(p_S(w) + 1e-12).

The gain is Δ_k(d) = LL(allowed) − LL(allowed \ {k}) = Σ_w n_w · [log p_A(w) − log p_{A\k}(w)].

Now take a token w that only topic k explains (a foreground signature token — no other allowed topic
puts meaningful mass on it). Remove k and p_{A\k}(w) collapses toward zero, so it hits the floor:
log(p_{A\k}(w) + 1e-12) ≈ log(1e-12) ≈ −27.6. The token's contribution to Δ_k becomes
n_w · (log p_A(w) − log 1e-12) ≈ n_w · (log p_A(w) + 27.6).

So the magnitude is set by the **arbitrary floor constant 1e-12**, not by the model — a signature (or
bursty) token can inject 20–30 nats each, and a handful dominate the whole document's Δ. Δ is
measuring "how far does removing this topic push a token into the floor," not "how much predictive
information does the topic carry." The same floor cliff makes the objective locally pathological, which
is why the one-Newton-step downdate lands in a different basin from the cold solve (the audit): both
symptoms share one root cause.

## Proposed fix: score against a background-smoothed predictive distribution

A held-out predictive distribution should never assign a token literally-zero probability — that is
what a proper perplexity computation smooths away. The natural smoothing here is to back off to the
corpus unigram (the marginal token distribution m_w), which we already have:

    p_S^smooth(w) = (1 − ε) · Σ_{k in S} θ_k^(S) β_kw  +  ε · m_w

Then removing the only topic that explained w degrades it to **ε · m_w** — a real, bounded
probability — instead of the 1e-12 floor. The k-exclusive token's contribution becomes

    n_w · [ log((1−ε) p_A(w) + ε m_w) − log(ε m_w) ]  ≈  n_w · log( p_A(w) / (ε m_w) ),

which is a genuine **log-gain of topic k over the background/marginal model** for that token — bounded,
interpretable, and (we think) more principled than the current quantity, not less. Δ_k becomes "how
much predictive lift does topic k give over a background-only predictor," which is arguably the thing
presence and depth were always meant to capture.

We expect this to fix both symptoms at once: the magnitudes bound (killing the spike-domination that
flattened presence/depth), and the objective loses its floor cliff, so the cheap Newton downdate
should track the cold solve again (the audit should collapse back toward zero).

## What we'd value your read on

1. **Is marginal back-off the right smoothing** for this contrast, or would you reach for a different
   held-out smoother (Jelinek–Mercer / absolute-discounting from language modeling, a Dirichlet-
   smoothed predictive, etc.)? The mixing weight enters both terms of the contrast, so it does not
   fully cancel as a common mode — it bounds the difference rather than canceling it. Is that the
   right trade, or does it smuggle in a bias we should worry about?
2. **How should ε be set** — a fixed small constant, or calibrated the same way we pinned the scale
   (sweep ε, take the held-out-LL argmax, so the smoother is self-consistent with the data)?
3. **Scale interaction.** c* was calibrated under the unsmoothed (floored) score. Under the smoothed
   score the held-out-LL surface shifts — do we recalibrate c* jointly with ε, or are they cleanly
   separable (c* governs the η prior in inference; the smoother governs scoring)?
4. **The null band** (permuted-topic Δ) is computed with the same score, so it also becomes bounded —
   which should make the presence comparison (real Δ vs null Δ) cleaner. Any reason the permutation
   null interacts badly with marginal smoothing?
5. **Does this subsume the downdate-audit failure**, or should we also harden the one-Newton-step
   ablation independently (e.g. a couple more steps / a trust region) for the still-steep cases?

Alternatives we considered and set aside: the unique-token (dedup) variant we already emit caps
burstiness but does **not** fix the floor — a single k-exclusive token still injects the full 27.6 nats.
Winsorizing / capping Δ per document would hide the symptom without a principled basis. The
background-mixing fix seems to address the actual cause.
