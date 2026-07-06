# Predictive-gain presence / depth / prominence — converged design (build spec)

Supersedes `2026-07-06-responsibility-presence-metric-proposal.md`. Same data shape: documents are
user-day bag-of-words aggregates; a gated logistic-normal topic model (background block ∪ per-group
foreground blocks, hard-masked); per-document θ = softmax(η), η ~ Normal(Γᵀx, Σ). This is the design
we build, after Fable's predictive-contrast pivot. It is domain-agnostic and doubles as the
implementation spec.

## 1. The principle: measure a predictive CONTRAST, not a functional of the point posterior

Every axis in the previous proposal — presence, depth, the smoothing-floor correction, the
uncertainty integration — was a functional of a biased point posterior: infer θ̂ (or its Laplace
covariance), read a quantity off it, discover the MAP/Laplace bias contaminates that quantity, add a
correction. Three quantities, three biases, three patches. The tower is inherent to the CLASS of
estimator, not to any patch. So the principled move is not a better correction — it is to change the
class of quantity to one whose bias structure is benign.

We already proved that class works, on the scale. A posterior MOMENT (the variance-EM: between-doc
variance of the modes) inherited the Laplace shrinkage directly — compressed 30–60%, ran away. A
predictive CONTRAST (held-out predictive LL) was unbiased and bounded, because the shrinkage appears
in both terms and cancels as common mode: the truth pays the same prior penalty the estimate does,
so the difference is clean. Presence and depth get the same treatment, one level down.

## 2. The core quantity: leave-one-topic-out held-out predictive gain

For document d, split its tokens into visible / held-out (the same seeded split the scale sweep
already produces). Let LL(d | S) be the held-out predictive log-likelihood when θ is inferred over
allowed topic set S on the visible tokens. Define

    Δ_k(d) = LL(d | allowed) − LL(d | allowed \ {k})

— the nats of held-out predictive power topic k contributes to d over and above every other topic.
Up to the inference approximation, Δ_k(d) is the pointwise conditional mutual information between
topic k and d's held-out tokens given the other topics: the predictive information k carries about
the document. No threshold anywhere; a continuous quantity with a unit (nats).

## 3. The three readouts, all predictive-gain

- **Presence** (headline, "how widely") — the fraction of documents where Δ_k clears a
  model-GENERATED null band (see §6, permuted-topic ablation), and/or the mean gain in nats. The
  threshold is replaced by a calibrated noise floor the model produces itself.
- **Depth** (companion, "how much") — topic k's share of the document's total predictive structure,
  aggregated as (Σ_d Δ_k(d)) / (Σ_d Σ_j Δ_j(d)) — numerator and denominator aggregated SEPARATELY
  across documents, never a per-document ratio (§5, caveat 2).
- **Prominence distribution** (replaces the θ̂ topic-mass histogram) — the aggregate binned
  distribution of Δ_k(d) across documents. The previous "topic mass distribution" was the histogram
  of the biased point θ̂_k; this is its principled replacement, on the same predictive-information
  footing as presence and depth.

Depth-as-share measures UNIQUE contribution, not raw attribution: a broad topic that overlaps others
ablates cheaply (the others cover for it) → low depth; a niche topic nothing else explains hurts when
removed → high depth. The "common topics get less depth, niche topics dominate their documents"
intuition falls out by construction, because the contrast discounts the overlap that raw
responsibility double-counted.

## 4. What this RETIRES

Not augments — retires: the responsibility/plug-in presence metric; the θ̂ topic-mass histogram; and
the whole diagnostic battery those needed (the smoothing-floor measurement, the S-sample posterior
uncertainty integration, the saturation caveat, the separate depth construction). Each dissolves
because the contrast never contained the bias:

- **The smoothing floor auto-subtracts.** A topic with none of its tokens in d predicts d's held-out
  tokens no better than the rest, so Δ_k ≈ 0. We no longer measure a floor and subtract it; the
  contrast never contained it.
- **Uncertainty weighting is intrinsic** — no posterior sampling. A thin document has little held-out
  signal, so every Δ_k is small and its evidence is automatically light. The "don't let a 3-token
  document speak with a 300-token document's confidence" property comes from the amount of held-out
  data, not a bolted-on posterior variance.
- **Depth stops fighting saturation.** Predictive gain does not saturate (more tokens give more gain,
  honestly), and the share normalizes length out entirely.

## 5. The honest limit + the two load-bearing approximations

**What cancels vs what does not.** The common-mode Laplace shrinkage — the roughly-uniform pull of
every θ̂ toward the prior mean, exactly the first-order effect that compressed the plug-in metric —
cancels. What does NOT cancel is DIFFERENTIAL bias: if topic k's inference is biased differently from
the topics it is compared against. That is second-order, and it is now the ONLY thing the calibration
plant must test (§7). This is not provably zero-bias; it is "the dominant bias is removed by
construction, and the residual is a single measurable question."

Two approximations carry weight and must be validated, not assumed:

1. **One Newton step may under-capture the "covering" for high-mass topics.** The redistribution that
   lets other topics cover for an ablated k is only partially realized by a single warm-started step
   when θ̂_k is large — so Δ_k can be UNDER-estimated for exactly the dominant topics. Mitigation:
   take 2–3 steps (or a convergence check) when θ̂_k exceeds a small bound; make the step count one of
   the things the calibration plant measures.
2. **Depth-share is unstable per document.** Δ_k / Σ_j Δ_j blows up when the denominator is near zero
   (a document with little held-out signal — the case the contrast correctly calls uninformative). Do
   NOT ship a per-document ratio; aggregate numerator and denominator separately across documents and
   divide the sums (§3).

## 6. Mechanism & distributed implementation (~2× the single-inference pass)

It is the same embarrassingly-parallel-over-documents idiom every per-document pass here already uses
(`mapPartitions(local).treeReduce(combine)`; cf. `corpus_theta_gated_rdd`, the scale sweep). Per
document, locally (β, Γ, Σ-at-c* broadcast):

1. Seeded visible/held-out split — REUSE the sweep's split + seed convention.
2. Infer the full mode θ̂ over the allowed (gated) set A on visible tokens — the inference we already
   do — AT THE CALIBRATED SCALE c* (ADR 0034 addendum: the mode AND the Hessian/covariance must be at
   c*, never the unit fit scale).
3. Score LL_full = held-out predictive LL under θ̂.
4. For each k in A: warm-start from θ̂, drop k, take ONE Newton step → θ̂₋ₖ; score LL₋ₖ;
   Δ_k(d) = LL_full − LL₋ₖ. (Plus a small number of PERMUTED-topic ablations — k's β shuffled across
   the vocabulary — to get the null-Δ band for presence.)
5. Accumulate per-topic partials: Σ Δ_k, Σ(Δ_k over Σ_j Δ_j numerator/denominator separately),
   null-band counts, the prominence-histogram bin counts, a per-topic length-correlation
   accumulator, and a dedup-variant Δ (§7). Foreground topics accumulate WITHIN their group.

`treeReduce` sums the per-topic accumulators → corpus aggregates.

**The cost trick (why it is ~2× a single inference, not |A|×).** Naively step 4 is |A| ablation
re-inferences (|A| ≈ 40–60, gated, not all K). Instead reuse the full mode's Hessian factorization
(the Laplace covariance we already compute): each ablation is "remove one coordinate + one Newton
step" = a rank-1 DOWNDATE of that factorization (O(|A|²)) plus one solve, so all |A| ablations
together cost ~O(|A|³) — the same order as the one full inference. Net per document ≈ 2× the current
single-inference θ-histogram pass. The downdate approximates the softmax-renormalization change from
dropping a topic — fine for one Newton step, but it is an approximation to validate (fallback: cold
one-step solve at |A|× cost, still tractable). Vectorize the |A| ablations in numpy; sample documents
to a cap (the aggregate is a mean); one split, not three.

**Export (terms-of-service constraint).** Only per-topic AGGREGATES leave — the (presence, depth)
scalars, the aggregate prominence histogram (a binned summary over documents, like the θ̂ histogram
before it), and the per-topic diagnostic scalars (§7). No per-document quantities. Δ_k is a difference
of two noisy LLs so it is noisy per document, but the aggregate mean is clean and per-document noise
never leaves — the ToS constraint helps here.

## 7. Validation — collapsed to one known-truth question + real-corpus checks

- **Slim calibration plant** (the one thing needing known truth): ONE synthetic corpus, planted
  per-topic membership, three document-length bands. The single question: does predictive-contrast
  presence recover planted presence with MATERIALLY LESS length-dependence than the plug-in version,
  and how does its residual error move with length? This tests the differential bias (§5) and the
  one-Newton-step approximation — the only failures that survive the contrast. Not a factorial
  harness; a sharp instrument aimed at the second-order question.
- **Real-corpus triangle** (upgraded): per-topic (NOT pooled) length-correlation — broad topics and
  narrow topics have very different length exposure, and a pooled number averages away exactly the
  per-topic exposure you need when a rate drifts over time; the floor as the null-Δ reading; and the
  READ-THE-DOCUMENTS internal audit — sample documents at high/middle/low presence for a few topics
  and have a human judge whether the ranking tracks "this document is actually about that." That
  audit is internal (nothing ships, ToS-safe) and is the ONLY check in the battery that can catch "the
  metric is well-calibrated to a model that mis-describes the documents." Precision-at-k by eyeball is
  the ground truth.
- **Dedup burstiness diagnostic** (free, real data): emit a unique-token variant of the gain (n_w
  capped at 1) next to the raw-count version, per topic. Topics where the two diverge are the
  burstiness-sensitive ones, quantified on the real repeat distribution with zero modeling
  assumptions. Small divergence → the parked misspecification test loses urgency; large divergence on
  niche topics → prioritize it (and consider sublinear count weighting n_w → log(1+n_w), but only if
  the diagnostic says so).
- **Parked follow-on**: deliberately-misspecified synthetic generation (non-logistic-normal θ, bursty
  tokens) for the SHAPE of the failure — no longer load-bearing for the ship decision.

## 8. Build order

1. `spark-vi`: `corpus_predictive_gain_gated_rdd` — per-document LOO predictive gain at c* (reuse the
   held-out split; warm-start one-Newton-step ablations via Hessian-factorization downdate; permuted
   null; within-group accumulation); returns per-topic aggregates (mean gain, depth num/denom,
   prominence-histogram bins, null-band, per-topic length-correlation, dedup-variant gain).
2. Unit tests on fixtures: Δ correctness on a tiny corpus; downdate-ablation vs cold-solve agreement;
   within-group denominators; the multi-step trigger for high-mass topics.
3. The slim calibration plant (§7) — predictive-contrast vs plug-in recovery across length bands.
4. Export: per-topic presence/depth/prominence aggregates + diagnostic scalars (ToS-safe), wired into
   the bundle builders (both, parity).
5. Frontend: the (presence, depth) plane + the prominence distribution, replacing the τ-threshold
   prevalence readout AND the θ̂ topic-mass histogram.
6. Real-corpus diagnostics readout + the read-the-documents audit before trusting the numbers.
