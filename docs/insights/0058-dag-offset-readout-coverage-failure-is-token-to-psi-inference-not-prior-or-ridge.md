# Insight 0058 — The DAG-offset read-out's calibrated-coverage failure (0057) is token→ψ inference weakness under gating, NOT prior/ridge misspecification: given the true ψ field the identical engine recovers at corr 0.996 with perfectly calibrated intervals; recovery is token-limited while calibration is structurally overconfident (the interval conditions on latent ψ as if observed)

**Date:** 2026-07-14
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | read-out | coverage | calibration | gibbs | identifiability
**Status:** Observed
**Relates to / refines:** insight 0057 (the read-out recovers ordering but fails calibrated
coverage; hypothesized the mean attenuation came from "the depth-scaled ridge parameterization
biasing the point, which a prior — not more samples — addresses"). **This insight refutes that
specific hypothesis** and localizes 0057's finding 3 to a different mechanism. Also relates to
0050/0051 (only increments identified; ridge intervals order-right but overconfident), 0044/0047
(mean-field Σ attenuation — a different, now-excluded mechanism), 0056 (design wall under exact
Gibbs). Directly answers the coverage-residual decomposition Fable pre-registered.

**Setting context:** DAG-offset gated PG-STM read-out engine (`dag_readout.py`,
`pg_stm_dag_gibbs.py`), 2-foreground synthetic plant (`TopicBlockPartition` bg K=3, foreground
A×3, B×3; K−1=8), DAG 0→{1,2}, 1→3, 2→4, docs at nodes {1:200, 3:200, 4:250}, so node 3 (child
of 1) is genuinely identified on its own foreground sticks [3,4] via the node-1-vs-node-3
contrast, while nodes 2,4 form the design-wall chain. **Well-specified plant throughout:** the
planted increments are drawn from the fit's OWN prior N(0, penalty_u⁻¹) and Σ_true = I, so any
residual failure cannot be prior/Σ misspecification by construction. Coverage on node 3's
identified block, R=50 unless noted, Σ pinned at truth in the decomposition cells.

## Findings

1. **The coverage residual is neither Σ-estimation nor coordinate frame — both of Fable's
   candidate lanes are excluded.** Decomposition, R=50, matched plant: Σ **sampled** → marginal
   0.06 / joint 0.08; Σ **pinned at truth I** → marginal 0.04 / joint 0.04. Pinning Σ at the
   truth changed nothing (predicted 0.5–0.7 if Σ were the middle chunk). Joint (Mahalanobis-HDR)
   coverage is equally broken, so it is not a rotation with fine marginals. And the earlier
   "matched-prior → 0.20" that suggested prior-misspecification was *dominant* was R=10
   small-sample noise: at R=50 the matched-prior coverage is 0.06, statistically indistinguishable
   from the mismatched 0.00. Matching the increment prior barely moved coverage.

2. **The read-out estimate is right-scale but uncorrelated with the planted increment — genuine
   non-recovery, not a frame rotation.** Well-specified, Σ pinned, n_iter=120: est SD 1.24 vs
   truth SD 1.16 (right magnitude), but corr(est,truth) ≈ 0 and slightly negative;
   corr(est3,truth3)=−0.29, corr(est4,truth4)=−0.36, off-diagonal ≈ 0, best 2×2 linear map
   R²=0.07 with fitted B ≈ diag(−0.37). Off-diagonal ≈ 0 rules out a within-block rotation/swap
   (which would have shown high off-diagonal / high R²). The intervals are ~6× too narrow
   (miss/half-width = 6.4).

3. **The decisive cut: given the TRUE ψ field, the identical engine recovers perfectly and is
   perfectly calibrated.** Feed the engine's own ridge regression C = (WᵀW + diag(penalty))⁻¹ WᵀM
   the planted closure-sum means plus the true Σ=I residual (M = μ + N(0,I)) instead of
   token-inferred ψ (R=200): corr(est,truth) = 0.996 on both sticks, best 2×2 map R² = 0.992 with
   B ≈ identity, and ridge posterior SD 0.100 = RMS miss 0.100 → **calibration ratio 1.01.** So
   the compiler (node 3 IS identified), the ridge, the matrix-normal draw, the per-coordinate
   schema, and the interval construction are all correct and calibrated. **The entire failure is
   upstream, in token→ψ inference** — this exonerates the depth-scaled ridge that 0057 finding 3
   blamed (same penalty, perfect recovery given ψ).

4. **Recovery is token-limited; calibration is a separate, structural overconfidence.** Real token
   engine, n_iter=120, node-3 R² vs doc_len: 80 → −0.11, 320 → +0.20, 1280 → +0.51 (monotone,
   climbing toward the oracle ceiling 0.99). So the point estimate *is* recoverable — it is
   starved of token evidence under the gated likelihood, where each document's foreground-stick ψ
   is dominated by its prior μ = CᵀW_d (the current increment) rather than by its ~80 tokens, so
   token evidence about the increment barely propagates into the chain. But **coverage stays ~0
   even at doc_len=1280 where R²=0.51** (0/10): the interval is Σ/√N computed *conditioning on
   latent ψ as if it were observed data*, so it captures only residual-regression uncertainty and
   structurally omits the ψ-*inference* uncertainty. More tokens sharpen the point but never widen
   the interval to match, so calibration does not follow recovery. Only in the true-ψ limit does
   Σ/√N become the correct interval (finding 3).

## Interpretation

One mechanism produces both failures: when the gated token likelihood is weak relative to the ψ
prior, the co-sampled Gibbs conditions each sweep on an under-informed, under-dispersed ψ. The
increment point is starved (token-fixable — finding 4's climbing R²) and the emitted interval
omits ψ-inference variance (structurally overconfident, NOT token-fixable — coverage flat at ~0).
This is the design wall of insight 0056 relocated: the increment is identified in the *design*
(the true-ψ oracle recovers it at corr 0.996), but not learnable to calibration from the available
token evidence under the gated likelihood.

## Consequences

- **Fable's round-2 reframe (prior-scale misspecification → learned τ-per-tier + joint τ+Σ
  hierarchy) is refuted as the fix for this failure.** τ addresses prior scale; the matched-prior
  + Σ-pinned + true-ψ-oracle chain shows prior scale is not the cause. Building the τ+Σ hierarchy
  would not move coverage. Recommend NOT starting that build on this evidence.
- **The compiler and the read-out schema are vindicated** — correct and calibrated under oracle ψ.
  No change needed there.
- **The real levers are two, and they are different axes:** (i) *recovery* wants more token
  evidence per document / effective sample (doc length, pooling, or a stronger-than-token
  information source about the foreground sticks); (ii) *calibration* wants the coefficient
  posterior to propagate the latent ψ-inference uncertainty — the current per-sweep C-draw treats
  ψ as observed, so its interval is structurally too narrow whenever ψ is inferred rather than
  known. A correct interval must integrate the ψ-posterior spread (which the co-sampled chain
  under-disperses because ψ is prior-dominated).
- The committed coverage test stays xfail (reason now points here as well as 0057); the acceptance
  gate remains honestly unmet, not loosened.

**Does not claim:** anything about real data (synthetic, model-matched). Does NOT claim node 3 is
unidentified — it is identified (oracle corr 0.996). Does NOT claim more tokens fix calibration —
they demonstrably do not (finding 4). The negative slope in finding 2 at n_iter=120 is not
characterized beyond "≈ uncorrelated"; at longer chains (0057's n_iter=250) recovery corr rises to
~0.6, consistent with slow token→ψ propagation, but calibration still fails.
