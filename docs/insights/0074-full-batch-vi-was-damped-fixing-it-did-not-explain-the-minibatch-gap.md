# 0074 — Full-batch VI was silently damped for its whole life; fixing it did NOT explain the mini-batch gap (an honest negative), and the fit log proves topics are dying while the dead-node check says EMPTY

**Date:** 2026-07-29
**Topic:** core | optimizer | multidomain | diagnostics
**Status:** Confirmed (defect + fix), Refuted (my causal hypothesis)
**Context:** chasing insight 0073's Finding 6 — mini-batch (exp 0072) beat full-batch (exp 0071) on case-finding PR in 5 of 6 diseases, in an A/B designed as a cost check, not a quality one. Both experiments re-run on the corrected optimizer (commit 49cd7ad).

## Finding 1 — the defect: `VIRunner` applied a decaying step to full batches

`VIRunner.fit` computed the Robbins-Monro step `rho_t = (tau0 + t + 1)^-kappa`
**unconditionally**, including when `mini_batch_fraction is None`. With the whole
corpus every iteration, `target_stats` is a *deterministic* map `T(params)`, so
`update_global`'s `new = (1-rho)·old + rho·T(old)` moves by `rho·‖T(old) - old‖`. As
`rho_t → 0` the parameters freeze while the fixed-point residual is still large. Two
consequences: the fit lands short of the batch-VB fixed point at any finite budget,
and the relative-ELBO early stop fires on the *vanishing step* rather than on
convergence.

`Σ_t rho_t` diverges for `kappa < 1`, so the damped iteration would reach the same
fixed point *eventually*. **It is a finite-budget defect, not an asymptotic one** —
which is why it survived: nothing about it is wrong in the limit.

Measured on a 1600-doc gated synthetic (random init, 200 iterations):

| schedule | early stop fires | ELBO at stop | ELBO @200 | placement MRR | top2 |
|---|---|---|---|---|---|
| `lr = 1.0` (batch VB) | iter 5 | −389794 | −389536 | **0.910** | **0.988** |
| `(1+t+1)^-0.7` (exp 0071's) | iter 15 | −390532 | −389772 | 0.856 | 0.828 |

Note the second, independent problem visible there: **the full-batch early stop is
far too loose.** `convergence_tol=1e-4` *relative* on an ELBO of ~390,000 is ~39 nats
absolute, crossed at iteration **5** even in the clean run, which then gains another
258 nats. Mini-batch is immune only because it never early-stops.

Fixed: full batch now uses `rho = 1` (batch variational EM). `learning_rate_tau0` and
`learning_rate_kappa` are inert on that path.

## Finding 2 (the honest negative, and the more useful half) — the fix did NOT explain Finding 6

`rho=1.0000` confirmed in both re-run logs. exp 0071 PR-AUC, defective → corrected:

| disease | `all` | `drop:observation` |
|---|---|---|
| Ehlers-Danlos | 0.038 → 0.036 | 0.138 → 0.133 |
| Sarcoidosis | 0.050 → 0.051 | 0.066 → 0.065 |
| Systemic lupus | 0.104 → 0.112 | 0.146 → 0.147 |
| Scleroderma | 0.090 → 0.084 | 0.088 → 0.081 |
| Myasthenia gravis | 0.012 → 0.012 | 0.023 → 0.018 |
| Amyloidosis | 0.017 → 0.017 | 0.038 → 0.038 |

Within ±0.008, no systematic direction. **The prediction that exp 0071 was materially
under-fit, and that Finding 6 was an artifact of a mis-specified baseline, is refuted.**
Insight 0073's Finding 6 stands (see its amendment).

**Why the synthetic overstated it, which is the transferable lesson:** the
reproduction used **random** init, where the seed is far from any optimum and damping
costs a great deal. The experiments use **spectral** init, which starts near a good
solution — so the first few large-`rho` steps that the damped schedule "wasted" were
not carrying much of the work. A synthetic that reproduces a mechanism does not
license a claim about that mechanism's *magnitude* under a different initialization.
The caveat was flagged before the re-run; the re-run confirmed it mattered.

The fix is still correct — it is the regime every SVI-vs-Gibbs gate validates, and the
defect was real — it simply was not the cause of the gap.

## Finding 3 — a coverage gap, not just an instance

Every SVI-vs-Gibbs equivalence gate runs through `tests/_stm_synth.fit_gated_svi_local`,
whose docstring reads *"Full-batch lr=1.0 each iteration = variational EM"*. So the
validated full-batch regime was `rho=1` all along, while the **production** path
(`VIRunner`, used by every cloud driver) used something else. Nothing tested
`VIRunner`'s full-batch step size at all. Fixing the instance without closing the gap
leaves the next one free to happen: a `VIRunner`-vs-`fit_gated_svi_local` equivalence
test is owed.

Two latent issues surfaced in passing, both recorded in code, neither fixed:

- **Resume equivalence is now vacuous** for conjugate models. Under `rho=1` a conjugate
  model reaches its exact posterior in one step, so every run lands on the same fixed
  point regardless of history and `test_auto_checkpoint_then_resume_via_kwarg`'s
  parameter comparison can no longer detect a broken resume.
- **Mini-batch resume replays the wrong draws.** `VIRunner` seeds its sampling RNG once
  per `fit()` from `cfg.random_seed` and consumes draws by `step`, not by the global
  `t`, so a resumed mini-batch fit replays draws 0,1,2 instead of 3,4,5. That matters
  for checkpointed cloud fits, which is a real workflow here.

## Finding 4 — the fit log proves topics are dying, and `dead_node_report` says EMPTY

exp 0071 (K=180), from the per-iteration diagnostics:

```
Σλ_k[m0: min=27.8]   Σλ_k[m1: min=7.17]   Σλ_k[m2: min=8.33]
η_m = 0.005556,  V = [5000, 1291, 1500]
0.005556 × 5000 = 27.78   × 1291 = 7.17   × 1500 = 8.33
```

All three minima sit **exactly** at `η_m · V_m` — the prior with **zero** data assigned.
Topics 102/103/104 are byte-identical at `Σλ = 43.3` with all-zero condition
probabilities. And the sanity read printed `dead-node report: EMPTY (every node
concentrated in >=1 domain; init OK)`.

This is the blind spot noted in insights 0072 and 0073, now **arithmetic rather than a
hunch** — and the earlier diagnosis in those two entries was **wrong**. They said
`dead_node_report` "detects flatness, not mass-starvation, because the spectral seed
plants a peak." But a topic with zero assigned data has an *exactly constant* row
(`λ_k[m][v] = η_m + count`, and η_m does not vary with v), so its peak/mean ratio is
exactly 1 and the flatness test *would* catch it.

The real gap is **granularity**. `dead_node_report` reports per NODE with an
ANY-topic-alive rule: it marks a node alive the moment one of its `tpn` topics
concentrates, then stops looking. With `tpn=5`, a node can carry one live topic and
four sitting at the prior and read "alive" — which is exactly what exp 0071 did.

Fix shipped as `starved_topic_report` in `analysis/cloud/multidomain_cloud.py`,
reported beside the node-level read and recorded in the manifest as `starved_topics`.
It needs no knowledge of η_m: a topic is starved iff `(max - min) ≤ tol · max` in every
domain. exp 0070 shows no fully-starved topics (min row sums 207 vs prior 56, and 47.4
vs 12.8), so this is specific to the deep-DAG K=180 fit rather than universal, and at
large emergent K some slack is expected — the point is to REPORT it instead of reading
"dead-node report: EMPTY" as "every topic carries signal".

## Finding 5 — the LR readout cannot score a single-anchor cohort

exp 0070's readout printed every table empty with `0 rare6 anchors present`. Not a fit
problem: `charmpheno.omop.cohorts.disease_anchors` documents that a single-disease
cohort (diabetes, eds, cancer) yields *one* anchor with **the DAG rooted directly at
it**, so that anchor is engine id 0 — which has no entry in `lay.nodes` and is by
construction not a scoreable node. rare6 works only because its six anchors hang under
a *synthetic* forest root and are therefore ordinary nodes. For a single-anchor cohort
the scoreable targets are the anchor's **children** (the type taxonomy beneath it),
which is what per-node placement means there.

## Implications

- Full batch should be understood as the **reference/oracle path**, not an alternative
  optimizer: deterministic, exact on conjugate models (one `rho=1` step reaches the
  closed-form posterior), and the only path that can answer "has this converged?".
  Framed that way, its correctness must be tested *through the production runner* —
  which is exactly what was missing.
- Finding 6's leading explanation is now a **compute-shape** difference rather than a
  bug: full batch *converged* at iteration 13 (a genuine local fixed point reached from
  the spectral seed), while mini-batch never early-stops and ran 200 × 0.1 = 20 epochs
  of noisy updates. That is SVI-noise-as-regularizer escaping the seed's basin, and it
  fits this arc's history of seed-dependent basins. Testable: run full batch from
  several seeds, or with the early stop disabled, and see whether it reaches
  mini-batch's quality.
- The full-batch early-stop tolerance (`1e-4` relative on ~4e5) deserves its own look;
  it fires while the ELBO is still moving materially.
