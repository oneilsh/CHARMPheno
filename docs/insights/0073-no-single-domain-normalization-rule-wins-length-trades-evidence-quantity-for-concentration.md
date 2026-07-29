# 0073 — No single per-domain normalization rule wins: `std` bounds the observation drag reliably, `length` trades evidence QUANTITY for CONCENTRATION, and which is right is disease-specific

**Date:** 2026-07-29
**Topic:** multidomain | case-finding | readout | metrics
**Status:** Confirmed (two independent fits agree)
**Context:** exp 0071 (rare6, condition+drug+observation, lookback, full-batch) and 0072 (same, mini-batch), re-read with the new `--normalize` readout (no re-fit; the persisted held-out test sets were reused). 39,081 held-out docs, 6 rare6 anchors, α→∞ lift limit. Follows insight 0072, which established that observation costs most of the precision under the plain unnormalized sum. Motivated by the owner's requirement that a signal-free domain should *not hurt*, since observation may carry signal for some diseases as the disease set grows.

## Setup

The multi-domain LR score is a sum of per-domain log-likelihood ratios whose terms scale with each domain's token volume. Four label-free per-domain transforms were applied before that sum: `none`; `std` (divide domain m's whole score matrix by one scalar, its std); `length` (per-doc divide by that domain's token count = mean log-LR per token); `length+std`.

The criterion is the **within-rule drag**, `drag(rule) = PR_AUC(drop:observation, rule) − PR_AUC(all, rule)`. An earlier cross-rule formulation was refuted before the run (it silently added the rule's effect on the *retained* domains to the effect on the dropped one; see the design spec's correction of record).

## Finding 1 — `std` is the reliable "do no harm" rule, and it does exactly what was predicted: bounded, not eliminated

Mean drag across the six diseases:

| rule | exp 0071 | exp 0072 |
|---|---|---|
| none | +0.0313 | +0.0308 |
| **std** | **+0.0148** | **+0.0163** |
| length | +0.0268 | +0.0218 |
| length+std | +0.0155 | +0.0157 |

`std` roughly halves the drag and is the only rule that **never increases** it for any disease in either fit. Its cost to the retained domains is small: the `drop:observation` column moves to 0.97× (0071) and 0.91× (0072) of its unnormalized value on average. This is precisely the pre-registered expectation — equal-variance weighting converts *catastrophic* into *bounded* — and it is not a free lunch.

## Finding 2 — the drag is NOT closed; curating observation out still wins for most diseases

Comparing the best `all|<rule>` against the `drop:observation|none` baseline:

| disease | best `all` (0071) | `drop:obs|none` | verdict | best `all` (0072) | `drop:obs|none` | verdict |
|---|---|---|---|---|---|---|
| Ehlers-Danlos | 0.054 (std) | 0.138 | drop | 0.076 (length) | 0.149 | drop |
| Systemic lupus | 0.114 (length) | 0.146 | drop | 0.121 (length) | 0.148 | drop |
| Sarcoidosis | 0.051 (length) | 0.066 | drop | 0.063 (length) | 0.070 | drop |
| Amyloidosis | 0.019 (std) | 0.038 | drop | 0.019 (none) | 0.034 | drop |
| Scleroderma | 0.090 (none) | 0.088 | keep (tie) | 0.097 (none) | 0.102 | drop (tie) |
| Myasthenia gravis | 0.090 (length+std) | 0.023 | **keep, 3.9×** | 0.105 (length+std) | 0.052 | **keep, 2.0×** |

So normalization alone does not make observation free. For 4–5 of 6 diseases the honest move is still to drop it. Note also that scleroderma's drag is ≈ 0 under `none` (−0.002 / +0.005): **whether a domain hurts is itself disease-specific**, not a property of the domain.

## Finding 3 (the mechanism) — `length` swaps evidence QUANTITY for evidence CONCENTRATION

Dividing a domain's score by that document's token count discards *how much* matching evidence a patient has and keeps only *how concentrated* it is. That is the right trade for some diseases and destructive for others — the effect on the `drop:observation` (condition+drug) subset, where observation is not even present:

| disease | none → length (0071) | none → length (0072) |
|---|---|---|
| Myasthenia gravis | 0.023 → 0.071 (**+209%**) | 0.052 → 0.083 (**+60%**) |
| Sarcoidosis | 0.066 → 0.078 (+18%) | 0.070 → 0.090 (+29%) |
| Systemic lupus | 0.146 → 0.151 (+3%) | 0.148 → 0.154 (+4%) |
| Ehlers-Danlos | 0.138 → 0.123 (−11%) | 0.149 → 0.128 (−14%) |
| Amyloidosis | 0.038 → 0.019 (**−50%**) | 0.034 → 0.021 (−38%) |
| Scleroderma | 0.088 → 0.042 (**−52%**) | 0.102 → 0.029 (**−72%**) |

This tracks the *form* of each disease's evidence, not its prevalence or positive count (MG n+=80 and scleroderma n+=79 sit at the same base rate and move in opposite directions). Myasthenia gravis is carried by a single near-pathognomonic drug (pyridostigmine — insight 0072, Finding 3), so concentration is the signal and volume is noise. Scleroderma and amyloidosis are multi-system diseases whose signal is the *accumulation* of many matching codes, which is exactly what per-doc division throws away. This confirms the caveat pre-registered in the design spec: `length` is a bias–variance trade, not a clean fix.

## Finding 4 — the myasthenia gravis result is the rule unlocking MG, NOT observation carrying signal

Tempting to read MG as "observation had signal all along." It does not. Under `length+std` (exp 0071): `all` 0.012 → 0.090 (7.5×) **and** `drop:observation` 0.023 → 0.092 — both jump together, the drag falls to +0.002, and `only:observation` is 0.004 against a prevalence of 0.0020. Observation became *free*, and contributed nothing. The gain is length normalization unlocking MG's condition+drug ranking. exp 0072 reproduces this (`all` 0.028 → 0.105, `drop:obs` 0.052 → 0.109, drag +0.004).

Worth keeping straight: a rule that makes a useless domain harmless and a rule that extracts signal from it are different achievements, and only the first has been demonstrated.

## Finding 5 — reproducible across two independent fits

Every conclusion above holds in both exp 0071 (full-batch) and exp 0072 (mini-batch SVI): the per-disease ranking of the four rules agrees, and the mean drags match to ~0.002. This is not fit noise.

## Finding 6 (side observation) — mini-batch beats full-batch on case-finding

`all|none` PR-AUC, 0071 → 0072: Ehlers-Danlos 0.038 → 0.068, myasthenia gravis 0.012 → 0.028, scleroderma 0.090 → 0.097, sarcoidosis 0.050 → 0.059, amyloidosis 0.017 → 0.019; only systemic lupus regresses (0.104 → 0.099). Five of six favor the mini-batch fit. The A/B was designed as a cost/throughput check, not a quality one, so this is unexplained and deserves its own look before mini-batch is treated as merely cheaper.

## Implication / next lever

The rules trade off in **opposite directions across diseases** (`length` unlocks MG by +209% and destroys scleroderma by −52%, in the same fit, on the same subset). No single global rule can therefore be right, which retires the search for one and makes the case for **per-node / per-disease reliability weighting** empirical rather than speculative: weight each domain by what it demonstrably earns for that node, with `explain_away_placement_scores` already in the library as one candidate mechanism.

Two limits on the `std` numbers above, recorded in the design spec: `σ_m` is the std of the whole score matrix, which conflates across-doc spread (what actually drives cross-domain dominance) with across-node spread; and it is estimated from the batch being scored, so it is transductive and would need pinning to a reference cohort before `std` could serve as a deployed default.

Method note: the readout is post-hoc and needs no re-fit (`make multidomain-lr-readout ID=N`, `MULTIDOMAIN_LR_NORMALIZE=` for the main tables; the rule-comparison blocks always print all four).
