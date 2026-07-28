# 0071 — Multi-domain LR readout: condition carries rare-disease case-finding, drug is a mild disease-specific positive, observation is net-negative — and the LR lift beats the observation-dominated θ-mass everywhere

**Date:** 2026-07-28
**Topic:** multidomain | case-finding | readout
**Status:** Confirmed
**Context:** The per-domain LR placement-lift readout (`analysis/cloud/multidomain_lr_readout.py`, branch `multidomain-spectral-init`) run post-hoc on exp 0071 (rare6, condition+drug+observation, lookback, full-batch) and exp 0072 (same, mini-batch). 39,463 held-out docs; 6 rare6 anchors. Per-disease detection = AUC of max-over-subtree(anchor) LR score at the α→∞ lift limit, vs frontier∩subtree. Columns: θ-mass baseline; then LR-AUC for domain subsets {all, only:X, drop:X}.

## Finding

Four results, all robust across the full-batch (0071) and mini-batch (0072) fits (every cell agrees within ~0.02, so the domain story is not an optimizer artifact):

1. **LR ≫ θ-mass for every disease.** `all` (LR, all domains) beats the model's native θ-mass placement everywhere, large for the sharp diseases: Scleroderma 0.95 vs 0.79 (+0.16), Amyloidosis 0.76 vs 0.66 (+0.10). The α→∞ lift recovers disease signal that θ-mass buries under observation's ~58%-of-θ volume (insight 0069: θ-contribution ≡ ω_m·V_m; observation is the high-volume domain). Confirms the single-domain "fork-settler" (LR-AUC ≫ θ-AUC ⇒ signal present but buried) on the multi-domain artifact.

2. **Condition is necessary and nearly sufficient.** `only:condition` ≈ or > `all` for 5/6 diseases (conditions alone match all three domains combined), and `drop:condition` collapses to near chance (EDS 0.60, Scleroderma 0.70). The condition domain carries the case-finding signal.

3. **Observation is net-negative — for all six diseases.** `drop:observation ≥ all` in every row (removing observation *improves* placement): EDS 0.81→0.82, MG 0.75→0.77, etc. `only:observation` is the weakest single domain everywhere (0.53–0.68, barely off chance). Observation has weak intrinsic signal but dilutes the condition signal — consistent with the topic dump showing its vocab is dominated by AoU survey/SDOH/admin/generic-vitals tokens (high volume, low specificity). The best-scoring column in every row is `drop:observation` (= condition + drug).

4. **Drug is a mild, disease-specific positive.** `drop:observation` (cond+drug) ≥ `only:condition` for the diseases with signature drugs: **MG +0.045** (pyridostigmine), **lupus +0.028** (hydroxychloroquine), **EDS +0.019** (POTS meds); ~neutral for scleroderma/amyloid.

Also confirmed: the **overall detection sweep is at chance until α=∞** (0.50 → ~0.72), the expected gate artifact (the hard gate under-represents common codes, so an unshrunk log-ratio over-penalizes them; the signal lives in the α→∞ lift limit only) — the same α-shape the single-domain readout shows.

## Implication

- For rare-disease case-finding, the deployable model is **condition + drug**; observation should be curated (strip `vocabulary_id='PPI'` — the AoU survey vocabulary — and/or cap document-frequency) or dropped.
- **The ω-sweep (SP4) is largely retired by this result.** ω_m re-weights each domain's contribution to the shared θ *during fitting*; the LR readout is θ-free and lets you select domains *post-hoc* (the `drop:` columns), so the deployment question ("which domains to use") is answered for free without a per-ω re-fit. ω would only add the second-order question of whether fitting-time down-weighting further sharpens the *kept* domains' λ — and the condition λ already scores 0.80–0.96, so it is not obviously distorted. Deprioritize the sweep.
- Open: these are AUC (ranking); at rare-disease base rates (scleroderma 79/39463 ≈ 0.2%) high AUC need not mean deployable precision. A precision/recall readout (post-hoc, no re-fit) is the next test, and the place where cond+drug's marginal AUC gain over cond-alone either does or doesn't become operationally real.
