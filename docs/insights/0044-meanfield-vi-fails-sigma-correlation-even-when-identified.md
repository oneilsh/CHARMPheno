# Insight 0044 — Mean-field PG-VI does not recover the stick-space Σ correlation, even when it is identified

**Date:** 2026-07-12
**Branch:** `pg-stm`
**Tests:** `spark-vi/tests/test_pg_stm_stick_native.py`; corpus
`gated_ln_corpus_stick` in `spark-vi/tests/_stm_synth.py`
**Relates to:** exp 0049 (PG-STM milestone-1), Task 7 VI-vs-Gibbs xfail

## The question

The PG-STM core's mean-field variational posterior factorizes across documents
(q(ψ_d) independent per doc). Task 7 observed that this VI does not reproduce exact
Gibbs on the background-stick correlation block, but that observation was CONFOUNDED:
the corpus (`gated_ln_corpus`) plants η in SOFTMAX space, so the stick-space Σ is only
weakly identified — the VI≠Gibbs gap could be blamed on non-identification, not on
mean field itself. To decide, we need a corpus where the stick-space Σ is genuinely
identified.

## Setup

`gated_ln_corpus_stick` draws ψ ~ N(0, Σ_true) in the model's OWN (K−1)-dim stick space
and composes θ = gated_theta(ψ) (nested stick-breaking), so the planted Σ is identified
by the likelihood. Config: bg_k=3 (a single, well-observed r01 background correlation;
higher-index background sticks are under-observed under stick-breaking and noisy even
for Gibbs), planted r01 = +0.30, D=1000, doc_len=40. Fit both the mean-field PG-VI and
the exact PG-Gibbs sampler; compare the recovered r01 to the planted value.

## Result

| corpus seed | planted r01 | exact Gibbs r01 | mean-field VI r01 |
|---|---|---|---|
| 0 | +0.30 | −0.37 (under-sampled draw) | −0.68 |
| 1 | +0.30 | +0.34 (d=0.04) | −0.72 |
| 2 | +0.30 | +0.27 (d=0.03) | −0.70 |

- **Exact Gibbs recovers the planted correlation** in sign and magnitude on the
  adequately-sampled draws (seeds 1/2: d ≈ 0.03–0.04). Seed 0 is a noisier draw where
  even Gibbs flips — the background sticks get only the σ(gate) fraction of each doc's
  tokens, so their correlation is at the edge of identifiability at D=1000. This is the
  POSITIVE CONTROL: the corpus is identified.
- **Mean-field VI recovers the WRONG SIGN in every seed** (r01 ≈ −0.7 vs planted +0.30).
  Not merely attenuated or noisy — systematically inverted, ~1.0 away from the truth and
  ~0.7–1.0 away from exact Gibbs on the same corpus.

The bg_k=4 (3×3 block) version shows the same pattern: Gibbs lands near planted on the
low-index sticks while VI reads spurious ±0.8–0.9 across the whole block.

## Interpretation

The VI≠Gibbs disagreement is a GENUINE mean-field limitation, NOT the softmax-planting
confound. Mechanism (see `test_pg_stm_gibbs.py` docstring): the per-doc posterior
q(ψ_d) = N(m_d, V_d) has a FULL V_d = (Σ⁻¹ + diag(ω))⁻¹, but the PG data precision
diag(ω) is ADDED to the correlated prior precision, so as ω grows the posterior is
dominated by its diagonal data term and the prior correlation is attenuated; the
delta-method E[log θ] and the between-doc factorization (Σ sees only the scatter of
per-doc MEANS, not their posterior spread) compound it. The net effect here is strong
enough to invert the sign of a modest correlation.

## Consequence

The Σ correlation read-out is the comorbidity deliverable (and a load-bearing asset for
the KG rare-disease case-finding thesis, where patient-level correlated structure is the
point). This insight says: **do not trust mean-field VI's Σ correlations.** For the
correlation read-out, use exact Gibbs, or replace the between-doc mean field with a
structured / collapsed variational posterior that carries the prior correlation through
the ω-weighting. β recovery is unaffected (both estimators recover the planted topics),
so mean-field VI remains fine for topic content — the limitation is specific to the Σ
correlation.

This also cleanly explains the Task-7 xfail: it is not loosened or a sampler bug; it is
this mean-field limitation, now established on an identified corpus rather than a
confounded one. It sharpens the sub-project-2 (distributed PG-SVI) design question:
whether the SVI kernel needs a structured per-doc posterior for the comorbidity read-out,
or whether Σ correlations are reported from a Gibbs pass while SVI carries β/Γ/gating.
