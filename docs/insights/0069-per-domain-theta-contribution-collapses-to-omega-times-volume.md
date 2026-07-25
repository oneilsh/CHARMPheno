# 0069 — The per-domain θ-contribution instrument collapses to ω_m × V_m exactly, so it carries no fit-dependent information

**Date:** 2026-07-25
**Topic:** svi | diagnostics
**Status:** Confirmed
**Context:** SP2 of the multi-domain gated DAG LDA arc (branch `multidomain-spectral-init`), plan Task 7. The arc design called for a per-modality θ-contribution instrument on the grounds that ω_m is a tuned dial and cannot be tuned without seeing what each domain contributes to the shared θ.

## Finding

The instrument was built to the specified definition — the exact per-domain partition of `Σ_k (γ_k − α_k)`, grouped by each token's domain — and the definition is met exactly. Working from the CAVI recurrence

    γ = α + expElogθ_d · (eb_d @ (ω_tok · counts / phi_norm))

and the fact that `phi_norm` is recomputed from the returned γ, the emitted quantity is

    Σ_n ω_n · counts_n · (phi_norm_n − 1e-100) / phi_norm_n

grouped by domain. Verified numerically against the partition computed the long way, without the `1e-100` rearrangement: **max absolute difference 0.0**.

The problem is what that expression reduces to. The guard factor `(phi_norm − 1e-100) / phi_norm` is **bit-exactly 1.0** in float64 for any `phi_norm` not at the underflow floor, so the instrument equals

    ω_m × (token volume in domain m)

for **every** ω, not merely for ω = 1. Measured: stat `[4.0, 4.2]` against `ω·volume` `[4.0, 4.2]`, difference 0.0. It is a deterministic function of the token histogram and the ω dial — **computable without fitting anything.**

## Why it matters

1. **The instrument cannot do the job the design wanted.** It cannot detect θ domination by one modality, because it reports nothing a per-domain token count does not already report. Whatever the posterior does with that evidence — whether the shared θ actually ends up dominated — is invisible to it. SP3/SP4 must not lean on it as a volume-imbalance diagnostic.
2. **It is still worth keeping, as a different thing than advertised.** It is an exact trace of the dial's effect on the γ accumulation: it confirms ω was applied, to which domains, and in what proportion, and it is the cheapest possible regression guard on "ω weights θ" (it scales exactly 0.25× when ω_m does, and goes to exactly 0 at ω_m = 0). That is a real if modest use. What it is not is a posterior diagnostic.
3. **The general lesson: a quantity defined as a partition of the γ *increment* is a partition of the evidence, not of the posterior.** `Σ_k (γ_k − α_k)` telescopes back to the weighted token count by construction, because the CAVI γ-update conserves evidence mass across topics (each token's φ is a distribution over k). Any per-domain summary of that increment therefore cannot depend on how the mass was distributed. Measuring domination requires a quantity sensitive to *where* the mass landed — for example each domain's marginal contribution to the fitted θ of documents, or a leave-one-domain-out refit — not a partition of the input.

**Bottom line:** the instrument is exact and correctly documented, and it is a dial-effect trace rather than the volume-imbalance diagnostic the arc design assumed. Tuning ω against a downstream task metric remains the only validated route; a per-domain token count is the free equivalent of this instrument.
