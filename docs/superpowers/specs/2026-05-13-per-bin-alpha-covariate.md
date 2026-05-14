# Per-Bin Asymmetric α Covariate — Design Stub

**Date:** 2026-05-13
**Status:** Stub, parked. Captures the design idea so it can be acted on later without re-deriving the math. **Not** scheduled for implementation; the [dashboard work](2026-05-13-dashboard-design.md) ships against the current covariate-free model.
**Scope:** Add a single doc-level categorical covariate (age bin, sex, condition cohort, etc.) to OnlineLDA while preserving Dirichlet conjugacy and the existing Spark-VI inference engine. Generalizes to any patient-level categorical via a `bin_id` column on the doc spec. Does **not** cover content covariates (covariate-dependent β), continuous covariates, or any structural-topic-model (STM) work.

---

## Context

The conversation that produced this stub started from a "demographic breakdowns of phenotypes by age" idea, which we cut from the dashboard scope on data-export grounds. That cut left a real modeling question parked: can we make age (or another patient-level categorical) a first-class part of the model, rather than something we measure after the fact?

The textbook answer is the structural topic model (STM, Roberts et al.): a logistic-normal prior on θ with a regression on covariates, optionally a second covariate path modulating β. STM is well-defined and well-studied, but it does not fit the existing Spark-VI framework cheaply:

- The current inference engine relies on **Dirichlet-multinomial conjugacy**. The per-doc variational update has a closed form: `γ_k = α_k + Σ_w n_w · φ_{w,k}`, with `φ_{w,k} ∝ E[β_{k,w}] · exp(ψ(γ_k))`. Cheap, vectorizable, mapPartitions-friendly.
- STM uses a logistic-normal prior on θ, which is not conjugate to multinomial. The per-doc update becomes a numerical optimization (typically Laplace approximation on η_d or a quasi-Newton inner loop), repeated per doc per outer iteration.
- Porting that to Spark is a significant lift: the inner loop becomes per-doc optimization, the M-step gains a regression, the ELBO bookkeeping changes. It is not an extension of the existing engine; it is a new engine.

The "more direct way to sneak in an age param" is to keep Dirichlet conjugacy and change *only* what's needed to make the prior covariate-dependent. That is this stub.

## Goals

1. **Add doc-level covariate support to OnlineLDA** via an asymmetric, per-bin Dirichlet prior on θ. β stays shared across the corpus.
2. **Preserve the existing inference machinery.** E-step is unchanged in shape — only the prior lookup differs per doc. β update is unchanged. The only new computation is per-bin empirical-Bayes α estimation.
3. **Add a `bin_id` column to the doc spec** so the engine can route docs to their per-bin prior. Generalize to any patient-level categorical, not just age.
4. **Document the path from "per-bin α" to "full STM"** so future work that needs covariate-dependent β has a clear next step.

## The Math

### Generative process (covariate-augmented LDA)

Let `b(d)` be the bin index for document `d` (e.g., age decile of the patient who owns the doc; or sex; or condition cohort). Bins are exogenous, observed, and fixed.

Each bin `b` has its own asymmetric Dirichlet prior `α^(b) ∈ ℝ_+^K`. The generative process becomes:

```
For each phenotype k: β_k ~ Dirichlet(η)              # unchanged, shared
For each document d:
  θ_d ~ Dirichlet(α^(b(d)))                          # per-bin prior
  For each token n in document d:
    z_{dn} ~ Categorical(θ_d)
    w_{dn} ~ Categorical(β_{z_{dn}})                  # unchanged
```

The only structural change from vanilla LDA is the prior on θ_d. β is shared; the assignment-and-emission machinery is shared.

### Variational E-step (per doc)

Unchanged in shape; only the prior lookup is parametrized by `b(d)`:

```
γ_{d,k} = α^(b(d))_k + Σ_w n_{d,w} · φ_{d,w,k}
φ_{d,w,k} ∝ E[β_{k,w}] · exp(ψ(γ_{d,k}))
```

Iterate to convergence as before. The per-doc cost is identical to vanilla LDA modulo one indexed gather of `α^(b(d))` from the prior table.

### Variational M-step

Two pieces:

1. **β update — unchanged.** Aggregate `φ`-weighted counts across all docs:
   ```
   λ_{k,w} = η + Σ_d n_{d,w} · φ_{d,w,k}
   ```
   Bin structure does not enter; β is shared.

2. **Per-bin α update — new.** For each bin `b`, run the existing empirical-Bayes Newton/digamma fixed-point on the subset of docs with `b(d) == b`:
   ```
   For b in bins:
     γ_b = { γ_d : b(d) == b }          # variational posteriors for this bin's docs
     α^(b) ← fixed_point(γ_b)            # same routine as the global α update today
   ```
   B independent calls to the existing α-optimizer. Trivially parallelizable across bins.

### ELBO

The ELBO gains B independent prior terms (one per bin) in place of the single global term. Each is structurally identical to the existing prior term; the sum decomposes by bin.

### Identifiability

Per-bin α captures bin-specific *prevalence* of phenotypes (which phenotypes are common in which bins). It does *not* capture covariate-dependent *content* (the same phenotype's β does not change with the bin). If we want content covariates later, we either move to STM or add a per-bin β multiplier — both are larger lifts and out of scope here.

A practical consequence: phenotypes that would have been age-specific (e.g., a chemo-cytopenia phenotype that mainly appears in peds onc) still emerge under per-bin α, because the model can give that phenotype high `α^(b)` only in the relevant bins. What it cannot do is *split* a phenotype across bins (e.g., "chemo-cytopenia in adults" vs. "chemo-cytopenia in peds" as two β rows). For v1 this is the right tradeoff.

## Engineering

### Doc spec plumbing

The doc spec currently emits `(doc_id, codes)` (or `(person_id, doc_id, codes)`). Add an optional `bin_id` column:

```
DocSpec:
  person_id: long
  doc_id: long
  codes: array<int>
  bin_id: int       # NEW, optional. Absent or null → fall back to global α.
```

Concrete doc specs that want covariate support compute `bin_id` from the patient table. Doc specs that don't bother leave it null; the engine then uses the existing global α and behaves exactly as it does today. This keeps the change backward-compatible.

### Engine changes

In `spark_vi`:

- `core/types.py` (or wherever `BOWRow` lives): add an optional `bin_id: int` field.
- `models/topic/lda.py`:
  - `OnlineLDA.fit` accepts an `n_bins: int` parameter. If `n_bins == 1` (default), behavior is identical to today.
  - State carries `alpha_` as shape `(n_bins, K)`. When `n_bins == 1` this reduces to a `(1, K)` matrix and the existing global-α path is recovered as a special case.
  - The per-doc E-step accepts `bin_id` and indexes into `alpha_` accordingly.
  - The α-update step iterates over bins and calls the existing fixed-point routine per bin.
- `mllib/`: the shim's `LDAModel` exposes `alpha_` as a `(n_bins, K)` matrix; `transform` accepts a bin_id column when `n_bins > 1`.
- Persistence (`VIResult`): the state schema grows an `n_bins` field and the α tensor becomes 2D. Backward-compatible via a `version` bump.

### Bin design (caller's responsibility)

The engine takes bins as exogenous integers and does not interpret them. Choosing bin edges (age deciles? quintiles? clinical bands?) is a domain decision, made in the doc spec layer. Bin count should stay modest (5-20 bins typical); very high B with sparse per-bin doc counts degrades the empirical-Bayes α fit.

A `charmpheno/charmpheno/omop/bins.py` helper holds bin-construction utilities (age band selection, sex coding, condition-cohort indicators). Out of scope for this stub; planned alongside the engine work when implementation is scheduled.

## Scope

### In scope when this is scheduled

- Engine changes above.
- A new ADR documenting the choice and the path to full STM if ever needed.
- A small evaluation: train OnlineLDA with and without per-bin α on the same corpus, compare per-bin θ distributions, NPMI, and ELBO.

### Not in scope (even when scheduled)

- **Covariate-dependent β.** Same-phenotype, different-content-by-bin is a content covariate; that's STM proper.
- **Continuous covariates.** Bin discretization is the contract.
- **Multiple covariates simultaneously.** A single categorical bin per doc. Multi-covariate is a cross-product of bins (which gets sparse fast); the engine doesn't need to know that's how the caller built the bin_id.
- **Hierarchical priors on α** (e.g., `α^(b) ~ Gamma(...)`). A flat empirical-Bayes per bin is enough for v1; hierarchical pooling is a follow-on if some bins are under-sampled.

## Open Questions (for the future plan, not blocking parking)

- **Default bin count and edges** for the canonical "age" application. Probably 10 age deciles from the corpus's empirical age distribution; needs a quick look at the OMOP person table when implementation is scheduled.
- **Cold-start for empty bins.** A bin with zero docs (or fewer than some threshold) needs a fallback prior. Options: pool with neighboring bins, fall back to global α, or refuse to fit. Decide when we see the empirical bin sizes.
- **Interaction with HDP.** This stub specifies OnlineLDA. OnlineHDP has its own prior structure (GEM sticks); a per-bin variant would put a per-bin base distribution on G_0. Plausible but its own design — defer.
