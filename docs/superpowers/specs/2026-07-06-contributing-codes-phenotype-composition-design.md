# Contributing-Codes Phenotype-Composition View — Design

**Date:** 2026-07-06
**Status:** Design (approved; not yet planned/built — implementation blocked on a concurrent `stm` thread)
**Scope:** Dashboard-only visualization + interaction upgrade to the Simulator → Explore Cohort / Patient "Top contributing codes" panel. No model, export, or data-format changes.

## Motivation

The patient panel already shows a phenotype profile bar (the patient's θ) and a "top
contributing codes" list. Today, clicking a phenotype band lists the patient's codes
sorted by that one phenotype's contribution, each drawn as a single monochrome
magnitude bar.

The model can say more than "how much does this code contribute to the *selected*
phenotype." For a given patient it can attribute each individual code across *all*
phenotypes — the per-token responsibility φ. Surfacing that turns each code into a
miniature phenotype-composition bar that mirrors the profile bar above, giving a
post-hoc, per-code answer to "which phenotypes does this condition speak to, for this
patient." This is an interpretability view (post-hoc attribution), not a change to how
the model is fit or how patients are generated.

## The quantity

For patient d, code w, phenotype k:

    φ(w,k) = θ(d,k) · β(k,w) / Σ_j θ(d,j) · β(j,w)

This is P(topic k generated this code | code w, patient d). It is **patient-conditioned**:
the same code splits differently for two patients because θ differs — the patient's own
phenotype mix disambiguates their codes. φ(w,·) sums to 1 by construction (every code is
fully attributed among the topics that can emit it).

This is already computed in
[ContributingCodes.svelte:43-52](../../../dashboard/src/lib/patient/ContributingCodes.svelte#L43-L52)
for the single selected phenotype (`pzkw`); this design generalizes it to the full
vector and moves it into a testable module.

## What we reuse unchanged

- **φ formula, β, and θ** — all client-side in `$bundle.model` (`beta`, `K`); no export change.
- **`phenotypeHue`** ([palette](../../../dashboard/src/lib/palette.ts)) — the shared
  phenotype→hue map, so a phenotype is the same color here, in the profile bar, and in the atlas.
- **`OTHER_THRESHOLD = 0.05`** and the tail-bucketing convention shared with `ProfileBar`,
  so both views collapse the long tail into "Other" identically.
- **The "Other" hatch idiom** — the repeating-linear-gradient fill already used for the
  Other band and link dot ([ContributingCodes.svelte:180-190](../../../dashboard/src/lib/patient/ContributingCodes.svelte#L180-L190))
  — reused for the prior-residual overlay.
- **`copy.ts` contributingCodes strings** and the panel's grid/list layout.

## Decisions (locked)

1. **Interaction: always-on composition + focus.** Every code shows its full phenotype
   split at all times; the panel no longer requires a selection to render. Selecting a
   phenotype band *sorts and emphasizes*, it never hides codes.
2. **Bar scaling: normalized composition.** Every code's bar has the same total length and
   shows only the split proportions (φ(w,·)). Chosen for at-a-glance specificity reading.
   Consequence: bars do **not** visually sum to θ, so the prior residual (below) is an
   aggregate reconciliation, not the leftover of the stack.
3. **v1 extra: prior-residual segment** only. Specificity sort/group and the
   ground-truth-z validation overlay are deferred.

## Components

### A. Pure module — `dashboard/src/lib/patient/codeComposition.ts`

Extract the math out of the Svelte component so it is unit-testable in isolation (mirrors
the `cohort.ts` / `projection.ts` + `.test.ts` pattern). Two exports:

- **`codeComposition(theta, codeBag, beta, K)` → `CodeRow[]`**
  For each unique code w in `codeBag`: its occurrence count `c`, and its full split
  φ(w,·) reduced to the phenotypes above `OTHER_THRESHOLD` plus an aggregated "Other"
  bucket (same hue map + threshold as the profile bar). Each row's segments sum to 1.
  A code emitted by no expressed topic (z = 0) yields an empty/again-"Other" split and
  must not divide by zero.

- **`explainedVsPrior(theta, codeBag, beta, K)` → `{ explained: number[]; prior: number[] }`**
  The aggregate reconciliation behind the residual overlay:
  - `θ_data(k) ∝ Σ_w c(w) · φ(w,k)` — the phenotype mass the patient's codes explain
    (occurrence-weighted, normalized over k).
  - Per phenotype, the **prior-supported remainder** is `max(0, θ(k) − θ_data(k))` — the
    part of θ(k) the codes do not account for (the Γ / population-prior pull). Clamp at 0:
    where `θ_data(k) > θ(k)` the codes *over*-explain that phenotype (the prior pulled mass
    elsewhere), so there is no prior band to draw — `explained` is capped at θ(k) and the
    residual is 0. Returned per-k so the profile bar can split each band into a
    code-explained portion and a prior portion.
  - **Exactness caveat (important):** this additive "prior + code evidence" split is *exact*
    only for the Dirichlet/LDA engine, where γ(k) = α(k) + Σ_w c·φ(w,k) is genuinely a sum
    of a prior pseudocount and code evidence. For **STM** (the current dashboard models)
    θ = softmax(η̂) is **not** an additive sum, so `explainedVsPrior` there is a principled
    *heuristic*, not an identity — it should be labeled as an approximate "evidence vs.
    prior" indication, not a decomposition. It becomes exact if/when the panel is fed a
    gated-LDA model. See insight
    [0028](../../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md).

Both functions are deterministic and Svelte-free.

### B. Component — `ContributingCodes.svelte`

- Consumes `codeComposition(...)`; renders each code as a **normalized stacked bar**: one
  segment per phenotype in the shared hue, a 2px surface gap between segments, "Other" in
  gray. Renders with **no selection** (new empty state — the old "click a band" hint is
  removed; a short caption explains the composition instead).
- **Selection = focus:** when `selectedPhenotypeId` is set, (a) sort the code rows by
  φ(w, selected) descending, and (b) desaturate every non-selected segment so the selected
  hue reads across all bars. Codes are never dropped. The "Other" band selection
  (`id === -1`) emphasizes the aggregated tail, consistent with today's behavior.
- Default sort when nothing is selected: occurrence count (today's implicit order). Cap at
  top-N (12) with the existing expand affordance.
- `open in atlas`, search-match highlight (`searchedConditionIdx`), and the count column
  are preserved.

### C. Profile-residual overlay — `ProfileBar.svelte` (small addition)

Using `explainedVsPrior`, render each θ band as a solid **code-explained** portion plus a
**hatched prior-supported** portion (reusing the Other hatch gradient). One glance shows
whether a phenotype call rests on this patient's codes or on the population prior — most
visible on short synthetic records where the prior dominates. Because the split is a
heuristic under STM (see the exactness caveat), the caption/tooltip must frame it as an
approximate "evidence vs. prior" cue, not an exact decomposition.

## Data flow

`theta` + `code_bag` (already passed into the panel) → `codeComposition` / `explainedVsPrior`
(pure) → stacked bars (component) + band split (profile bar). β and θ are read from
`$bundle.model`. Nothing leaves the client; nothing is re-fit.

## Testing

**`codeComposition.test.ts`** (pure, deterministic):
- each code row's segments sum to 1 (within epsilon);
- occurrence-weighted aggregate of φ reproduces `θ_data`, and `explained + prior`
  reconciles with θ per phenotype;
- the over-explain case (`θ_data(k) > θ(k)`) clamps: `prior(k) === 0` and
  `explained(k) === θ(k)`, never negative;
- a code emitted by no expressed topic (z = 0) is handled without NaN;
- "Other" bucketing matches `ProfileBar`'s threshold;
- empty `codeBag` returns no rows.

**Component test:** renders all codes with no selection; a selection sorts by the selected
phenotype and desaturates non-selected segments without removing any code.

## Deferred (recorded, not built)

- **Specificity sort/group** — order/group codes by concentration (entropy of φ(w,·)):
  phenotype-defining vs shared/background. The natural next increment; ties to the
  background/foreground gating intuition.
- **Ground-truth-z validation overlay** — synthetic patients carry the true generating
  topic per token; compare post-hoc φ against it as a credibility diagnostic (impossible
  on real data).
- **Occurrence-weighted / normalized toggle** — a length-∝-count mode in which the stacked
  code bars literally sum to the profile bar (composition = decomposition).

## Notes / caveats

- φ inherits the θ point estimate. In STM, θ = softmax(η̂) is a Laplace/MAP estimate pulled
  toward the Γ prevalence regression, so on short records the split leans on the prior —
  which is exactly what the residual overlay is meant to expose, not hide.
- Show soft weights, never a single argmax tag: the fractional split is both more honest
  and more informative.
