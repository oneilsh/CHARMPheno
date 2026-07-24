# Design-wall verdict + compiler validation (queue items 1 and 0)

Setup: nodes form an is-a DAG; each carries an additive offset over its ancestral closure; a
document attests a most-specific node. "Increment" = a node's coefficient over its parent (the
drop-root reparameterization). Queue: (0) design-wall probe → (1) identifiability compiler →
(2) read-out engine. This report closes item (0); item (1) is already built and validated, so I
recap that first.

## Queue item 1 — the identifiability compiler is built and validated

Identifiability = the null-space of the closure Gram `G = Σ_d z_d z_dᵀ` (z = the non-root closure
indicator). The compiler reads increment-identifiability off G before any fit and rewrites the DAG
to its identified quotient. Validated on realistic-overlap corpora (including the exact corpus
below):

1. **"If you can fit, you can compile" is literal.** `closure_gram` equals the offset block of the
   augmented design moment the fit already accumulates (`XtX[P:, P:]`) to machine zero — the
   compiler adds no new moment, it reads the one the fit forms.
2. **It reproduces the hard case deterministically.** On the corpus with a no-own-documents anchor
   B and its sole subtype B1, it auto-collapses `{B, B1}` (`z_B ≡ z_B1`), keeps the identified
   A/A1 distinction (A has own documents → `d(A,A1) = #distinguishing docs`), reports zero flagged
   residual, and the correctness invariant (quotient-then-moment ≡ moment-then-project) holds
   exactly. What previously took a hand isolation-probe now emits from a count reduce.
3. **The fragility spectrum is quantitative.** As an anchor gains own documents, the
   column-equality metric `d(anchor, subtype)` equals exactly that count, the collapse decision
   flips at a strict threshold, and the smallest eigenvalue of the (anchor, subtype) block grows
   monotonically (own-docs 0/5/25/100 → min-eig 0.0/2.5/12.3/47.5) — a real, monotone measure of
   identification strength, the raw material for a fragility tier and a cheap pre-fit surrogate for
   interval widths.
4. **The safety split holds under adversarial attestation.** Two non-adjacent subtypes under
   different anchors that are only ever co-attested (identical support columns) are FLAGGED, not
   merged — the exact spurious cross-structural merge the design set out to prevent. Deep
   no-own-documents chains collapse whole; the threshold boundary has no off-by-one; mixed
   collapse + keep + flag composes.

One open seam: the quotient's merged node carries the identified sum only up to a
depth-scaled-ridge reparameterization gap (the merged node gets one penalty, the collapsed chain
had two). Tiny under a light ridge (correlation ~1, `< 1e-2`), fixed when the read-out engine
lands by setting the merged node's penalty to match the chain it replaces.

## Queue item 0 — the design-wall probe

### The question you posed

The compiler reads increment-identifiability off G statically. You flagged that the offset
coefficients are fit through a ridge over a **shared** design moment (one XtX across all K-1 stick
columns) with a **co-sampled** latent, so a per-column rank argument is muddied. The arbiter you
named: run the exact Gibbs and watch whether the un-identified direction's posterior **contracts
with N** (estimator wall — a better engine or more data fixes it) or **stays pinned at the prior
width regardless of N** (design wall — no engine, no corpus volume touches it).

### What I ran

A faithful full Gibbs: the already-validated blocked PG-Gibbs sweep, with the **only** change
being that the offset block is now a proper matrix-normal **draw** instead of a ridge point —
`C ~ MN(mean = ridge, row-cov = (WᵀW + diag(penalty))⁻¹, col-cov = Σ)`, drawn and fed back every
sweep (a proper joint chain, not a plug-in). Corpus: an anchor A with its own attesting documents
plus a subtype A1; an anchor B with **no** own documents plus its sole subtype B1. Ladder: the
same attestation pattern scaled over a **16× document-count range**.

### The verdict — design wall, decisively

**Only the increment contracts; every anchor level is flat.** Posterior SD across 16× N (a 1/√N
estimator wall would shrink 4×):

| direction | SD shrinkage over 16× | reading |
|---|---|---|
| A1 increment (subtype over its parent) | **3.16×** | contracts ~1/√N → identified |
| anchor A level | 0.86× | flat → design wall |
| anchor B level (= B[B]+B[B1]) | 0.85× | flat → design wall |
| B/B1 split (= B[B]−B[B1], the Gram null) | 0.96× | flat → design wall |

**The linear algebra agrees to the coordinate.** The augmented design has **nullity 2**:
(i) the partition identity `intercept ≡ A + B` (every document sits under exactly one anchor)
confounds anchor *levels* with the intercept; (ii) `B1-column ≡ B-column` (B has no own documents)
confounds the B/B1 split. **Both null vectors load exactly zero on the A1-increment coordinate**,
so the increment is the *sole* offset coordinate in the design's row space — precisely the one
that contracted. The muddied-rank worry resolves in the clean direction: the shared XtX doesn't
rescue the level; the increment is genuinely the only thing there.

**Design-wall directions are wide-but-STATIONARY** (your outcome (a), not (b)). Split-chain
improved-R̂ on the null direction = **1.01** under a proper prior (SD 8) and **1.00** under a
near-flat prior (SD 130) — no wandering under either. And the width tracks the prior almost
exactly (`8 × √(0.25 / 1e-3) ≈ 128 ≈ 130`): **the posterior along the design wall is the prior** —
data-independent, N-independent. That is the definitive signature, and it's the good version: the
direction is un-identified but the chain is proper, so reporting its width is honest and carries no
divergence / no export hole.

### What this fixes about the engine's promises

1. **Acceptance is per-coordinate-class, confirmed.** Increment directions get a number + a
   nominal-coverage claim; level/null directions report **UNRESOLVED** (a wide interval), never a
   point. This isn't a fallback — it's what the posterior actually is.
2. **The static compiler is validated as a pre-fit surrogate.** Gibbs confirms exactly what the
   compiler computes without a fit (closure-Gram null = the split; intercept-collinearity of the
   with-intercept per-group Gram = the levels). So the compiler can gate the read-out's coordinate
   list up front, and the engine never *can* correlate over an un-identified direction.
3. **The offset draw is the engine kernel.** The matrix-normal block I built for this probe is the
   co-sampled Gibbs Γ-block scoped for item (2) — one conjugate block in the existing sweep.
   Promoting it into the library (with the merged-node penalty fix above) is the engine's first
   task; the coverage plant is its gate.

## The one decision I'd put to you

The verdict makes the engine's honest deliverable **calibrated increments + a visibly unresolved
level**, exactly as you predicted. Before I build: do you want the read-out to **emit** the
design-wall directions at all (as UNRESOLVED, width-only, machine-flagged) so a consumer sees
"this coordinate exists but the design can't resolve it" — or **suppress** them from the exported
object entirely and only surface them through the compiler's pre-fit report? Both are honest; the
first keeps the object complete, the second keeps it un-misusable. I lean emit-but-flag (it
parallels the compiler's flagged_dim), but it's a reporting-contract call and it shapes the schema.

## In practical terms

We can quantify, with a calibrated interval, **how a child node's offset differs from its
parent's** whenever the corpus contains documents attested at the parent node itself. But a node's
**absolute level** — its overall offset, not its difference from its parent — cannot be pinned down
by more documents, because every document sits under exactly one top-level node, so a node's
absolute level and the global baseline are the same unknown under two labels. A parent node with a
single child and no documents of its own is mathematically the child under another name — the
compiler already collapses it. So the engine returns sharp answers for parent→child differences and
an explicit "unresolved by this data" for absolute levels; no amount of extra data changes which is
which — only attesting documents at the parent level would.
