# DAG-offset read-out engine — design (step 2 of the identifiability queue)

**Date:** 2026-07-14
**Branch:** pg-stm
**Status:** Design (brainstormed + approved section-by-section; awaiting spec review → plan)
**Relates to:** the identifiability compiler (spec/plan 2026-07-14, module `dag_identify.py`,
insight 0055); the design-wall verdict (insight 0056, report
`docs/reports/2026-07-14-design-wall-verdict.md`); the PG-STM DAG model
(`pg_stm_dag.py`); the PG-STM Gibbs sampler (`pg_stm.py::pg_stm_gibbs`); insights 0050/0052
(only increments identified; three-wall taxonomy); Fable's locked engine contract (memory
`project_dag_ontology_pg_stm.md`).

## What this is

Queue item (2) after the compiler: the **read-out engine**. It takes a corpus + a DAG and emits,
per ontology node, a calibrated posterior for the node's parent→child offset **increment** where
the design identifies it, and an explicit, machine-flagged non-answer where it does not. The
engine is a warm-started, co-sampled Gibbs pass that emits offset-increment draws on the
compiler's identified quotient, assembled into a per-coordinate-class read-out object and gated by
a coverage plant.

Terminology note: throughout this project and in reports to Fable we say **identifiability** for
the estimability of the offset contrasts (which linear combinations of node offsets the
attestation design pins down). This is distinct from the topic-model "identifiability" literature,
which concerns uniqueness of the topic-word matrix β; that β-identifiability is not at issue here.
The classical theory underneath is Searle's estimable functions of a rank-deficient design.

## Fable's engine contract (locked — the requirements this design satisfies)

1. **Emit-but-flag**, because suppression destroys actionable information: an UNRESOLVED coordinate
   is a data-collection instruction with a price tag (`d(anchor, subtype)` counts the documents
   that would fix it), and a fixed coordinate set keeps downstream artifacts stable as data accrues
   (a coordinate flips `UNRESOLVED → number` in place when parent-level documents arrive).
2. **Status carries the cause**, not just the state: `design_null(partition_identity)` vs
   `design_null(no_own_documents)` vs `fragile(margin)`.
3. **No point estimate for design-wall coordinates** — not even the posterior mean (it is the prior
   mean in disguise). Emit width, status, cause only.
4. **GAUGE ≠ UNRESOLVED** — two null directions with different semantics and distinct labels:
   GAUGE = the partition identity (a gauge freedom no attestation can resolve; report the fixed
   convention) vs UNRESOLVED = a contingent collinearity (real corpora break it; report the
   attestation recipe). The compiler already distinguishes them mechanically; carry the distinction
   into the exported vocabulary.
5. **Merged-node penalty seam** — fix at promotion AND add the invariant as a test: the quotient
   fit's posterior over identified coordinates must match the original design's posterior projected
   onto its row space, penalty included.
6. **Soft-gate composition** — the coverage plant must include a soft-gated (fractional-z) cell, so
   the engine's first validated configuration is the one production runs.

## §1 Architecture — a two-phase pipeline

The soft-gate membership posterior must converge before the quotient can be computed
(identifiability is conditional on the labeling posterior), and the warm-start β pins topic labels
through co-sampling (insights 0045/0049). Hence two phases:

```
PHASE A  (warm-start, mean-field VI = existing PGSTMDag.fit + a new fractional-z E-step)
  → converged β (label anchor), Γ, offset point, Σ; membership posterior p(c|doc) per soft-gated doc

PHASE B1 (compile, once, on the EXPECTED design)
  expected indicator  z̄_d = Σ_c p(c|doc)·z(c)           (labeled docs: z̄_d = z_d; unlabeled: root)
  expected closure Gram  Ḡ = Σ_d Σ_c p_c · z_c z_cᵀ      (E[z zᵀ]: carries the within-doc spread)
  → dag_identify → { identified quotient DAG, node_map, GAUGE dirs, UNRESOLVED dirs (+recipe) }

PHASE B2 (co-sampled quotient Gibbs — the engine kernel)
  remap each doc's closure to quotient coords; warm-start β from Phase A;
  each sweep: sample β, ψ, ω, Σ, membership c_d ~ p(c|doc, current params);
    DRAW the offset-INCREMENT block  C ~ MN(dag_offset_ridge mean, A⁻¹, Σ),
    A = WᵀW + diag(depth-scaled penalty), merged-node penalty = Σ(chain it replaces)   [0055 fix]
  → post-burn increment draws for the identified quotient coordinates

PHASE C  (read-out assembly)
  map quotient-coord draws back to original nodes → per-coordinate-class schema + prevalence estimand

GATE:  coverage plant {populated, scarce, soft-gated, design-wall} + merged-node posterior invariant
```

The compiler runs **once** on the converged expected moment (not recomputed mid-Gibbs — membership
is stable after Phase A). Fitting the quotient means un-identified directions are simply absent from
Phase B2; GAUGE/UNRESOLVED coordinates are reported entirely from Phase B1's static analysis, never
fit. Membership uncertainty enters twice, consistently: the compiler's `Ḡ = E[z zᵀ]` carries the
within-doc spread (pulling a marginally-supported coordinate toward the null), and Phase B2 samples
membership so the reported increment widths integrate that uncertainty exactly.

## §2 Modules

| Module | Responsibility | Interface |
|---|---|---|
| `pg_stm_dag.py` *(extend)* | `_softgate_estep_doc` — the fractional-z E-step: mixture over a doc's candidate closures scored by marginal likelihood → `p(c\|doc)` + expected indicator `z̄_d`. Beside `_bg_estep_doc`; shared by Phase A (VI) and Phase B2 (Gibbs). | `(doc, candidates, β, Γ, Σ) → (weights, z̄, estep_stats)` |
| `dag_identify.py` *(extend)* | expected/weighted-Gram entry (`closure_gram` accepting fractional `z̄` + the within-doc variance term); `classify_null_directions` labelling each null direction GAUGE vs UNRESOLVED (partition/intercept-collinearity vs closure-Gram null) with the attestation recipe (`d` count). | `Ḡ, dag → {quotient, node_map, gauge_dirs, unresolved_dirs(+recipe)}` |
| `pg_stm_dag_gibbs.py` *(new)* | the co-sampled quotient Gibbs sweep — the engine kernel. The probe's MN offset-increment draw promoted with the 0055 merged-node penalty fix; warm-starts β; samples β/ψ/ω/Σ/membership; emits post-burn increment draws. | `PGSTMDagGibbs(...).run(docs, quotient, warm_start) → increment_draws` |
| `dag_readout.py` *(new)* | assemble the per-coordinate-class schema from draws + classification; map quotient coords back to original nodes; compute the prevalence estimand. No point estimate for GAUGE/UNRESOLVED. | `(draws, classification, node_map) → ReadOut` |
| orchestrator | wire Phase A→B1→B2→C (a `dag_offset_readout(...)` function, mirroring `pg_stm_sigma_readout`). | `(corpus, dag) → ReadOut` |
| `tests/_stm_synth.py` *(extend)* | soft-gate corpus generator + the coverage plant (populated / scarce / soft-gated / design-wall cells). | — |

Soft-gate accumulators are written empty/accumulate/combine so a later mllib/Spark shim is a
drop-in, but **no distributed code ships in v1** (in-process local, validated on plants — as the
compiler was).

## §3 The read-out schema (coordinate-class object)

A **fixed coordinate set** — every non-root original node always appears, with a status — so
downstream artifacts stay stable across corpora and a coordinate flips `unresolved → number` in
place when parent-level documents arrive.

```
ReadOut
  calibration: "absolute"                 # increments are coverage-validated (vs the old "ordinal")
  coordinates: { node_id → Coordinate }   # FIXED key set = all non-root original nodes
  prevalence:  { node_id → {labeled_mass, inferred_total, recall_ratio} }
  meta: { corpus_fingerprint, membership_converged, n_draws, ci_level }

Coordinate — tagged union on `status`; each node = its increment over its parent:
  status: identified | fragile | unresolved | gauge
  node, parent
  identified → increment_mean, ci_low, ci_high, covered
  fragile    → increment_mean, ci_low, ci_high, covered, fragility{ margin, min_eig }
  unresolved → width, reason:"design_null(no_own_documents)", recipe:{ attest_node, docs_needed }   # NO point estimate
  gauge      → reason:"design_null(partition_identity)", convention:"<the coordinate convention fixed>"  # NO number, NO recipe
```

Status is assigned by the compiler's two analyses, never a heuristic: closure-Gram null → the
merge/`unresolved` axis (carries the recipe); foreground-Gram intercept-collinearity → the `gauge`
axis; surviving identified coordinates → `identified`, demoted to `fragile` when the compiler's
min-eigenvalue margin is below the fragility threshold (the insight-0055 spectrum). `gauge` and
`unresolved` both omit the point estimate but are semantically distinct (unfixable-with-a-convention
vs fixable-with-a-recipe). **Prevalence** rides free on the soft-gate machinery: per node,
`labeled_mass` (hard-attested) + `inferred_total` (adds partial-label membership resolving toward
that node) → `recall_ratio`.

## §4 The soft gate (partial-label fractional-z)

Candidate closure set per document, from the labeling (not inferred):
- **labeled** → one candidate, its attested closure, `p=1` (hard z; the current path).
- **partial label** ("under B, subtype unknown") → candidates = closures at/under B (a mask over B's
  descendants). **This is the soft gate** — anchor membership is known, only the depth is soft.
- **unlabeled** → **root only** (background member; the insight-0054 bg-only path). No anchor
  inference.

Membership posterior `p(c|doc) ∝ π_c · L(doc|closure c)`, where `L(doc|c)` is the document's
marginal likelihood under candidate `c` (the existing seed-completion machinery), and `π_c` is a
**fixed** prior kept upstream of and independent from model outputs (no sample↔model feedback).
Expected indicator `z̄_d = Σ_c p_c z(c)`. Membership uncertainty enters the compiler via
`Ḡ = E[z zᵀ]` (within-doc spread) and the Gibbs via per-sweep membership sampling (widths integrate
it exactly).

**Deferred (not v1):** the latent-anchor MNAR recall correction (inferring anchor membership for
apparently-unlabeled documents) — a stronger modeling claim, its own later step. v1's "prevalence"
is only partial-label mass redistribution within a labeled subtree.

## §5 Coverage plant (acceptance gate) + honesty rules

**Redraw-truth protocol (insight 0051):** each replicate redraws node offsets from the prior,
generates a corpus, runs the full engine, checks whether each coordinate's credible interval covers
its planted value; empirical coverage is tallied per coordinate class over R replicates.

| cell | construction | pass criterion |
|---|---|---|
| **populated** | node with many own docs + a branching sibling → identified increment | 90% CI covers ~90% (band `[0.85, 0.95]`), tight |
| **scarce** | node with few distinguishing docs → identified but wide | covers ~90% — wide-but-covers |
| **soft-gated** | docs partial-labeled at an internal node → fractional z over its subtree | subtree coordinate covers ~90% conditional on membership |
| **design-wall** | a no-own-docs anchor (insight 0054) → GAUGE/UNRESOLVED | schema assertion, not coverage: no `increment_mean`; `width` present; `recipe`/`convention` present |

Plus the **merged-node posterior invariant** as its own correctness test (contract req 5): fit the
quotient and the original DAG, project the original's posterior onto its row space (penalty-matched),
assert agreement on identified coordinates within MC error.

**Honesty rules (standing constraints):**
- Test-honesty: every plant labels planted-vs-real; coverage is a math-correctness property on
  synthetic, no transfer-to-real claim.
- Stick-space rule (insight 0053): read out at block granularity or in θ-space, never per-stick
  across positions; stick ordering is a frozen model constant.
- Cite literature in docstrings (Polson-Scott-Windle / Linderman-Johnson-Adams / Searle /
  Vehtari-R̂ / the 0051 coverage protocol); no LaTeX (Unicode Greek); engine layer domain-neutral
  (integer ids only); hash ids in any row-level log.

## Scope

**In v1:** the fractional-z partial-label soft gate; the co-sampled quotient Gibbs increment engine
(MN offset draw promoted to the library + merged-node penalty fix); the per-coordinate-class read-out
schema with GAUGE/UNRESOLVED/identified/fragile; the coverage plant (four cells incl. soft-gated) +
the merged-node posterior invariant; the prevalence estimand from partial-label resolution.

**Deferred:** the mllib/Spark shim; the latent-anchor MNAR recall correction; many-to-many
(multi-anchor) membership; the OMOP DAG-builder + real-corpus run; Piece B (node-owned distinct
topics); the LDA-vs-PG bake-off; the LLM-facing reporting/tiering layer.

## Open questions (with defaults — will not gate the plan)

1. **CI level** — default 90% credible intervals (matches the 0051 coverage protocol); revisit if a
   consumer needs 95%.
2. **Fragility threshold** — default: the compiler's min-eigenvalue margin below which a coordinate
   is `fragile`; set from the insight-0055 spectrum on the coverage plant (calibrate so scarce-cell
   coordinates that still cover are `identified`, not `fragile`).
3. **Membership convergence gate** — default: run Phase A to the existing VI stopping rule; expose
   `membership_converged` in `meta` so a non-converged run is machine-visible.

## References

Polson, Scott & Windle (2013); Linderman, Johnson & Adams (2015); Searle (1971); Roberts, Stewart &
Airoldi (2016, STM); Ramage, Manning & Dumais (2011, PLDA); Vehtari et al. (2021, improved R̂). See
`docs/references.md` (§ Pólya-Gamma augmentation & identifiability; § Topic hierarchies, supervision
& gating).
