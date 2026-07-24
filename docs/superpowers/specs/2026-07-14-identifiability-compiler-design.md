# Identifiability Compiler — Design (v1)

**Date:** 2026-07-14
**Branch:** pg-stm
**Status:** approved (brainstorm), spec written; plan next
**Relates to:** the DAG/ontology PG-STM engine (`pg_stm_dag.py`, `DagGate`); insights 0050
(anchor offsets un-identified under a partitioning gate), 0052 (information vs estimator walls),
0054 (background-only members flip the identification flag but anchor levels stay
intercept-confounded; no-branching nodes are collapsible); Fable's amended directive
(three-wall taxonomy; the compiler as a first-class pre-fit stage).

## 1. Purpose and framing

Identifiability in this project has been discovered *after the fact*, one insight number at a
time (0033, 0050, 0054). The identifiability compiler turns that lesson into infrastructure:
compute — **before any fit, from the design alone** — which increment coordinates the corpus can
actually resolve; rewrite the node DAG to exactly those; and let the model only ever fit questions
the data can answer. Read-outs then consume the surviving identified-coordinate list, so
correlating over an unidentified direction is not a mistake one *can* make.

The key technical insight: **identifiability is a property of the design moment**, so it can be
read off the Gram matrix the fit already computes. "If you can fit, you can compile" — the
compiler's cost is a subset of the fit's cost.

This is a **domain-agnostic** engine component: integer node/token ids only, no vocabulary from
any application domain in core code, comments, or docstrings.

## 2. Placement, layering, and scope

Follows the existing model-layering pattern (as `hdp`/`lda`/`stm`/`pg_stm` do):

- **Core (v1, this spec):** single-machine, domain-agnostic, in `spark_vi/models/topic/`.
- **mllib shim (DEFERRED):** distributes the single corpus reduce; built only once the approach
  validates. Not in v1.
- **Format adapters (DEFERRED):** map real ontology sources (OMOP concept_ancestor / SNOMED /
  MONDO / HPO via Monarch / hand-authored edge lists / CSV) onto the abstract integer-node input.
  Not in v1.

**Three purity tiers inside v1** — the boundary that keeps the theoretical core theoretical
(thresholds and human labels are pushed outward):

1. **Math kernel — threshold-free.** Pure linear algebra: Grams, singular spectra, null-space
   bases. No cutoffs, no names.
2. **Quotient builder — exactly one numeric parameter.** Takes a numerical rank tolerance `tol`
   (an `rcond`-style number, NOT a semantic label) and constructs the quotient DAG + node-mapping
   + stability margins. Deterministic.
3. **Reporting / interpretation layer (thin; sits with the LLM labeling pipeline; DEFERRED /
   minimal in v1).** Owns the identified / fragile / unidentified tiering (bucketing the kernel's
   margins), human-readable node names, conditionality annotations, and hysteresis *labeling*.

The numeric rank tolerance is the ONLY threshold that touches the compiler, and it lives in tier 2
(the builder), not the kernel. The kernel commits to no cutoff at all.

## 3. The identifiability object (math kernel, threshold-free)

Input from the corpus: for each document, its node placement(s) → **closure indicator** `z_d`
(0/1 over the non-root nodes, exactly `DagGate.offset_indicator`), and its group. No fitting — a
single lightweight reduce over the corpus.

- **Pooled Gram** `G = sum_d z_d z_d^T` — the offset block of the same `XtX` the fitter forms
  (`pg_accumulate_doc` already accumulates `outer(w, w)` with `w = [covariates; z]`), a small dense
  `(U_off x U_off)` matrix (`U_off` = number of non-root nodes). Its **null-space** = the
  exactly-confounded increment directions; its **singular spectrum** = the conditioning of every
  surviving direction (small-but-nonzero singular values are the weakly-identified directions —
  the continuous raw material the reporting layer later buckets; here they are just numbers).
- **Per-group foreground Grams** `G_g` — the same reduce restricted to the documents that activate
  group `g`'s sticks, WITH the intercept column included. Each `G_g`'s null-space surfaces the
  level-vs-intercept direction *for that group*, so the absolute-level design wall (0054) is named
  per-node, not blanket. Cost: `U_off` tiny masked Grams instead of one — still milliseconds.

The kernel emits: the Grams, their singular values + vectors, and the null-space bases. Nothing
thresholded, nothing named.

**Conditionality metadata (recorded, no logic in v1).** Two facts the kernel carries as plain
metadata so the eventual report cannot be misread: whether the counts are **enrichment-weighted**
(a corpus that oversamples minority nodes yields a Gram reflecting the enriched design — fine for
"what does THIS corpus identify," but comparisons across differently-enriched corpora must not
read resolvability differences as ontology differences), and whether any `z_d` are **fractional**
(soft-gate expected indicators → "expected identifiability under the labeling model"). Both are
labelable fields now; the associated logic is deferred.

## 4. The quotient builder (one numeric tolerance) + correctness + determinism

Given the kernel's spectrum + null-space and a numeric rank tolerance `tol`:

- **Classify each null direction by graph-locality.** A direction supported on **graph-adjacent**
  nodes (a parent->child chain) → **auto-collapse** to one quotient node. A direction spanning
  **non-adjacent / cross-tree** nodes (coincidental identical support, or a genuinely ambiguous
  multi-parent confound) → **flag, do NOT auto-merge**. Structure the compiler *understands* it
  collapses; structure it merely *detects* it escalates. This is the safety split that prevents
  spurious cross-tree collapses and the O(nodes^2) support-set comparison the naive scheme required.
- **Multi-parent is native** — a null direction simply names which nodes it spans; diamonds and
  multipath contribute their actual Gram structure, no tree assumption, no blanket human-punt.
- **Emit:** the quotient `DagGate` (fewer offset coordinates), a **node-mapping** original<->quotient,
  and a **per-decision stability margin** (distance of the deciding singular value from `tol`).

**Correctness invariant (the headline test).** *Quotient-then-form-the-moment ≡
form-the-moment-then-project onto the identified subspace.* Forming the quotient DAG's Gram must
equal projecting the original `G` onto its identified complement (residual ≈ 0). If this holds on
the plants, "map back to the original for the report" is provably faithful, not hopefully so.

**Determinism across refits.** With an effective-rank tolerance, a node sitting right at the cutoff
could flip collapsed<->kept between corpus snapshots, churning every downstream artifact. The
builder emits the stability margin, and applies **hysteresis** when handed the previous quotient:
do NOT un-collapse a node unless its margin clears `tol` plus a band. The band/policy is
deterministic and lives in the builder; the tiering *labels* it might feed still live in reporting.

## 5. Modules and interfaces

New module `spark_vi/models/topic/dag_identify.py`, next to `pg_stm_dag.py`. Four units, each
independently testable. The accumulator mirrors the distributable sufficient-stats idiom
(`empty -> accumulate -> combine`) so the deferred mllib shim is a drop-in.

- **Gram accumulator (kernel):**
  - `closure_gram(dag, doc_nodes) -> G` — pooled `sum_d z_d z_d^T`, `(U_off x U_off)`.
  - `foreground_grams(dag, doc_nodes, doc_groups, partition) -> {g: G_g}` — per-group, intercept
    column included.
  Input is exactly the `(doc_nodes, groups)` the fit already consumes — a pre-pass, no fitting.
- **Spectrum (kernel, threshold-free):**
  - `identifiability_spectrum(G) -> {singular_values, vectors, null_basis}` via a rank-revealing
    factorization. Same call for pooled and per-block Grams.
- **Quotient builder (one numeric tol):**
  - `build_quotient(dag, spectrum, *, tol, prev=None) -> {quotient_dag, node_map, margins, flagged}`
    — adjacency-classifies null directions (collapse vs flag), applies hysteresis if `prev` given.
- **Invariant check:**
  - `quotient_moment_matches_projection(dag, G, quotient) -> residual` — the §4 correctness
    equation; returns the residual the plant test asserts ≈ 0.

Output plugs straight into the existing fit: `PGSTMDag(..., dag=quotient_dag).fit(...)`, with
`node_map` carried for the report. The reporting layer (tiers / names / conditionality
annotations) consumes `margins` + `spectrum` + the conditionality metadata — deferred, not in v1's
core.

## 6. Validation plants (synthetic -> math-correctness only)

Honesty rule on every test: state what is planted vs real, where it sits on the synthetic->real
spectrum, and the claim it supports AND does not; no transfer claim from a synthetic result.

1. **Known confound recovery** — plant a single-child parent with no own-level attestation
   (`z_parent ≡ z_child`); assert the pooled null-space is exactly that direction and
   `build_quotient` collapses exactly those two, keeping the rest.
2. **Multi-parent native** — a diamond where the multi-parent increment is separable in one config
   and confounded in another; assert the null-space names the confounded span with no tree
   assumption.
3. **Cross-tree coincidence -> flag, not merge** — two non-adjacent nodes with identical planted
   support; assert the direction is flagged, never auto-collapsed (the safety split).
4. **Per-block level wall (0054 per-node)** — one anchor with parent-level attestation, one without;
   assert the per-group foreground Gram names the level-vs-intercept direction for the second and
   not the first.
5. **Correctness invariant (headline)** — `quotient_moment_matches_projection` residual ≈ 0 on the
   plants.
6. **Determinism / hysteresis** — perturb counts near `tol`; without hysteresis a near-threshold
   node flips, with `prev` it holds.
7. **Identity sanity** — a fully-identified DAG quotients to itself, and fitting the quotient equals
   fitting the original.

## 7. Deferred (explicit)

- **mllib shim** — distribute the Gram reduce (the accumulator is written `empty/accumulate/combine`
  so this is a drop-in later).
- **Format adapters** — OMOP / SNOMED / MONDO / HPO / hand-authored edge lists → the abstract
  integer-node input.
- **Reporting layer beyond minimal** — tiers (identified / fragile / unidentified), human node
  names, conditionality annotations, hysteresis labeling. Lives with the LLM labeling pipeline.
- **Soft-gate fractional `z` + enrichment-weight logic** — metadata fields exist in v1; the logic
  is later.
- **Fragility ↔ Gibbs-interval-width validation cell** — the payoff that proves the compiler is a
  cheap surrogate for the step-2 read-out engine's uncertainty map; needs that engine to exist.

## 8. Dependencies and boundaries

- **Node count owned upstream.** The compiler resolves the DAG it is given; it assumes a manageable
  node count (`U_off` ~ 10^3, not 10^5). Taming a raw multipath ontology into a sane node set is the
  node-mapping layer's job (the deferred OMOP DAG-builder, which already rejects full-ancestor
  closure for exactly this reason). Letting that leak into the compiler would bloat it.
- **Compiler cost ⊆ fit cost.** The Grams ride on the same moment the fit accumulates; the
  rank-revealing factorization is a millisecond dense op on a hundreds-to-low-thousands square.
