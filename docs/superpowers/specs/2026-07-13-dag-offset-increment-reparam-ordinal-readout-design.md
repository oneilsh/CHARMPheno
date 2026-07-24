# DAG Offset Increment Reparameterization + Ordinal Read-Out — Design

**Status:** approved (brainstorm 2026-07-13). Step 1 of the post-v1 DAG plan
(increment reparam → read-out engine → Piece B).

**Goal:** Make the parent→child **increment** the primary parameterization of the
DAG offset block (the identified coordinates, per insight 0050), removing the
unidentified direction the ridge currently splits arbitrarily; and reshape the v1
offset-uncertainty read-out into an honest **ordinal** artifact (no absolute widths,
machine-visible calibration status), per Fable's two rules and insight 0052.

**Scope:** one focused refactor of `spark-vi/spark_vi/models/topic/pg_stm_dag.py`,
`spark-vi/tests/_stm_synth.py`, and `spark-vi/tests/test_pg_stm_dag.py` on branch
`pg-stm`. No new module. **Out of scope:** the exact-conditional (Gibbs) read-out
engine that produces *calibrated absolute* intervals (step 2), and node-owned topics
/ multi-level gate-Σ composition (Piece B, step 3).

---

## Background (why)

Insight 0050: under a partitioning gate the individual node offsets are un-identified
(the root's offset column equals the covariate intercept; the anchor columns sum to
the intercept), so the ridge splits a phantom direction arbitrarily. What **is**
identified is each node's parent→child increment. The additive-over-closure mean
`μ_d = Γᵀx_d + Σ_{u∈closure(v_d)} η_u` is already a telescoping sum of increments —
this change reads the model in its own coordinates rather than treating "increments"
as a post-hoc read-out convention.

Insight 0051/0052: the cheap ridge-conditional interval built from mean-field
posterior means orders uncertainty correctly but is severely overconfident in
absolute terms (it is blind to the mean-field bias). Until the step-2 engine lands,
the read-out must ship as **ordinal only**, and must be shaped so it cannot be
mistaken for a calibrated interval.

---

## Part A — Increment reparameterization

**Drop the root's offset column.** The offset design becomes the closure indicator
with the root entry omitted; the root's constant contribution is carried by the Γ
intercept. `B[root] ≡ 0` by construction (not estimated). Every non-root node's
coefficient is its **increment over its parent** (additive-over-closure already makes
`B[u]` the marginal contribution on top of u's ancestors; dropping the redundant root
column makes the augmented design full-rank on those increments).

- **Unchanged:** the E-step, the Σ M-step, the β M-step, the gate. This is purely the
  offset design matrix and its per-row penalty. Piece-A isolation is preserved.
- **Soft identification of internal nodes with no direct documents.** When an internal
  node (anchor) has no directly-attesting documents (all members are coded at more
  specific descendants), the likelihood identifies only the leaf **path-sums**
  `η_anchor + η_subtype`, not the split. The depth-scaled ridge resolves the split in
  the intended direction: because `λ_u = λ_base·(1+depth_u)^γ` grows with depth, the
  shallower anchor is cheaper, so the shared subtype level is attributed to the
  anchor (`η_anchor/η_subtype = λ_subtype/λ_anchor > 1`). The anchor absorbs the
  common signal ("the general node inferred from its subtypes"); the subtype
  increments capture only subtype-specific deviations. No anchor-only documents and no
  sum-to-zero constraint required — the interpretation "increment → 0 ⇔ child equals
  parent unless evidence" is retained.
- **Honesty consequence:** for a no-direct-docs anchor the split is prior-driven, so
  the anchor's own level is only weakly identified and its uncertainty is **wide**;
  the leaf path-sums and subtype-vs-subtype contrasts stay tight. The read-out (Part
  B) surfaces this via the `identified` flag.
- **Forward benefit:** the full-rank design also makes the step-2 Gibbs Γ-block
  well-conditioned.

**Representation:** keep offset arrays (`B`, `node_norms`, internal variances) indexed
by node id `0..U-1` with the root entry defined as 0/not-estimated, so downstream can
index by node id uniformly. `root_only_dag()` now yields an **empty** offset block
(0 non-root nodes) → PGSTMDag on a root-only DAG is exactly PGSTMVI with no offset
augmentation (a cleaner equivalence than the prior collinear-intercept version).

## Part B — Ordinal read-out object

`fit()` **stops returning `offset_cov_diag`** (no absolute widths in the exported
object, Rule 1). It returns `offset_uncertainty`, an object with:

- per node: `rank` — global uncertainty ordering (1 = most resolved … N = least),
  the "which parts of the ontology the corpus resolves" diagnostic;
- per node: `parent_ratio` — this node's internal uncertainty ÷ its min-depth
  parent's (the local pooling comparison; `None` for the root and for any node whose
  parent is the root-absorbed intercept);
- per node: `identified` — boolean, data-identified vs prior-dominated, set from
  whether the ridge posterior variance sits appreciably below the prior variance
  `1/λ_u` (i.e., the likelihood added information); the exact ratio threshold is a
  documented default;
- top-level: `calibration: "ordinal"` — machine-visible status (Rule 2), so consumers
  gate programmatically and flipping to `"absolute"` when the step-2 engine lands is a
  schema change, not archaeology.

The raw per-node variances are still computed internally (to derive rank / ratio /
identified, and to feed the step-2 engine) but are **not** exported as widths.
`node_norms` (point-estimate increments) stay in the return.

**Multi-parent nodes (diamonds):** `parent_ratio` uses the min-depth parent; this is a
documented v1 default (rare at this stage).

---

## Validation (synthetic → math-correctness only; honesty rule intact)

The plant (`dag_offset_corpus` / a variant) gains a **no-direct-docs anchor**
(all members at subtypes) alongside an anchor **with** direct documents and a
background/root-only arm. Tests assert:

1. **Path-sums recovered in both anchor cases** — `η_anchor + η_subtype` recovered
   for a with-direct-docs anchor and a no-direct-docs anchor (the always-identified
   quantity).
2. **With-direct-docs anchor:** its own level identified/tight and flagged
   `identified = True`.
3. **No-direct-docs anchor:** its own level prior-dominated and flagged
   `identified = False` (the honesty case).
4. **Ordinal ranking:** the `rank` orders a data-scarce node above a well-populated
   one (the design-moment property that already gives width-ratio ≈ 2.0), asserted on
   the ordinal object — not on raw widths.
5. **Calibration status:** `offset_uncertainty.calibration == "ordinal"`.

Existing tests adapt:
- **Test 1 (equivalence):** root-only DAG → empty offset block → exactly PGSTMVI;
  update the `B` shape expectation accordingly (the β/Σ match is unchanged, in fact
  cleaner).
- **Test 4:** asserts on the new ordinal object (rank ordering / parent_ratio) instead
  of raw `offset_cov_diag` widths. The docstring keeps the 0050/0051 honesty framing.

Every test states planted-vs-real and refuses transfer claims; synthetic proves math
correctness only. Domain-agnostic engine layer preserved: integer node/token ids only.

---

## Success criteria

- `PGSTMDag.fit` no longer estimates a root offset; the augmented offset design is
  full-rank; root-only DAG reproduces PGSTMVI.
- `fit()` returns `offset_uncertainty` (rank / parent_ratio / identified +
  `calibration: "ordinal"`) and no longer exports raw offset variances.
- The no-direct-docs anchor is validated: path-sum recovered, anchor level flagged
  `identified = False`, scarce nodes rank above populated nodes.
- Full `test_pg_stm_dag.py` green; no absolute-coverage claim anywhere.
