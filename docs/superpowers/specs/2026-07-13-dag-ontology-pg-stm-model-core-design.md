# DAG/Ontology-Structured PG-STM — Model Core Design

**Goal:** Generalize the gated PG-STM's single-per-group topic offset into an
**additive offset summed over a document's ancestral closure in an is-a DAG**, so an
ontology (SNOMED now, MONDO/HPO later) of disease groups provides structured partial
pooling — validated to a fallback guarantee (unearned structure deactivates) on
real-β-seeded plants and a real-data spurious-edge check.

**Architecture:** A new `PGSTMDag` that reuses the validated pg_stm primitives (PG
augmentation, `psi_posterior`, block-Σ assembly, the pure-numpy PG sampler) and adds
(1) a `DagGate` structure (nodes, is-a edges, per-node owned topics, closure
computation), (2) an additive-η mean model `μ_d = Γᵀx_d + Σ_{u∈closure(v_d)} η_u` with a
sparse depth-scaled shrinkage prior on the node offsets, and (3) a closure-indicator
hierarchical-ridge M-step. The degenerate DAG (root + flat group nodes) reproduces the
current model.

**Tech stack:** Python / NumPy / SciPy (single-machine core, mirroring the existing
`spark_vi/models/topic/pg_stm.py`); the distributed SVI + cloud wiring are a later spec.

## Global Constraints

- **Test-honesty rule (load-bearing for this project):** every test's docstring and any
  report it feeds states (i) what is *planted* (synthetic ground truth) vs *real*
  (β borrowed from an existing fit, real length/group distributions), (ii) where it sits
  on the synthetic→real spectrum, and (iii) the claim it supports **and** the claim it
  explicitly does not. A hard line separates *math-correctness* claims (synthetic is
  appropriate — "the estimator recovers what's planted") from *transfer* claims (need
  real data — "this helps our corpus"). **No test's report may assert a transfer claim
  from a synthetic result.**
- **Ontology-agnostic:** the core consumes an is-a edge list + a many-to-many
  patient→node map as given inputs. SNOMED vs MONDO/HPO is a data swap, not a redesign.
- **DAG-native, many-to-many from the start:** a document may attest multiple leaves
  (multiple closures); a node may have multiple parents (diamonds handled by closure
  *sets*, so a shared ancestor is summed once).
- Domain-agnostic engine layer: the core sees integer node ids and integer token ids,
  never concept names/ids (consistent with the rest of `spark_vi`).
- Cite any method/default/constant from the literature in its docstring; a value with no
  citable source is labeled a heuristic, not a grounded default.

---

## Background & motivation

The read-out arc (insights 0044–0049) established that mean-field VI is unfit to feed the
Σ read-out (wrong correlation sign, collapsed rank/scale, β over-sharpened under topic
overlap), that exact Gibbs given correct β recovers Σ, and that the remaining wall is
β-accuracy for **data-scarce, overlapping** foreground topics — precisely the rare/
minority-subgroup case the gated model exists to serve (insight 0049). Partial pooling is
the one honest way to add information to a scarce group without adding documents, and an
is-a ontology supplies the pooling structure for free. The bet is calibrated because the
arc's failures were *estimator* failures, not architecture failures (runaway cured, scale
triangulated, heterogeneity bounded, scarce block located with the oracle ceiling proven
reachable). This spec builds the model core that makes that pooling structure real.

## Scope

**In:** the additive-η-over-closure construction, the sparse depth-scaled offset prior,
the closure-indicator M-step, the `DagGate` input structure, and the four validation
tests (Section: Validation).

**Out (each a later spec):** the OMOP DAG-builder (real `concept_ancestor` + leaf list →
`DagGate`); the read-out honesty layer (shipped posterior intervals over Σ; LKJ/half-t
priors on gated blocks); the soft-gate for unlabeled documents (prevalence / MNAR /
case-finding); the distributed SVI + cloud driver; the hierarchical-gated-LDA-vs-PG
bake-off. The core consumes a `DagGate` as a *given input* (synthetic or real-seeded), so
it is fully testable with no OMOP wiring.

---

## The construction

### Node / DAG representation and closures

A `DagGate` is defined by:
- `nodes`: integer node ids `0..U-1`, with `0` = root (background).
- `parents[u]`: the immediate is-a parents of node `u` (root has none). Multiple parents
  allowed (DAG).
- `owned_topics[u]`: the topic ids attached to node `u` (root owns the background topics;
  each disease/subtype node owns its own topics). Owned-topic sets are disjoint across
  nodes and partition the `K` topics.
- `closure(v)`: the ancestral closure of node `v` = `{v} ∪ all ancestors`, computed as a
  **set** (so a diamond's shared ancestor appears once).
- A document `d` carries `label_nodes[d]`: the set of most-specific attested nodes it is
  gated to (many-to-many). Its full closure is `∪_{v∈label_nodes[d]} closure(v)`, and its
  **allowed topics** are `∪_{u∈closure} owned_topics[u]`.

`depth[u]` = length of the shortest root→u path (used by the offset prior).

### Additive-η mean model (the clean, new part)

Let `Z` be the `D × U` binary **closure-indicator** matrix (`Z[d,u]=1` iff `u` is in
document d's closure), `B` the `U × (K-1)` node-offset matrix (row `u` = η_u over the
stick dimensions), `X` the `D × P` covariates, `Γ` the `P × (K-1)` covariate coefficients.
The per-document mean logits are

    M = X Γ + Z B          (D × (K-1)),   i.e.  μ_d = Γᵀ x_d + Σ_{u∈closure} η_u

and the per-document active sticks are drawn `ψ_d ~ N(μ_d[active_d], Σ[active_d, active_d])`
with PG augmentation on top — identical to the current model except that the mean gains
the additive closure-sum term. This composes with the PG stack because the augmentation
acts on the doc-level likelihood, which is agnostic to how μ_d was assembled.

### Sparse, depth-scaled offset prior (the fallback mechanism)

Each node offset row η_u is shrunk toward 0, harder for deeper nodes:

- **v1 (default):** a per-node ridge penalty `λ_u = λ_base · (1 + depth[u])^γ` on `‖η_u‖²`
  (an adaptive-ridge / hierarchical-Gaussian prior `η_u ~ N(0, (λ_u)^{-1} I)`). Soft
  shrinkage; unearned nodes → η_u ≈ 0 in norm. `λ_base`, `γ` are **structural, inspectable**
  knobs (not inference hyperparameters), defaulted and swept in the fallback plant.
- **Upgrade path (only if the fallback plant shows v1 under-shrinks):** a group-lasso
  penalty on `‖η_u‖` (exact zeros → true deactivation) or a per-node half-t scale
  (horseshoe-style). The prior is *chosen by the fallback plant's shrinkage measurement*,
  not asserted — consistent with the project's replace-the-guess-with-a-measurement rule.

Depth-scaling encodes "prefer general explanations, specialize only on evidence" and gives
diamonds a canonical resolution (deeper of two explanations pays more). The fitted `‖η_u‖`
per node is a shipped diagnostic ("which parts of the ontology the corpus actually uses").

### Covariance and gate over per-document-varying allowed sets (the subtle part — flagged)

**Risk, stated plainly:** flat stick-breaking is not closed under subsetting the allowed
topic set, which is why the current model uses *nested* stick-breaking; the DAG's
allowed set varies per document (it is the closure's owned topics), so the additive-η
mean does **not** by itself resolve Σ/gate consistency. Resolution: **reuse the existing
block-Σ / active-sub-block mechanism** (`assemble_sigma` + per-doc active-set restriction),
keyed on the **closure** instead of the flat group. A document uses the sub-block of Σ
over its allowed sticks; the block scatter for a node's owned sticks is accumulated over
all documents whose closure contains that node (generalizing the current per-group
accumulation, which is the single-level case). Multi-level closures mean a stick can be
active for documents at several depths; the block accumulation sums over the correct
document set per node. **Whether this composes correctly for multi-level closures is the
one construction claim the tests must prove** — Test 1 (equivalence at the flat DAG) and
Test 2 (offset recovery through a two-level closure) are designed to catch a failure here.

### M-step (hierarchical ridge — Γ's machinery generalized)

Given the current ψ means `M` (VI expectations or Gibbs samples), jointly estimate `[Γ; B]`
by penalized least squares on the stacked design `W = [X | Z]`:

    [Γ; B] = (Wᵀ W + Λ)^{-1} Wᵀ M

where `Λ = blockdiag(0_P, diag(λ_u))` (no penalty on covariates; depth-scaled penalty on
node offsets). This is exactly `pg_gamma_ridge_moments` with `W` in place of `X` and a
block penalty — reused, not rebuilt. Σ and β M-steps are unchanged from `pg_stm`.

### Degenerate case = the current model

With `DagGate` = { root owns background topics; one flat child node per group owning that
group's foreground topics; every document's closure = {root, its group node} }, the
closure-indicator design reduces to the current per-group structure and `PGSTMDag`
reproduces `PGSTMVI` numerically. This is Test 1.

---

## Validation (the airtight part)

Four tests. Each carries the Global-Constraints honesty labeling.

### Test 1 — Equivalence to the current flat model (regression; math-correctness)

- **Construction:** the degenerate `DagGate` above; same corpus, seed, and iterations fed
  to `PGSTMVI` and `PGSTMDag`.
- **Proves:** `PGSTMDag` is a strict generalization — the new closure/offset machinery does
  not perturb validated flat-model behavior (β, Γ, Σ match to numerical tolerance).
- **Does not prove:** anything about multi-level DAG behavior.

### Test 2 — Offset recovery through a multi-level closure (math-correctness, realistic overlap)

- **Construction:** a synthetic DAG **structure** (root → anchors → one subtype level) with
  **planted node offsets** `η_u`; topics **β borrowed from a real fit** (e.g. exp 0030 EDS
  or 0050 export) and **real document-length + group-size distributions**; ψ drawn with a
  planted Σ; documents distributed across depths (some at anchors, some at subtypes,
  mirroring the ~50%-general/50%-specific split).
- **Proves:** given a known closure structure, the estimator recovers the offsets that are
  actually present — under *realistic topic overlap* (β is real) and through a two-level
  closure (subtype inherits anchor). Directly exercises the flagged Σ/gate composition.
- **Does not prove:** that real data's true offsets are recoverable (unknown), nor transfer.

### Test 3 — Fallback / spurious-edge shrinkage (the headline)

**3a (synthetic, with ground truth):**
- **Construction:** plant offsets on a **tree only** (a subset of nodes), then fit with a
  DAG that adds **spurious extra edges/nodes** (offsets truly 0); real-β-seeded as in Test 2.
- **Proves:** spurious node offsets shrink to ≈0 in norm **and** tree-region estimates match
  a tree-only fit — "reduces to the simpler model where the data is tree-like," measured.
- **Does not prove:** that surviving structure is correct — only that noise dies.

**3b (real-data, no ground truth — the antidote to plant-distrust):**
- **Construction:** take the (later) real DAG, **inject random cross-edges between
  unrelated diseases**, fit on the **real corpus**, measure the injected offsets' norms.
- **Proves, on real data, with no planted truth:** injected-spurious offsets die (they are
  spurious *by construction* — added at random). This is the fallback guarantee's real-data
  form; it does not depend on any synthetic plant. (Runs when the OMOP builder exists; the
  spec records it here so the plan wires the hook, even though it executes in a later phase.)
- **Does not prove:** recovery magnitudes or coverage (unmeasurable on real data — Tests 2/4).

### Test 4 — Coverage / calibration (honesty of shipped width)

- **Construction:** on the Test-2 real-β plant, over repeated planted draws, check the
  posterior offset (and Σ) credible intervals cover the planted values at nominal rate
  (90% interval ⊃ planted ~90% of the time), reported **separately for well-populated vs
  scarce nodes**.
- **Proves:** the width we would ship is calibrated — scarce nodes come back wide-and-
  covering (honest UQ) rather than tight-and-wrong.
- **Does not prove:** coverage on real data (unmeasurable — which is why 3b exists). Depends
  on posterior draws from the Σ/offset read-out; if the read-out honesty layer is a later
  spec, Test 4 covers *offset* intervals here and Σ intervals move to that spec. (Planning
  decision — see Open questions.)

---

## File layout & interfaces

- **New:** `spark-vi/spark_vi/models/topic/pg_stm_dag.py`
  - `DagGate` (dataclass): `parents`, `owned_topics`, `depth`, `closure(v)`,
    `closure_indicator(label_nodes) -> (U,) binary`, `allowed_topics(label_nodes)`.
    Includes `dump()` → per-node `{id, depth, parents, owned_topics, n_docs}` for audit
    (the inspectable-knob requirement).
  - `dag_offset_mstep(M, X, Z, *, lam_base, gamma, depth) -> (Gamma, B)` — the closure-
    indicator hierarchical ridge (reuses `pg_gamma_ridge_moments`' solve).
  - `PGSTMDag` — driver mirroring `PGSTMVI`: per-doc E-step reuses `pg_estep_doc` with the
    allowed set from the closure and mean `Γᵀx + Bᵀz`; M-step calls `dag_offset_mstep` +
    the unchanged Σ/β steps; degenerate DAG == `PGSTMVI`.
  - `degenerate_flat_dag(partition) -> DagGate` — builds the flat DAG from a
    `TopicBlockPartition` (for Test 1 and as the migration bridge).
- **Extend:** `spark-vi/tests/_stm_synth.py` — `dag_offset_corpus(*, dag, offsets, beta,
  length_dist, node_assignment, sigma_true, seed)` that plants offsets on a given `DagGate`
  using **supplied real β** and real length/assignment distributions; a `real_beta_from(...)`
  loader that pulls β from an existing export bundle.
- **New:** `spark-vi/tests/test_pg_stm_dag.py` — one test per Section-Validation item, each
  with the planted-vs-real / proves-vs-doesn't docstring.
- **Reused unchanged:** `pg_stm.py` primitives (`omega`, `psi_posterior`, `assemble_sigma`,
  `_pg.py` sampler, `pg_gamma_ridge_moments`), `_mcmc_diag.py` (coverage/convergence).

Interfaces the later specs consume: `DagGate` (built by the OMOP builder), `PGSTMDag.fit`
returning `{beta, Gamma, B, Sigma, node_norms, ...}`.

---

## Load-bearing risks

1. **Σ/gate composition over multi-level closures** (flagged above) — the one genuinely new
   correctness claim. Mitigation: Tests 1 + 2 catch it before any real run; if it fails, the
   fallback is per-closure-block Σ assembly with explicit multi-membership accounting, to be
   designed in the plan.
2. **Prior shrinkage strength** — v1 depth-scaled ridge may under-shrink spurious edges
   (ridge → near-zero, not exact zero). Mitigation: Test 3a *measures* it; upgrade to
   group-lasso/horseshoe only if the measurement demands it.
3. **Diamond identifiability** (offset decomposition at document-poor multi-parent nodes) —
   real but *localized* and honestly represented by posterior width (Test 4). Not a v1
   blocker; multi-parent nodes are earned, and the depth-scaled prior gives diamonds a
   canonical resolution.

## Deferred / follow-on specs

OMOP DAG-builder · read-out honesty (Σ posterior intervals + LKJ/half-t) · soft-gate for
unlabeled docs (prevalence/MNAR/case-finding) · distributed SVI + cloud · hierarchical
gated-LDA-vs-PG bake-off.

## Open questions for planning

- **Prior form for v1:** depth-scaled ridge (default) vs group-lasso from the start. Default
  ridge; revisit after Test 3a.
- **Coverage scope (Test 4):** offset intervals only in this spec vs also Σ intervals (pulls
  the read-out honesty layer partly forward). Default: offset intervals here; Σ intervals in
  the read-out spec.
- **VI vs Gibbs for the core:** the read-out arc says Σ needs Gibbs, but the *offset/fallback*
  math is estimator-agnostic. Default: build/validate the offset + fallback machinery under
  the existing VI M-step (fast, and the fallback is a mean-model property); the Σ read-out's
  Gibbs requirement is inherited from the read-out spec, not re-litigated here.
