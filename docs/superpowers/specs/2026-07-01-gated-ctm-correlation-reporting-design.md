# Gated CTM: Honest Correlation Reporting + Multi-Membership Representation — Design

**Goal:** Surface a trustworthy topic-correlation matrix R in the dashboard for the
gated Correlated Structural Topic Model, sourced from the fitted logistic-normal Σ
with a support-keyed "identified" mask, and enable a multi-membership document
representation so cross-foreground (cross-disease) correlations become estimable from
comorbid patients where they exist.

**Architecture:** Three layers, respecting the existing domain boundary. (1) The
domain-agnostic engine (`spark-vi`) already estimates the within-group Σ blocks and
computes the logistic-normal correlation; it gains persisted per-pair support counts
and a correlation-with-mask helper. (2) The OMOP layer (`charmpheno`) gains a
multi-membership DocSpec and a correlation export. (3) The dashboard (`dashboard/`)
gains one heatmap panel. The model fit is unchanged in its mechanics — `pd_complete`
stays as the fit-time completion; only the misleading conditioning diagnostics and
their framing are removed.

**Tech Stack:** Python / NumPy / SciPy (`spark-vi` engine), PySpark (`charmpheno`
OMOP prep + cloud drivers), TypeScript / Vite (`dashboard/`).

## Global Constraints

- `spark-vi` stays domain-agnostic: integer token ids and topic indices only; no
  OMOP/EHR vocabulary, cohort names, or clinical semantics in the library. The
  correlation matrix, the identified mask, and the support counts are all expressed
  over topic indices — domain-free. Cohort/group semantics live only in `charmpheno`.
- No LaTeX in docs or docstrings. Use plain text and Unicode Greek (Σ, η, θ, μ, Γ, δ,
  λ, ν).
- Any method, default, or constant drawn from the literature cites its source in the
  docstring. No citable source means it is a heuristic and is labelled as one.
- No personal information in committed artifacts (code, docs, ADRs, specs).
- Code references in prose use markdown links (`[name](path#Lstart-Lend)`).
- Row-level document output that prints identifier columns hashes them
  (SHA-256-truncated) before display; aggregates and probabilities may print raw.
- Execution follows subagent-driven development with TDD per task.

---

## Background — why this design

Three findings from the investigation drive the design; they are recorded in full in
the doc amendments (Component 5), summarized here as motivation.

1. **The gated model is genuinely a Correlated Structural Topic Model.** Each
   document's η-inference uses the full marginal covariance over its allowed topic
   set, `Σ[allowed, allowed]` ([stm.py:777](../../spark-vi/spark_vi/models/topic/stm.py#L777)),
   whose off-diagonals couple topics — the defining CTM mechanism (Blei & Lafferty
   2007). The "Structural" mean regression μ_d = Γᵀx_d and the block-gating are intact.
   Reporting correlations is therefore reporting a real, identified model quantity.

2. **The gated Σ's cross-foreground block is a matter of identifiability, not a
   structural zero.** A topic-pair covariance Σ_ij is identified when documents can
   co-realize both topics. Background↔background, background↔foreground, and
   within-foreground blocks are identified from single-group documents. The
   cross-foreground block (a cancer-specific topic vs a dementia-specific topic) is
   identified only from documents that carry BOTH groups — comorbid patients. Under
   the current split representation ([PatientCohortDocSpec](../../charmpheno/charmpheno/omop/doc_spec.py#L232),
   `doc_id = {source_cohort}:{person_id}`) no document is multi-group, so the
   cross-foreground block is structurally unobserved and `pd_complete` fabricates it.

3. **The full-matrix condition number the earlier arc fought is a reporting
   artifact.** The fit only ever inverts within-allowed-set marginal sub-blocks
   (`safe_inverse(Σ[allowed, allowed])`), which are repaired per document. The
   cross-foreground entries never enter any single-group document's inference. The
   full assembled Σ and its condition number appear only in reporting
   ([topic_correlation](../../spark-vi/spark_vi/models/topic/_linalg.py) and the
   `sigma_cond` / `max_abs_offdiag_corr` diagnostics). Topic recovery was empirically
   invariant to that condition number (held at 4–5/8 planted topics even at cond
   ≈ 2e9 on the synthetic harness). The exp 0022/0023/0024 lever hunt and the
   conditioning framing of insight 0032 optimized a number the model does not use.

The design follows: gate the fit (unchanged, it works and surfaces minority
sub-phenotypes), report correlations honestly from the within-group Σ blocks with an
identified mask, keep `pd_complete` as the principled fit-time completion (now
genuinely used once multi-group documents exist), and enable multi-membership so the
cross-foreground block can be estimated from comorbid patients rather than fabricated.

---

## Component 1 — Multi-membership representation

**What it does.** Adds a document representation in which a comorbid patient becomes
ONE document spanning all their cohorts, so the cross-foreground covariance block is
estimated from real joint data instead of fabricated.

**Unit.** New `PatientMultiCohortDocSpec` (`doc_unit = "patient_multicohort"`) in
[charmpheno/charmpheno/omop/doc_spec.py](../../charmpheno/charmpheno/omop/doc_spec.py),
a sibling of `PatientCohortDocSpec`. It is additive: `patient_cohort` (split) stays
and remains the default; experiments opt in via the `doc_unit` frontmatter key.

**Interface / data flow.**
- `derive_docs`: `doc_id = person_id` (one document per patient, merging all their
  events across cohorts), and it carries a **`groups` array column** = the sorted
  distinct `source_cohort` values that patient belongs to. It does NOT encode the
  cohort in `doc_id`.
- Downstream, the STMDocument's `groups` frozenset is built directly from the `groups`
  array column, replacing the `doc_id.split(":")` recovery at
  [stm_bigquery_cloud.py:303-305](../../analysis/cloud/stm_bigquery_cloud.py#L303-L305).
  The split-representation path (reading a single `source_cohort`) is preserved for
  `patient_cohort`; the multi-cohort path reads the array.
- **Covariate keying** simplifies to `person_id`. With one document per patient, the
  patient-level covariates (age, sex) key cleanly on `person_id`, removing the
  `(person_id, source_cohort)` composite keying that exists only because split docs
  double-count comorbid patients ([stm_bigquery_cloud.py:307-310](../../analysis/cloud/stm_bigquery_cloud.py#L307-L310)).
- **Eval per-cohort NPMI** ([eval_coherence_cloud.py:272](../../analysis/cloud/eval_coherence_cloud.py#L272))
  currently splits `doc_id` to assign a document to one cohort's reference corpus. It
  reads the `groups` array instead: **a comorbid document contributes to BOTH cohorts'
  reference corpora** (confirmed design choice). Single-group documents are unchanged.

**Consequences (intended, require a re-fit).** Comorbid patients move from two
half-documents to one merged document; total document count drops; the merged
documents carry both groups' foreground topics in their allowed set, giving the
cross-foreground block real support. The gating variable stays out of the covariate
formula (ADR 0026), so no rank-deficiency interaction. The re-fit is already planned
in the finalization arc.

**Edge cases.** A patient with a single cohort under `patient_multicohort` yields a
single-group document identical in spirit to today's — the representation degrades
gracefully to the split behavior for non-comorbid patients. A comorbid patient with
very few events still produces one document; the cross-foreground support it
contributes is counted honestly (Component 3's mask handles thin support).

---

## Component 2 — Fit side

**What changes.** Almost nothing in mechanics; the change is removing a misleading
diagnostic and its narrative, and persisting one statistic.

- **Keep `pd_complete`** ([_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py))
  as the fit-time completion of the assembled Σ. Once multi-group documents exist, a
  multi-group document's prior needs a value for the cross-foreground block in
  `Σ[allowed, allowed]`; `pd_complete` supplies the zero-precision
  (conditional-independence) completion where support is thin, which is the principled
  max-entropy fill (Dempster 1972; Grone et al. 1984; Lauritzen 1996). `safe_inverse`
  continues to stabilize each per-document sub-block.
- **Remove the conditioning-optimization apparatus.** The `sigma_cond` and
  `max_abs_offdiag_corr` diagnostics that framed the full-matrix condition number as a
  pathology are removed from the iteration diagnostics and eval readouts
  ([stm.py:812-818](../../spark-vi/spark_vi/models/topic/stm.py#L812-L818),
  [850-858](../../spark-vi/spark_vi/models/topic/stm.py#L850-L858)). They measured a
  reporting artifact the fit does not use. If a diagnostic is retained for monitoring,
  it is computed over the within-group (identified) blocks only and labelled as a
  reporting statistic, not a fit-health signal. (The IW-prior and diag-shrink knobs are
  already gone from the branch; this removes their remaining diagnostic scaffolding and
  the framing, not additional levers.)
- **Persist per-pair support N.** The final `n_pairs_stat` (K×K per-pair document
  support already accumulated in the M-step,
  [stm.py:572](../../spark-vi/spark_vi/models/topic/stm.py#L572)) is saved with the
  model so the reporting layer can build the identified mask without a re-pass. It is
  a topic-index-keyed integer matrix — domain-agnostic.

---

## Component 3 — Honest correlation reporting

**What it does.** Produces the dashboard's correlation artifact from the fitted Σ,
with every cell labelled identified or NA by its support.

- **Source: logistic-normal Σ.** R_ij = Σ_ij / sqrt(Σ_ii · Σ_jj), the Blei & Lafferty
  2007 topic correlation, already implemented as
  [topic_correlation](../../spark-vi/spark_vi/models/topic/_linalg.py) and
  `OnlineSTM.topic_correlation_matrix`. This is preferred over an empirical θ
  correlation: it is the model's native quantity, needs no separate document pass, and
  avoids the simplex/compositional bias of θ-correlation.
- **Identified mask.** Cell (i,j) is identified iff `N_ij ≥ min_pair_support` — the
  same floor the M-step uses to decide estimated-vs-completed
  ([stm.py:711](../../spark-vi/spark_vi/models/topic/stm.py#L711)). Identified cells
  carry R; unidentified cells are NA. Under the split representation the entire
  cross-foreground block is NA; under multi-membership its cells populate wherever
  comorbid support clears the floor.
- **Engine helper (domain-agnostic).** A `spark-vi` function returns `(R, identified)`
  from `(Σ, N, min_pair_support)` over topic indices. No cohort semantics.
- **Export (`charmpheno`).** New
  [charmpheno/charmpheno/export/correlation.py](../../charmpheno/charmpheno/export/)
  (sibling of `gating.py`) writes **`correlation.json`**:
  `R` (K×K, NA where unidentified), `identified` (K×K bool), `support` (K×K int N),
  `topic_order` (kept-topic ids in block order), and `block_labels` (background /
  group labels — the domain layer attaches names). Wired into
  [analysis/local/build_dashboard.py](../../analysis/local/build_dashboard.py) and the
  cloud dashboard builder alongside the existing sidecars.

**Error handling.** A degenerate diagonal (Σ_ii ≤ 0) cannot occur for a fitted model
(variances are positive), but the helper guards it: a non-positive diagonal marks that
topic's row/column NA rather than dividing by zero. Suppressed/gated-out topics
(already dropped from the dashboard via `gating.json`) are excluded from `topic_order`.
The **reference topic** (K−1 parameterization, ADR 0031) is inert in Σ — unit-variance
diagonal, zero cross-entries by construction — so its correlations are not meaningful;
it is excluded from `topic_order` in the reported matrix (consistent with how the
dashboard already treats it), rather than shown as a spurious zero-correlation row.

---

## Component 4 — Dashboard heatmap

**What it does.** One heatmap panel rendering `correlation.json`.

- Topics ordered by block (background, then each group's foreground), with block
  separators; block labels from `block_labels`.
- Diverging color scale over R from −1 to 1; **NA cells greyed** with a tooltip
  ("no joint support: N < min_pair_support"). Identified cells show R on hover with
  the pair's support N.
- Reads the sidecar through [dashboard/src/lib/bundle.ts](../../dashboard/src/lib/bundle.ts),
  following the existing sidecar-loading pattern. Scope is one panel; no reorder/toggle
  interactivity (that would be a separate frontend effort).

---

## Component 5 — Documentation

- **Amend insight 0032** — add a resolution section: the fit uses only within-group
  marginal sub-blocks (repaired per-document by `safe_inverse`); the cross-foreground
  block is unidentified under the split representation and never enters single-group
  inference; the full-matrix condition number is a reporting artifact and topic
  recovery is invariant to it (with the synthetic-harness evidence). This supersedes
  the Findings-4-through-6 framing that treated conditioning as a fit pathology.
- **Amend ADR 0033** — reframe `pd_complete`: it is the fit-time completion that gives
  multi-group documents a coherent cross-foreground prior, NOT a conditioning cure.
  The exp 0022/0023/0024 lever hunt is recorded as having optimized a reporting
  artifact.
- **Amend exp 0025** — reframe its success criteria: it validates topic recovery and
  the honest correlation report, not a condition-number target.
- **New ADR** — "Topic-correlation reporting via identified mask + multi-membership
  representation": records the CTM framing, the identifiability-by-support principle,
  the logistic-normal source choice, the keep-pd_complete-drop-diagnostics decision,
  and the multi-membership representation with the both-references eval choice.

All amendments follow the impersonal, no-LaTeX, cite-the-literature conventions.

---

## Component 6 — Testing strategy

- **Retained (already written):** the Layer-1 deterministic conditioning tests and the
  term-sharing synthetic generator (Jaccard ≈ 0.33, matching the real HF β) in
  [spark-vi/tests/test_stm_pd_completion_conditioning.py](../../spark-vi/tests/test_stm_pd_completion_conditioning.py)
  and [_stm_synth.py](../../spark-vi/tests/_stm_synth.py). They correctly characterize
  `pd_complete` on completable and non-completable observed patterns and stay.
- **Engine (`spark-vi`, domain-agnostic):**
  - correlation-with-mask helper: identified cells equal `topic_correlation`;
    `N < min_pair_support` cells are NA; unit diagonal; symmetry.
  - persisted support N round-trips through save/load.
  - **"recovery is invariant to full-Σ cond"**: on the harness, planted-topic recovery
    holds while the full-matrix condition number varies by orders of magnitude — the
    finding, on the record.
- **OMOP layer (`charmpheno`):**
  - `PatientMultiCohortDocSpec.derive_docs`: one doc per patient; `groups` array = the
    patient's distinct cohorts; a comorbid patient yields one merged document.
  - covariate keying on `person_id` for the multi-cohort path; per-cohort eval
    reference assignment puts a comorbid doc in both cohorts' references.
  - `correlation.py` export: cross-foreground cells NA under `patient_cohort`;
    populated where comorbid support clears the floor under `patient_multicohort`.
- **Dashboard (`dashboard/`):** `bundle.ts` loads `correlation.json`; the panel greys
  NA cells and renders identified cells (component/render test following the existing
  test pattern).

---

## Component independence and sequencing

Components 1 (representation) and 3 (reporting) are independent: reporting works under
either representation, differing only in how many cross-foreground cells are NA.
Sequencing is therefore flexible; a natural order is engine reporting (2 + 3) first —
delivering the honest heatmap immediately under the current split data — then the
multi-membership representation (1) to light up the cross-foreground cells, then docs
(5) and the dashboard panel (4) once the artifact is stable. The implementation plan
will fix the concrete order.
