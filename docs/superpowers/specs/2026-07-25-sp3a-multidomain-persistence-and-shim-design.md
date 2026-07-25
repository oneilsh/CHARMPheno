# SP3a — Multi-domain persistence + mllib shim — Design

**Date:** 2026-07-25
**Status:** Approved (user, 2026-07-25). Ready for a plan.
**Branch:** `multidomain-spectral-init` (SP2 complete at `28f1f71`, unmerged)
**Arc:** `docs/superpowers/specs/2026-07-24-multidomain-gated-lda-arc-design.md` — this expands that document's SP3 stub, which is split here into SP3a (this doc, `spark-vi` only) and SP3b (`charmpheno` + `analysis/cloud`).

## Goal

Make a multi-domain gated LDA fit **persistable and deployable**: a fit can be saved, loaded back, and served through the MLlib shim with the fitted ω applied, so that a cloud fit produces an artifact the export/dashboard path can consume. SP2 built the statistical core; nothing downstream can use it yet because a per-domain dict λ cannot be written and the shim cannot read one.

## Why this is SP3a and not all of SP3

The arc's SP3 stub spans three layers: `spark-vi` engine persistence, the MLlib shim, and `charmpheno` drug-domain assembly plus a cloud driver. The third depends on the first two and lives in a different package with different constraints (clinical vocabulary is *permitted* there and *forbidden* here). Splitting keeps each plan reviewable and lets SP3a land without waiting on cohort-assembly decisions.

## Decision: the scalable-init blocker is dissolved, not deferred

The arc design recorded as SP3's first item that the per-domain candidate floor exists only on the dense init path, that the production shim routes to the *scalable* path above `spectralMaxVocab`, and that the floor was worth "recovery 0.005 vs 0.675" — leaving production multi-domain fits unvalidated on the code path they take. It pre-registered an untested **immunity hypothesis** and instructed that it be tested first rather than assumed.

**It was tested (2026-07-25, throwaway probe) and the answer is stronger than the hypothesis.** One planted two-domain corpus, three seeds compared, post-EM per-domain recovery (50 full-batch iterations, the same quantity the dense acceptance test asserts):

| seed | domain 0 (nodes 1/2/3) | domain 1 (nodes 1/2/3) |
|---|---|---|
| dense **with** per-domain floor | 0.895 / 0.701 / 0.492 | 0.826 / 0.699 / 0.995 |
| dense **without** the floor | 0.933 / 0.703 / 0.530 | 0.970 / 0.737 / 0.882 |
| **scalable** (no floor available) | 0.951 / 0.686 / 0.589 | 0.984 / 0.706 / 0.964 |

The scalable path matches or beats the dense+floor seed on five of six cells. **No per-domain rule is needed on the scalable path**, and no `domain_bounds` plumbing there: SP3a's deliverable is the *test* plus a docstring recording the equivalence, which is exactly the arc design's "if immune" branch.

Two secondary observations from the same probe, both to be confirmed by the committed test before anything is claimed durably:

1. **The floor's value looks like a plant artifact.** On a *degenerate* corpus (no background documents, evenly-split closure signatures — the regime insights 0065/0066 were measured in), dense-without-floor has three near-dead cells post-EM (0.015 / 0.022 / 0.004) and the floor rescues them. On a well-specified corpus the two are indistinguishable. If the committed multi-seed test reproduces this, it questions insight 0065's framing of the floor as load-bearing and earns an insight entry — the third instance of the pattern behind 0067 and 0068. **Do not write that insight until the multi-seed test confirms it**; the probe is one corpus at one fit seed.
2. **The scalable path was the most robust of the three on the degenerate plant** (no cell at the uniform floor), consistent with ADR 0032's reasoning for preferring an absolute document-frequency floor over a mean-relative one.

## Components

### 1. Dict-λ persistence (`spark_vi/io/export.py`)

- `save_result` writes one `params/lambda_<m>.npy` per domain and records the per-domain sizes in the manifest; `load_result` reconstructs `{m: (K, V_m)}`.
- The existing `UnsupportedGlobalParamError` guard stays as the fallback for values that are genuinely unsupported (it currently catches the dict λ; after this change it should no longer fire for that reason). Its `dtype.hasobject` check is unchanged.
- `format_version` bumps; `load_result` must still read the previous version's single-array λ.

### 2. Metadata completion (`get_metadata`)

Add `domains`, `omega`, `eta_m`. Without these a saved multi-domain result is not reconstructable even once the writer exists — the model cannot be rebuilt to interpret its own λ.

### 3. η provenance (arc blocker 4) — resolved by invariant, not by plumbing

Multi-domain `update_global` / `compute_elbo` read per-domain η from instance state (`self._eta_domains`); the single-domain path reads `global_params["eta"]`, and `OnlineLDA.compute_elbo`'s docstring says that is deliberate so an η-optimization update mid-fit feeds back.

`optimize_eta` is already rejected for `GatedOnlineLDA`, so **η cannot change during a fit and the two sources are equivalent by construction.** Therefore: write η_m into the manifest, document that a reconstructed fit takes η from the model, and **add a guard that raises if η optimization is ever enabled in multi-domain mode.** The guard is the deliverable — it is what keeps the invariant true, and it converts a latent divergence into a loud failure if someone later enables the feature.

### 4. Shim: per-domain feature columns (`spark_vi/mllib/topic/gated_lda.py`)

**User decision:** separate per-domain feature columns, not one concatenated column.

- New `featuresCols` Param: an ordered list of column names, each holding a sparse vector over that domain's own vocabulary. `featuresCols=None` keeps the existing single-`featuresCol` path **byte-identical**.
- `domainBounds` is **derived once from the per-column vector sizes of the fit dataset's first row** (a single Spark action at `_fit` entry), recorded on the estimator, and then **every row is validated against it** inside the existing row mapper. A row whose per-column vector sizes disagree with the recorded layout raises a named error identifying the column and both sizes. An explicit `domainBounds` Param may override the derivation, in which case the first row is validated like any other — this is the escape hatch for a dataset whose first row is unrepresentative, and it keeps derivation from being the only path. This validation is not optional: silently re-laying-out the vocabulary would corrupt a fit invisibly, and it is the one failure mode the concatenated-column option would not have had.
- Ingest concatenates the per-domain vectors into the engine's single concatenated id space (the engine's representation is unchanged — domain membership is `searchsorted(domain_bounds, w)`).
- `omega` and per-domain `eta` become Params.
- `GatedLDAModel._transform` learns dict λ — it currently raises via `lam.sum(axis=1)` — and **applies the fitted ω**, so deployed θ matches fitted θ. SP2's review confirmed there is no train/serve skew inside the engine (`infer_local` applies ω); this closes the same gap at the shim.
- The Model persists the bounds and ω it was fit with, so `transform` cannot disagree with `fit` about the layout.

## Validation / acceptance

1. **Scalable-vs-dense per-domain recovery, at least 3 fit seeds** — the arc's pre-registered acceptance item, now expected to pass. The gate is that the scalable seed is not materially worse than the dense+floor seed per node and domain (a stated tolerance, justified by the observed seed-to-seed spread, not a single-seed comparison — SP2 lost a review round to exactly that mistake). Records the equivalence in a docstring, and reports whether the dense floor's own value reproduces as plant-dependent (the finding above, which earns an insight entry only if it does).
2. **save → load → identical `nodeAffinity`** on a multi-domain fit: the round-trip must reproduce the read-out, not merely load without error.
3. **Spark-local multi-domain fit through the shim** with two feature columns, extending `tests/test_gated_lda_shim.py`'s existing unmarked-smoke convention.
4. **Row-validation failure** raises a named error for a mis-sized vector.
5. **Backward compatibility:** `featuresCols=None` byte-identical; previous-format `load_result` still works; `tests/test_lda_contract.py` green.

## Out of scope

- **Mid-fit checkpoint/resume.** The writer lands here; the resume loop does not. A preempted cloud fit still loses its progress — named as an SP3b/SP4 prerequisite if fit sizes make preemption real.
- Drug-domain assembly and the cloud driver — SP3b.
- ω tuning against a task metric — SP4. Per insight 0069 the `theta_contribution_by_domain` stat **cannot** substitute for it.

## Constraints carried forward

- **Domain-neutral naming is binding in `spark_vi/**` and `spark-vi/tests/**`:** integer token ids and domain sizes only; no clinical/OMOP/EHR vocabulary in code, comments, docstrings or tests; domains are `0`/`1`/`m` (or `a`/`b`). (There is a known pre-existing violation in `dag_placement.py` reaching the API surface — `disease_mass`, `auc_disease_mass`, `ap_disease_mass` — which is *not* to be fixed opportunistically here; renaming those keys breaks consumers and needs its own scoped decision.)
- Backward compatibility is hard: `domains=None` byte-identical, base `OnlineLDA`/vanilla LDA/HDP untouched, `omega=None` identity.
- No LaTeX; Unicode Greek or the file's existing ASCII spellings. Cite literature for any method, default or constant.
- **Any planted corpus uses `bg_frac` > 0 and `ancestor_signature_decay` < 1** (insights 0067, 0068). A background-starved or label-unidentifiable plant produces convincing false negatives; two were escalated as engine defects during SP2 before being traced to the plant.
- **Verbatim from the SP2 plan, because it did more work than any other line in it:** if short of a gate, STRENGTHEN THE PLANT, NEVER loosen the assertion; a strong-plant failure is a genuine negative — STOP and report.
- **New for this plan, per SP2's final review:** each task that adds a gate must name one mutation of the code under test and show the assertion fires. SP2 lost four fix rounds to gates that passed for reasons unrelated to the deliverable; the two places a mutation check was done up front produced its two strongest tests.
