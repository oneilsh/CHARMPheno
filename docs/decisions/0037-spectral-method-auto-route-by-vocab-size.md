# 0037 — `spectral_method="auto"` routes dense↔scalable by vocabulary size

**Status:** Accepted (supersedes the "explicit knob, no magic threshold" clause of
[ADR 0032](0032-scalable-spectral-init-random-projection-over-maxv.md))

## Context

ADR 0032 introduced the scalable (random-projection) spectral-init path and made a
deliberate choice: expose **both** dense and scalable behind an **explicit knob**
(`spectral_method`), with **dense as the default**, and explicitly *rejected* a
"magic V-threshold" auto-selector. The reasoning at the time: dense is the exact,
validated path, and an implicit threshold hides a correctness-relevant choice.

Two things have changed since:

1. **Scalability is a first-class goal of this library**, and the explicit-knob
   design has a sharp failure mode: a genuinely large-vocabulary run (V≈100k) that
   forgets to pass `spectral_method="scalable"` silently attempts to materialize a
   V×V float64 co-occurrence matrix on the driver (8·V² bytes ≈ **80 GB** at
   V=100k) and OOMs. The safe default for a scalability-targeted library is to *not*
   OOM by default.
2. **The main risk that made scalable look dicey is gone at the default.** Exp 0017
   / insight 0031 showed the scalable path's one real deviation from dense was a
   single-topic Σ escape under a *free-variance* Σ. Under the block-wise
   unit-diagonal Σ that is now the production default ([ADR 0034](0034-blockwise-unit-diagonal-correlation-sigma.md)),
   Σ_ii is pinned by construction and that escape cannot occur. Topic-recovery
   equivalence was already confirmed (all 40 phenotypes, NPMI +0.166 vs dense
   +0.173).

## Decision

Add `spectral_method="auto"` and make it the **default**. `"auto"` resolves at fit
time, where the vocabulary size is known:

- **V < `SPECTRAL_AUTO_VOCAB_THRESHOLD` (default 10,000) → dense** (exact, the
  validated path). At V=10,000 a dense V×V float64 co-occurrence matrix is
  8·10⁴² ≈ 0.8 GB on the driver; the threshold sits near a ~1 GB single-matrix
  footprint (peak is a small multiple with per-group matrices).
- **V ≥ threshold → scalable**, and a `logging.warning` is emitted naming V, the
  threshold, and the dense-memory reason, so the switch is never silent.

`"dense"` and `"scalable"` remain valid explicit overrides that pass through
unchanged. The **resolved** method (what actually ran) is recorded in
`metadata["stm_hardening"]["spectral_method"]`; the requested value is preserved
in `spectral_method_requested`.

The threshold is a single tunable module constant, not a fitted quantity. This is a
heuristic memory guard, not a correctness boundary — both paths are correct; the
threshold only trades dense's exactness against dense's driver memory.

## Consequences

- **No current model changes behavior.** Every present cohort (cancer V=3691,
  population V=6115) is below the threshold and stays on the exact dense path.
  `"auto"` only diverges from dense for genuinely large future vocabularies.
- **Large-V runs stop OOMing by default**, which is the point.
- **This reverses ADR 0032's specific "no magic threshold / dense default" clause.**
  The rest of ADR 0032 (the scalable algorithm, the projection dimension, the
  doc-frequency floor) stands. The scalable path's own large-V *quality/memory*
  validation on real data is still outstanding (a high-V cohort; exp 0017 was a
  small-V equivalence run) — `"auto"` routes to it, so that validation gates how
  much to trust the auto-selected path at scale, and the threshold can be revised
  when it lands.
- Drivers (`stm_bigquery_cloud.py`, `fit_stm_local.py`) and `run_experiment`'s
  `build_stm_args` default `--spectral-method` to `auto`; explicit `dense`/`scalable`
  are emitted, `auto` is the implicit (omitted) default.
