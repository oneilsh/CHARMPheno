# 0030 — Diagnostic traces persist faithfully (any rank, no size cap)

**Status:** Accepted
**Date:** 2026-06-25
**Related:** ADR 0021 (producer-side vocab trim / egress budget); ADR 0023 (STM inference); the 2026-05-13 `iteration_diagnostics` broadening.

## Context

A `VIModel` emits small per-iteration values via `iteration_diagnostics(global_params)`; the runner accumulates them into `VIResult.diagnostic_traces` and `io/export.save_result` persists them. The original `_classify_trace` **rejected any per-iteration array with `ndim > 1`** (an explicit "YAGNI" punt), and because STM's per-iter `Gamma` snapshot is 2-D `(P, K)`, `STMModel.save` worked around the rejection by **dropping its diagnostic traces wholesale**. The Γ/Σ/topic-block-label trajectories were therefore lost across save→load. Lifting the rejection raised a follow-on question: should a **size cap** be imposed to prevent checkpoint bloat or accidental sensitive data?

## Decision

Persist `np.ndarray` per-iter values of **any rank** — stacked to `(n_iter, *dims)` in `traces/<name>.npy` — and rename the internal trace kind `"vector"` → `"array"`. **No size cap.** Emitting a trace is opt-in (the base `iteration_diagnostics` returns `{}`), so any per-iter state is a deliberate model choice the framework persists faithfully; a model that should not carry heavy state suppresses it itself (return `{}` / omit the key). `STMModel.save` stops dropping traces.

## Alternatives considered

- **Fixed element/byte cap.** Rejected: an arbitrary magic number that would false-reject legitimate large-but-valid traces (STM's Γ at large `K`/`P` already exceeds any modest cap), guarding only a hypothetical.
- **Per-model opt-in flag for N-D traces.** Rejected: emission is *already* the opt-in (overriding `iteration_diagnostics`); a second flag is redundant.
- **STM-specific Γ/Σ sidecar.** Rejected: the generic stack/load path already supports any rank, so a special case is the *less* clean option.

## Consequences

- STM's per-iter `(P, K)` Γ + `K` Σ + topic-block labels now round-trip (small: `P·K + K` floats per iter).
- No *mechanical* guard against an author ignoring the "keep it small" docstring norm — that is a code-review concern, not a framework one. The homogeneity and mixed-kind checks remain (correctness, not size).
- Safe by construction: `iteration_diagnostics` receives only `global_params` (never per-document data), and the checkpoint is distinct from the whitelisted, producer-side-trimmed dashboard egress bundle (ADR 0021), so traces neither carry per-patient data nor auto-egress.
- **Update (2026-06-30):** the `Σ[min…max]` scalar-range trace is generalized to
  full-matrix signals under the full-covariance Σ arc ([ADR 0033](0033-stm-full-covariance-sigma.md)):
  eigenvalue range + condition number of Σ, max |off-diagonal correlation|, and
  imputed fraction (share of entries below the min_pair_support floor). The per-iter
  Σ artifact shape changes from a K-vector to a (K−1)×(K−1) matrix.
