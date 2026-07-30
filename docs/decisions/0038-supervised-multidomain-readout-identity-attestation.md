# 0038 — Supervised multidomain readouts require attested row and domain identity

**Status:** Accepted
**Date:** 2026-07-30

## Context

The hybrid domain-reliability readout uses repeated nested cross-validation over
persisted held-out rows. Row-level folds are honest only when each row represents
a distinct person. The legacy exp 0071/0072 sidecars persist aligned BOWs and
frontiers but no proof of this identity invariant, so their supervised results
would be unverifiable.

The same artifact stores per-domain λ and BOW matrices under ordinal keys. Those
integers are storage coordinates, not semantic domain identity. A fixed
condition-plus-drug baseline selected by position could silently become
condition-plus-observation if manifest order changes.

## Decision

The fit-time writer receives person IDs from the same collected rows used to
construct the persisted BOWs. Before writing any sidecar, it asserts that the ID
count matches the row count and that every ID is unique. It persists no raw ID or
hash. Instead, `test_meta.json` carries only:

- `row_count`;
- `unique_person_count`; and
- `one_row_per_person: true`.

The supervised loader requires a mapping with the true flag and both counts equal
to `n_docs`. Missing, false, or inconsistent attestations abort with no legacy
override. Such an artifact must be refit; the attestation must never be inferred
or retrofitted after the aligned IDs have been discarded.

Manifest domain names are the authoritative semantic identity of ordinal λ/BOW
keys. They must be unique strings, one per ordinal domain, and include
`condition` and `drug`. Policies and reports use these names.
`fixed:condition_drug` resolves the two named domains rather than positional
indices.

These are analysis-layer persistence and evaluation contracts. The
domain-agnostic `spark_vi` primitives continue to operate on caller-supplied
domain keys and do not import clinical names or person identity.

## Alternatives considered

- **Persist raw person IDs or stable hashes.** Rejected because the readout needs
  only proof of uniqueness; either representation increases disclosure risk.
- **Assume the document-unit configuration proves uniqueness or add a CLI
  override.** Rejected because configuration intent is not evidence about the
  exact persisted rows.
- **Retrofit counts into legacy metadata.** Rejected because uniqueness cannot be
  reconstructed after person IDs are discarded.
- **Keep positional domain semantics.** Rejected because a valid reordered
  manifest can silently change the scientific baseline while preserving shapes.

## Consequences

- Legacy exp 0071/0072 artifacts remain valid historical inputs for their prior
  unsupervised and LR analyses, but are invalid specifically for this supervised
  row-fold readout.
- Fresh exp 0073/0074 fits preserve the 0072/0071 configurations while producing
  attestable sidecars.
- Any future artifact with multiple rows per person needs a new grouped-fold
  identity contract; it cannot weaken this row-level one.
- Reordered domain manifests retain correct named policy and fixed-baseline
  semantics.
