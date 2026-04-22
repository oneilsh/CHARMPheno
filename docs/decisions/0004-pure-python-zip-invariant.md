# 0004 — Pure-Python zip-compatibility invariant for both packages

**Status:** Accepted
**Date:** 2026-04-22

## Context
Spark executors in Dataproc can receive additional Python code via
`--py-files` or `sc.addPyFile()`, which accept `.py`, `.zip`, and `.egg`
archives only. Wheels with native code are not supported via this mechanism.
The project needs `--py-files`-compatible archives for both packages.

## Decision
Both `spark-vi` and `charmpheno` stay **pure-Python, flat-layout** packages:
- No C extensions.
- No build-time code generation.
- No conditional imports requiring non-standard dependencies at import time.

The `make zip` target in each package produces a flat `zip -r <pkg>.zip <pkg>/`
archive suitable for `--py-files` / `addPyFile`. The `make build` target
produces the standard wheel + sdist for `pip install`.

## Alternatives considered
- Allow C extensions / rely on conda-pack (rejected for bootstrap: adds a
  cluster-side setup dependency and complicates the deploy story).
- Skip zip delivery; rely on `pip install` only (rejected: Dataproc
  wheels-on-executors requires cluster properties set at creation time,
  which is less portable than `addPyFile`).

## Consequences
- If a future dependency introduces native code, the zip target fails,
  forcing an explicit re-discussion rather than silent breakage.
- Both `dist/<pkg>.whl` and `dist/<pkg>.zip` must stay green on every
  package-touching commit.
