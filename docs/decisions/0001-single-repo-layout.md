# 0001 — Single-repo layout with two poetry projects

**Status:** Accepted
**Date:** 2026-04-22

## Context
The project comprises a reusable PySpark framework (`spark-vi`), a clinical
specialization (`charmpheno`), and analysis / notebook work that drives
development of both. All three co-evolve tightly in the early phase; each has
its own eventual release cadence.

## Decision
Adopt a single-repo monorepo that contains two independent poetry projects
side-by-side (`spark-vi/` and `charmpheno/`), plus top-level `analysis/`,
`notebooks/`, and `scripts/` directories. Each poetry project has its own
`pyproject.toml`, `Makefile`, tests, and `dist/` output directory.

## Alternatives considered
- Three separate repos from day 1 (rejected: cross-repo friction when the
  two packages are co-evolving and need joint iteration).
- Single pyproject with both packages in one `src/` tree (rejected: fuses
  release boundaries; splitting later becomes a refactor rather than an
  extraction).

## Consequences
- Forward-compatible: either package can be split into its own repo later
  with minimal work.
- Dev workflow requires two editable installs; one-time cost.
- Top-level `Makefile` delegates to each package (`make -C spark-vi test`).
