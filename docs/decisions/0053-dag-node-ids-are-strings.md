# 0053 — DAG node ids are strings

**Status:** Accepted
**Date:** 2026-11-23

## Context

ADR 0052 §1 (and spec 2026-10-30 decision 2) accepted node ids as `str` or `int`, kept
as given, so `1` and `"1"` were different nodes. Supporting that cost accounting in
every layer: a mixed-type sort key, numpy-int unwrapping and integral-float recovery
for pandas, a labels-column check against the DAG's id types, `describeNodes` /
`nodeStatistics` columns that were LONG, STRING, or STRING-with-ambiguity, and tests
for each. The clinical caller hands over Mondo CURIEs (strings); the int SNOMED path is
retired in the driver. The maintainer's walkthrough (2026-11-23) found the generality
was not worth its cost.

## Decision

Node ids are `str`. `parse_dag`, the labels, `nodeTerms` / `gateSeedProfile` node keys,
the featurizer's label sources and `DagHeadsEstimator` refuse anything else (int,
bool, numpy int, float) with an error that names the cast (`str(id)`,
`.cast('array<string>')`). A numpy `str_` is unwrapped. Ids sort as plain strings,
`ROOT` first. Node-id columns in outputs are STRING. TERM ids (`vocabularies` and the
terms named by the seed extras) stay `str | int`, matched exactly, as before.

## Alternatives considered

- Keep mixed ids: the accounting above, for no caller that needs it.
- Normalize ints to strings silently: hides a type mismatch between a DAG and its
  labels instead of naming it.

## Consequences

- Supersedes the "str or int, kept as given" clause of ADR 0052 §1; the rest of 0052
  stands.
- Saved node sets with string ids load unchanged (same JSON, same engine order); a
  model saved with int ids no longer loads (no backward compatibility, ADR 0052 §7).
- A caller with integer ids (the source-hashed SNOMED bundle path,
  `charmpheno.omop.condition_dag` / `case_finding_assembly`) casts at its seam.

## Addendum (2026-11-23, same walkthrough): term ids are strings too

`vocabularies` and the terms named by `nodeTerms` / `gateSeedProfile` (bare, or the term
of a `(featureColumn, term)` pair) are `str`; ints, bools and numpy ints are refused
with the cast hint, a numpy `str_` is unwrapped. Vectorizer vocabularies
(`CountVectorizer`, `DagGatedFeaturizer`) already hold strings, so the only cost was
`_TermIndex`'s "twin" detection (`10` vs `"10"`), now deleted, with the mixed-type term
sort key. Every id the estimator takes is a string. Stored JSON is unchanged for string
terms.

**Addendum (2026-11-23, walkthrough):** a document LABEL that is not a `str` raises
`TypeError` (with the cast hint) in the support pass, the frontier, `NodeSet.land` and
the readout labeler, under either `unknownLabels` policy. Before, it counted as an
unknown label, so `"ignore"` silently dropped it and the document could turn into
background. `unknownLabels` now governs only strings that are not DAG nodes.
