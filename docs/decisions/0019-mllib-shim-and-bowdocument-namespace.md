# ADR 0019 — MLlib shim and BOWDocument under `topic/` namespace

**Status:** Accepted
**Date:** 2026-05-13
**Related:** ADR 0009 (LDA MLlib shim), ADR 0012 (HDP MLlib shim),
ADR 0016 (`spark_vi.models.topic` namespace).

## Context

ADR 0016 introduced a `topic/` subpackage convention for topic-model
content under `spark_vi.models`, mirrored later by `spark_vi.eval.topic`
when the NPMI coherence work landed (ADR 0017). The pattern keeps
generic framework primitives in their parent namespace and topic-specific
content visibly scoped — a future non-topic VIModel family (factor
analysis, Gaussian mixtures) fits cleanly without retrofitting names.

Two pieces of pre-existing layout drifted from that convention:

- `spark_vi.mllib.lda` and `spark_vi.mllib.hdp` (the MLlib shims) lived
  directly under `spark_vi.mllib`, not under `spark_vi.mllib.topic`,
  despite being topic-only shims. `spark_vi.mllib._common` mixed generic
  shim infrastructure (`_PersistenceParams`, `_PersistableModel`,
  `apply_persistence_params`) with the topic-specific
  `_vector_to_bow_document` helper.
- `BOWDocument` lived in `spark_vi.core.types`, suggesting "generic
  framework row type." It isn't — it's a bag-of-words type consumed only
  by topic models. A future non-topic VIModel would have to skip past it
  to find what `core` actually owns.

This ADR records the move that brings both into line with the ADR 0016
convention.

## Decisions

### MLlib shims move under `spark_vi.mllib.topic`

```
spark_vi/mllib/lda.py          → spark_vi/mllib/topic/lda.py
spark_vi/mllib/hdp.py          → spark_vi/mllib/topic/hdp.py
spark_vi/mllib/topic/__init__.py    (new — re-exports the four classes)
```

`spark_vi/mllib/__init__.py` keeps the top-level convenience re-exports
(`from spark_vi.mllib import OnlineLDAEstimator` still works) by
re-importing from `spark_vi.mllib.topic`. The public surface for users
who hit the package at its conventional name is unchanged.

### `mllib/_common.py` splits along the same seam

The topic-specific helper splits out:

```
spark_vi/mllib/_common.py          # generic — _PersistenceParams,
                                   #   apply_persistence_params,
                                   #   _PersistableModel
spark_vi/mllib/topic/_common.py    # topic-specific — _vector_to_bow_document
```

The split parallels `models/_common`-vs-`models/topic/...` (no
`models/_common` exists today; the principle applies if one ever does).

### `BOWDocument` moves to `spark_vi.models.topic.types`

```
spark_vi/core/types.py             → spark_vi/models/topic/types.py
spark_vi/core/__init__.py             # drops the re-export
spark_vi/models/topic/__init__.py     # gains a re-export so
                                      # `from spark_vi.models.topic import BOWDocument`
                                      # works for users hitting the package name
```

`spark_vi.core` no longer publishes any row type — it owns `VIConfig`,
`VIModel`, `VIResult`, `VIRunner` only. Future non-topic models will
define their own row type in their own package alongside their model
class.

### No back-compat aliases

Per ADR 0014 and ADR 0016's precedent: hard move, update all imports
across analysis drivers, tests, probes, charmpheno, and the framework
itself. The framework has no external consumers; an alias-and-deprecate
shim would just be code to delete later. All in-tree call sites are
updated in the same commit.

## Alternatives considered

- **Leave the layout as-is.** Mismatch with ADR 0016 was small and the
  code worked. Rejected because the inconsistency makes the namespacing
  rule ("topic-specific content gets a `topic/` subpackage") less
  reliable as a heuristic — readers can't trust that they've found
  everything topic-related by walking the `topic/` subpackages.
- **Move only the shims; leave `BOWDocument` in `core/types.py`.**
  Halfway move. Rejected — the same heuristic argument applies to the
  row type as to the shim; doing one without the other leaves an
  obvious gap.
- **Make `core/types.py` define a generic protocol and put the concrete
  type next to the models.** Cleanest in the limit, but no second row
  type exists today to motivate the protocol. Would be premature
  abstraction. Reopen if/when a non-topic VIModel family arrives.
- **Keep `mllib/_common.py` mixed.** Pragmatic option (one file, two
  consumers). Rejected because the whole point of the move is to
  realign content with location; leaving a generic-named module
  half-full of topic content would defeat that.
- **Drop the top-level `spark_vi.mllib` re-exports** and require
  `from spark_vi.mllib.topic.lda import OnlineLDAEstimator`. Rejected
  — short import path is a small but real ergonomic, and the
  re-export costs nothing.

## Consequences

- Public import surface for top-level users is unchanged:
  `from spark_vi.mllib import OnlineLDAEstimator` still works.
- Deep imports change: `from spark_vi.mllib.lda` →
  `from spark_vi.mllib.topic.lda` (and same for `hdp`). All in-tree
  call sites updated in the same commit. External code using the deep
  path must update — the framework has no external consumers, so this
  cost is internal-only.
- `from spark_vi.core import BOWDocument` no longer resolves. Use
  `from spark_vi.models.topic import BOWDocument` instead. All
  in-tree call sites updated.
- `spark_vi.core` becomes strictly framework-generic — no topic-specific
  content remains. Future non-topic VIModel families can be added under
  `spark_vi.models.<family>/` without naming friction.
- Test count unchanged (250 spark-vi + 41 charmpheno + 26 scripts =
  317 total; all pass post-move).
- The split of `mllib/_common.py` means topic-specific shim helpers
  (BOWDocument converter) and generic shim helpers (persistence Params,
  `_PersistableModel`) no longer share a module. A future non-topic
  shim would inherit only the generic half automatically.
