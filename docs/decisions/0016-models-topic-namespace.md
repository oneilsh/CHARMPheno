# ADR 0016 — `spark_vi.models.topic` namespace

**Status:** Accepted
**Date:** 2026-05-11
**Supersedes:** none (refines the implicit layout established by ADR 0001-0015)
**Superseded by:** none

## Context

The `spark_vi` framework was originally laid out with topic-model
implementations directly under `spark_vi.models.{lda,online_hdp,counting}`.
This made sense when the only models in flight were topic models. As we
add evaluation surface (held-out coherence, future term relevance, future
synthetic-recovery testing — see ADR 0017) the eval namespace wants a
`spark_vi.eval.topic.*` shape to leave room for non-topic-model eval modules
later. An asymmetric `models.lda` / `eval.topic.coherence` layout is
needlessly confusing; we want the topic-model scope to be visible on both
the model side and the eval side.

## Decision

Move the three topic-model implementation files to a `topic` subpackage:

```
spark_vi/models/lda.py        -> spark_vi/models/topic/lda.py
spark_vi/models/online_hdp.py -> spark_vi/models/topic/online_hdp.py
spark_vi/models/counting.py   -> spark_vi/models/topic/counting.py
```

`CountingModel` is not a topic model in the LDA/HDP sense but shares the
bag-of-words input shape and is used as a contract-conformance fixture for
topic-model infrastructure; it lives under `topic/` for that reason.

The top-level `spark_vi.models.__init__` re-exports `OnlineLDA`, `OnlineHDP`,
and `CountingModel` so `from spark_vi.models import OnlineLDA` continues to
work. The new canonical import path is `from spark_vi.models.topic import
OnlineLDA` but the umbrella import is the supported public surface.

## Consequences

**Breaking:** any external code importing `spark_vi.models.lda`,
`spark_vi.models.online_hdp`, or `spark_vi.models.counting` directly (as
submodules, not via the umbrella) breaks. The framework is early enough
that no external consumers exist, and within-repo callers (`mllib`, tests,
probes, `analysis/`, ADR 0013 prose) are migrated in this same PR.

**Non-breaking:** umbrella imports `from spark_vi.models import OnlineLDA`
are preserved by the re-export. Historical specs that name old paths
(2026-04-22, 2026-05-04) are left as-is — they are point-in-time documents.

**Forward:** ADR 0017 introduces `spark_vi.eval.topic.*` mirroring this
layout.

## Alternatives considered

- **Leave the layout asymmetric.** Rejected: the dissonance compounds with
  every new eval module added under `eval/topic/`.
- **Re-namespace under `spark_vi.topic.{models,eval}` (collapse the inner
  split).** Rejected: framework code that is generic over model class
  (`spark_vi.core`, `spark_vi.io`, `spark_vi.diagnostics`) already shapes
  the top-level under capability, not domain. `topic` is one domain among
  potential others; better to keep it as a leaf of the `models` and `eval`
  capabilities than as a domain-rooted competing axis.
