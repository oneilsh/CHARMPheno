"""Source-hash identity of the corpus-shaping modules that no cache key named before.

`cohort_defs_version()` (cohorts.py) and the assembler hashes cover how cohorts and
bundles are ASSEMBLED, but three modules shape every corpus without being hashed:
`bigquery` (the loaders: which rows, which columns, which person sample),
`measurement_tokens` (how a measurement becomes a token) and `topic_prep` (how events
become a bag of words). An edit there used to serve a stale cached corpus under an
unchanged key (audit 2026-10-26 §6). `corpus_code_version()` folds all three; the
bundle, corpus and covariate keys each fold it (ADR 0051 step 3's cache drop).

Coarse by design, like `cohort_defs_version`: any edit to one of these modules
invalidates every cached corpus (correctness over cache reuse).
"""
from __future__ import annotations

import hashlib
import inspect


def corpus_code_version() -> str:
    """16-hex digest over the source of the three corpus-shaping modules.

    Falls back to a constant per module if its source is unavailable (a byte-
    compiled deploy), as `cohort_defs_version` does."""
    from charmpheno.omop import bigquery, measurement_tokens, topic_prep

    h = hashlib.sha256()
    for mod in (bigquery, measurement_tokens, topic_prep):
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            src = f"{mod.__name__}:src-unavailable"
        h.update(mod.__name__.encode("utf-8"))
        h.update(src.encode("utf-8"))
    return h.hexdigest()[:16]
