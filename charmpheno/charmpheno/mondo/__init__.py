"""The Mondo label space: Mondo's own is-a graph as the case-finding label DAG.

  * `ontology` — fetch a Mondo release (version-keyed cache) and read CDR tables;
  * `omop_mapping` — Mondo -> OMOP standard condition concepts (a port of
    monarch-initiative/mondo2omop);
  * `native_dag` — the native build (exp 0110, redesigned by spec 2026-10-30): the
    code ladder (each OMOP code -> its most-specific Mondo terms) and the relevant
    subgraph (the coded terms and their Mondo ancestors), handed to the DAG-gated
    estimator in Mondo curies. Which nodes get a topic block (powering, landing,
    reduction, splice) is the estimator's, learned from the training documents.

Depends on `spark_vi.dag.ops` for the generic graph operations. These modules are
folded into the bundle cache key by source hash (`analysis/cloud/_case_finding_cache.py`):
an edit here is a cache drop (AGENTS.md). Moved from `analysis/cloud/mondo_*.py` with
ADR 0051; the splice module `collapse` was removed with spec 2026-10-30 (the
estimator's splice subsumes it).
"""
