"""The Mondo label space: Mondo's own is-a graph as the case-finding label DAG.

  * `ontology` — fetch a Mondo release (version-keyed cache) and read CDR tables;
  * `omop_mapping` — Mondo -> OMOP standard condition concepts (a port of
    monarch-initiative/mondo2omop);
  * `native_dag` — the native build (exp 0110): power Mondo terms by closure support,
    take the transitive reduction over the kept set, splice single-child classes
    (`collapse`), and resolve each OMOP code to the label nodes it attests;
  * `collapse` — the splice-to-fixpoint reduction.

Depends on `spark_vi.dag` for the generic graph operations and on
`charmpheno.omop.condition_dag` for the engine DAG type. These modules are folded into
the bundle cache key by source hash (`analysis/cloud/_case_finding_cache.py`): an edit
here is a cache drop (AGENTS.md). Moved from `analysis/cloud/mondo_*.py` with ADR 0051.
"""
