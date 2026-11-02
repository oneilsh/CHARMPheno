"""Generic label-DAG mechanics (ADR 0051): the topic-block layout over a DAG, the
graph operations the label-DAG builders, the gated engine and the readout share, and
the fit-time node-set learning.

`ops` holds closure, descendants, depth, transitive reduction and the closure matrix
over `{child: [parents]}` maps (exported here). `nodeset` turns a user's DAG (dict,
pandas or Spark `(child, parent)`, ids kept as given) and labelled documents into the
learned node set: validation, support, prune, land, reduce, splice, engine ids and the
per-document frontier (spec 2026-10-30, R1/R3/R4); it is the estimator's internal
helper, imported from `spark_vi.dag.nodeset`. `layout.DagLayout` lays a DAG over dense
engine ids (root 0) out as topic blocks and owns the gate (`allowed_set`); it and
`layout.frontier_from_coded` are ENGINE internals (spec decision 5), imported from
`spark_vi.dag.layout` and not exported here. Depends on numpy only (pandas / pyspark
only when handed such inputs); knows nothing of any ontology or of what the documents
are.
"""
from spark_vi.dag.ops import (ancestor_closure, children_and_min_depth, closure_matrix,
                              descendants, nearest_kept_ancestors, nearest_kept_parents,
                              parent_adjacency, parents_over_range,
                              transitive_reduction_parents)

__all__ = ["ancestor_closure", "children_and_min_depth", "closure_matrix",
           "descendants", "nearest_kept_ancestors", "nearest_kept_parents",
           "parent_adjacency", "parents_over_range", "transitive_reduction_parents"]
