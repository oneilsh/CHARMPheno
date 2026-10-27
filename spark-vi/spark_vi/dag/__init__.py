"""Generic label-DAG mechanics (ADR 0051): the topic-block layout over a DAG and the
graph operations the label-DAG builders, the gated engine and the readout share.

`layout.DagLayout` lays a DAG over integer node ids (root 0) out as topic blocks and
owns the gate (`allowed_set`); `ops` holds closure, descendants, depth, transitive
reduction and the closure matrix over `{child: [parents]}` maps. Depends on numpy only;
knows nothing of Mondo, OMOP or patients (those builders live in `charmpheno.mondo`).
"""
from spark_vi.dag.layout import DagLayout, frontier_from_coded
from spark_vi.dag.ops import (ancestor_closure, children_and_depth, closure_matrix,
                              descendants, nearest_kept_ancestors, nearest_kept_parents,
                              parent_adjacency, parents_over_range,
                              transitive_reduction_parents)

__all__ = ["DagLayout", "ancestor_closure", "children_and_depth", "closure_matrix",
           "descendants", "frontier_from_coded", "nearest_kept_ancestors",
           "nearest_kept_parents", "parent_adjacency", "parents_over_range",
           "transitive_reduction_parents"]
