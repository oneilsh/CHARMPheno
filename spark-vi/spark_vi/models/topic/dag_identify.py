"""Identifiability compiler (v1): read increment-identifiability off the design-moment
(closure-indicator Gram) and rewrite a node DAG to the coordinates the corpus can resolve.

Domain-agnostic: integer node ids only. The math kernel (closure_gram, foreground_grams,
identifiability_spectrum) is threshold-free; the only numeric threshold (the rank tolerance
tol) lives in the quotient builder (detect_confounds / build_quotient). See
docs/superpowers/specs/2026-07-14-identifiability-compiler-design.md and insights 0050/0052/0054.

Index convention: offset index i (0-based) corresponds to node id i+1 (the root, node 0, has
no offset column). Grams are offset-index-ordered, shape (U, U) with U = dag.n_offset_nodes.
"""
import numpy as np


def closure_gram(dag, doc_nodes):
    """Pooled closure-indicator Gram G = sum_d z_d z_d^T over the corpus, where
    z_d = dag.offset_indicator(nodes_d) is a document's non-root closure indicator. This is
    the offset block of the design moment the fit accumulates, so the compiler's cost is a
    subset of the fit's. Returns a dense (U, U) array, U = dag.n_offset_nodes."""
    U = dag.n_offset_nodes
    G = np.zeros((U, U), dtype=np.float64)
    for nodes in doc_nodes:
        z = dag.offset_indicator(nodes)
        G += np.outer(z, z)
    return G
