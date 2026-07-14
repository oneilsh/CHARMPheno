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


def foreground_grams(dag, doc_nodes, doc_groups, partition):
    """Per-group foreground Grams, each accumulated over the documents that activate that
    group's sticks (i.e. belong to the group), with the intercept column included. The
    design row is w = [1.0, z_d]; each group's Gram is (1+U, 1+U). A group whose documents
    all attest its anchor makes the intercept column equal the anchor column -> a zero
    eigenvalue naming that group's absolute-level design wall per node (insight 0054)."""
    U = dag.n_offset_nodes
    out = {g: np.zeros((1 + U, 1 + U), dtype=np.float64) for g in partition.groups}
    for nodes, g in zip(doc_nodes, doc_groups):
        if g not in out:
            continue
        z = dag.offset_indicator(nodes)
        w = np.concatenate([np.array([1.0]), z])
        out[g] += np.outer(w, w)
    return out


def identifiability_spectrum(G):
    """Raw, threshold-free symmetric eigen-spectrum of a closure Gram. Returns eigenvalues
    ascending and their unit eigenvectors (columns), via numpy.linalg.eigh (G is symmetric
    PSD). No cutoff and no naming happen here -- the small-but-nonzero eigenvalues are the
    weakly-identified directions, left as raw numbers for the quotient builder (which owns
    the one numeric tolerance) and the reporting layer (which owns any tiers)."""
    G = np.asarray(G, dtype=np.float64)
    evals, evecs = np.linalg.eigh(G)          # ascending, orthonormal columns
    return {"eigenvalues": evals, "eigenvectors": evecs}


def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, a, b):
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[min(ra, rb)] = max(ra, rb)      # deterministic: point smaller root at larger


def detect_confounds(dag, G, spectrum, *, tol, prev_collapsed=None, band=0.0):
    """Detect the confounds the compiler can safely auto-collapse (parent-child
    column-equality chains: ||z_parent - z_child||^2 < tol) and count the residual
    confounded dimension it must instead flag. Hysteresis: with prev_collapsed given, a
    fresh edge needs ||.||^2 < tol - band to collapse and a previously-collapsed edge stays
    collapsed unless ||.||^2 > tol + band, so near-threshold decisions do not churn between
    snapshots. Root edges are skipped (no offset column; anchor-level walls are the
    foreground Grams' concern). flagged_dim = null_dim - collapse_dims counts confounds not
    explained by auto-collapsed chains (multi-child / diamond / cross-tree / multi-parent)."""
    G = np.asarray(G, dtype=np.float64)
    prev_collapsed = set() if prev_collapsed is None else set(prev_collapsed)
    n = dag.n_nodes
    parent = list(range(n))
    collapsed_edges = set()
    margins = {}
    for c in range(1, n):
        i = c - 1
        for p in dag.parents[c]:
            if p == 0:
                continue
            j = p - 1
            d = float(G[i, i] + G[j, j] - 2.0 * G[i, j])       # ||z_c - z_p||^2 >= 0
            margins[(p, c)] = tol - d
            was = (p, c) in prev_collapsed
            thresh = (tol + band) if was else (tol - band)     # hysteresis
            if d < thresh:
                _uf_union(parent, p, c)
                collapsed_edges.add((p, c))
    groups = {}
    for u in range(1, n):
        groups.setdefault(_uf_find(parent, u), set()).add(u)
    collapse_sets = [frozenset(s) for s in groups.values() if len(s) >= 2]
    collapse_dims = sum(len(s) - 1 for s in collapse_sets)
    null_dim = int(np.sum(spectrum["eigenvalues"] < tol))
    flagged_dim = max(0, null_dim - collapse_dims)
    return {"collapse_sets": collapse_sets, "collapsed_edges": collapsed_edges,
            "margins": margins, "flagged_dim": flagged_dim}
