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

from spark_vi.models.topic.pg_stm_dag import DagGate


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


def expected_closure_gram(dag, doc_candidates):
    """Expected closure Gram Ḡ = sum_d sum_c p_c z_c z_c^T = sum_d E[z_d z_d^T] under a
    soft membership posterior. doc_candidates[d] = list of (weight, nodes) candidate
    closures for document d (labeled docs: a single (1.0, nodes)). Carries the within-doc
    spread (a doc split across candidates adds fractional curvature to each), so a
    soft-gated coordinate is appropriately closer to the design null than a hard one.
    Reduces to closure_gram on hard membership. Shape (U, U), U = dag.n_offset_nodes."""
    U = dag.n_offset_nodes
    G = np.zeros((U, U), dtype=np.float64)
    for cands in doc_candidates:
        for p, nodes in cands:
            z = dag.offset_indicator(nodes)
            G += float(p) * np.outer(z, z)
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
    foreground Grams' concern). Only edges into a
    SINGLE-parent child auto-collapse (a multi-parent child is not a chain edge -- merging it
    would transitively fuse its distinct parents; left for flagged_dim instead).
    flagged_dim = null_dim - collapse_dims counts confounds not explained by auto-collapsed
    chains (multi-child / diamond / cross-tree / multi-parent)."""
    G = np.asarray(G, dtype=np.float64)
    prev_collapsed = set() if prev_collapsed is None else set(prev_collapsed)
    n = dag.n_nodes
    parent = list(range(n))         # hysteresis decisions -> the QUOTIENT
    parent_bare = list(range(n))    # bare-tol decisions -> the flagged_dim accounting
    collapsed_edges = set()
    margins = {}
    for c in range(1, n):
        ps = dag.parents[c]
        if len(ps) != 1:                # multi-parent (or orphan) child: not a chain edge -> flag
            continue
        (p,) = ps
        if p == 0:                      # anchor's parent is the root (no offset column) -> skip
            continue
        i, j = c - 1, p - 1
        d = float(G[i, i] + G[j, j] - 2.0 * G[i, j])           # ||z_c - z_p||^2 >= 0
        margins[(p, c)] = tol - d
        if d < tol:                     # genuine bare-tol null direction (hysteresis-independent)
            _uf_union(parent_bare, p, c)
        was = (p, c) in prev_collapsed
        thresh = (tol + band) if was else (tol - band)         # hysteresis
        if d < thresh:
            _uf_union(parent, p, c)
            collapsed_edges.add((p, c))
    groups = {}
    for u in range(1, n):
        groups.setdefault(_uf_find(parent, u), set()).add(u)
    collapse_sets = [frozenset(s) for s in groups.values() if len(s) >= 2]
    # flagged_dim uses BARE-tol collapse dims, not the hysteresis quotient: a
    # hysteresis-retained-but-not-actually-null collapse (d in [tol, tol+band]) removes a node
    # from the quotient but is NOT a null direction, so counting it here would let it mask a
    # genuine separate confound. Each bare-tol chain dim IS a null dim, so this never
    # under-reports. (Equals collapse_dims when band == 0.)
    bare_groups = {}
    for u in range(1, n):
        bare_groups.setdefault(_uf_find(parent_bare, u), set()).add(u)
    collapse_dims_bare = sum(len(s) - 1 for s in bare_groups.values() if len(s) >= 2)
    null_dim = int(np.sum(spectrum["eigenvalues"] < tol))
    flagged_dim = max(0, null_dim - collapse_dims_bare)
    return {"collapse_sets": collapse_sets, "collapsed_edges": collapsed_edges,
            "margins": margins, "flagged_dim": flagged_dim}


def build_quotient(dag, detected):
    """Rewrite the DAG to its identified quotient: merge each detected parent-child
    column-equality set into one node, keep every other node, and re-number quotient nodes
    in a topological order (root first) so the resulting DagGate satisfies parent-id <
    child-id. Returns the quotient DagGate and a node_map (original node id -> quotient node
    id). The merge is faithful by construction because merged columns are (numerically)
    equal; Task 6's invariant test proves it against the moment."""
    n = dag.n_nodes
    # 1. representative per original node: min id of its collapse set, else itself
    rep = list(range(n))
    for s in detected["collapse_sets"]:
        r = min(s)
        for u in s:
            rep[u] = r
    # 2. quotient adjacency among representatives (original edges lifted through rep)
    reps = sorted(set(rep))                       # includes 0 (root is its own rep)
    radj_parents = {r: set() for r in reps}
    for child in range(n):
        rc = rep[child]
        for p in dag.parents[child]:
            rp = rep[p]
            if rp != rc:
                radj_parents[rc].add(rp)
    # 3. topological order of the quotient (Kahn), root first, deterministic by id
    indeg = {r: len(radj_parents[r]) for r in reps}
    children_of = {r: set() for r in reps}
    for r in reps:
        for p in radj_parents[r]:
            children_of[p].add(r)
    order = []
    ready = sorted(r for r in reps if indeg[r] == 0)   # root (0) has indeg 0
    while ready:
        r = ready.pop(0)
        order.append(r)
        for ch in sorted(children_of[r]):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)
        ready.sort()
    # 4. assign quotient ids in topo order; force root (rep 0) to quotient 0
    if order[0] != 0:               # fail loud (survives python -O): root (id 0) must sort first
        raise ValueError(f"quotient topological order must start at the root, got {order[0]}")
    qid = {r: i for i, r in enumerate(order)}
    node_map = np.array([qid[rep[u]] for u in range(n)], dtype=np.int64)
    # 5. build the quotient DagGate
    new_parents = [tuple(sorted(qid[p] for p in radj_parents[r])) for r in order]
    quotient_dag = DagGate(new_parents)
    return {"quotient_dag": quotient_dag, "node_map": node_map}


def quotient_moment_matches_projection(dag, G, quotient, doc_nodes):
    """Correctness invariant: quotient-then-form-the-moment == form-the-moment-then-project.
    Recompute the quotient DAG's pooled Gram from the corpus mapped through node_map, and
    compare it to the original Gram restricted to one representative original offset index
    per quotient offset node. Returns the max abs difference; ~0 (machine precision) for
    exact column-equality collapses, which certifies the quotient faithfully represents the
    identified part of the original design."""
    G = np.asarray(G, dtype=np.float64)
    node_map = quotient["node_map"]
    quotient_dag = quotient["quotient_dag"]
    # G_q: recompute on the quotient DAG from the remapped corpus
    q_doc_nodes = [frozenset(int(node_map[u]) for u in nodes) for nodes in doc_nodes]
    G_q = closure_gram(quotient_dag, q_doc_nodes)
    # projection: one representative ORIGINAL offset index per quotient OFFSET node.
    # quotient offset node q (id 1..Uq) <- pick any original node u with node_map[u]==q;
    # its offset index is u-1. Order reps by quotient offset id so rows/cols align with G_q.
    Uq = quotient_dag.n_offset_nodes
    reps_off = np.empty(Uq, dtype=np.int64)
    seen = {}
    for u in range(dag.n_nodes):
        q = int(node_map[u])
        if q >= 1 and q not in seen:
            seen[q] = u - 1                       # original offset index for quotient node q
    for q in range(1, Uq + 1):
        reps_off[q - 1] = seen[q]
    G_proj = G[np.ix_(reps_off, reps_off)]
    return float(np.max(np.abs(G_q - G_proj)))
