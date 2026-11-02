"""Learning a label DAG's node set from labelled documents (spec 2026-10-30, R1/R3/R4).

The DAG-gated estimator takes a DAG and documents labelled with DAG nodes, both in the
USER's node ids, and learns at fit time which nodes get a topic block (as
`StringIndexer` learns its vocabulary). This module is that learning, kept apart from
the estimator so it is testable without Spark and reusable by the heads:

1. `parse_dag`: the DAG as a dict, a pandas DataFrame or a Spark DataFrame
   `(child, parent)` -> a validated `ParsedDag` (self-loops and duplicate edges
   dropped, parentless nodes hung from `ROOT`, cycles refused by name).
2. `LabelIndex` / `doc_support` / `count_support` / `count_support_spark`: one pass
   over the documents. Unknown labels are refused by name (`unknown_labels="error"`)
   or dropped (`"ignore"`); a node's SUPPORT is the number of documents whose labels'
   closure (the labels plus all their ancestors) contains it.
3. `learn_node_set`: prune (support < `min_docs_per_node`), land the dropped nodes'
   labels on their nearest kept ancestors, transitively reduce over the kept nodes,
   splice single-child chains to a fixpoint, and number the kept nodes as dense
   engine ids `1..N` (root 0). Returns a `NodeSet`.
4. `NodeSet.frontier`: a document's labels -> its engine-id frontier, the
   most-specific landed nodes (what the gate reads).

Node ids are `str` or `int`, kept exactly as given: `1` and `"1"` are different nodes
(spec decision 2). Nothing here knows what a node means.

Engine-side structures (`DagLayout`, `frontier_from_coded`) stay private to the
engine (decision 5); this module produces the `{engine child: [engine parents]}` map
`DagLayout` takes.

Executor-side use (ADR 0047): the per-document work is `doc_support(labels, index)` and
`FrontierFn.__call__(labels)`; both read only small dicts of tuples carried by a
picklable object meant to be BROADCAST (`count_support_spark` does this), never a
numpy array captured in a task closure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from spark_vi.dag.ops import (ancestor_closure, nearest_kept_ancestors,
                              transitive_reduction_parents)


# --------------------------------------------------------------------------------
# The synthetic root
# --------------------------------------------------------------------------------

class _Root:
    """The synthetic root every parentless node hangs from.

    A private singleton, not a `str` or `int`, so it can never equal a user node id
    (user ids are restricted to `str` / `int`). It pickles BY NAME (`__reduce__`
    returns the module-level name), so a `ParsedDag` or `NodeSet` shipped to an
    executor and back still holds the very same object and `is ROOT` keeps working.
    In engine ids it is always 0."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "ROOT"

    def __reduce__(self):
        return "ROOT"


ROOT = _Root()


# --------------------------------------------------------------------------------
# Node ids
# --------------------------------------------------------------------------------

def _coerce_id(x, *, what="node id"):
    """A user node id as given, after unwrapping numpy scalars (np.int64(1) -> 1,
    np.str_("a") -> "a": same value and hash, so this is NOT a normalization between
    1 and "1"). `bool` is refused because `True == 1` would silently merge two
    nodes in every dict; anything that is not `str` / `int` is refused by repr."""
    if isinstance(x, bool):
        raise TypeError(f"{what} {x!r}: bool is not a valid node id (True == 1)")
    if isinstance(x, (str, int)):
        return x
    # numpy scalars without importing numpy at module scope.
    t = type(x).__module__
    if t == "numpy":
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None
        if np is not None:
            if isinstance(x, np.bool_):
                raise TypeError(f"{what} {x!r}: bool is not a valid node id")
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.str_):
                return str(x)
    raise TypeError(f"{what} {x!r} has type {type(x).__name__}; node ids must be "
                    f"str or int")


def node_sort_key(x):
    """A total, deterministic order over mixed `str` / `int` ids that never compares
    a `str` with an `int`: every int before every str, ints numerically, strs
    lexicographically. (`ROOT` sorts first.)"""
    if x is ROOT:
        return (0, 0, "")
    if isinstance(x, int):
        return (1, x, "")
    return (2, 0, x)


def _sorted_ids(xs):
    return sorted(xs, key=node_sort_key)


# --------------------------------------------------------------------------------
# R1: parsing and validating the DAG
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedDag:
    """A validated DAG in user ids.

    `parents`: `{node: tuple of parents}` for EVERY user node (a node that appears
    only as a parent is included); a parentless node's parents are `(ROOT,)`. Parent
    tuples are sorted by `node_sort_key`. `ROOT` is not a key.
    `nodes`: every user node, sorted by `node_sort_key`.
    `n_self_loops` / `n_duplicate_edges`: what parsing dropped (reported, not fatal).
    """
    parents: dict
    nodes: tuple
    n_self_loops: int = 0
    n_duplicate_edges: int = 0

    def __contains__(self, node):
        return node in self.parents

    def __len__(self):
        return len(self.nodes)


def _is_spark_df(obj) -> bool:
    # Duck typing keeps pyspark an optional import: a Spark DataFrame has `rdd`,
    # `collect` and `columns`; a pandas DataFrame has no `rdd`.
    return all(hasattr(obj, a) for a in ("rdd", "collect", "columns", "select"))


def _is_pandas_df(obj) -> bool:
    return (type(obj).__module__.split(".")[0] == "pandas"
            and hasattr(obj, "columns") and hasattr(obj, "itertuples"))


def _missing(v) -> bool:
    """None, NaN, pandas NA / NaT: a missing parent (the row's child is parentless).
    Checked by type name so pandas is never imported here."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:          # also numpy float NaN
        return True
    return type(v).__name__ in ("NAType", "NaTType")


def _pandas_cell(v, *, what):
    """A pandas cell as a node id. An int column with a missing value is upcast to
    float64 by pandas (`[None, 1]` -> `[nan, 1.0]`), so an INTEGRAL float is read
    back as the int pandas turned it from; a non-integral float is refused."""
    if _missing(v):
        return None
    if isinstance(v, float) or type(v).__name__.startswith("float"):
        fv = float(v)
        if fv.is_integer():
            return int(fv)
        raise TypeError(f"{what} {v!r} is a non-integral float; node ids must be "
                        f"str or int")
    return _coerce_id(v, what=what)


def _edges_from_input(dag):
    """`(edges, isolated)` from any accepted input: `edges` a list of
    `(child, parent)` pairs, `isolated` the nodes given with no parent."""
    edges, isolated = [], []
    if isinstance(dag, ParsedDag):
        for c, ps in dag.parents.items():
            for p in ps:
                if p is ROOT:
                    isolated.append(c)
                else:
                    edges.append((c, p))
        return edges, isolated
    if isinstance(dag, dict):
        for c, ps in dag.items():
            c = _coerce_id(c, what="child")
            if _missing(ps):
                isolated.append(c)
                continue
            if isinstance(ps, (list, tuple, set, frozenset)):
                ps = list(ps)
                if not ps:
                    isolated.append(c)
                for p in ps:
                    if _missing(p):
                        isolated.append(c)
                    else:
                        edges.append((c, _coerce_id(p, what=f"parent of {c!r}")))
            else:
                edges.append((c, _coerce_id(ps, what=f"parent of {c!r}")))
        return edges, isolated
    if _is_spark_df(dag):
        _require_columns(dag.columns)
        # A label DAG is small: one collect on the driver (R1).
        rows = dag.select("child", "parent").collect()
        pairs = [(r[0], r[1]) for r in rows]
        conv = _coerce_id
    elif _is_pandas_df(dag):
        _require_columns(list(dag.columns))
        pairs = list(zip(dag["child"].tolist(), dag["parent"].tolist()))
        conv = _pandas_cell
    else:
        raise TypeError(
            f"the DAG must be a dict {{child: parent | [parents] | None}}, a pandas "
            f"DataFrame or a Spark DataFrame with columns (child, parent); got "
            f"{type(dag).__name__}")
    for c, p in pairs:
        if _missing(c):
            raise ValueError("the DAG has a row with a missing child id")
        c = conv(c, what="child")
        if _missing(p):
            isolated.append(c)
        else:
            edges.append((c, conv(p, what=f"parent of {c!r}")))
    return edges, isolated


def _require_columns(cols):
    missing = [c for c in ("child", "parent") if c not in cols]
    if missing:
        raise ValueError(f"the DAG DataFrame needs columns (child, parent); missing "
                         f"{missing}, has {list(cols)}")


def _find_cycle(parents):
    """One cycle of `{child: (parents)}` as a list of nodes, or None. Iterative DFS in
    `node_sort_key` order, so the reported cycle is deterministic."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in parents}
    for start in _sorted_ids(parents):
        if color[start] != WHITE:
            continue
        path = [start]
        color[start] = GREY
        iters = [iter(parents[start])]
        while iters:
            nxt = next(iters[-1], None)
            if nxt is None:
                color[path.pop()] = BLACK
                iters.pop()
                continue
            if nxt is ROOT:
                continue
            if color[nxt] == GREY:
                return path[path.index(nxt):] + [nxt]
            if color[nxt] == WHITE:
                color[nxt] = GREY
                path.append(nxt)
                iters.append(iter(parents[nxt]))
    return None


def parse_dag(dag) -> ParsedDag:
    """The DAG as given (R1) -> a validated `ParsedDag`.

    Accepts a `dict` `{child: parent | [parents] | None}`, a pandas DataFrame or a
    Spark DataFrame with columns `(child, parent)` (one row per edge; a missing
    parent marks a parentless node; the Spark frame is collected once on the
    driver), or an already-parsed `ParsedDag`. Ids are `str` / `int` kept as given
    (1 and "1" are different nodes). Self-loops and duplicate edges are dropped and
    counted; a node that appears only as a parent is a node; every parentless node
    hangs from `ROOT`. A cycle is refused with a `ValueError` naming its nodes."""
    if isinstance(dag, ParsedDag):
        return dag
    edges, isolated = _edges_from_input(dag)
    # `==` never equates a str with an int, so 1 -> "1" is an edge, not a loop.
    n_self = sum(1 for c, p in edges if c == p)
    raw = [(c, p) for c, p in edges if c != p]
    uniq = set(raw)
    n_dup = len(raw) - len(uniq)
    parents: dict = {}
    for c, _ in edges:             # a child whose only edge was a self-loop stays
        parents.setdefault(c, set())
    for c, p in uniq:
        parents.setdefault(c, set()).add(p)
        parents.setdefault(p, set())
    for c in isolated:
        parents.setdefault(c, set())
    out = {c: tuple(_sorted_ids(ps)) if ps else (ROOT,) for c, ps in parents.items()}
    cyc = _find_cycle(out)
    if cyc is not None:
        raise ValueError("the DAG has a cycle: " + " -> ".join(repr(n) for n in cyc))
    return ParsedDag(parents=out, nodes=tuple(_sorted_ids(out)),
                     n_self_loops=n_self, n_duplicate_edges=n_dup)


# --------------------------------------------------------------------------------
# R3.1-2: label validation and support
# --------------------------------------------------------------------------------

UNKNOWN_LABEL_POLICIES = ("error", "ignore")


def _check_policy(unknown_labels):
    if unknown_labels not in UNKNOWN_LABEL_POLICIES:
        raise ValueError(f"unknown_labels must be one of {UNKNOWN_LABEL_POLICIES}; got "
                         f"{unknown_labels!r}")


def _iter_labels(labels):
    """One document's labels as an iterable of ids; None / empty = background. A
    bare string is refused (it would iterate as characters)."""
    if labels is None:
        return ()
    if isinstance(labels, (str, bytes)):
        raise TypeError(f"a document's labels must be an array of node ids, not the "
                        f"scalar {labels!r}")
    return labels


def _msg_key(u):
    if isinstance(u, (str, int)) and not isinstance(u, bool):
        return node_sort_key(u)
    return (3, 0, repr(u))


def _unknown_error(unknown, n_unknown):
    shown = ", ".join(repr(u) for u in sorted(unknown, key=_msg_key))
    more = "" if n_unknown <= len(unknown) else f" (and {n_unknown - len(unknown)} more)"
    return ValueError(
        f"{n_unknown} label(s) are not nodes of the DAG: {shown}{more}. Ids are "
        f"matched exactly (1 and '1' are different nodes); set "
        f"unknown_labels='ignore' to drop them.")


@dataclass(frozen=True)
class LabelIndex:
    """What the per-document support step needs, small and picklable (broadcast it).

    `nodes`: the DAG's nodes in `node_sort_key` order (a node's index is its
    position); `closure`: `{node: tuple of indices of its reflexive ancestor
    closure}`, root excluded. Plain dicts of tuples, no arrays."""
    nodes: tuple
    closure: dict

    @classmethod
    def from_dag(cls, dag) -> "LabelIndex":
        dag = parse_dag(dag)
        idx = {n: i for i, n in enumerate(dag.nodes)}
        clo = {n: tuple(sorted(idx[a] for a in ancestor_closure(n, dag.parents)
                               if a is not ROOT))
               for n in dag.nodes}
        return cls(nodes=dag.nodes, closure=clo)


def doc_support(labels, index: LabelIndex):
    """ONE document: `(set of node indices its labels' closure covers, list of
    unknown labels)`. The executor-side half of the support pass; reads only
    `index` (broadcast) and the row."""
    covered, unknown = set(), []
    for lab in _iter_labels(labels):
        try:
            lab = _coerce_id(lab, what="label")
        except TypeError:
            unknown.append(lab)
            continue
        clo = index.closure.get(lab)
        if clo is None:
            unknown.append(lab)
        else:
            covered.update(clo)
    return covered, unknown


_UNKNOWN_CAP = 20      # unknown labels carried back for the error message


@dataclass
class SupportCounts:
    """The reduced support pass: `n_docs`, `counts` (node index -> documents), the
    first few `unknown` labels seen (at most `_UNKNOWN_CAP`) and `n_unknown`, the
    number of unknown label OCCURRENCES. Mergeable (`merge`), which is the driver
    reducer; empty is the identity."""
    n_docs: int = 0
    counts: Counter = field(default_factory=Counter)
    unknown: set = field(default_factory=set)
    n_unknown: int = 0

    def add(self, labels, index):
        covered, unknown = doc_support(labels, index)
        self.n_docs += 1
        self.counts.update(covered)
        if unknown:
            self.n_unknown += len(unknown)
            for u in unknown:
                if len(self.unknown) >= _UNKNOWN_CAP:
                    break
                try:
                    self.unknown.add(u)
                except TypeError:      # unhashable junk: keep its repr
                    self.unknown.add(repr(u))
        return self

    def merge(self, other: "SupportCounts") -> "SupportCounts":
        self.n_docs += other.n_docs
        self.counts.update(other.counts)
        self.n_unknown += other.n_unknown
        for u in other.unknown:
            if len(self.unknown) >= _UNKNOWN_CAP:
                break
            self.unknown.add(u)
        return self

    def by_node(self, index: LabelIndex) -> dict:
        """`{node: support}` over every DAG node (0 where no document reaches it)."""
        return {n: int(self.counts.get(i, 0)) for i, n in enumerate(index.nodes)}


def _finish(sc: SupportCounts, index, unknown_labels):
    if sc.n_unknown and unknown_labels == "error":
        raise _unknown_error(sc.unknown, sc.n_unknown)
    return sc.by_node(index), sc.n_docs


def count_support(docs_labels, dag, *, unknown_labels="error"):
    """The support pass over an in-memory iterable of per-document label arrays.

    Returns `(support, n_docs)`: `support` is `{node: number of documents whose labels'
    closure contains node}` over every DAG node. Unknown labels raise a `ValueError`
    naming them (`unknown_labels="error"`) or are dropped (`"ignore"`); a document
    left with no known label is background."""
    _check_policy(unknown_labels)
    index = LabelIndex.from_dag(dag)
    sc = SupportCounts()
    for labels in docs_labels:
        sc.add(labels, index)
    return _finish(sc, index, unknown_labels)


def _partition_support(rows, index_bc, col):
    sc = SupportCounts()
    index = index_bc.value
    for r in rows:
        sc.add(r[col], index)
    yield sc


def count_support_spark(df, labels_col, dag, *, unknown_labels="error"):
    """The support pass over a Spark DataFrame's `labels_col` (an array column).

    Same result as `count_support`. One job: each partition folds its rows into a
    `SupportCounts` (reading the `LabelIndex` from a BROADCAST, so no structure rides
    the task closure, ADR 0047) and the per-partition counts merge with `treeReduce`.
    Every partition yields exactly one (possibly empty) `SupportCounts`, so the
    reduce has no zero value to get wrong."""
    _check_policy(unknown_labels)
    index = LabelIndex.from_dag(dag)
    sc_ctx = df.sparkSession.sparkContext
    bc = sc_ctx.broadcast(index)
    try:
        rdd = df.select(labels_col).rdd
        if rdd.getNumPartitions() == 0:
            sc = SupportCounts()
        else:
            sc = (rdd.mapPartitions(lambda it, _bc=bc: _partition_support(it, _bc, 0))
                  .treeReduce(lambda a, b: a.merge(b)))
    finally:
        bc.unpersist()
    return _finish(sc, index, unknown_labels)


# --------------------------------------------------------------------------------
# R3.3-6 and R4: prune, land, reduce, splice, engine ids
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontierFn:
    """A document's labels (user ids) -> its frontier in ENGINE ids: the
    most-specific landed nodes, as a frozenset (empty = background).

    Landing: a kept label stays; a pruned or spliced label moves to its nearest kept
    ancestors (`landing`); one with none lands nowhere. The frontier then drops every
    landed node that is a strict ancestor of another landed node (`strict_anc`), so
    `allowed_set(frontier)` is the gate of the landed labels' closure: an ancestor's
    closure is inside its descendant's. Small dicts of tuples; picklable for a
    broadcast. Unknown labels follow `unknown_labels`."""
    landing: dict          # user id -> tuple of engine ids
    strict_anc: dict       # engine id -> frozenset of strict-ancestor engine ids (no 0)
    unknown_labels: str = "error"

    def __call__(self, labels) -> frozenset:
        landed = set()
        unknown = []
        for lab in _iter_labels(labels):
            try:
                tgt = self.landing.get(_coerce_id(lab, what="label"))
            except TypeError:
                tgt = None
            if tgt is None:
                unknown.append(lab)
            else:
                landed.update(tgt)
        if unknown and self.unknown_labels == "error":
            raise _unknown_error(unknown[:_UNKNOWN_CAP], len(unknown))
        covered = set()
        for e in landed:
            covered |= self.strict_anc[e]
        return frozenset(landed - covered)


@dataclass(frozen=True)
class NodeSet:
    """The node set learned from (DAG, labelled documents) (R3, R4).

    In user ids:
      `parents`  — the kept DAG `{node: tuple of kept parents}`, transitively reduced;
                   a top-level node's parents are `(ROOT,)`.
      `pruned`   — `{dropped node: tuple of landing nodes}` for nodes with support <
                   `min_docs_per_node` (an empty tuple: its documents lose that label,
                   background when it was their only one).
      `spliced`  — the kept-by-support nodes removed as single-child chain links.
      `support`  — `{kept node: documents reaching it}`; `n_docs` the documents seen.
    Engine ids (internal to the engine, R4): kept nodes are numbered `1..N` in
    TOPOLOGICAL order (by longest path from the root, then `node_sort_key`), the
    root is 0.
      `engine_id` — `{user id: engine id}`; `user_ids` — tuple, `user_ids[e]` is the
                   user id of engine id `e` (`user_ids[0] is ROOT`).
      `engine_parents` — `{engine child: [engine parents]}`, root omitted as a key
                   and given as 0: the map `DagLayout` takes.
      `frontier` — a `FrontierFn`: doc labels -> engine-id frontier.
    """
    parents: dict
    pruned: dict
    spliced: tuple
    support: dict
    n_docs: int
    engine_id: dict
    user_ids: tuple
    engine_parents: dict
    frontier: FrontierFn
    min_docs_per_node: int = 1

    @property
    def nodes(self) -> tuple:
        """Kept user ids in engine-id order (engine ids 1..N)."""
        return self.user_ids[1:]

    def land(self, labels) -> tuple:
        """A document's labels landed on the kept node set, in user ids (sorted,
        deduplicated). Unknown labels follow the frontier's policy."""
        eng = set()
        unknown = []
        for lab in _iter_labels(labels):
            try:
                tgt = self.frontier.landing.get(_coerce_id(lab, what="label"))
            except TypeError:
                tgt = None
            if tgt is None:
                unknown.append(lab)
            else:
                eng.update(tgt)
        if unknown and self.frontier.unknown_labels == "error":
            raise _unknown_error(unknown[:_UNKNOWN_CAP], len(unknown))
        return tuple(self.user_ids[e] for e in sorted(eng))


def _children_of(red):
    """`{node: [children]}` from a `{child: [parents]}` map (parents outside it
    ignored)."""
    ch = {n: [] for n in red}
    for c, ps in red.items():
        for p in ps:
            if p in ch:
                ch[p].append(c)
    return ch


def _longest_depth(parents):
    """Longest path from the root (top-level nodes = 1) over an acyclic
    `{node: tuple(parents)}` map containing `ROOT` as the top parent."""
    depth = {}
    for start in parents:
        if start in depth:
            continue
        stack = [(start, False)]
        while stack:
            n, done = stack.pop()
            if n in depth:
                continue
            ps = [p for p in parents[n] if p is not ROOT]
            if done or all(p in depth for p in ps):
                depth[n] = 1 + max((depth[p] for p in ps), default=0)
            else:
                stack.append((n, True))
                stack.extend((p, False) for p in ps if p not in depth)
    return depth


def learn_node_set(dag, support, *, n_docs=None, min_docs_per_node=1,
                   unknown_labels="error") -> NodeSet:
    """Prune, land, reduce and splice (R3.3-6), then number the kept nodes (R4).

    `support` is `{node: documents whose labels' closure contains it}` from
    `count_support` / `count_support_spark` (a missing node counts 0).

    PRUNE: drop nodes with support < `min_docs_per_node`. Closure support never
    decreases going up (a document reaching a node reaches its ancestors), so the
    kept set is closed upward: pruning removes whole lower regions and never an
    intermediate between two kept nodes (that is what the splice does).
    REDUCE: the kept DAG is `transitive_reduction_parents(kept, full DAG)`: each kept
    node's parents are its nearest kept ancestors, minus any that is an ancestor of
    another (read from the full DAG).
    SPLICE (to a fixpoint): remove a kept node `n` with exactly one kept child `c` and
    no document landing on `n` itself. Why the test is `support[n] == support[c]`:
    landing preserves which kept nodes a document reaches (a dropped label lands on
    the first kept node on each upward path, so every kept ancestor stays in the
    closure, and nothing new enters it), so support is unchanged by landing; and
    every kept descendant of `n` lies under `c`, so the documents reaching `n` are
    exactly those reaching `c` plus those whose landed frontier contains `n`. Equal
    support therefore means `n` admits exactly `c`'s documents: its block would train
    on identical data. All qualifying nodes are removed per pass (a chain goes at
    once) and the reduction is recomputed from the full DAG, until a pass finds none.
    LAND: every non-kept node (pruned or spliced) lands on its nearest kept ancestors
    in the FINAL kept set (`nearest_kept_ancestors`; nearest of nearest is nearest,
    so resolving against the final set equals landing step by step). A spliced node
    never carries a frontier label, so its landing only drops a redundant ancestor.
    """
    _check_policy(unknown_labels)
    dag = parse_dag(dag)
    min_docs = int(min_docs_per_node)
    if min_docs < 0:
        raise ValueError(f"min_docs_per_node must be >= 0; got {min_docs_per_node!r}")
    sup = {n: int(support.get(n, 0)) for n in dag.nodes}
    # The graph work runs on POSITIONS in `dag.nodes` (ints), not on user ids: the
    # `ops` helpers sort node lists, and a mixed str/int id set cannot be sorted
    # natively. Positions follow `node_sort_key`, so sorting them IS the type-aware
    # order. The root is implicit there (a top-level node has no parents).
    nodes = dag.nodes
    pos = {n: i for i, n in enumerate(nodes)}
    P = {pos[c]: [pos[p] for p in ps if p is not ROOT] for c, ps in dag.parents.items()}
    sup_i = [sup[n] for n in nodes]

    kept = {i for i in range(len(nodes)) if sup_i[i] >= min_docs}
    pruned_ids = [nodes[i] for i in range(len(nodes)) if i not in kept]
    spliced = set()
    while True:
        red = transitive_reduction_parents(kept, P)
        ch = _children_of(red)
        doomed = {n for n in kept
                  if len(ch[n]) == 1 and sup_i[n] == sup_i[ch[n][0]]}
        if not doomed:
            break
        spliced |= doomed
        kept -= doomed

    parents = {nodes[c]: tuple(nodes[p] for p in sorted(ps)) if ps else (ROOT,)
               for c, ps in red.items()}
    depth = _longest_depth(parents)
    order = sorted((nodes[i] for i in kept),
                   key=lambda n: (depth[n], node_sort_key(n)))
    engine_id = {n: i + 1 for i, n in enumerate(order)}
    user_ids = (ROOT,) + tuple(order)
    engine_parents = {engine_id[c]: sorted(0 if p is ROOT else engine_id[p]
                                           for p in parents[c])
                      for c in order}

    memo: dict = {}
    landing_user = {n: tuple(nodes[a] for a in
                             sorted(nearest_kept_ancestors(pos[n], kept, P, memo)))
                    for n in nodes}
    landing = {n: tuple(sorted(engine_id[a] for a in t))
               for n, t in landing_user.items()}
    eng_par = {e: [p for p in ps if p != 0] for e, ps in engine_parents.items()}
    strict_anc = {}
    for e in range(1, len(user_ids)):
        acc, stack = set(), list(eng_par[e])
        while stack:
            a = stack.pop()
            if a not in acc:
                acc.add(a)
                stack.extend(eng_par[a])
        strict_anc[e] = frozenset(acc)

    return NodeSet(
        parents={n: parents[n] for n in order},
        pruned={n: landing_user[n] for n in pruned_ids},
        spliced=tuple(nodes[i] for i in sorted(spliced)),
        support={n: sup[n] for n in order},
        n_docs=int(n_docs) if n_docs is not None else 0,
        engine_id=engine_id, user_ids=user_ids, engine_parents=engine_parents,
        frontier=FrontierFn(landing=landing, strict_anc=strict_anc,
                            unknown_labels=unknown_labels),
        min_docs_per_node=min_docs)


def fit_node_set(dag, docs_labels, *, min_docs_per_node=1,
                 unknown_labels="error") -> NodeSet:
    """The whole of R3 + R4 on in-memory data: parse, one support pass, learn.
    `docs_labels` is an iterable of per-document label arrays (iterated once)."""
    dag = parse_dag(dag)
    support, n_docs = count_support(docs_labels, dag, unknown_labels=unknown_labels)
    return learn_node_set(dag, support, n_docs=n_docs,
                          min_docs_per_node=min_docs_per_node,
                          unknown_labels=unknown_labels)
