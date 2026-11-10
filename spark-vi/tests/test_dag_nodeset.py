"""spark_vi.dag.nodeset: DAG parsing, label validation, support, prune / land /
reduce / splice and engine ids (spec 2026-10-30, R1/R3/R4); dropping kept nodes after
a fit (`drop_nodes`, R14.2)."""
import pickle
import random

import pytest

from spark_vi.dag.layout import DagLayout
from spark_vi.dag.nodeset import (ROOT, FrontierFn, LabelIndex, ParsedDag,
                                  SupportCounts, count_support, doc_support,
                                  drop_nodes, fit_node_set, learn_node_set, node_sort_key,
                                  parse_dag)


# ---------------------------------------------------------------- R1 parsing

def test_dict_scalar_list_and_none_parents():
    d = parse_dag({"b": "a", "c": ["a", "b"], "a": None, "z": []})
    assert d.parents == {"a": (ROOT,), "b": ("a",), "c": ("a", "b"), "z": (ROOT,)}
    assert d.nodes == ("a", "b", "c", "z")


def test_nodes_only_as_parents_are_nodes_and_hang_from_root():
    d = parse_dag({"b": "a", "c": "a"})
    assert "a" in d and d.parents["a"] == (ROOT,)


def test_mixed_str_int_ids_kept_distinct():
    d = parse_dag({1: "1", "1": None, 2: 1})
    assert set(d.parents) == {1, "1", 2}
    assert d.parents[1] == ("1",)                 # an edge, not a self-loop
    assert d.parents["1"] == (ROOT,)
    assert d.n_self_loops == 0
    # ints before strs, never compared with each other
    assert d.nodes == (1, 2, "1")


def test_self_loops_and_duplicate_edges_dropped_and_counted():
    import pandas as pd
    df = pd.DataFrame({"child": ["b", "b", "b", "a"], "parent": ["a", "a", "b", None]})
    d = parse_dag(df)
    assert d.parents == {"a": (ROOT,), "b": ("a",)}
    assert d.n_self_loops == 1 and d.n_duplicate_edges == 1
    d2 = parse_dag({"x": ["x"]})
    assert d2.parents == {"x": (ROOT,)} and d2.n_self_loops == 1


def test_pandas_int_column_with_missing_parent():
    import pandas as pd
    # pandas upcasts [None, 1] to float64 [nan, 1.0]; read back as the int 1
    df = pd.DataFrame({"child": [1, 2, 3], "parent": [None, 1, 1]})
    d = parse_dag(df)
    assert d.parents == {1: (ROOT,), 2: (1,), 3: (1,)}
    assert all(type(n) is int for n in d.nodes)


def test_pandas_object_column_mixed_ids():
    import pandas as pd
    df = pd.DataFrame({"child": ["1", 1, "x"], "parent": [None, "1", 1]},
                      dtype=object)
    d = parse_dag(df)
    assert d.parents == {"1": (ROOT,), 1: ("1",), "x": (1,)}


def test_pandas_missing_columns_refused():
    import pandas as pd
    with pytest.raises(ValueError, match="child, parent"):
        parse_dag(pd.DataFrame({"node": [1], "parent": [None]}))


def test_cycle_refused_naming_nodes():
    with pytest.raises(ValueError, match="cycle") as e:
        parse_dag({"a": "c", "b": "a", "c": "b", "d": "a"})
    msg = str(e.value)
    for n in ("'a'", "'b'", "'c'"):
        assert n in msg
    assert "'d'" not in msg


def test_bad_ids_refused():
    with pytest.raises(TypeError, match="bool"):
        parse_dag({True: None})
    with pytest.raises(TypeError, match="str or int"):
        parse_dag({1.5: None})
    with pytest.raises(TypeError):
        parse_dag([("a", "b")])


def test_numpy_ids_unwrapped():
    import numpy as np
    d = parse_dag({np.int64(2): np.int64(1)})
    assert d.parents == {2: (1,), 1: (ROOT,)}


def test_root_sentinel_pickles_by_identity():
    d = parse_dag({"b": "a"})
    d2 = pickle.loads(pickle.dumps(d))
    assert d2.parents["a"][0] is ROOT
    assert repr(ROOT) == "ROOT"


def test_node_sort_key_total_on_mixed():
    xs = ["b", 3, "a", 1, "10", 10]
    assert sorted(xs, key=node_sort_key) == [1, 3, 10, "10", "a", "b"]


# ---------------------------------------------------------------- R3.1-2 labels, support

# a -> b -> d ; a -> c -> d (diamond) ; e top-level
DIAMOND = {"b": "a", "c": "a", "d": ["b", "c"], "e": None}


def test_support_counts_ancestor_closure():
    docs = [["d"], ["b"], ["b", "a"], [], None, ["e", "e"]]
    sup, n = count_support(docs, DIAMOND)
    assert n == 6
    assert sup == {"a": 3, "b": 3, "c": 1, "d": 1, "e": 1}


def test_unknown_labels_error_names_them():
    with pytest.raises(ValueError, match=r"'zz'.*") as e:
        count_support([["a", "zz"], [1]], DIAMOND)
    assert "'zz'" in str(e.value) and "1" in str(e.value)


def test_unknown_labels_int_vs_str_distinct():
    with pytest.raises(ValueError, match="not nodes"):
        count_support([[1]], {"1": None})
    sup, _ = count_support([["1"]], {"1": None})
    assert sup == {"1": 1}


def test_unknown_labels_ignore_drops_and_backgrounds():
    sup, n = count_support([["zz"], ["zz", "b"]], DIAMOND, unknown_labels="ignore")
    assert n == 2
    assert sup["b"] == 1 and sup["a"] == 1 and sup["d"] == 0


def test_bad_policy_and_scalar_labels_refused():
    with pytest.raises(ValueError):
        count_support([], DIAMOND, unknown_labels="drop")
    with pytest.raises(TypeError):
        count_support(["abc"], DIAMOND)


def test_doc_support_and_reducer_merge():
    idx = LabelIndex.from_dag(DIAMOND)
    covered, unk = doc_support(["d", "q"], idx)
    assert {idx.nodes[i] for i in covered} == {"a", "b", "c", "d"}
    assert unk == ["q"]
    a = SupportCounts().add(["d"], idx).add(["e"], idx)
    b = SupportCounts().add(["b"], idx)
    m = a.merge(b).merge(SupportCounts())
    assert m.n_docs == 3
    assert m.by_node(idx) == {"a": 2, "b": 2, "c": 1, "d": 1, "e": 1}
    # small and picklable (broadcast)
    assert pickle.loads(pickle.dumps(idx)) == idx


# ---------------------------------------------------------------- R3.3-6 prune, land, reduce, splice

def test_prune_lands_on_multiple_parents():
    # x has two incomparable parents p, q; x is rare -> pruned; lands on both
    dag = {"p": None, "q": None, "x": ["p", "q"], "y": "p", "w": "q"}
    docs = [["x"]] + [["y"]] * 2 + [["w"]] * 2
    ns = fit_node_set(dag, docs, min_docs_per_node=2)
    assert ns.pruned == {"x": ("p", "q")}
    assert ns.land(["x"]) == ("p", "q")
    lay = DagLayout(ns.engine_parents)
    f = ns.frontier(["x"])
    assert {ns.user_ids[e] for e in f} == {"p", "q"}
    assert set(lay.allowed_set(f)) >= set(lay.block[ns.engine_id["p"]])


def test_prune_lands_to_background():
    dag = {"a": None, "b": None, "c": "b"}
    docs = [["a"]] + [["c"]] * 3 + [["b"]]
    ns = fit_node_set(dag, docs, min_docs_per_node=2)
    assert ns.pruned == {"a": ()}
    assert ns.frontier(["a"]) == frozenset()
    assert ns.land(["a"]) == ()


def test_default_min_docs_prunes_unreached_nodes_only():
    ns = fit_node_set(DIAMOND, [["b"], ["e"], ["c"]])
    assert set(ns.pruned) == {"d"}
    assert ns.pruned["d"] == ("b", "c")


def test_transitive_reduction_over_kept():
    # a -> b -> d plus the redundant a -> d: d's parents reduce to {b}.
    dag = {"a": None, "b": "a", "d": ["a", "b"]}
    docs = [["d"]] * 3 + [["b"]] + [["a"]]
    ns = fit_node_set(dag, docs)
    assert ns.pruned == {} and ns.spliced == ()
    assert ns.parents == {"a": (ROOT,), "b": ("a",), "d": ("b",)}


def test_support_is_monotone_so_pruning_keeps_ancestors():
    # closure support never decreases upward: a kept node's ancestors are all kept
    dag = {"a": None, "b": "a", "c": "b"}
    ns = fit_node_set(dag, [["c"]] * 2 + [["b"]] * 3, min_docs_per_node=3)
    assert ns.pruned == {"c": ("b",)}
    assert ns.spliced == ("a",)                  # a admits exactly b's 5 docs
    assert ns.support == {"b": 5}


def test_reduction_after_splice_drops_sibling_edge():
    # a -> b -> d, a -> c -> d; b (one child d, no own docs) is spliced: d's nearest
    # kept are a (past b) and c, and a is an ancestor of c, so d's parents reduce
    # to {c}: the sibling edge a -> d is the redundant one.
    dag = {"a": None, "b": "a", "c": "a", "d": ["b", "c"]}
    docs = [["d"]] * 3 + [["c"]] * 3 + [["a"]] * 3
    ns = fit_node_set(dag, docs)
    assert ns.pruned == {} and ns.spliced == ("b",)
    assert ns.parents == {"a": (ROOT,), "c": ("a",), "d": ("c",)}
    assert ns.frontier(["b"]) == frozenset({ns.engine_id["a"]})


def test_splice_chain_to_fixpoint():
    # r -> m1 -> m2 -> leaf, nobody labelled at r, m1, m2: the whole chain splices.
    dag = {"r": None, "m1": "r", "m2": "m1", "leaf": "m2", "other": None}
    docs = [["leaf"]] * 3 + [["other"]]
    ns = fit_node_set(dag, docs)
    assert ns.spliced == ("m1", "m2", "r")
    assert ns.parents == {"leaf": (ROOT,), "other": (ROOT,)}
    assert ns.pruned == {}
    # labels at a spliced node land up (here nowhere: r was top-level)
    assert ns.frontier(["leaf", "m1"]) == frozenset({ns.engine_id["leaf"]})


def test_splice_spares_node_with_own_documents():
    dag = {"r": None, "m": "r", "leaf": "m"}
    docs = [["leaf"]] * 3 + [["m"]]                 # one doc labelled at m itself
    ns = fit_node_set(dag, docs)
    assert ns.spliced == ("r",)                      # r admits exactly m's docs
    assert set(ns.parents) == {"m", "leaf"}


def test_redundant_ancestor_label_does_not_block_splice():
    # a doc listing both m and its child leaf has frontier {leaf}: m still splices
    dag = {"m": None, "leaf": "m"}
    ns = fit_node_set(dag, [["leaf", "m"], ["leaf"]])
    assert ns.spliced == ("m",)


def test_splice_after_prune_landing_is_blocked():
    # p -> m -> {leaf, rare}; rare pruned and lands on m, so m has a frontier doc
    # and one kept child: NOT spliced. p (one kept child m, same docs) is spliced.
    dag = {"p": None, "m": "p", "leaf": "m", "rare": "m"}
    docs = [["leaf"]] * 3 + [["rare"]]
    ns = fit_node_set(dag, docs, min_docs_per_node=2)
    assert ns.pruned == {"rare": ("m",)}
    assert ns.spliced == ("p",)
    assert ns.parents == {"m": (ROOT,), "leaf": ("m",)}
    assert ns.frontier(["rare"]) == frozenset({ns.engine_id["m"]})


def test_splice_multi_parent_child():
    # a -> s -> x ; b -> x. s has one child x, no own docs: spliced, x keeps a and b.
    dag = {"a": None, "b": None, "s": "a", "x": ["s", "b"], "y": "a"}
    docs = [["x"]] * 2 + [["y"]] + [["b"]]
    ns = fit_node_set(dag, docs)
    assert ns.spliced == ("s",)
    assert ns.parents["x"] == ("a", "b")


def test_learn_node_set_takes_precomputed_support():
    sup = {"a": 5, "b": 5, "c": 0, "d": 0, "e": 2}
    ns = learn_node_set(DIAMOND, sup, n_docs=7, min_docs_per_node=1)
    assert ns.n_docs == 7
    assert ns.spliced == ("a",) and set(ns.pruned) == {"c", "d"}
    with pytest.raises(ValueError):
        learn_node_set(DIAMOND, sup, min_docs_per_node=-1)


# ---------------------------------------------------------------- R4 engine ids, frontier

def test_engine_ids_topological_and_deterministic():
    dag = {"z": None, 3: "z", "a": 3, 1: None, 2: [1, "z"]}
    docs = [["a", 2], [3, 1], ["z"], [2]]
    ns = fit_node_set(dag, docs)
    assert ns.user_ids[0] is ROOT
    # depth 1: ints first (1), then "z"; depth 2: 2, 3; depth 3: "a"
    assert ns.nodes == (1, "z", 2, 3, "a")
    assert ns.engine_id == {1: 1, "z": 2, 2: 3, 3: 4, "a": 5}
    assert ns.engine_parents == {1: [0], 2: [0], 3: [1, 2], 4: [2], 5: [4]}
    # same result from the dict in another insertion order
    ns2 = fit_node_set(dict(reversed(list(dag.items()))), docs)
    assert ns2.engine_id == ns.engine_id and ns2.engine_parents == ns.engine_parents
    DagLayout(ns.engine_parents)                     # the layout accepts it


def test_frontier_unknown_label_policy():
    ns = fit_node_set(DIAMOND, [["d"], ["e"]])
    with pytest.raises(ValueError, match="'nope'"):
        ns.frontier(["nope"])
    ns_i = fit_node_set(DIAMOND, [["d"], ["e"]], unknown_labels="ignore")
    assert ns_i.frontier(["nope", "e"]) == frozenset({ns_i.engine_id["e"]})
    assert ns.frontier(None) == frozenset() and ns.frontier([]) == frozenset()


def test_frontier_fn_pickles():
    ns = fit_node_set(DIAMOND, [["d"], ["e"]])
    f2 = pickle.loads(pickle.dumps(ns.frontier))
    assert isinstance(f2, FrontierFn)
    assert f2(["d", "a"]) == ns.frontier(["d", "a"])


def test_node_set_json_round_trip_keeps_id_types():
    """`to_dict` -> JSON -> `from_dict` is the identity, with 1 and "1" kept apart
    (the model persists its node set this way)."""
    import json
    from spark_vi.dag.nodeset import NodeSet
    dag = {"A": None, 1: "A", "1": "A", 2: [1, "1"], "x": 2, "y": 2, "z": "y"}
    ns = fit_node_set(dag, [[2], ["x"], ["x", 1], ["1"], ["z"], []],
                      min_docs_per_node=2)
    back = NodeSet.from_dict(json.loads(json.dumps(ns.to_dict())))
    assert back == ns
    assert 1 in back.spliced and "1" in back.engine_id      # 1 and "1" stay apart


def _random_dag(rng, n):
    ids = [i if rng.random() < 0.5 else f"n{i}" for i in range(n)]
    dag = {}
    for j, v in enumerate(ids):
        k = rng.choice([0, 1, 1, 2, 3]) if j else 0
        dag[v] = rng.sample(ids[:j], min(k, j)) or None
    return dag, ids


@pytest.mark.parametrize("seed", range(12))
def test_frontier_gate_equals_closure_of_raw_labels(seed):
    """allowed_set(frontier(labels)) is the gate over (closure of the raw labels) ∩
    kept, for random DAGs, labels and thresholds: landing, splicing and the frontier
    reduction change no document's set of reached kept nodes."""
    rng = random.Random(seed)
    dag, ids = _random_dag(rng, 30)
    docs = [rng.sample(ids, rng.choice([0, 1, 1, 2, 3])) for _ in range(60)]
    ns = fit_node_set(dag, docs, min_docs_per_node=rng.choice([1, 2, 4]))
    P = parse_dag(dag).parents
    lay = DagLayout(ns.engine_parents, n_bg=1, tpn=1)
    from spark_vi.dag.ops import ancestor_closure
    kept = set(ns.engine_id)
    for labels in docs:
        raw = set()
        for l in labels:
            raw |= ancestor_closure(l, P)
        want = sorted({0} | {lay.block[ns.engine_id[n]][0] for n in raw & kept})
        f = ns.frontier(labels)
        assert sorted(lay.allowed_set(f).tolist()) == want
        # the frontier is an antichain
        for e in f:
            assert not (set(lay.closure(e)) - {e}) & f
    # every kept node's support equals the number of docs reaching it
    for n in kept:
        reach = sum(1 for labels in docs
                    if n in set().union(*(ancestor_closure(l, P) for l in labels)))
        assert ns.support[n] == reach
    # no surviving node is a splice candidate
    ch = {n: [c for c, ps in ns.parents.items() if n in ps] for n in ns.parents}
    for n, cs in ch.items():
        if len(cs) == 1:
            assert ns.support[n] > ns.support[cs[0]]


# ---------------------------------------------------------------- R14.2 drop_nodes

# r is top-level; x, y under r; m under x only; z under x AND y; w under m.
# Labels make every node kept and none a splice candidate (each has its own docs).
DROP_DAG = {"r": None, "x": "r", "y": "r", "m": "x", "z": ["x", "y"], "w": "m"}
DROP_DOCS = [["r"], ["x"], ["y"], ["m"], ["z"], ["w"], ["w"], ["z", "m"], []]


def _drop_ns():
    ns = fit_node_set(DROP_DAG, DROP_DOCS)
    assert set(ns.nodes) == set(DROP_DAG) and not ns.pruned and not ns.spliced
    return ns


def _check_engine_numbering(ns):
    """Engine ids follow learn_node_set's rule: (longest depth, node_sort_key)."""
    from spark_vi.dag.nodeset import _longest_depth
    depth = _longest_depth(ns.parents)
    keys = [(depth[n], node_sort_key(n)) for n in ns.nodes]
    assert keys == sorted(keys)
    assert ns.engine_id == {n: i + 1 for i, n in enumerate(ns.nodes)}
    for n, ps in ns.parents.items():
        assert ns.engine_parents[ns.engine_id[n]] == sorted(
            0 if p is ROOT else ns.engine_id[p] for p in ps)


def test_drop_a_leaf_lands_its_labels_on_its_parent():
    ns = _drop_ns()
    out = drop_nodes(ns, ["w"])
    assert "w" not in out.engine_id and "w" not in out.support
    assert out.unsupported == {"w": ("m",)}
    assert out.frontier(["w"]) == {out.engine_id["m"]}
    assert out.land(["w", "y"]) == ("y", "m")             # engine order
    assert out.parents["m"] == ("x",)
    assert ns.unsupported == {} and "w" in ns.engine_id   # the input is unchanged
    _check_engine_numbering(out)


def test_drop_a_node_with_kept_children_reattaches_and_rereduces():
    """x has a single-parent child m and a multi-parent child z whose other parent y is
    kept. m re-attaches to r; z's nearest survivors are r (through x) and y, and r is
    an ancestor of y, so the re-reduction leaves z under y only."""
    ns = _drop_ns()
    out = drop_nodes(ns, ["x"])
    assert out.parents["m"] == ("r",)
    assert out.parents["z"] == ("y",)
    assert out.parents["w"] == ("m",)
    assert out.unsupported == {"x": ("r",)}
    assert out.frontier(["x"]) == {out.engine_id["r"]}
    lay = DagLayout(out.engine_parents, n_bg=0, tpn=1)
    (fz,) = out.frontier(["z"])
    assert {out.user_ids[e] for e in lay.closure(fz)} - {ROOT} == {"z", "y", "r"}
    assert out.support == {n: ns.support[n] for n in out.nodes}
    _check_engine_numbering(out)


def test_drop_a_top_level_node():
    ns = _drop_ns()
    out = drop_nodes(ns, ["r"])
    assert out.parents["x"] == (ROOT,) and out.parents["y"] == (ROOT,)
    assert out.unsupported == {"r": ()}
    assert out.frontier(["r"]) == frozenset()             # lands nowhere: background
    assert out.land(["r"]) == ()
    _check_engine_numbering(out)


def test_chained_drops_equal_one_drop_and_reland_earlier_drops():
    ns = _drop_ns()
    step = drop_nodes(drop_nodes(ns, ["w"]), ["m"])
    assert step.unsupported == {"w": ("x",), "m": ("x",)}  # w re-landed past m
    assert step == drop_nodes(ns, ["m", "w"]) == drop_nodes(ns, ["w", "m"])
    # a parent and its child dropped together: w falls to r, z to y
    both = drop_nodes(ns, ["x", "m"])
    assert both.parents["w"] == ("r",) and both.parents["z"] == ("y",)
    assert both.unsupported == {"m": ("r",), "x": ("r",)}


def test_drop_with_mixed_str_and_int_ids():
    dag = {"A": None, 1: "A", "1": "A", 2: [1, "1"], 3: 2}
    docs = [["A"], [1], ["1"], [2], [3], [3]]
    ns = fit_node_set(dag, docs)
    assert not ns.spliced and not ns.pruned
    out = drop_nodes(ns, [1])
    assert 1 not in out.engine_id and "1" in out.engine_id     # 1 and "1" apart
    assert out.parents[2] == ("1",)                            # A is above "1"
    assert out.unsupported == {1: ("A",)}
    assert out.land([1]) == ("A",)
    _check_engine_numbering(out)
    with pytest.raises(KeyError, match="already dropped"):
        drop_nodes(out, ["1", 1])                              # 1 is no longer kept


def test_pruned_and_spliced_labels_that_landed_on_a_dropped_node_land_further_up():
    # leaf is pruned (support 1 < 2) and lands on b; s is spliced (one kept child t,
    # no documents of its own) and lands on b.
    dag = {"a": None, "b": "a", "c": "a", "leaf": "b", "s": "b", "t": "s"}
    docs = [["a"], ["b"], ["b"], ["c"], ["c"], ["leaf"], ["t"], ["t"]]
    ns = fit_node_set(dag, docs, min_docs_per_node=2)
    assert ns.pruned == {"leaf": ("b",)} and ns.spliced == ("s",)
    assert ns.land(["s"]) == ("b",)
    out = drop_nodes(ns, ["b"])
    assert out.pruned == {"leaf": ("a",)}
    assert out.spliced == ("s",) and out.land(["s"]) == ("a",)
    assert out.parents["t"] == ("a",)
    assert out.unsupported == {"b": ("a",)}


def test_drop_refuses_a_node_that_is_not_kept_and_empty_is_identity():
    ns = _drop_ns()
    assert drop_nodes(ns, []) is ns
    with pytest.raises(KeyError, match="not a kept node"):
        drop_nodes(ns, ["nope"])
    with pytest.raises(KeyError, match="already dropped"):
        drop_nodes(drop_nodes(ns, ["w"]), ["w"])


def test_drop_is_deterministic():
    ns = _drop_ns()
    a = drop_nodes(ns, ["x", "w"])
    b = drop_nodes(fit_node_set(DROP_DAG, list(reversed(DROP_DOCS))), ["w", "x"])
    assert a == b
    assert a.to_dict() == b.to_dict()


def test_node_set_json_round_trip_keeps_unsupported():
    import json
    from spark_vi.dag.nodeset import NodeSet
    dag = {"A": None, 1: "A", "1": "A", 2: [1, "1"], 3: 2}
    ns = fit_node_set(dag, [["A"], [1], ["1"], [2], [3], [3]])
    out = drop_nodes(ns, [1, 2])
    back = NodeSet.from_dict(json.loads(json.dumps(out.to_dict())))
    assert back == out
    # 2 lands on the first survivor of each upward path (through 1: A; through
    # "1": "1"), as a pruned node's labels land in learn_node_set (not reduced: the
    # frontier drops the redundant A, so the gate is the same)
    assert back.unsupported == {1: ("A",), 2: ("1", "A")}
    assert back.frontier([2]) == {back.engine_id["1"]}
    # a node set saved before R14 has no key: never cleaned
    d = ns.to_dict()
    del d["unsupported"]
    assert NodeSet.from_dict(d).unsupported == {}


@pytest.mark.parametrize("seed", range(12))
def test_drop_gate_equals_closure_of_raw_labels_over_survivors(seed):
    """After dropping random kept nodes (in one or two calls; the kept set is then not
    closed upward), allowed_set(frontier(labels)) is the gate over (closure of the raw
    labels in the FULL DAG) ∩ survivors, reachability among survivors is the full
    DAG's, support is unchanged, and the kept DAG is a transitive reduction."""
    from spark_vi.dag.ops import ancestor_closure
    rng = random.Random(100 + seed)
    dag, ids = _random_dag(rng, 30)
    docs = [rng.sample(ids, rng.choice([0, 1, 1, 2, 3])) for _ in range(60)]
    ns = fit_node_set(dag, docs, min_docs_per_node=rng.choice([1, 2, 4]))
    kept = list(ns.nodes)
    k = rng.randint(1, max(1, len(kept) // 2))
    gone = rng.sample(kept, k)
    if seed % 2 and k > 1:
        out = drop_nodes(drop_nodes(ns, gone[: k // 2]), gone[k // 2:])
    else:
        out = drop_nodes(ns, gone)
    assert set(out.unsupported) == set(gone)
    assert set(out.nodes) == set(kept) - set(gone)
    _check_engine_numbering(out)
    P = parse_dag(dag).parents
    lay = DagLayout(out.engine_parents, n_bg=1, tpn=1)
    surv = set(out.engine_id)
    for labels in docs:
        raw = set()
        for lab in labels:
            raw |= ancestor_closure(lab, P)
        want = sorted({0} | {lay.block[out.engine_id[n]][0] for n in raw & surv})
        f = out.frontier(labels)
        assert sorted(lay.allowed_set(f).tolist()) == want
        for e in f:
            assert not (set(lay.closure(e)) - {e}) & f
    full_anc = {n: ancestor_closure(n, P) - {n} for n in surv}
    for n in surv:
        new_anc = {out.user_ids[e] for e in lay.closure(out.engine_id[n])} - {n, ROOT}
        assert new_anc == full_anc[n] & surv
        ps = [p for p in out.parents[n] if p is not ROOT]   # reduced: no comparable pair
        for p in ps:
            assert not any(p in full_anc[q] for q in ps if q != p)
    assert out.support == {n: ns.support[n] for n in out.nodes}


# ---------------------------------------------------------------- Spark inputs (slow)

def test_spark_dag_and_support_match_python(spark):
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType
    from spark_vi.dag.nodeset import count_support_spark
    edges = spark.createDataFrame(
        [("b", "a"), ("c", "a"), ("d", "b"), ("d", "c"), ("d", "c"), ("e", None),
         ("x", "x")],
        StructType([StructField("child", StringType()),
                    StructField("parent", StringType(), True)]))
    d = parse_dag(edges)
    assert d.parents == {"a": (ROOT,), "b": ("a",), "c": ("a",), "d": ("b", "c"),
                         "e": (ROOT,), "x": (ROOT,)}
    assert d.n_self_loops == 1 and d.n_duplicate_edges == 1
    docs = [["d"], ["b"], ["b", "a"], [], None, ["e", "e"], ["x"]] * 3
    schema = StructType([StructField("labels", ArrayType(StringType()), True)])
    df = spark.createDataFrame([(l,) for l in docs], schema).repartition(5)
    assert count_support_spark(df, "labels", d) == count_support(docs, d)
    bad = spark.createDataFrame([(["d", "zz"],)], schema)
    with pytest.raises(ValueError, match="'zz'"):
        count_support_spark(bad, "labels", d)
    sup, n = count_support_spark(bad, "labels", d, unknown_labels="ignore")
    assert n == 1 and sup["d"] == 1
    empty = spark.createDataFrame([], schema)
    sup0, n0 = count_support_spark(empty, "labels", d)
    assert n0 == 0 and set(sup0.values()) == {0}


def test_spark_int_ids_distinct_from_str(spark):
    from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType
    from spark_vi.dag.nodeset import count_support_spark
    edges = spark.createDataFrame(
        [(2, 1), (1, None)],
        StructType([StructField("child", LongType()),
                    StructField("parent", LongType(), True)]))
    d = parse_dag(edges)
    assert d.parents == {1: (ROOT,), 2: (1,)}
    df = spark.createDataFrame(
        [([2],), ([1],)], StructType([StructField("labels", ArrayType(LongType()))]))
    sup, n = count_support_spark(df, "labels", d)
    assert sup == {1: 2, 2: 1} and n == 2
    str_df = spark.createDataFrame(
        [(["1"],)], StructType([StructField("labels", ArrayType(StringType()))]))
    with pytest.raises(ValueError, match="not nodes"):
        count_support_spark(str_df, "labels", d)
    ns = learn_node_set(d, sup, n_docs=n)
    assert ns.nodes == (1, 2) and ns.engine_parents == {1: [0], 2: [1]}
