"""Tests for the compact anchor-induced hierarchy reduction (pure, no Mondo)."""
import sys
from pathlib import Path

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)


def _adj(edges):
    """child -> [parents] from (child, parent) pairs."""
    a: dict = {}
    for child, parent in edges:
        a.setdefault(child, []).append(parent)
    return a


def test_ancestors_stops_at_ceiling():
    from anchor_hierarchy import ancestors
    # leaf -> mid -> top -> ROOT
    adj = _adj([("leaf", "mid"), ("mid", "top"), ("top", "ROOT")])
    assert ancestors(adj, "leaf") == {"mid", "top", "ROOT"}
    assert ancestors(adj, "leaf", stop=frozenset({"ROOT"})) == {"mid", "top"}
    assert ancestors(adj, "leaf", stop=frozenset({"top"})) == {"mid"}


def test_two_anchors_sharing_a_midlevel_form_one_class_and_chain_collapses():
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    # CTD is the branch point for eds+marfan; a linear chain CTD->sub->eds collapses.
    adj = _adj([
        ("eds", "ctd_sub"), ("ctd_sub", "ctd"),   # chain above eds
        ("marfan", "ctd"),
        ("ctd", "ROOT"),
    ])
    h = reduce_to_anchor_hierarchy(
        ["eds", "marfan"], adj, stop=["ROOT"], min_class_size=2)
    # exactly one class covering both anchors
    assert h["n_classes"] == 1
    (cid, info), = h["classes"].items()
    assert info["members"] == ["eds", "marfan"] and info["size"] == 2
    # chain node ctd_sub (covers only eds) is NOT a class; ctd is the class.
    assert cid == "ctd"
    assert h["terminal_class"] == {"eds": "ctd", "marfan": "ctd"}
    assert h["parent_of"]["eds"] == ["ctd"] and h["parent_of"]["marfan"] == ["ctd"]
    assert h["parent_of"]["ctd"] == []  # top class -> synthetic root


def test_singleton_anchor_has_no_class_and_attaches_to_root():
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    adj = _adj([
        ("eds", "ctd"), ("marfan", "ctd"), ("ctd", "ROOT"),
        ("lone", "other"), ("other", "ROOT"),   # shares nothing with the pair
    ])
    h = reduce_to_anchor_hierarchy(
        ["eds", "marfan", "lone"], adj, stop=["ROOT"], min_class_size=2)
    assert h["terminal_class"]["lone"] is None
    assert h["parent_of"]["lone"] == []
    assert h["n_classes"] == 1  # only the eds/marfan class


def test_nested_classes_link_specific_to_general():
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    # autoimmune > {ctd > {eds, scleroderma}, vasculitis-leaf gpa}
    adj = _adj([
        ("eds", "ctd"), ("scleroderma", "ctd"),
        ("ctd", "autoimmune"),
        ("gpa", "autoimmune"),
        ("autoimmune", "ROOT"),
    ])
    h = reduce_to_anchor_hierarchy(
        ["eds", "scleroderma", "gpa"], adj, stop=["ROOT"], min_class_size=2)
    # ctd covers {eds, scleroderma}; autoimmune covers all three.
    assert h["classes"]["ctd"]["members"] == ["eds", "scleroderma"]
    assert h["classes"]["autoimmune"]["size"] == 3
    # ctd's parent is the broader autoimmune class; autoimmune is top.
    assert h["parent_of"]["ctd"] == ["autoimmune"]
    assert h["parent_of"]["autoimmune"] == []
    # gpa (only under autoimmune) -> autoimmune; eds -> the specific ctd.
    assert h["terminal_class"]["gpa"] == "autoimmune"
    assert h["terminal_class"]["eds"] == "ctd"


def test_max_class_fraction_suppresses_the_umbrella():
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    adj = _adj([
        ("eds", "ctd"), ("scleroderma", "ctd"),
        ("ctd", "autoimmune"), ("gpa", "autoimmune"),
        ("autoimmune", "ROOT"),
    ])
    # drop classes covering >2/3 of anchors -> autoimmune (covers all 3) dropped.
    h = reduce_to_anchor_hierarchy(
        ["eds", "scleroderma", "gpa"], adj, stop=["ROOT"],
        min_class_size=2, max_class_fraction=0.67)
    assert "autoimmune" not in h["classes"]
    assert "ctd" in h["classes"]                 # covers 2/3 -> kept
    assert h["terminal_class"]["gpa"] is None     # its only class was dropped


def test_hierarchy_to_edges_wires_root_classes_and_anchors():
    from anchor_hierarchy import reduce_to_anchor_hierarchy, hierarchy_to_edges
    # concept-id space: 100/101 anchors under class 200, which is under class 300.
    adj = _adj([("anchor:100", "200"), ("anchor:101", "200"),
                ("200", "300"), ("300", "9999")])
    h = reduce_to_anchor_hierarchy(
        ["anchor:100", "anchor:101"], adj, stop=["9999"], min_class_size=2)
    edges = hierarchy_to_edges(h, 0)  # root concept-id 0
    es = set(edges)
    # 200 covers {100,101}; 300 covers the same -> chain collapses to one class 200.
    assert (0, 200) in es               # top class -> root
    assert (200, 100) in es and (200, 101) in es  # anchors under their class
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in edges)


def test_hierarchy_to_edges_unclustered_anchor_attaches_to_root():
    from anchor_hierarchy import reduce_to_anchor_hierarchy, hierarchy_to_edges
    adj = _adj([("anchor:100", "200"), ("anchor:101", "200"), ("200", "9999"),
                ("anchor:500", "600"), ("600", "9999")])
    h = reduce_to_anchor_hierarchy(
        ["anchor:100", "anchor:101", "anchor:500"], adj, stop=["9999"],
        min_class_size=2)
    edges = set(hierarchy_to_edges(h, 0))
    assert (0, 500) in edges  # singleton anchor -> root directly


def test_reports_raw_ancestor_count_we_avoid():
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    adj = _adj([
        ("eds", "a"), ("a", "b"), ("b", "c"), ("c", "ROOT"),
        ("marfan", "a"),
    ])
    h = reduce_to_anchor_hierarchy(
        ["eds", "marfan"], adj, stop=["ROOT"], min_class_size=2)
    # raw ancestors across both = {a, b, c}; compact keeps 1 class (a).
    assert h["n_raw_ancestors"] == 3
    assert h["n_classes"] == 1
