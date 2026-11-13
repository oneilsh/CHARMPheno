"""charmpheno.export.dag_gated: codecs, term selection, support banding and derivation,
and the build on a small node set (pure numpy; no Spark). The end-to-end tests over a
saved run dir and the committed browser fixture are in
analysis/cloud/tests/test_export_dag_gated.py."""
import numpy as np
import pytest

from charmpheno.export import dag_gated as dg


# --------------------------------------------------------------------------- codecs

@pytest.mark.parametrize("s,band", [(0, 0), (1, 10), (19, 10), (20, 10), (21, 30),
                                    (40, 30), (41, 50), (1000, 990), (1001, 1010),
                                    (None, None)])
def test_band_support_midpoints(s, band):
    assert dg.band_support(s) == band


def test_delta_round_trip():
    ids = [0, 3, 4, 17, 400, 401]
    enc = dg.delta_encode(ids)
    assert enc == [0, 3, 1, 13, 383, 1]
    assert dg.delta_decode(enc) == ids
    assert dg.delta_encode([]) == [] and dg.delta_decode([]) == []


def test_log_byte_round_trip_within_half_a_step():
    rng = np.random.default_rng(0)
    x = np.exp(rng.uniform(-12, -0.5, size=2000))
    lo, hi = dg.log_byte_range(x)
    b = dg.log_byte_encode(x, lo, hi)
    assert b.dtype == np.uint8 and b.min() == 0 and b.max() == 255
    err = np.abs(np.log(dg.log_byte_decode(b, lo, hi)) - np.log(x))
    assert err.max() <= dg.log_byte_max_log_error(lo, hi) + 1e-12


def test_log_byte_degenerate_range():
    lo, hi = dg.log_byte_range([0.25, 0.25])
    assert lo == hi
    b = dg.log_byte_encode([0.25, 0.25], lo, hi)
    assert list(b) == [0, 0]
    assert np.allclose(dg.log_byte_decode(b, lo, hi), 0.25)


# --------------------------------------------------------------------------- selection

def test_select_terms_top_beta_then_lift_within_pool():
    # one topic, 6 terms. β order: 0 > 1 > 2 > 3 > 4 > 5.
    beta = np.array([[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]])
    # term 5 has the highest lift but is outside the pool of 4;
    # inside the pool (0..3), beyond the top 2, term 3 out-lifts term 2.
    pw = np.array([0.30, 0.30, 0.30, 0.01, 0.08, 0.001])
    got = dg.select_terms(beta, pw, top_beta=2, top_lift=1, lift_pool=4)
    assert got[0].tolist() == [0, 1, 3]
    got = dg.select_terms(beta, pw, top_beta=2, top_lift=5, lift_pool=4)
    assert got[0].tolist() == [0, 1, 2, 3]            # pool exhausted, no duplicates


def test_select_terms_matches_brute_force_and_breaks_ties_by_index():
    rng = np.random.default_rng(3)
    lam = 0.05 + rng.pareto(1.2, size=(20, 300)) * (rng.random((20, 300)) < 0.2)
    beta, pw = dg.domain_beta(lam), dg.term_marginal(lam)
    got = dg.select_terms(beta, pw, top_beta=7, top_lift=5, lift_pool=40)
    for k in range(20):
        order = sorted(range(300), key=lambda w: (-beta[k, w], w))
        head, pool = order[:7], order[7:40]
        lift = sorted(pool, key=lambda w: (-(beta[k, w] / pw[w]),
                                           order.index(w)))[:5]
        assert got[k].tolist() == sorted(head + lift)
        assert len(set(got[k].tolist())) == len(got[k])


def test_select_terms_small_vocabulary_ships_everything():
    beta = np.full((2, 5), 0.2)
    got = dg.select_terms(beta, np.full(5, 0.2), top_beta=25, top_lift=25,
                          lift_pool=200)
    assert all(g.tolist() == [0, 1, 2, 3, 4] for g in got)


# --------------------------------------------------------------------------- support

def test_spliced_support_is_derived_exactly_and_pruned_stays_unknown():
    from spark_vi.dag.nodeset import count_support, fit_node_set, parse_dag
    dag = {"a": [], "b": ["a"], "c": ["b"], "d": ["c"], "e": ["c"], "f": ["a"],
           "g": ["f"], "h": ["a", "g"]}
    labels = [["d"]] * 30 + [["e"]] * 25 + [["h"]] * 40 + [["g"]] * 3 + [["f"]] * 2
    ns = fit_node_set(dag, labels, min_docs_per_node=20)
    assert set(ns.spliced) >= {"b"}                    # b -> c chain, equal support
    true, _ = count_support(labels, parse_dag(dag))
    sup = dg.node_supports(dag, ns)
    for n in ns.spliced:
        assert sup[n] == true[n]
    for n in ns.nodes:
        assert sup[n] == true[n]
    for n in ns.pruned:
        assert sup[n] is None


def test_unsupported_support_comes_from_extra_or_is_unknown():
    from spark_vi.dag.nodeset import drop_nodes, fit_node_set
    dag = {"a": [], "b": ["a"], "c": ["b"], "d": ["a"]}
    ns = drop_nodes(fit_node_set(dag, [["c"]] * 5 + [["b"]] * 2 + [["d"]] * 4), ["b"])
    assert dg.node_supports(dag, ns)["b"] is None
    assert dg.node_supports(dag, ns, {"b": 7})["b"] == 7


# --------------------------------------------------------------------------- build

def _small(tpn=1):
    from spark_vi.dag.nodeset import fit_node_set
    dag = {"x": [], "y": ["x"], "z": ["x"], "w": ["y", "z"], "v": ["w"]}
    ns = fit_node_set(dag, [["w"]] * 40 + [["y"]] * 30 + [["z"]] * 20 + [["v"]] * 10,
                      min_docs_per_node=20)
    n_bg = 2
    K = n_bg + len(ns.nodes) * tpn
    rng = np.random.default_rng(1)
    lam = {0: 0.1 + rng.gamma(0.3, 20, size=(K, 40)), 1: 0.1 + rng.gamma(0.3, 20, size=(K, 30))}
    vm = [{str(1000 + i): i for i in range(40)}, {str(2000 + i): i for i in range(30)}]
    return dag, ns, n_bg, lam, vm


def test_build_with_tpn_two_points_nodes_at_their_first_topic():
    dag, ns, n_bg, lam, vm = _small(tpn=2)
    f = dg.build_export(lam, ns, n_bg=n_bg, tpn=2, given_dag=dag,
                        domains=["a", "b"], vocab_maps=vm, top_beta=3, top_lift=2,
                        lift_pool=10, created_utc="t")
    nodes, topics = f["nodes.json"], f["topics.json"]
    assert len(topics["node"]) == n_bg + 2 * len(ns.nodes)
    for i, t in enumerate(nodes["topic"]):
        if t is None:
            continue
        assert t >= n_bg and (t - n_bg) % 2 == 0
        assert topics["node"][t] == i and topics["node"][t + 1] == i
    assert topics["node"][:n_bg] == [None] * n_bg


def test_build_refuses_a_lambda_that_does_not_fit_the_layout():
    dag, ns, n_bg, lam, vm = _small()
    bad = {m: a[:-1] for m, a in lam.items()}
    with pytest.raises(ValueError, match="topics"):
        dg.build_export(bad, ns, n_bg=n_bg, tpn=1, given_dag=dag, domains=["a", "b"],
                        vocab_maps=vm)


def test_embed_in_page_round_trips_and_cannot_close_the_script(tmp_path):
    """The self-contained page carries the four files in a JSON script block; a string
    holding "</script>" must not end the block early."""
    import json
    from charmpheno.export.dag_gated import FILES, embed_in_page
    for name in FILES:
        (tmp_path / name).write_text(json.dumps({"name": name, "evil": "</script><b>x"}))
    page = "<html><head><title>t</title></head><body><script>boot()</script></body></html>"
    out = embed_in_page(page, tmp_path)
    start = out.index('<script id="dag-data" type="application/json">') + len(
        '<script id="dag-data" type="application/json">')
    block = out[start:out.index("</script>", start)]
    data = json.loads(block)
    assert set(data) == set(FILES)
    assert data["nodes.json"]["evil"] == "</script><b>x"
    assert out.index('id="dag-data"') < out.index("</head>") < out.index("boot()")


def test_npmi_block_aligns_topics_and_keeps_only_scalars():
    import math
    from charmpheno.export.dag_gated import npmi_block
    rows = [{"domain": 2, "topic": 0, "npmi_gated": 0.25, "npmi_all": 0.2, "n_pairs": 45},
            {"domain": 2, "topic": 2, "npmi_gated": math.nan, "npmi_all": -0.0414},
            {"domain": 0, "topic": 1, "npmi_gated": -0.1235, "npmi_all": None},
            {"domain": 0, "topic": 9, "npmi_gated": 0.5, "npmi_all": 0.5}]   # out of range
    block, meta = npmi_block({"top_n": 10, "split": "train", "min_pair_count": 20,
                              "rows": rows}, K=3)
    assert block == {"domains": [0, 2],
                     "gated": [[None, -124, None], [250, None, None]],
                     "all": [[None, None, None], [200, None, -41]]}
    assert meta == {"top_n": 10, "split": "train", "min_pair_count": 20,
                    "domains": [0, 2], "scale": 1000}


def test_build_refuses_a_node_set_floored_below_the_egress_floor():
    """Code review A2: kept nodes must rest on >= EGRESS_FLOOR documents."""
    import pytest
    from spark_vi.dag.nodeset import fit_node_set
    dag, _, n_bg, lam, vm = _small()
    ns = fit_node_set(dag, [["w"]] * 4 + [["y"]] * 3 + [["z"]] * 2 + [["v"]] * 1,
                      min_docs_per_node=2)
    K = n_bg + len(ns.nodes)
    lam = {m: a[:K] for m, a in lam.items()}
    with pytest.raises(ValueError, match=r"min_docs_per_node=2 is below the egress floor 20"):
        dg.build_export(lam, ns, n_bg=n_bg, tpn=1, given_dag=dag, domains=["a", "b"],
                        vocab_maps=vm, created_utc="t")


def test_npmi_ships_only_at_or_above_the_egress_floor(capsys):
    """Code review A1: min_pair_count is the pair-cell floor; below 20 the block is
    dropped with the reason printed."""
    dag, ns, n_bg, lam, vm = _small()
    rows = [{"domain": 0, "topic": 0, "npmi_gated": 0.25, "npmi_all": 0.2}]
    assert dg.npmi_exportable({"min_pair_count": 20})[0]
    assert not dg.npmi_exportable({"min_pair_count": 19})[0]
    assert not dg.npmi_exportable({})[0]
    for mpc, shipped in ((3, False), (20, True)):
        f = dg.build_export(lam, ns, n_bg=n_bg, tpn=1, given_dag=dag,
                            domains=["a", "b"], vocab_maps=vm, created_utc="t",
                            npmi={"top_n": 10, "split": "train", "min_pair_count": mpc,
                                  "rows": rows})
        assert ("npmi" in f["topics.json"]) is shipped
        assert ("npmi" in f["manifest.json"]) is shipped
    assert "NPMI dropped" in capsys.readouterr().out


# --------------------------------------------------------------------------- marks

def test_positive_profile_sets_drops_not_annotated_and_contested():
    rows = [("MONDO:1", 10, False), ("MONDO:1", 11, False), ("MONDO:1", 12, True),
            ("MONDO:1", 11, True),                       # 11 contested: listed both ways
            ("MONDO:2", 20, True),                       # only a NOT term: no profile
            ("MONDO:3", 30, False), ("MONDO:3", 30, False)]
    prof, st = dg.positive_profile_sets(rows)
    assert prof == {"MONDO:1": {10}, "MONDO:3": {30}}
    assert st == {"n_rows": 7, "n_neg_rows": 3, "n_contested": 1}


def test_marks_local_keeps_only_vocabulary_concepts():
    vm = {"1000": 0, "1001": 1, "1005": 2}
    idx, cnt = dg.marks_local({"a": {1005, 1000, 9999}, "b": {9999}, "c": {"1001"}}, vm)
    assert idx == {"a": [0, 2], "c": [1]} and cnt == {"a": 2, "c": 1}


def test_build_with_marks_ships_indices_counts_and_extra_terms():
    dag, ns, n_bg, lam, vm = _small()
    kw = dict(n_bg=n_bg, tpn=1, given_dag=dag, domains=["a", "b"], vocab_maps=vm,
              top_beta=3, top_lift=2, lift_pool=10, created_utc="t",
              term_names={1000 + i: f"term {i}" for i in range(40)})
    plain = dg.build_export(lam, ns, **kw)
    # own codes on the kept nodes (with a code outside the vocabulary), a profile on
    # every node (kept or not)
    own = {u: {1000 + 3 * i, 1039, 555555} for i, u in enumerate(sorted(ns.nodes))}
    prof = {u: {1001, 1038, 777777} for u in ["x", "y", "z", "w", "v"]}
    f = dg.build_export(lam, ns, own_codes=own, profile=prof,
                        marks_meta={"profile": {"source": "pm.tsv", "hpo_release": "v1"}},
                        **kw)
    nodes, vocab, man = f["nodes.json"], f["vocab.json"], f["manifest.json"]
    assert "marks" not in plain["manifest.json"] and "own" not in plain["nodes.json"]
    # the topics are untouched by the marks (same terms, same weights; a later domain's
    # vocab indices shift by the added terms); the vocab grows at the domain's end only
    def coded(files):
        v = files["vocab.json"]
        return [[([(v["domain"][j], v["code"][j]) for j in dg.delta_decode(ids)], w)
                 for ids, w in per] for per in files["topics.json"]["d"]]
    assert coded(f) == coded(plain)
    assert [per[0] for per in f["topics.json"]["d"]] == \
        [per[0] for per in plain["topics.json"]["d"]]          # domain 0: byte-identical
    n0 = man["domains"][0]["shipped_terms"]
    assert n0 == plain["manifest.json"]["domains"][0]["shipped_terms"]
    extra = man["domains"][0]["mark_terms"]
    assert extra > 0 and "mark_terms" not in man["domains"][1]
    assert vocab["code"][:n0] == plain["vocab.json"]["code"][:n0]
    assert vocab["domain"].count(0) == n0 + extra
    added = vocab["code"][n0:n0 + extra]
    assert all(vocab["name"][n0 + j] == f"term {c - 1000}" for j, c in enumerate(added))
    used = set()
    for per in f["topics.json"]["d"]:
        used.update(dg.delta_decode(per[0][0]))
    assert not used & set(range(n0, n0 + extra))
    # indices, counts, nulls
    code_of = vocab["code"]
    for i, u in enumerate(nodes["id"]):
        want_own = sorted(c for c in own.get(u, ()) if str(c) in vm[0])
        got = nodes["own"][i]
        assert (sorted(code_of[j] for j in got) if got else []) == want_own
        assert (got is None) == (not want_own)
        assert nodes["own_n"][i] == len(want_own)
        assert all(vocab["domain"][j] == 0 for j in got or [])
        want_prof = sorted(c for c in prof.get(u, ()) if str(c) in vm[0])
        assert nodes["profile_n"][i] == len(want_prof)
        if nodes["status"][i] == 0:
            assert sorted(code_of[j] for j in nodes["profile"][i]) == want_prof
        else:
            assert nodes["profile"][i] is None              # listed for topic nodes only
    assert any(s != 0 and n for s, n in zip(nodes["status"], nodes["profile_n"]))
    assert man["marks"] == {
        "domain": 0,
        "own": {"source": "code_map.tsv",
                "nodes": sum(1 for x in nodes["own"] if x is not None)},
        "profile": {"source": "pm.tsv", "hpo_release": "v1",
                    "nodes": sum(1 for x in nodes["profile"] if x is not None)}}
    # one block alone
    g = dg.build_export(lam, ns, profile=prof, **kw)
    assert "own" not in g["nodes.json"] and "profile" in g["nodes.json"]
    assert set(g["manifest.json"]["marks"]) == {"domain", "profile"}
