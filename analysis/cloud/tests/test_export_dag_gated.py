"""export_dag_gated.py end to end on a synthetic run dir saved as the fit driver saves
one (spec 2026-11-11, `dag-gated-export/1`): the four files' schema, the selection, the
codecs, the support banding, index consistency, no per-document data, and the
committed browser fixture (`dag-dashboard/fixture/`) staying current. No Spark."""
import json
from pathlib import Path

import numpy as np
import pytest

import dag_export_synth as synth

REPO = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO / "dag-dashboard" / "fixture"

MANIFEST_KEYS = {"schema", "run", "created_utc", "mondo_version", "cdr", "K", "n_bg",
                 "tpn", "domains", "selection", "weight_codec", "freq_codec",
                 "disclosure", "sizes", "notes", "min_docs_per_node"}
NODES_KEYS = {"id", "name", "parents", "kept_parents", "depth", "status", "landing",
              "support", "support_frac", "topic"}
VOCAB_KEYS = {"domain", "code", "name", "freq"}
TOPICS_KEYS = {"node", "d"}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("dagexp")
    run = synth.fixture_run(root)
    out = root / "export"
    files, sizes = synth.export_run(run, out, created_utc=synth.FIXTURE_CREATED_UTC)
    from spark_vi.dag.nodeset import NodeSet
    from spark_vi.io.export import load_result
    res = load_result(str(run / "model" / "result"))
    ns = NodeSet.from_dict(res.metadata["dag_node_set"])
    from charmpheno.export.dag_gated import read_export
    return {"run": run, "out": out, "files": read_export(out), "sizes": sizes,
            "lam": res.global_params["lambda"], "ns": ns,
            "n_bg": int(res.metadata["dag_node_set"]["n_bg"])}


def test_schema_of_the_four_files(exported):
    f = exported["files"]
    man, nodes, vocab, topics = (f["manifest.json"], f["nodes.json"], f["vocab.json"],
                                 f["topics.json"])
    # npmi is optional (the synthetic run carries a topic-NPMI result, so it is there)
    assert set(man) == MANIFEST_KEYS | {"npmi"} and man["schema"] == "dag-gated-export/1"
    assert set(nodes) == NODES_KEYS and set(vocab) == VOCAB_KEYS
    assert set(topics) == TOPICS_KEYS | {"npmi"}
    nb = topics["npmi"]
    assert nb["domains"] == man["npmi"]["domains"] == [0, 1]
    assert all(len(a) == man["K"] for a in nb["gated"] + nb["all"])
    assert nb["gated"][0][0] is None                     # an unscored topic stays null
    assert all(v is None or isinstance(v, int) for a in nb["gated"] for v in a)
    assert man["selection"] == {"top_beta": 25, "top_lift": 25, "lift_pool": 200}
    assert man["weight_codec"]["kind"] == man["freq_codec"]["kind"] == "log-byte"
    assert [d["name"] for d in man["domains"]] == list(synth.DOMAINS)
    assert man["disclosure"]["min_cell"] == 20
    n = len(nodes["id"])
    assert all(len(v) == n for v in nodes.values())
    assert all(len(v) == len(vocab["code"]) for v in vocab.values())
    assert len(topics["node"]) == len(topics["d"]) == man["K"]
    assert all(len(per) == len(man["domains"]) for per in topics["d"])
    for m, d in enumerate(man["domains"]):
        assert d["shipped_terms"] == sum(1 for x in vocab["domain"] if x == m)
    # minified with separators=(",", ":")
    for name in ("nodes.json", "vocab.json", "topics.json", "manifest.json"):
        raw = (exported["out"] / name).read_text(encoding="utf-8")
        assert raw == json.dumps(json.loads(raw), separators=(",", ":"),
                                 ensure_ascii=False)


def test_sizes_match_the_files_on_disk(exported):
    import gzip
    sizes = exported["files"]["manifest.json"]["sizes"]
    for name, (raw, gz) in sizes.items():
        b = (exported["out"] / name).read_bytes()
        assert raw == len(b)
        assert gz == len(gzip.compress(b, compresslevel=9, mtime=0))


def test_statuses_and_index_consistency(exported):
    f, ns = exported["files"], exported["ns"]
    nodes, topics = f["nodes.json"], f["topics.json"]
    n, K, n_bg = len(nodes["id"]), f["manifest.json"]["K"], f["manifest.json"]["n_bg"]
    assert set(nodes["status"]) == {0, 1, 2, 3}
    assert any(len(p) > 1 for p in nodes["parents"])                   # multi-parent
    pos = {i: k for k, i in enumerate(nodes["id"])}
    for i in range(n):
        st = nodes["status"][i]
        for lst in (nodes["parents"][i], nodes["kept_parents"][i] or [],
                    nodes["landing"][i] or []):
            assert all(0 <= j < n and j != i for j in lst)
        assert all(nodes["depth"][j] < nodes["depth"][i] for j in nodes["parents"][i])
        if st == 0:
            t = nodes["topic"][i]
            assert nodes["landing"][i] is None and nodes["kept_parents"][i] is not None
            assert n_bg <= t < K and topics["node"][t] == i
            assert all(nodes["status"][j] == 0 for j in nodes["kept_parents"][i])
            assert 0 <= nodes["support_frac"][i] <= 1000
        else:
            assert nodes["topic"][i] is None and nodes["kept_parents"][i] is None
            assert nodes["support_frac"][i] is None
            assert all(nodes["status"][j] == 0 for j in nodes["landing"][i])
    for t, i in enumerate(topics["node"]):
        if t < n_bg:
            assert i is None
        else:
            assert nodes["topic"][i] == t
    # statuses and landings agree with the node set
    st = dict(zip(nodes["id"], nodes["status"]))
    assert {u for u, s in st.items() if s == 0} == set(ns.nodes)
    assert {u for u, s in st.items() if s == 1} == set(ns.pruned)
    assert {u for u, s in st.items() if s == 2} == set(ns.spliced)
    assert {u for u, s in st.items() if s == 3} == set(ns.unsupported)
    for u, land in ns.pruned.items():
        assert nodes["landing"][pos[u]] == sorted(pos[x] for x in land)
    for u in ns.spliced:
        want = sorted(pos[ns.user_ids[e]] for e in ns.frontier.landing[u])
        assert nodes["landing"][pos[u]] == want
    # kept_parents is the kept DAG; a top-level kept node has []
    for u, ps in ns.parents.items():
        got = sorted(nodes["id"][j] for j in nodes["kept_parents"][pos[u]])
        assert got == sorted(p for p in ps if isinstance(p, str))


def test_support_is_banded_and_unknown_only_where_not_recorded(exported):
    nodes = exported["files"]["nodes.json"]
    for s, st in zip(nodes["support"], nodes["status"]):
        if st == 1:
            assert s is None                                          # pruned
        else:
            assert s is not None and (s == 0 or s % 20 == 10)
    ns = exported["ns"]
    from charmpheno.export.dag_gated import band_support
    for u, sup in ns.support.items():
        assert nodes["support"][nodes["id"].index(u)] == band_support(sup)


def test_selection_and_codec_round_trip(exported):
    from charmpheno.export import dag_gated as dg
    f, lam = exported["files"], exported["lam"]
    man, vocab = f["manifest.json"], f["vocab.json"]
    vm = json.loads((exported["run"] / "bundle_meta.json").read_text())["vocab_maps"]
    gid = {(d, c): i for i, (d, c) in enumerate(zip(vocab["domain"], vocab["code"]))}
    dec = dg.decode_topics(f)
    for m in range(len(man["domains"])):
        beta, pw = dg.domain_beta(lam[m]), dg.term_marginal(lam[m])
        picks = dg.select_terms(beta, pw, **man["selection"])
        inv = {i: int(c) for c, i in vm[m].items()}
        err = dg.log_byte_max_log_error(man["weight_codec"]["per_domain"][m]["lo"],
                                        man["weight_codec"]["per_domain"][m]["hi"])
        for k in range(man["K"]):
            ids, w = dec[k][m]
            assert ids == sorted(ids) and len(set(ids)) == len(ids)
            assert ids == sorted(gid[(m, inv[li])] for li in picks[k])   # exact
            assert all(vocab["domain"][i] == m for i in ids)
            local = {gid[(m, inv[li])]: li for li in picks[k]}
            true = np.array([beta[k, local[i]] for i in ids])
            assert np.abs(np.log(w) - np.log(true)).max() <= err + 1e-12
        # the vocab's p(w)
        fr = dg.decode_freq(f)
        sel = [i for i, d in enumerate(vocab["domain"]) if d == m]
        true = np.array([pw[vm[m][str(vocab["code"][i])]] for i in sel])
        ferr = dg.log_byte_max_log_error(man["freq_codec"]["per_domain"][m]["lo"],
                                         man["freq_codec"]["per_domain"][m]["hi"])
        assert np.abs(np.log(fr[sel]) - np.log(true)).max() <= ferr + 1e-12


def test_vocab_ships_only_selected_terms_with_names(exported):
    f = exported["files"]
    vocab = f["vocab.json"]
    used = set()
    for per in f["topics.json"]["d"]:
        for deltas, _ in per:
            used.update(np.cumsum(deltas).tolist())
    assert used == set(range(len(vocab["code"])))
    assert all(isinstance(c, int) for c in vocab["code"])
    assert all(vocab["name"])
    names = f["nodes.json"]["name"]
    assert "atrial fibrillation" in names and "MONDO:" not in " ".join(names)


def test_no_per_document_data(exported):
    f = exported["files"]
    blob = json.dumps(f)
    for word in ('"n_docs"', "person", "doc_key", "patient", "train_fg_docs"):
        assert word not in blob
    # every number per node is banded, a topic reference, a depth, an index or the
    # sharpness: nothing else rides nodes.json
    assert set(f["nodes.json"]) == NODES_KEYS


def test_without_node_statistics_unsupported_support_is_null(tmp_path):
    run = synth.fixture_run(tmp_path)
    (run / "node_statistics.tsv").unlink()
    files, _ = synth.export_run(run, tmp_path / "out")
    nodes = files["nodes.json"]
    for s, st in zip(nodes["support"], nodes["status"]):
        if st in (1, 3):
            assert s is None


def test_names_fall_back_to_the_bundle_meta_without_the_mondo_file(tmp_path):
    import export_dag_gated as drv
    run = synth.fixture_run(tmp_path)
    for p in (tmp_path / "mondo").iterdir():
        p.unlink()
    files, _ = drv.main(["--run-dir", str(run), "--out", str(tmp_path / "o"),
                         "--no-fetch-mondo"])
    nodes = files["nodes.json"]
    for u, nm, st in zip(nodes["id"], nodes["name"], nodes["status"]):
        assert (nm != u) if st == 0 else (nm == u)
    # no names CSV given: the run's concept_names.csv is picked up
    assert all(files["vocab.json"]["name"])


def test_committed_fixture_is_current(tmp_path):
    """dag-dashboard/fixture/ is the generator's output, byte for byte (regenerate:
    `python analysis/cloud/tests/dag_export_synth.py --fixture dag-dashboard/fixture`)."""
    synth.main(["--fixture", str(tmp_path / "fx")])
    for name in ("manifest.json", "nodes.json", "vocab.json", "topics.json"):
        assert (tmp_path / "fx" / name).read_bytes() == \
            (FIXTURE_DIR / name).read_bytes(), name
    total = sum((FIXTURE_DIR / n).stat().st_size for n in
                ("manifest.json", "nodes.json", "vocab.json", "topics.json"))
    assert total < 200_000


@pytest.mark.slow
def test_scale_export_fits_the_size_target(tmp_path):
    """The real shape (2,895 topics; 5000 / 5000 / 1601 terms): about 2 MB gzipped."""
    run = synth.scale_run(tmp_path)
    files, sizes = synth.export_run(run, tmp_path / "out")
    assert files["manifest.json"]["K"] == 2895
    assert sum(g for _, g in sizes.values()) < 3_000_000
