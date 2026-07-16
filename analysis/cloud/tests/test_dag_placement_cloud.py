"""Tests for the dag_placement cloud driver (piece 3): the pure scoring adapter
and the arg surface. The end-to-end BQ+fit run is the cluster smoke."""


def test_profiles_from_scored_rows_maps_affinity_and_frontier():
    from pyspark.ml.linalg import DenseVector
    from spark_vi.models.topic.dag_placement import DagLayout
    from dag_placement_cloud import profiles_from_scored_rows
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)   # nodes = [1,2,3]
    # a "row" needs __getitem__ by name; use dicts (the driver indexes by name).
    rows = [
        {"nodeAffinity": DenseVector([0.5, 0.3, 0.2]), "frontier": [3]},
        {"nodeAffinity": DenseVector([0.1, 0.8, 0.1]), "frontier": [2, 1]},
    ]
    profiles, labels = profiles_from_scored_rows(rows, lay)
    assert profiles[0] == {1: 0.5, 2: 0.3, 3: 0.2}
    assert labels[0] == {3}
    assert labels[1] == {1, 2}
    # profiles feed evaluate cleanly
    from spark_vi.models.topic.dag_placement import evaluate
    ev = evaluate(profiles, labels, lay)
    assert "auc_by_depth" in ev and "mrr" in ev


def test_parse_args_surface():
    from dag_placement_cloud import parse_args
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--anchor", "201820",
        "--min-n", "50", "--n-bg", "2", "--tpn", "1", "--person-mod", "10",
        "--vocab-size", "5000", "--init", "spectral", "--out-dir", "/tmp/x",
        "--strip-mode", "both",
    ])
    assert a.anchor == 201820 and a.min_n == 50 and a.n_bg == 2 and a.tpn == 1
    assert a.init == "spectral" and a.out_dir == "/tmp/x"
    assert a.strip_mode == "both"
    # K is emergent: there must be NO --K arg.
    assert not hasattr(a, "K")


def test_main_importable():
    import dag_placement_cloud
    assert hasattr(dag_placement_cloud, "main")
