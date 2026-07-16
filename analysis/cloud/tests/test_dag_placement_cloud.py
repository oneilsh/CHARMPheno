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
        "--cdr", "p.d", "--billing", "bp", "--disease", "rare6",
        "--min-n", "50", "--n-bg", "2", "--tpn", "1", "--person-mod", "10",
        "--vocab-size", "5000", "--init", "spectral", "--out-dir", "/tmp/x",
        "--strip-mode", "both",
    ])
    assert a.disease == "rare6" and a.min_n == 50 and a.n_bg == 2 and a.tpn == 1
    assert a.init == "spectral" and a.out_dir == "/tmp/x"
    assert a.strip_mode == "both"
    # K is emergent: there must be NO --K arg.
    assert not hasattr(a, "K")


def test_main_importable():
    import dag_placement_cloud
    assert hasattr(dag_placement_cloud, "main")


def test_topic_evolution_logger_prints_named_terms_and_blocks(capsys):
    """The per-iter logger prints one line per topic with its DAG-node block label
    and top vocab TERMS by name (STM parity), every_n-throttled."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from dag_placement_cloud import _make_topic_evolution_logger, _topic_node_labels
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)          # K=4: 2 bg + node1 + node2
    int2cid = {0: 100, 1: 200, 2: 300}                     # engine-id -> concept-id
    name_by_id = {100: "dm", 200: "Type 2", 300: "Type 1"}
    labels = _topic_node_labels(lay, int2cid, name_by_id, n_bg=2)
    assert labels[:2] == ["bg", "bg"] and set(labels[2:]) == {"Type 2", "Type 1"}
    idx_to_cid = {0: 5001, 1: 5002, 2: 5003}
    vocab_names = {5001: "Hyperglycemia", 5002: "Neuropathy", 5003: "Retinopathy"}
    lam = np.array([[9.0, 1.0, 0.5], [0.5, 8.0, 1.0],
                    [1.0, 1.0, 7.0], [3.0, 3.0, 3.0]])
    logger = _make_topic_evolution_logger(
        top_n=2, every_n=5, idx_to_cid=idx_to_cid, vocab_names=vocab_names,
        topic_labels=labels)
    logger(3, {"lambda": lam}, [])                         # not a multiple of 5 -> silent
    assert capsys.readouterr().out == ""
    logger(5, {"lambda": lam}, [])                         # fires
    out = capsys.readouterr().out
    assert "topics @ iter 5" in out
    assert "Hyperglycemia" in out and "Neuropathy" in out  # vocab names, not ids
    assert "Type 2" in out and "bg" in out                 # block labels


def test_parse_args_topic_logging_flags():
    from dag_placement_cloud import parse_args
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x",
                    "--print-topics-every", "10", "--top-n-tokens", "6"])
    assert a.print_topics_every == 10 and a.top_n_tokens == 6
    # default off
    b = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x"])
    assert b.print_topics_every == 0


def test_log_corpus_stats_counts(spark):
    """Corpus stats: per-split doc counts, per-source_cohort breakdown, how many
    docs carry a frontier, and the vocab/topic dims."""
    from types import SimpleNamespace
    from spark_vi.models.topic.dag_placement import DagLayout
    from dag_placement_cloud import _log_corpus_stats
    train = spark.createDataFrame(
        [(1, "diabetes", [3]), (2, "diabetes", [2]), (3, "general", [])],
        ["person_id", "source_cohort", "frontier"])
    test = spark.createDataFrame(
        [(4, "diabetes", [3]), (5, "general", [])],
        ["person_id", "source_cohort", "frontier"])
    bundle = SimpleNamespace(train_df=train, test_df=test,
                             vocab_map={100: 0, 200: 1, 300: 2})
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=20, tpn=5)
    stats = _log_corpus_stats(bundle, lay)
    assert stats["train"]["n_docs"] == 3 and stats["train"]["n_frontier"] == 2
    assert stats["train"]["by_source_cohort"] == {"diabetes": 2, "general": 1}
    assert stats["test"]["n_docs"] == 2 and stats["test"]["n_frontier"] == 1
    assert stats["vocab_size"] == 3 and stats["n_bg"] == 20 and stats["tpn"] == 5
    assert stats["K"] == lay.K == 20 + 3 * 5
