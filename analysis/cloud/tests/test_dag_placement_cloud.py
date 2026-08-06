"""Tests for the dag_placement cloud driver (piece 3): the pure scoring adapter
and the arg surface. The end-to-end BQ+fit run is the cluster smoke."""


def test_profiles_from_scored_rows_maps_affinity_and_frontier():
    from pyspark.ml.linalg import DenseVector, Vectors
    from spark_vi.models.topic.dag_placement import DagLayout
    from dag_placement_cloud import profiles_from_scored_rows
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)   # nodes = [1,2,3]
    # a "row" needs __getitem__ by name; use dicts (the driver indexes by name).
    rows = [
        {"nodeAffinity": DenseVector([0.5, 0.3, 0.2]), "frontier": [3],
         "features": Vectors.sparse(5, {0: 1.0, 1: 2.0})},
        {"nodeAffinity": DenseVector([0.1, 0.8, 0.1]), "frontier": [2, 1],
         "features": Vectors.sparse(5, {2: 4.0})},
    ]
    profiles, labels, lengths = profiles_from_scored_rows(rows, lay)
    assert profiles[0] == {1: 0.5, 2: 0.3, 3: 0.2}
    assert labels[0] == {3}
    assert labels[1] == {1, 2}
    assert lengths == [3.0, 4.0]
    # profiles feed evaluate cleanly
    from spark_vi.models.topic.dag_placement import evaluate
    ev = evaluate(profiles, labels, lay)
    assert "auc_by_depth" in ev and "mrr" in ev


def test_profiles_from_scored_rows_returns_token_lengths():
    from pyspark.ml.linalg import Vectors
    from dag_placement_cloud import profiles_from_scored_rows
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: [0], 2: [0]}, n_bg=1, tpn=1)
    rows = [{"nodeAffinity": Vectors.dense([0.3, 0.1]), "frontier": [1],
             "features": Vectors.sparse(5, {0: 2.0, 3: 1.0})}]
    profiles, labels, lengths = profiles_from_scored_rows(rows, lay)
    assert lengths == [3.0]                        # 2 + 1


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
    assert a.node_alpha_scale == 1.0            # symmetric default
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


# --- Task 0: covariate sidecar routing into the case-finding corpus ----------

def test_covariates_from_scored_rows_absent_is_none_and_present_aligns():
    from pyspark.ml.linalg import Vectors
    from dag_placement_cloud import covariates_from_scored_rows
    # no covariates column -> None (baseline path)
    rows_no = [{"nodeAffinity": Vectors.dense([0.1, 0.2]), "frontier": [1],
                "features": Vectors.sparse(2, {0: 1.0})}]
    assert covariates_from_scored_rows(rows_no) is None
    # present -> aligned (D, P) matrix in row order
    rows_yes = [{"covariates": Vectors.dense([1.0, 2.0])},
                {"covariates": Vectors.dense([3.0, 4.0])}]
    X = covariates_from_scored_rows(rows_yes)
    assert X.shape == (2, 2)
    assert list(X[0]) == [1.0, 2.0] and list(X[1]) == [3.0, 4.0]


def test_covariates_from_scored_rows_zero_fills_null_rows():
    """A doc with no covariate match (null after a left join) becomes a zero
    vector, so the 2x2 keeps an IDENTICAL doc set across cells (never drops)."""
    from pyspark.ml.linalg import Vectors
    from dag_placement_cloud import covariates_from_scored_rows
    rows = [{"covariates": Vectors.dense([0.5, 1.0])},
            {"covariates": None},                         # unmatched doc
            {"covariates": Vectors.dense([2.0, 3.0])}]
    X = covariates_from_scored_rows(rows)
    assert X.shape == (3, 2)
    assert list(X[1]) == [0.0, 0.0]


def test_join_covariates_left_join_preserves_all_docs(spark):
    from pyspark.ml.linalg import Vectors
    from dag_placement_cloud import join_covariates, covariates_from_scored_rows
    scored = spark.createDataFrame(
        [(1, [3]), (2, [2]), (3, [])], ["person_id", "frontier"])
    cov = spark.createDataFrame(
        [(1, Vectors.dense([0.5, 1.0])), (2, Vectors.dense([-0.5, 2.0]))],
        ["person_id", "covariates"])
    joined = join_covariates(scored, cov, key="person_id")
    rows = joined.orderBy("person_id").collect()
    assert len(rows) == 3                                  # doc 3 kept (no cov match)
    X = covariates_from_scored_rows(rows)
    assert X.shape == (3, 2)
    assert list(X[0]) == [0.5, 1.0] and list(X[1]) == [-0.5, 2.0]
    assert list(X[2]) == [0.0, 0.0]                        # zero-filled


def test_parse_args_covariate_surface():
    from dag_placement_cloud import parse_args
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x",
                    "--covariate-formula", "age + sex",
                    "--covariate-continuous", "age",
                    "--covariate-categorical", "sex",
                    "--pred-cov", "on"])
    assert a.covariate_formula == "age + sex"
    assert a.covariate_continuous == ["age"] and a.covariate_categorical == ["sex"]
    assert a.pred_cov == "on"
    # defaults: no covariates -> baseline path
    b = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x"])
    assert b.covariate_formula is None and b.pred_cov == "off"
