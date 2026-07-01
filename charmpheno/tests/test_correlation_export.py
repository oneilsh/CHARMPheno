import numpy as np
from charmpheno.export.correlation import build_correlation_json
from spark_vi.models.topic.partition import TopicBlockPartition


def test_build_correlation_json_orders_blocks_and_nulls_unidentified():
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))  # K=4, ids 0..3
    R = np.array([[1.0, 0.3, 0.2, np.nan],
                  [0.3, 1.0, 0.1, np.nan],
                  [0.2, 0.1, 1.0, np.nan],
                  [np.nan, np.nan, np.nan, 1.0]])
    identified = np.array([[1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [1, 1, 1, 0],
                           [0, 0, 0, 1]], dtype=bool)
    support = np.array([[300, 300, 150, 0],
                        [300, 300, 150, 0],
                        [150, 150, 150, 0],
                        [0, 0, 0, 150]], dtype=float)
    kept = [0, 1, 2, 3]
    out = build_correlation_json(R, identified, support, part, kept)
    assert out["topic_order"] == [0, 1, 2, 3]
    assert out["block_labels"] == ["background", "background", "A", "B"]
    # cross-foreground (A id=2, B id=3) is unidentified -> null in R
    assert out["R"][2][3] is None and out["R"][3][2] is None
    assert out["identified"][2][3] is False
    assert out["support"][2][3] == 0
    assert out["R"][0][1] == 0.3            # identified cell preserved


def test_cross_foreground_all_null_under_split_representation():
    """Under today's split (single-group) corpus, no doc co-realizes an A and a
    B foreground topic, so the whole cross-foreground block is unidentified."""
    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A", 1), ("B", 1)))  # ids: bg=0,A=1,B=2
    R = np.array([[1.0, 0.4, 0.5], [0.4, 1.0, np.nan], [0.5, np.nan, 1.0]])
    identified = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool)
    support = np.array([[300, 200, 100], [200, 200, 0], [100, 0, 100]], dtype=float)
    out = build_correlation_json(R, identified, support, part, [0, 1, 2])
    assert out["R"][1][2] is None and out["identified"][1][2] is False


def test_build_dashboard_writes_correlation_json(tmp_path):
    """The dashboard build emits correlation.json from a saved STM model."""
    import json
    from spark_vi.models.topic._linalg import topic_correlation_identified
    # Minimal stand-in for the wiring: Sigma + n_pairs + partition -> json file.
    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A", 1), ("B", 1)))
    Sigma = np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 0.0], [1.0, 0.0, 4.0]])
    N = np.array([[300, 200, 0], [200, 200, 0], [0, 0, 100]], dtype=float)
    R, ident = topic_correlation_identified(Sigma, N, min_pair_support=10)
    out = build_correlation_json(R, ident, N, part, [0, 1, 2])
    (tmp_path / "correlation.json").write_text(json.dumps(out))
    loaded = json.loads((tmp_path / "correlation.json").read_text())
    assert loaded["R"][1][2] is None                # cross-foreground NA
    assert loaded["block_labels"] == ["background", "A", "B"]
