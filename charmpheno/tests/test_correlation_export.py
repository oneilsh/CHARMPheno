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


def test_build_correlation_json_excludes_reference_topic():
    """When reference_id is given, that topic's row/col is dropped entirely
    from topic_order/R/identified/support (Component 3: the reference topic
    is inert in Sigma but n_pairs marks it identified, so without exclusion it
    renders a spurious zero-correlation band)."""
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))  # K=4, ids 0..3
    R = np.array([[1.0, 0.3, 0.2, 0.0],
                  [0.3, 1.0, 0.1, 0.0],
                  [0.2, 0.1, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    identified = np.ones((4, 4), dtype=bool)
    support = np.full((4, 4), 300.0)
    kept = [0, 1, 2, 3]
    out = build_correlation_json(R, identified, support, part, kept, reference_id=0)
    assert 0 not in out["topic_order"]
    assert out["topic_order"] == [1, 2, 3]
    assert len(out["R"]) == 3 and all(len(row) == 3 for row in out["R"])
    assert len(out["identified"]) == 3 and all(len(row) == 3 for row in out["identified"])
    assert len(out["support"]) == 3 and all(len(row) == 3 for row in out["support"])
    assert out["block_labels"] == ["background", "A", "B"]


def test_build_correlation_json_reference_id_none_is_unchanged():
    """Default reference_id=None preserves prior behavior (no filtering)."""
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))
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
    out_default = build_correlation_json(R, identified, support, part, kept)
    out_explicit_none = build_correlation_json(R, identified, support, part, kept,
                                                reference_id=None)
    assert out_default == out_explicit_none
    assert out_default["topic_order"] == [0, 1, 2, 3]


def test_driver_mps_lookup_uses_nested_stm_hardening_floor():
    """The drivers read result.metadata.get("min_pair_support", 1), but
    min_pair_support is persisted nested under metadata["stm_hardening"]
    (spark-vi mllib/topic/stm.py ~lines 372-377), so a naive top-level .get
    always misses and silently floors at 1 even when the fit used 10. This
    test exercises the robust lookup both drivers must use:
        mps = metadata.get("min_pair_support") or \
              metadata.get("stm_hardening", {}).get("min_pair_support", 1)
    against a metadata dict shaped like the real persisted one, and confirms
    it drives the correct identified mask (a cell with N=5, between 1 and the
    fit floor of 10, must be unidentified/null)."""
    from spark_vi.models.topic._linalg import topic_correlation_identified

    metadata = {
        "K": 3, "V": 100, "P": 2,
        "stm_hardening": {
            "reference_topic": True,
            "min_pair_support": 10,
            "spectral_init": True,
            "spectral_method": "dense",
        },
    }
    # The naive lookup (pre-fix driver behavior) silently misses the nested value.
    naive_mps = int(metadata.get("min_pair_support", 1))
    assert naive_mps == 1

    # The robust lookup both drivers must use post-fix.
    mps = metadata.get("min_pair_support") or \
        metadata.get("stm_hardening", {}).get("min_pair_support", 1)
    mps = int(mps)
    assert mps == 10

    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A", 1), ("B", 1)))
    Sigma = np.eye(3) * 4.0
    Sigma[0, 1] = Sigma[1, 0] = 1.0
    N = np.array([[300, 5, 0], [5, 300, 0], [0, 0, 300]], dtype=float)

    # Under the naive floor of 1, N=5 >= 1 -> spuriously identified.
    R_naive, ident_naive = topic_correlation_identified(Sigma, N, min_pair_support=naive_mps)
    out_naive = build_correlation_json(R_naive, ident_naive, N, part, [0, 1, 2])
    assert out_naive["identified"][0][1] is True
    assert out_naive["R"][0][1] is not None

    # Under the correct fit floor of 10, N=5 < 10 -> unidentified/null.
    R_fixed, ident_fixed = topic_correlation_identified(Sigma, N, min_pair_support=mps)
    out_fixed = build_correlation_json(R_fixed, ident_fixed, N, part, [0, 1, 2])
    assert out_fixed["identified"][0][1] is False
    assert out_fixed["R"][0][1] is None
