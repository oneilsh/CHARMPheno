import numpy as np
from spark_vi.models.topic.partition import TopicBlockPartition
from charmpheno.export.gating import suppressed_topic_ids, build_gating_json


def test_suppressed_topic_ids_drops_sub_k_group():
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))  # K=6
    # rare_dx has 50 patients (>=20), ultrarare has 8 (<20) -> suppress topic id 5
    counts = {"rare_dx": 50, "ultrarare": 8}
    assert suppressed_topic_ids(part, counts, k=20) == {5}


def test_build_gating_json_keeps_only_above_k_groups():
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))
    counts = {"rare_dx": 50, "ultrarare": 8}
    supp = suppressed_topic_ids(part, counts, k=20)              # {5}
    kept_ids = [i for i in range(part.K) if i not in supp]       # [0,1,2,3,4]
    gj = build_gating_json(part, counts, k=20, kept_topic_ids=kept_ids)
    assert gj["group_var"] == "source_cohort"
    assert gj["groups"] == ["rare_dx"]                          # ultrarare suppressed
    assert gj["topic_blocks"] == ["background", "background", "background",
                                  "rare_dx", "rare_dx"]


def test_adapt_stm_excludes_suppressed_topics():
    from charmpheno.export.model_adapter import adapt_stm
    # build a minimal STM-like result object
    class R:
        global_params = {"lambda": np.abs(np.random.default_rng(0).normal(
            size=(6, 8))) + 0.1, "Gamma": np.zeros((2, 6))}
        metadata = {"covariate_manifest": {"covariate_names": ["Intercept", "age"]}}
    part = TopicBlockPartition("source_cohort", background_k=3,
                               foreground=(("rare_dx", 2), ("ultrarare", 1)))
    exp = adapt_stm(R(), partition=part, suppressed=frozenset({5}))
    assert exp.beta.shape[0] == 5                                # 6 - 1 suppressed
    assert list(exp.topic_indices) == [0, 1, 2, 3, 4]
    assert exp.topic_blocks == ["background", "background", "background",
                                "rare_dx", "rare_dx"]


def test_adapt_stm_subsets_theta_arrays_to_kept():
    """theta_histogram and theta_percentiles must be subset by `kept` so their
    axis-0 length equals len(topic_indices), not the full K.

    Regression: before the fix, both arrays were passed through at length K=6
    while topic_indices had length 5 (one topic suppressed) — a silent
    misalignment.
    """
    from charmpheno.export.model_adapter import adapt_stm

    rng = np.random.default_rng(42)
    K, n_bins = 6, 3

    # theta_histogram: K × n_bins list-of-lists (as stored in metadata)
    hist_raw = rng.random((K, n_bins)).tolist()
    # theta_percentiles: K × 5 list-of-dicts (as _parse_theta_percentiles expects)
    cols = ["p5", "p25", "p50", "p75", "p95"]
    pct_raw = [{c: float(rng.random()) for c in cols} for _ in range(K)]

    class R:
        global_params = {
            "lambda": np.abs(rng.normal(size=(K, 8))) + 0.1,
            "Gamma": np.zeros((2, K)),
        }
        metadata = {
            "covariate_manifest": {"covariate_names": ["Intercept", "age"]},
            "theta_histogram": hist_raw,
            "theta_percentiles": pct_raw,
        }

    part = TopicBlockPartition(
        "source_cohort",
        background_k=3,
        foreground=(("rare_dx", 2), ("ultrarare", 1)),  # K=6; ultrarare → topic id 5
    )
    exp = adapt_stm(R(), partition=part, suppressed=frozenset({5}))

    n_kept = len(exp.topic_indices)  # should be 5
    assert n_kept == 5, f"expected 5 kept topics, got {n_kept}"
    assert exp.theta_histogram is not None
    assert exp.theta_histogram.shape[0] == n_kept, (
        f"theta_histogram axis-0 length {exp.theta_histogram.shape[0]} != {n_kept}"
    )
    assert exp.theta_percentiles is not None
    assert exp.theta_percentiles.shape[0] == n_kept, (
        f"theta_percentiles axis-0 length {exp.theta_percentiles.shape[0]} != {n_kept}"
    )


def test_build_gating_json_emits_labels_and_proportions():
    """Gated bundle carries humanized group_var_label + per-group labels, and a
    k-anon-safe group_proportions over kept groups (summing to 1) for the
    dashboard's per-patient group draw."""
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "source_cohort"
        groups = ["cancer", "dementia"]
        def topic_labels(self):
            return ["background"] * 30 + ["cancer"] * 10 + ["dementia"] * 10
        def block_indices(self, g):
            return range(30, 40) if g == "cancer" else range(40, 50)
    counts = {"cancer": 9000, "dementia": 2000}
    kept = list(range(50))
    out = build_gating_json(_P(), counts, k=20, kept_topic_ids=kept)
    assert out["group_var_label"] == "Source cohort"     # humanized
    assert out["group_labels"] == {"cancer": "Cancer", "dementia": "Dementia"}
    props = out["group_proportions"]
    assert abs(sum(props.values()) - 1.0) < 1e-9
    assert abs(props["cancer"] - 9000 / 11000) < 1e-9


def test_build_gating_json_label_override():
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "rare_dx"
        groups = ["rare_dx"]
        def topic_labels(self):
            return ["background"] * 2 + ["rare_dx"]
        def block_indices(self, g):
            return [2]
    out = build_gating_json(
        _P(), {"rare_dx": 100}, k=20, kept_topic_ids=[0, 1, 2],
        group_label_overrides={"rare_dx": "Rare diabetes cohort"},
    )
    assert out["group_labels"]["rare_dx"] == "Rare diabetes cohort"


def test_build_gating_json_background_only_proportion():
    """A background-only group (present in group_counts but NOT a foreground
    group, e.g. a 'general' population) is excluded from the foreground
    group_proportions, but its cohort share surfaces as
    background_only_proportion. Foreground shares are over the WHOLE cohort, so
    the sampler can draw a representative mix rather than collapsing to the
    (minority) foreground group."""
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "source_cohort"
        groups = ["cancer"]                     # only cancer is a foreground group
        def topic_labels(self):
            return ["background"] * 40 + ["cancer"] * 20
        def block_indices(self, g):
            return range(40, 60)
    # 9000 cancer + 51000 background-only 'general' = 60000 total cohort.
    counts = {"cancer": 9000, "general": 51000}
    out = build_gating_json(_P(), counts, k=20, kept_topic_ids=list(range(60)))

    assert out["groups"] == ["cancer"]          # 'general' is not a foreground group
    assert abs(out["group_proportions"]["cancer"] - 9000 / 60000) < 1e-9
    assert abs(out["background_only_proportion"] - 51000 / 60000) < 1e-9
    # foreground shares + background-only == whole cohort
    assert abs(sum(out["group_proportions"].values())
               + out["background_only_proportion"] - 1.0) < 1e-9


def test_build_gating_json_no_background_only_when_all_foreground():
    """When every counted group is a kept foreground group,
    background_only_proportion is 0 and group_proportions sum to 1 (the
    prior behavior is preserved for all-foreground cohorts)."""
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "source_cohort"
        groups = ["cancer", "dementia"]
        def topic_labels(self):
            return ["background"] * 30 + ["cancer"] * 10 + ["dementia"] * 10
        def block_indices(self, g):
            return range(30, 40) if g == "cancer" else range(40, 50)
    out = build_gating_json(_P(), {"cancer": 9000, "dementia": 2000},
                            k=20, kept_topic_ids=list(range(50)))
    assert out["background_only_proportion"] == 0.0
    assert abs(sum(out["group_proportions"].values()) - 1.0) < 1e-9
