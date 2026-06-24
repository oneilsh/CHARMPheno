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
