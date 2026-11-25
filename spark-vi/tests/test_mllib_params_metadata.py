"""spark-vi's own MLlib Params metadata writer/reader (ADR 0055).

Replaces pyspark's private `DefaultParamsWriter._get_metadata_to_save` /
`DefaultParamsReader._parseMetaData`. Pinned to MLlib's document two ways: against a
fixture written by MLlib's (private) writer (`data/metadata_v0`, made by
`data/make_metadata_v0.py`), which our writer reproduces and our reader reads, and,
while pyspark still has it, against that writer itself. The pin is for MLlib format
compatibility (a saved stage MLlib can read), not an old spark-vi layout: saved models
have no backward compatibility (ADR 0055)."""
import json
from pathlib import Path

import pytest
from pyspark.ml.util import DefaultParamsReader

from spark_vi.mllib._common import (params_metadata_json, parse_params_metadata,
                                    read_params_metadata)
from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator

FIXTURE = Path(__file__).parent / "data" / "metadata_v0"


def _fixture_estimator():
    est = DagGatedLDAEstimator(gateTpn=1, maxIter=7, topicConcentration=[0.1, 0.2],
                               gateWarm=4)
    est._resetUid("DagGatedLDAEstimator_fixture0001")
    return est


def test_writer_reproduces_the_private_writers_fixture_byte_for_byte():
    est = _fixture_estimator()
    text = params_metadata_json(est, "3.5.9", {"extraKey": {"a": [1, 2]}},
                                timestamp_ms=1764000000000)
    # pyspark's default seed is hash(class name), which varies per process
    # (PYTHONHASHSEED): put this process's value into the fixture, keeping its bytes
    # (key order, separators) otherwise.
    fixture = json.loads((FIXTURE / "metadata" / "part-00000").read_text())
    fixture["defaultParamMap"]["seed"] = est.getOrDefault("seed")
    assert text == json.dumps(fixture, separators=(",", ":"))


def test_reader_reads_mllibs_own_metadata():
    meta = read_params_metadata(str(FIXTURE),
                                "spark_vi.mllib.topic.dag_gated_lda.DagGatedLDAEstimator")
    assert meta["extraKey"] == {"a": [1, 2]}
    est = DagGatedLDAEstimator()
    est._resetUid(meta["uid"])
    DefaultParamsReader.getAndSetParams(est, meta)
    ref = _fixture_estimator()
    for p in ref.params:
        assert est.isSet(p) == ref.isSet(p), p.name
        if ref.isDefined(p) and p.name != "seed":     # default seed: per process
            assert est.getOrDefault(p.name) == ref.getOrDefault(p), p.name


def test_writing_leaves_the_instance_unchanged():
    est = _fixture_estimator()
    before = {p.name: (est.isSet(p), est.getOrDefault(p) if est.isDefined(p) else None)
              for p in est.params}
    meta = json.loads(params_metadata_json(est, "3.5.9"))
    after = {p.name: (est.isSet(p), est.getOrDefault(p) if est.isDefined(p) else None)
             for p in est.params}
    assert before == after
    # a set Param that also has a default appears in both maps, as MLlib writes it
    assert meta["paramMap"]["gateWarm"] == 4 and meta["defaultParamMap"]["gateWarm"] == 0


def test_matches_the_private_writer_while_pyspark_has_it():
    from pyspark.ml.util import DefaultParamsWriter
    if not hasattr(DefaultParamsWriter, "_get_metadata_to_save"):
        pytest.skip("this pyspark no longer has the private writer")

    class _SC:
        version = "3.5.9"
    est = _fixture_estimator()
    ours = json.loads(params_metadata_json(est, "3.5.9", {"x": 1}))
    theirs = json.loads(DefaultParamsWriter._get_metadata_to_save(est, _SC(), {"x": 1}))
    ours.pop("timestamp"), theirs.pop("timestamp")
    assert ours == theirs


def test_parse_refuses_another_class_and_a_malformed_line():
    line = (FIXTURE / "metadata" / "part-00000").read_text().strip()
    with pytest.raises(ValueError, match="expected class name"):
        parse_params_metadata(line, "some.Other")
    with pytest.raises(ValueError, match="'uid'"):
        parse_params_metadata(json.dumps({"class": "a", "sparkVersion": "3.5.0",
                                          "paramMap": {}}))
