"""Regenerates `metadata_v0/`: Params metadata exactly as spark-vi wrote it through
pyspark's private `DefaultParamsWriter._get_metadata_to_save` (pyspark 3.5), before
spark-vi's own writer replaced it (ADR 0055). Kept as a read-compatibility fixture: the
reader must keep loading what that writer wrote. Run once, with pyspark 3.5, from the
repo root: `python spark-vi/tests/data/make_metadata_v0.py`. The timestamp is pinned."""
import json
import os

from pyspark.ml.util import DefaultParamsWriter

from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator


class _SC:
    version = "3.5.9"


def main():
    est = DagGatedLDAEstimator(gateTpn=1, maxIter=7, topicConcentration=[0.1, 0.2],
                               gateWarm=4)
    est._resetUid("DagGatedLDAEstimator_fixture0001")
    txt = DefaultParamsWriter._get_metadata_to_save(est, _SC(), {"extraKey": {"a": [1, 2]}})
    d = json.loads(txt)
    d["timestamp"] = 1764000000000
    out = os.path.join(os.path.dirname(__file__), "metadata_v0", "metadata")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "part-00000"), "w") as f:
        f.write(json.dumps(d, separators=(",", ":")) + "\n")
    open(os.path.join(out, "_SUCCESS"), "w").close()


if __name__ == "__main__":
    main()
