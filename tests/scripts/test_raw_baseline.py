"""Exp 0147 raw-code baseline: token selection and the binary presence column
(gated_pc_readout.select_raw_codes / attach_raw_presence)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis" / "cloud"),
                str(REPO_ROOT / "charmpheno"), str(REPO_ROOT / "spark-vi"),
                os.environ.get("PYTHONPATH", "")) if p)

import gated_pc_readout as gpr  # noqa: E402


def test_presence_kernel_marks_selected_tokens_per_domain():
    from pyspark.ml.linalg import Vectors
    k = gpr.raw_presence_kernel(lambda: {(0, 3): 0, (1, 3): 1, (1, 7): 2}, 3)
    out = k(Vectors.sparse(10, [3, 4], [2.0, 1.0]), Vectors.sparse(10, [7], [1.0]))
    assert out.toArray().tolist() == [1.0, 0.0, 1.0]       # (1, 3) absent; counts -> 1


def test_select_and_attach_on_spark(spark):
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    rows = [Row(person_id=i,
                features_0=Vectors.sparse(6, [0] + ([1] if i < 3 else []), [1.0] * (1 + (i < 3))),
                features_1=Vectors.sparse(4, [2] if i < 5 else [], [1.0] if i < 5 else []))
            for i in range(8)]
    df = spark.createDataFrame(rows)
    cols = gpr.raw_feature_cols(df)
    assert cols == ["features_0", "features_1"]
    sel = gpr.select_raw_codes(df, 2, cols)
    assert sel == [(0, 0), (1, 2)]                          # df 8 and 5 beat 3
    got = {r.person_id: r.topicDistribution.toArray().tolist()
           for r in gpr.attach_raw_presence(df, sel, cols).collect()}
    assert got[0] == [1.0, 1.0] and got[7] == [1.0, 0.0]


def test_raw_features_run_through_the_distributed_stacked_readout(spark, tmp_path):
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    rng = np.random.default_rng(0)
    C = 3

    def frame(n, off):
        rows = []
        for i in range(n):
            a, b = rng.random() < 0.3, rng.random() < 0.2
            toks = set(rng.choice(np.arange(2, 12), 3, replace=False).tolist())
            if a:
                toks.add(0)
            if b:
                toks.add(1)
            y1 = float(a and rng.random() < 0.8)
            y2 = float(b and rng.random() < 0.8)
            rows.append(Row(person_id=off + i, label=[1.0, y1, y2],
                            labelMask=[1.0, 1.0, 1.0],
                            features_0=Vectors.sparse(12, sorted(toks), [1.0] * len(toks))))
        return spark.createDataFrame(rows)
    tr, te = frame(400, 0), frame(200, 10_000)
    cols = gpr.raw_feature_cols(tr)
    sel = gpr.select_raw_codes(tr, 6, cols)
    tr, te = gpr.attach_raw_presence(tr, sel, cols), gpr.attach_raw_presence(te, sel, cols)
    got = gpr.run_readout(tr, te, {"C": C, "K": len(sel), "weight_y": 0.0},
                          recall_targets=[0.5], fdr_targets=[0.5], min_count=0,
                          readout_mode="distributed", theta_topm=0, out_dir=tmp_path,
                          mask_tag="raw6_stacked", stacked=True,
                          parent_int={0: [], 1: [0], 2: [0]}, readout_l2=1.0)
    assert "gated_pc_stacked" in got and got["gated_pc_stacked"]
    z = np.load(tmp_path / "readout_heads_gated_pc_raw6_stacked.npz")
    assert z["V"].shape == (C, len(sel))


def test_masked_heads_drop_out_of_the_distributed_solve(spark, tmp_path, capsys):
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    rng = np.random.default_rng(1)

    def frame(n, off):
        rows = []
        for i in range(n):
            th = rng.dirichlet(np.ones(4))
            y = [1.0, float(th[0] > 0.3), float(th[1] > 0.3)]
            rows.append(Row(person_id=off + i, label=y, labelMask=[1.0, 1.0, 1.0],
                            topicDistribution=Vectors.dense(th)))
        return gpr.mask_to_nodes(spark.createDataFrame(rows), [True, True, False])
    gpr.run_readout(frame(300, 0), frame(150, 10_000),
                    {"C": 3, "K": 4, "weight_y": 0.0}, recall_targets=[0.5],
                    fdr_targets=[0.5], min_count=0, readout_mode="distributed",
                    theta_topm=0, out_dir=tmp_path, mask_tag="nodes_stacked",
                    stacked=True, parent_int={0: [], 1: [0], 2: [0]}, readout_l2=1.0)
    z = np.load(tmp_path / "readout_heads_gated_pc_nodes_stacked.npz")
    assert bool(z["degenerate"][2]) and not bool(z["degenerate"][1])
