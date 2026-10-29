"""Exp 0167: `framings.workup_frame` — the per-group workup window joined from
the conversion sidecar (first diagnosis of a member in the label year) and the workup
sidecar (first dates of every code after the index)."""
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "cloud"))


def _frames(spark):
    D = dt.date
    first = spark.createDataFrame([
        (1, 101, D(2020, 3, 1)),        # member 100's child, 60 days after index
        (2, 100, D(2019, 6, 1)),        # diagnosed before the index: prevalent
        (3, 100, D(2021, 6, 1)),        # after the label year
        (4, 555, D(2020, 2, 1)),        # an unrelated node
    ], "person_id long, node_cid long, first_attested_date date")
    hz = spark.createDataFrame([(p, D(2020, 1, 1), D(2023, 1, 1)) for p in (1, 2, 3, 4)],
                               "person_id long, index_date date, obs_end_date date")
    wk = spark.createDataFrame([
        (1, 0, 7, D(2020, 2, 1)), (1, 2, 500003, D(2020, 1, 15)),
        (1, 0, 8, D(2020, 3, 1)),     # the diagnosis day itself
        (1, 1, 9, D(2020, 4, 1)),     # after the diagnosis
        (2, 0, 7, D(2020, 2, 1)),
    ], "person_id long, dom int, token long, first_date date")
    return first, hz, wk


def test_workup_frame_windows(spark):
    import framings as gpr
    first, hz, wk = _frames(spark)
    padj = {"MONDO:0000101": ["MONDO:0000100"]}
    nodes = [101, 100, 555]
    strict = {r["person_id"]: r for r in gpr.workup_frame(
        first, hz, wk, nodes, [100], padj).collect()}
    assert set(strict) == {1}                    # prevalent and late persons: no window
    assert strict[1]["__wdays"] == 60
    assert sorted((x["d"], x["t"]) for x in strict[1]["__wk"]) == [(0, 7), (2, 500003)]
    inc = gpr.workup_frame(first, hz, wk, nodes, [100], padj, inclusive=True).collect()
    assert sorted((x["d"], x["t"]) for x in inc[0]["__wk"]) == [(0, 7), (0, 8),
                                                              (2, 500003)]
    assert gpr.workup_frame(first, hz, wk, nodes, [999], padj).count() == 0
