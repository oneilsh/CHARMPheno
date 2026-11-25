"""blas_probe's per-task runner, without Spark: the subprocess bench runs the real
CAVI under each thread setting and reports a time per shape."""
import blas_probe


def test_task_runs_bench_under_each_setting():
    for setting in ("spark", "1", "unset"):
        res = blas_probe._task(setting, 3)(0)
        assert "error" not in res, res.get("error")
        assert set(res["ms_per_doc"]) == {f"{k}x{n}" for k, n in blas_probe.SHAPES}
        assert all(v > 0 for v in res["ms_per_doc"].values())
