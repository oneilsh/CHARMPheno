"""BLAS thread probe on the executors (scoping report WP6, 2026-11-25).

Question: do the fits' Python workers on Dataproc oversubscribe BLAS threads, and
does it cost time? On a 4-core dev box an STM E-step ran ~30x slower with
OpenBLAS's default threads than with one. The DAG-gated E-step is
`_cavi_doc_inference` (spark_vi.models.topic.lda): per document, up to cavi_max_iter
matrix-vector products with the open topics' (K_open, n_unique) slice of E[log β],
sizes where OpenBLAS may start threading.

What it does: on every executor core at once (as a fit does), each task runs the real
`_cavi_doc_inference` over synthetic documents in a fresh Python subprocess, once per
thread setting, and reports per-document time. The settings are separate Spark jobs,
so every concurrent task runs the same setting:

  * spark   - the worker's environment as Spark leaves it (Spark sets OMP_NUM_THREADS
              to spark.executor.cores when that is set and OMP_NUM_THREADS is not);
  * 1       - OMP / OPENBLAS / MKL threads = 1;
  * unset   - those three removed: the library default (all of the VM's cores), what
              a user outside Spark gets.

A subprocess, because OpenBLAS reads its thread count when it loads and the Python
worker has numpy loaded already. Synthetic data only: no All of Us data is read, so
nothing here is subject to the egress floor.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# (K_open, n_unique): background 200 plus a document's open node blocks, by the
# number of distinct codes. Bracket the 0175/0176 configuration (n_bg 200, tpn 1).
SHAPES = ((200, 40), (200, 150), (400, 40), (400, 150))

_BENCH = r"""
import json, sys, time
import numpy as np
from spark_vi.models.topic.lda import _cavi_doc_inference
out = {"blas": None}
try:
    from threadpoolctl import threadpool_info
    out["blas"] = [{k: p.get(k) for k in ("internal_api", "num_threads", "version")}
                   for p in threadpool_info()]
except Exception as e:  # threadpoolctl is optional on the cluster
    out["blas"] = "threadpoolctl unavailable: %s" % e
shapes = json.loads(sys.argv[1]); n_docs = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
out["ms_per_doc"] = {}
for K, n in shapes:
    V = 4 * n
    eb = rng.gamma(1.0, 1.0, size=(K, V)); eb /= eb.sum(axis=1, keepdims=True)
    docs = []
    for _ in range(n_docs):
        idx = np.sort(rng.choice(V, size=n, replace=False))
        docs.append((idx, rng.integers(1, 4, size=n).astype(np.float64)))
    gamma0 = np.ones(K)
    t0 = time.perf_counter()
    for idx, cnt in docs:
        _cavi_doc_inference(idx, cnt, eb, 0.5, gamma0, max_iter=100, tol=1e-3)
    out["ms_per_doc"]["%dx%d" % (K, n)] = 1e3 * (time.perf_counter() - t0) / n_docs
print(json.dumps(out))
"""


def _task(setting: str, n_docs: int):
    """Spark map function: run the bench in a subprocess under `setting`."""
    def run(i: int):
        import os as _os
        import subprocess as _sp
        import sys as _sys
        env = dict(_os.environ)
        seen = {v: env.get(v) for v in _VARS}
        if setting == "1":
            env.update({v: "1" for v in _VARS})
        elif setting == "unset":
            for v in _VARS:
                env.pop(v, None)
        env["PYTHONPATH"] = _os.pathsep.join(p for p in _sys.path if p)
        r = _sp.run([_sys.executable, "-c", _BENCH, json.dumps(SHAPES), str(n_docs),
                     str(1000 + i)], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": r.stderr[-2000:], "worker_env": seen}
        res = json.loads(r.stdout.strip().splitlines()[-1])
        res.update(worker_env=seen, cpu_count=_os.cpu_count(), host=_os.uname()[1])
        return res
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", type=int, default=1500, help="documents per task per shape")
    ap.add_argument("--settings", default="spark,1,unset")
    ap.add_argument("--out", default=None, help="markdown report path (run dir)")
    a = ap.parse_args(argv)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("blas-probe").getOrCreate()
    sc = spark.sparkContext
    conf = sc.getConf()
    cores = int(conf.get("spark.executor.cores", "1"))
    inst = conf.get("spark.executor.instances")
    n_tasks = int(inst) * cores if inst else sc.defaultParallelism
    lines = [f"# BLAS probe {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"spark.executor.cores={cores}, instances={inst}, tasks per job={n_tasks}, "
             f"docs per task per shape={a.docs}", ""]
    summary = {}
    for setting in a.settings.split(","):
        t0 = time.time()
        res = sc.parallelize(range(n_tasks), n_tasks).map(_task(setting, a.docs)).collect()
        wall = time.time() - t0
        errs = [r for r in res if "error" in r]
        ok = [r for r in res if "error" not in r]
        lines.append(f"## setting `{setting}` ({len(ok)} tasks ok, {len(errs)} failed, "
                     f"job wall {wall:.0f} s)")
        if errs:
            lines.append(f"first error:\n```\n{errs[0]['error']}\n```")
        if ok:
            r0 = ok[0]
            lines.append(f"- worker env as Spark set it: {r0['worker_env']}; "
                         f"os.cpu_count() = {r0['cpu_count']}")
            lines.append(f"- BLAS: {r0['blas']}")
            for shape in r0["ms_per_doc"]:
                v = sorted(r["ms_per_doc"][shape] for r in ok)
                med = statistics.median(v)
                summary.setdefault(shape, {})[setting] = med
                lines.append(f"- {shape}: median {med:.2f} ms/doc "
                             f"(min {v[0]:.2f}, max {v[-1]:.2f})")
        lines.append("")
    lines.append("## Median ms per document (K_open x n_unique)")
    settings = a.settings.split(",")
    lines.append("| shape | " + " | ".join(settings) + " |")
    lines.append("|---|" + "---|" * len(settings))
    for shape, row in summary.items():
        lines.append(f"| {shape} | " + " | ".join(
            f"{row[s]:.2f}" if s in row else "-" for s in settings) + " |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        with open(a.out, "w") as f:
            f.write(report)
        print(f"wrote {a.out}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
