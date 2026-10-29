"""Gated NPMI driver (exp 0159): score every topic of a saved DAG-gated fit and write
the run-dir tables. The computation is `spark_vi.eval.topic.gated_npmi`; this module
holds the run-dir I/O, the egress rule and the CLI.

EGRESS: the per-topic table (with reference document counts) is written to the run
dir and is WORKSPACE-INTERNAL; a topic with fewer than 20 reference documents is
not scored. The markdown summary carries pooled figures only.

    dag_gated_readout.py ... --npmi [--npmi-top 10] [--npmi-split train] [--npmi-domains 0]
    python3 topic_npmi.py compare <run A> <run B> [tag]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from charmpheno.egress import EGRESS_FLOOR
from spark_vi.eval.topic.gated_npmi import (accumulate, build_index, finish_counts,
                                            gated_npmi_counts, new_acc,
                                            own_code_exclusions, score,
                                            summarize, top_words)


def run_npmi(train_df, test_df, run_dir, *, lay, lams, n_bg, int2cid, names,
             split="train", top_n=10, domains=(0,), tag=None, min_pair_count=3,
             exclude=None):
    """Score every topic of a saved fit; write npmi[_<tag>].json (per topic,
    workspace-internal) and .md (pooled)."""
    run_dir = Path(run_dir)
    df = train_df if split == "train" else test_df
    df = df.cache()
    cols = set(df.columns)
    stem = "npmi" + (f"_{tag}" if tag else "")
    rows, md = [], []
    topic_node = {k: u for u in lay.nodes for k in lay.block[u]}
    try:
        for d in domains:
            col = f"features_{d}" if f"features_{d}" in cols else "features"
            lam = lams[d] if d < len(lams) else lams[0]
            tops = top_words(lam, top_n, exclude=(exclude if d == 0 else None))
            gated, every = gated_npmi_counts(df, lay, tops, feat_col=col, n_bg=n_bg)
            sc = score(gated, every, min_pair_count=min_pair_count, floor=EGRESS_FLOOR)
            drows = []
            for k, s in sc.items():
                u = topic_node.get(k)
                cid = int2cid.get(u) if u is not None else None
                drows.append(dict(domain=int(d), topic=int(k),
                                  kind="bg" if u is None else "node",
                                  node=(None if u is None else int(u)),
                                  cid=cid, name=(names.get(cid) if cid is not None else None),
                                  depth=(lay.depth(u) if u is not None else -1), **s))
            rows += drows
            md.append(summarize(drows, title=f"{run_dir.name} — NPMI, domain {d}, "
                                f"top {top_n}, {split} split (gated = the documents "
                                f"each block's gate admits)"))
    finally:
        df.unpersist()
    (run_dir / f"{stem}.json").write_text(json.dumps(
        {"top_n": top_n, "split": split, "min_pair_count": min_pair_count,
         "rows": rows}, indent=1))
    (run_dir / f"{stem}.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[npmi]   wrote {run_dir / (stem + '.md')} (+ .json, workspace-internal)",
          flush=True)
    return rows


def compare(run_a, run_b, tag=None):
    """Pair two runs' node topics by Mondo id (and domain): median B − A of gated and
    all-docs NPMI, B better counts, by depth. Pooled only."""
    stem = "npmi" + (f"_{tag}" if tag else "")
    A = json.loads((Path(run_a) / f"{stem}.json").read_text())["rows"]
    B = json.loads((Path(run_b) / f"{stem}.json").read_text())["rows"]
    key = lambda r: (r["domain"], r["cid"])  # noqa: E731
    a = {key(r): r for r in A if r["kind"] == "node" and r["cid"] is not None}
    b = {key(r): r for r in B if r["kind"] == "node" and r["cid"] is not None}
    L = [f"# NPMI paired by Mondo id: {Path(run_a).name} (A) vs {Path(run_b).name} (B)"]
    for d in sorted({k[0] for k in a}):
        for m in ("npmi_gated", "npmi_all"):
            ks = [k for k in a if k[0] == d and k in b
                  and not math.isnan(a[k][m]) and not math.isnan(b[k][m])]
            if not ks:
                continue
            dl = np.array([b[k][m] - a[k][m] for k in ks])
            L.append(f"domain {d} · {m}: {len(ks)} nodes · A {np.median([a[k][m] for k in ks]):+.4f}"
                     f" · B {np.median([b[k][m] for k in ks]):+.4f} · median B−A "
                     f"{np.median(dl):+.4f} · B better on {int((dl > 0).sum())}/{len(ks)}")
            for dep in sorted({a[k]["depth"] for k in ks}):
                kd = [k for k in ks if a[k]["depth"] == dep]
                dd = np.array([b[k][m] - a[k][m] for k in kd])
                L.append(f"    depth {dep}: n={len(kd)} median B−A {np.median(dd):+.4f} "
                         f"B better {int((dd > 0).sum())}/{len(kd)}")
    print("\n".join(L))
    return L


def _vec_words(v):
    """Indices of a Spark ML vector read by pyarrow: a struct with type (0 sparse,
    1 dense), size, indices, values."""
    if v is None:
        return ()
    if v.get("type") == 0 or v.get("indices") is not None:
        return v.get("indices") or ()
    vals = v.get("values") or ()
    return [i for i, x in enumerate(vals) if x]


def run_local(run_dir, parquet_dir, *, top_n=10, domains=(0,), tag=None,
              exclude_own=False, min_pair_count=3):
    """The same NPMI without Spark: the bundle's TRAIN parquet copied to local disk
    (`hdfs dfs -get`, an HDFS client read, no YARN), streamed with pyarrow. Layout,
    vocab maps and names from `<run>/bundle_meta.json`; own codes from
    `<run>/code_map.tsv`."""
    from spark_vi.dag import DagLayout
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    meta = json.loads((run_dir / "bundle_meta.json").read_text())
    npz = np.load(run_dir / "gated_pc_result.npz")
    lk = sorted((k for k in npz.files if k.startswith("lambda_")),
                key=lambda k: int(k.split("_", 1)[1]))
    lams = [npz[k] for k in lk] if lk else [npz["lambda"]]
    n_bg = int(manifest["n_bg"])
    lay = DagLayout({int(c): [int(p) for p in ps] for c, ps in meta["parent_int"].items()},
                    n_bg=n_bg, tpn=int(manifest["tpn"]))
    int2cid = {int(k): int(v) for k, v in meta["int2cid"].items()}
    names = {int(k): v for k, v in (meta.get("name_by_id") or {}).items()}
    exclude = None
    if exclude_own:
        vm0 = {int(k): int(v) for k, v in meta["vocab_maps"][0].items()}
        cid2int = {c: e for e, c in int2cid.items()}
        own = {}
        with open(run_dir / "code_map.tsv") as fh:
            next(fh)
            for line in fh:
                std, node = (int(x) for x in line.split("\t")[:2])
                e, vi = cid2int.get(node), vm0.get(std)
                if e is not None and vi is not None:
                    own.setdefault(e, set()).add(vi)
        exclude = own_code_exclusions(lay, own)
    import pyarrow.dataset as ds
    pf = ds.dataset(str(parquet_dir), format="parquet")
    cols = set(pf.schema.names)
    stem = "npmi" + (f"_{tag}" if tag else "")
    topic_node = {k: u for u in lay.nodes for k in lay.block[u]}
    rows, md = [], []
    for d in domains:
        col = f"features_{d}" if f"features_{d}" in cols else "features"
        tops = top_words(lams[d] if d < len(lams) else lams[0], top_n,
                         exclude=(exclude if d == 0 else None))
        K, N = tops.shape
        index = build_index(tops, range(K))
        acc, n = new_acc(), 0
        for batch in pf.to_batches(columns=[col, "frontier"], batch_size=20000):
            for v, fr in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                accumulate(acc, _vec_words(v), fr or (), index, lay, n_bg, N)
                n += 1
            print(f"[npmi-local]   domain {d}: {n:,} documents", flush=True)
        gated, every = finish_counts(acc, K, N)
        sc = score(gated, every, min_pair_count=min_pair_count, floor=EGRESS_FLOOR)
        drows = []
        for k, s_ in sc.items():
            u = topic_node.get(k)
            cid = int2cid.get(u) if u is not None else None
            drows.append(dict(domain=int(d), topic=int(k),
                              kind="bg" if u is None else "node",
                              node=(None if u is None else int(u)), cid=cid,
                              name=(names.get(cid) if cid is not None else None),
                              depth=(lay.depth(u) if u is not None else -1), **s_))
        rows += drows
        md.append(summarize(drows, title=f"{run_dir.name} — NPMI, domain {d}, top "
                            f"{top_n}{', own codes excluded' if exclude_own and d == 0 else ''}"
                            f", train split (local)"))
    (run_dir / f"{stem}.json").write_text(json.dumps(
        {"top_n": top_n, "split": "train", "min_pair_count": min_pair_count,
         "exclude_own": bool(exclude_own), "rows": rows}, indent=1))
    (run_dir / f"{stem}.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[npmi-local]   wrote {run_dir / (stem + '.md')} (+ .json, workspace-internal)")


if __name__ == "__main__":
    import argparse
    if len(sys.argv) >= 2 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) >= 3 and sys.argv[1] == "key":
        # The run's bundle on HDFS: <cache_uri>/<key>, recomputed from the manifest
        # exactly as the readout does (no Spark session).
        from dag_gated_readout import bundle_key_from_manifest
        _m = json.loads((Path(sys.argv[2]) / "manifest.json").read_text())
        _cu = ((_m.get("corpus_manifest") or {}).get("cache_uri")
               or "hdfs:///user/dataproc/charm/case_finding_cache")
        print(f"{_cu}/{bundle_key_from_manifest(_m)}")
    elif len(sys.argv) >= 2 and sys.argv[1] == "local":
        ap = argparse.ArgumentParser(prog="topic_npmi.py local")
        ap.add_argument("run_dir")
        ap.add_argument("--parquet", required=True,
                        help="local copy of the bundle's train.parquet directory")
        ap.add_argument("--top", type=int, default=10)
        ap.add_argument("--domains", default="0")
        ap.add_argument("--tag", default=None)
        ap.add_argument("--exclude-own", action="store_true",
                        help="drop the node's and its descendants' own attesting codes "
                             "from each block's condition top-N before scoring")
        a = ap.parse_args(sys.argv[2:])
        run_local(a.run_dir, a.parquet, top_n=a.top,
                  domains=[int(x) for x in a.domains.split(",")], tag=a.tag,
                  exclude_own=a.exclude_own)
    else:
        sys.exit("usage: topic_npmi.py compare <run A> <run B> [tag] | "
                 "topic_npmi.py local <run> --parquet DIR [--exclude-own] ...")
