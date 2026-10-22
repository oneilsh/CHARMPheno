"""Gated NPMI: topic coherence with each node block scored on the documents its gate
admits (exp 0159).

Why a per-block reference. A gated fit trains node u's block only on documents whose
gate admits it: u in closure(frontier), i.e. a document coded somewhere in u's
subtree. Scoring that block's top words against the whole corpus asks whether they
co-occur in the general population; scoring it against the admitted documents asks
the question the fit answered, whether they co-occur beyond their base rates AMONG
THE PATIENTS THE BLOCK DESCRIBES. Both are computed here: `npmi_gated` (admitted
documents) and `npmi_all` (every document). Background topics are admitted on every
document, so for them the two coincide.

NPMI per topic is the spark-vi definition (`spark_vi.eval.topic.coherence._npmi_pair`,
binary document co-occurrence, the mean over the top-N word pairs with a joint
document count >= `min_pair_count`, insight 0007) with its own reference size as N.

One pass for every topic: an inverted index from each top word to its (topic,
position) entries rides a broadcast with the layout, so a document touches only the
topics whose top words it holds. Per-partition accumulators are allocated
executor-side; the reduction identity is a None sentinel; broadcasts are destroyed
after the pass (ADR 0047).

EGRESS: the per-topic table (with reference document counts) is written to the run
dir and is WORKSPACE-INTERNAL; a topic with fewer than 20 reference documents is
not scored. The markdown summary carries pooled figures only.

    gated_pc_readout.py ... --npmi [--npmi-top 10] [--npmi-split train] [--npmi-domains 0]
    python3 topic_npmi.py compare <run A> <run B> [tag]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

EGRESS_FLOOR = 20


def top_words(lam, top_n):
    """(K, top_n) word indices per topic, highest E[beta] first (row-normalized)."""
    lam = np.asarray(lam, dtype=np.float64)
    beta = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-300)
    return np.argsort(-beta, axis=1)[:, :top_n]


def build_index(tops, topics):
    """{word: [(topic, position), ...]} over the scored topics' top words."""
    idx = {}
    for k in topics:
        for pos, w in enumerate(tops[k]):
            idx.setdefault(int(w), []).append((int(k), pos))
    return idx


def doc_topic_hits(words, frontier, index, lay, n_bg):
    """{topic: sorted positions present} for one document, restricted to the topics
    its gate admits (background always; a node block when the node is in the closure
    of the frontier), and {topic: positions} ignoring the gate (the all-docs
    reference). `words` is the document's distinct index set."""
    admitted = set(range(n_bg))
    for f in frontier or ():
        for u in lay.closure(int(f)):
            if u != 0:
                admitted.update(lay.block[u])
    gated, every = {}, {}
    for w in words:
        for k, pos in index.get(int(w), ()):
            every.setdefault(k, []).append(pos)
            if k in admitted:
                gated.setdefault(k, []).append(pos)
    return gated, every, admitted


def npmi_from_counts(n_docs, dfc, pairs, *, min_pair_count=3):
    """Mean NPMI over the top-N pairs of one topic with joint count >= min_pair_count;
    (npmi, n_pairs_scored). NaN when no pair qualifies or n_docs is 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    if n_docs <= 0:
        return float("nan"), 0
    N = len(dfc)
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            c = pairs[i, j]
            if c >= min_pair_count and dfc[i] > 0 and dfc[j] > 0:
                vals.append(_npmi_pair(dfc[i] / n_docs, dfc[j] / n_docs, c / n_docs))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def gated_npmi_counts(df, lay, tops, *, feat_col, n_bg, frontier_col="frontier"):
    """One pass: per topic (gated reference) and per topic (all-docs reference),
    (n_docs, df counts (N,), pair counts (N, N)). Returns two dicts
    {topic: (n, dfc, pairs)}; the all-docs n is the corpus size for every topic."""
    K, N = tops.shape
    topics = list(range(K))
    sc = df.sparkSession.sparkContext
    lay_b = sc.broadcast(lay)
    idx_b = sc.broadcast(build_index(tops, topics))

    def _part(rows, _lay=lay_b, _idx=idx_b, _N=N, _nbg=n_bg):
        L, I = _lay.value, _idx.value
        acc = None
        for r in rows:
            v = r[0]
            words = v.indices if hasattr(v, "indices") else np.flatnonzero(v.toArray())
            gated, every, admitted = doc_topic_hits(words, r[1], I, L, _nbg)
            if acc is None:
                acc = {"n_all": 0, "n_gated": {}, "g": {}, "a": {}}
            acc["n_all"] += 1
            for k in admitted:
                acc["n_gated"][k] = acc["n_gated"].get(k, 0) + 1
            for key, hits in (("g", gated), ("a", every)):
                for k, ps in hits.items():
                    d = acc[key].get(k)
                    if d is None:
                        d = acc[key][k] = (np.zeros(_N, np.int64),
                                           np.zeros((_N, _N), np.int64))
                    p = np.asarray(sorted(set(ps)), dtype=np.int64)
                    d[0][p] += 1
                    d[1][np.ix_(p, p)] += 1
        yield acc

    def _comb(a, b):
        if a is None:
            return b
        if b is None:
            return a
        a["n_all"] += b["n_all"]
        for k, n in b["n_gated"].items():
            a["n_gated"][k] = a["n_gated"].get(k, 0) + n
        for key in ("g", "a"):
            for k, (d, p) in b[key].items():
                if k in a[key]:
                    a[key][k][0][:] += d
                    a[key][k][1][:] += p
                else:
                    a[key][k] = (d, p)
        return a

    try:
        tot = (df.select(feat_col, frontier_col).rdd
               .mapPartitions(_part).treeReduce(_comb, depth=2))
    finally:
        for b in (lay_b, idx_b):
            try:
                b.destroy()
            except Exception:
                b.unpersist()
    z = (np.zeros(N, np.int64), np.zeros((N, N), np.int64))
    tot = tot or {"n_all": 0, "n_gated": {}, "g": {}, "a": {}}
    gated = {k: (tot["n_gated"].get(k, 0),) + tot["g"].get(k, z) for k in topics}
    every = {k: (tot["n_all"],) + tot["a"].get(k, z) for k in topics}
    return gated, every


def score(gated, every, *, min_pair_count=3, floor=EGRESS_FLOOR):
    """{topic: dict(n_ref, npmi_gated, npmi_all, pairs_gated, pairs_all)}; a topic
    whose gated reference is under the egress floor is not scored."""
    out = {}
    for k in gated:
        n, dfc, pr = gated[k]
        na, dfa, pra = every[k]
        if n < floor:
            out[k] = dict(n_ref=int(n), npmi_gated=float("nan"),
                          npmi_all=float("nan"), pairs_gated=0, pairs_all=0)
            continue
        g, pg = npmi_from_counts(n, dfc, pr, min_pair_count=min_pair_count)
        a, pa = npmi_from_counts(na, dfa, pra, min_pair_count=min_pair_count)
        out[k] = dict(n_ref=int(n), npmi_gated=g, npmi_all=a, pairs_gated=pg,
                      pairs_all=pa)
    return out


def summarize(rows, *, title):
    """Pooled markdown: background vs node topics, node topics by depth (medians,
    counts of nodes). No per-topic reference counts."""
    def med(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return (float(np.median(xs)), len(xs)) if xs else (float("nan"), 0)
    L = [f"# {title}", ""]
    bg = [r for r in rows if r["kind"] == "bg"]
    nd = [r for r in rows if r["kind"] == "node"]
    for lab, rs in (("background", bg), ("node", nd)):
        g, ng = med([r["npmi_gated"] for r in rs])
        a, na = med([r["npmi_all"] for r in rs])
        L.append(f"{lab} topics: {len(rs)} · scored {ng} · median NPMI gated {g:+.4f} · "
                 f"all-docs {a:+.4f}")
    L += ["", "depth · nodes scored · median NPMI gated · median NPMI all-docs",]
    for dep in sorted({r["depth"] for r in nd}):
        rs = [r for r in nd if r["depth"] == dep]
        g, ng = med([r["npmi_gated"] for r in rs])
        a, _ = med([r["npmi_all"] for r in rs])
        L.append(f"{dep:>5} · {ng:>4} · {g:+.4f} · {a:+.4f}")
    return "\n".join(L) + "\n"


def run_npmi(train_df, test_df, run_dir, *, lay, lams, n_bg, int2cid, names,
             split="train", top_n=10, domains=(0,), tag=None, min_pair_count=3):
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
            tops = top_words(lam, top_n)
            gated, every = gated_npmi_counts(df, lay, tops, feat_col=col, n_bg=n_bg)
            sc = score(gated, every, min_pair_count=min_pair_count)
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


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        sys.exit("usage: python3 topic_npmi.py compare <run A> <run B> [tag]")
