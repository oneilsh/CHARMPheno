"""Gated NPMI: topic coherence with each node block of a DAG-gated fit scored on the
documents its gate admits (exp 0159; moved here from `analysis/cloud/topic_npmi.py`,
which keeps the run-dir I/O and the CLI).

Why a per-block reference. A gated fit trains node u's block only on documents whose
gate admits it: u in closure(frontier), i.e. a document coded somewhere in u's
subtree. Scoring that block's top words against the whole corpus asks whether they
co-occur in the general population; scoring it against the admitted documents asks
the question the fit answered, whether they co-occur beyond their base rates AMONG
THE DOCUMENTS THE BLOCK DESCRIBES. Both are computed: `npmi_gated` (admitted
documents) and `npmi_all` (every document). Background topics are admitted on every
document, so for them the two coincide.

NPMI per topic is this package's definition (`coherence._npmi_pair`, binary document
co-occurrence, the mean over the top-N word pairs with a joint document count >=
`min_pair_count`) with its own reference size as N.

One pass for every topic: an inverted index from each top word to its (topic,
position) entries rides a broadcast with the layout (`spark_vi.dag.DagLayout`), so a
document touches only the topics whose top words it holds. Per-partition
accumulators are allocated executor-side; the reduction identity is a None sentinel;
broadcasts are destroyed after the pass (ADR 0047).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["top_words", "own_code_exclusions", "build_index", "doc_topic_hits",
           "new_acc", "accumulate", "finish_counts", "npmi_from_counts",
           "gated_npmi_counts", "score", "summarize"]


def top_words(lam, top_n, exclude=None):
    """(K, top_n) word indices per topic, highest E[beta] first (row-normalized).
    `exclude` {topic: iterable of word idx} drops those words before ranking (the
    own-code-excluded variant: a node's own attesting codes are in nearly every
    document its gate admits, so their gated NPMI is ~0 by construction)."""
    lam = np.asarray(lam, dtype=np.float64)
    beta = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-300)
    if exclude:
        beta = beta.copy()
        for k, ws in exclude.items():
            ws = [int(w) for w in ws if 0 <= int(w) < beta.shape[1]]
            if ws:
                beta[int(k), ws] = -1.0
    return np.argsort(-beta, axis=1)[:, :top_n]


def own_code_exclusions(lay, own_idx):
    """{topic: vocab idx} = the attesting codes of the block's node and of every
    descendant (`own_idx` {node: [vocab idx]})."""
    out = {}
    for u in lay.nodes:
        ws = set(own_idx.get(u, ()))
        for c in lay.descendants(u):
            ws.update(own_idx.get(c, ()))
        if ws:
            for k in lay.block[u]:
                out[k] = sorted(ws)
    return out


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


def new_acc():
    return {"n_all": 0, "n_gated": {}, "g": {}, "a": {}}


def accumulate(acc, words, frontier, index, lay, n_bg, N):
    """Add one document to the counts (shared by the Spark pass and the local one)."""
    gated, every, admitted = doc_topic_hits(words, frontier, index, lay, n_bg)
    acc["n_all"] += 1
    for k in admitted:
        acc["n_gated"][k] = acc["n_gated"].get(k, 0) + 1
    for key, hits in (("g", gated), ("a", every)):
        for k, ps in hits.items():
            d = acc[key].get(k)
            if d is None:
                d = acc[key][k] = (np.zeros(N, np.int64), np.zeros((N, N), np.int64))
            p = np.asarray(sorted(set(ps)), dtype=np.int64)
            d[0][p] += 1
            d[1][np.ix_(p, p)] += 1


def finish_counts(tot, K, N):
    """Accumulator -> (gated, every) dicts {topic: (n, dfc, pairs)}."""
    z = (np.zeros(N, np.int64), np.zeros((N, N), np.int64))
    tot = tot or new_acc()
    gated = {k: (tot["n_gated"].get(k, 0),) + tot["g"].get(k, z) for k in range(K)}
    every = {k: (tot["n_all"],) + tot["a"].get(k, z) for k in range(K)}
    return gated, every


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
            if acc is None:
                acc = new_acc()
            accumulate(acc, words, r[1], I, L, _nbg, _N)
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
    return finish_counts(tot, K, N)


def score(gated, every, *, min_pair_count=3, floor=20):
    """{topic: dict(n_ref, npmi_gated, npmi_all, pairs_gated, pairs_all)}; a topic
    whose gated reference holds fewer than `floor` documents is not scored (callers
    under a disclosure rule pass theirs)."""
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
