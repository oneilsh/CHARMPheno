"""The HPO profile seed for the leaves-first gate schedule (exp 0157/0158; ADR 0049).

`build_profile_seed` turns a phenotype-marks table (``hpo_profile_marks.py``
output: mondo_id, concept_id, neg) into {engine node id: {vocab idx: weight}} for
the estimator's `gateSeedProfile`. DRIVER-SIDE ONLY: the mapping needs the bundle
meta (concept -> vocab index, mondo curie -> engine node id) that only the fit
driver holds. Pure logic, no Spark; unit-tested in tests/scripts/test_profile_seed.py.

The profile-eta word-side prior and the guided-anchor candidates that used to be
built here were removed by ADR 0050 (the module keeps its name for the run logs
and docs that cite it).

EGRESS: the returned stats are counts of nodes/concepts/rows only — safe for
driver logs and the run manifest.
"""
from __future__ import annotations

import numpy as np


def _neg_flags(values) -> np.ndarray:
    """Boolean polarity of the ``neg`` column, whatever a TSV round-trip made
    of it (ints, bools, or "True"/"1" strings) — the same tolerance as
    hpoa_stage2_probe's ``_neg_mask``, without importing that (Spark-heavy)
    driver module here."""
    out = np.empty(len(values), dtype=bool)
    for i, v in enumerate(values):
        if isinstance(v, str):
            out[i] = v.strip().lower() in ("true", "1")
        else:
            out[i] = bool(int(v))
    return out



def build_profile_seed(marks_df, *, eid_by_mondo, vocab_index_by_concept):
    """The leaves-first schedule's PROFILE seed (exp 0157/0158; ADR 0049): {engine
    node id: {condition-domain vocab idx: weight}} for every node of this run with
    an external phenotype profile (HPO) in the vocabulary. The estimator seeds the
    node's condition-domain start from it (leaves at iteration 0, later blocks when
    they open); unprofiled nodes keep the histogram seed.

    Rules: the concept is in the bundle's domain-0 vocabulary; NOT-annotated
    (``neg``) phenotypes never seed, and a concept listed both ways for a node is
    contested and dropped; the weight is an IDF over THIS run's profiled nodes,
    log(N / n_c), so a phenotype every profiled node shares (n_c = N) weighs 0 and
    drops out — the same knob-free exclusion of profile-generic terms the guided
    anchors use, computed here because the marks table carries no weight
    (``hpo_profile_marks.py`` is bundle-agnostic). Columns used: mondo_id,
    concept_id, neg. Returns ``(seed, stats)``; stats are counts only."""
    mids = marks_df["mondo_id"].astype(str).to_numpy()
    cids = marks_df["concept_id"].astype("int64").to_numpy()
    negs = _neg_flags(marks_df["neg"].tolist())
    pos, neg = {}, {}
    n_unmapped = 0
    for mid, cid, ng in zip(mids, cids, negs):
        eid = eid_by_mondo.get(mid)
        if eid is None:
            continue
        idx = vocab_index_by_concept.get(int(cid))
        if idx is None:
            idx = vocab_index_by_concept.get(str(int(cid)))
        if idx is None:
            n_unmapped += 1
            continue
        (neg if ng else pos).setdefault(int(eid), set()).add(int(idx))
    sets = {e: s - neg.get(e, set()) for e, s in pos.items()}
    sets = {e: s for e, s in sets.items() if s}
    N = len(sets)
    df = {}
    for s in sets.values():
        for i in s:
            df[i] = df.get(i, 0) + 1
    seed = {}
    for e, s in sets.items():
        w = {i: float(np.log(N / df[i])) for i in s if df[i] < N}
        if w:
            seed[e] = w
    stats = {"n_nodes_profiled_in_dag": int(len({eid_by_mondo[m] for m in set(mids)
                                                if m in eid_by_mondo})),
             "n_nodes_with_vocab_profile": int(N),
             "n_nodes_seeded": int(len(seed)),
             "n_rows_concept_not_in_vocab": int(n_unmapped),
             "median_concepts_per_seeded_node": (
                 float(np.median([len(v) for v in seed.values()])) if seed else 0.0)}
    return seed, stats
