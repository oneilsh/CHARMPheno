"""Profile-eta boost builder — WP-3 of the 2026-09-06 profile-eta-prior plan.

Turns the ``hpoa_stage2_probe --emit-eta`` TSV (per-(node, standard Condition
concept, polarity) rows: ``mondo_id, concept_id, weight, neg, coverage``; freq
x IDF already folded into ``weight``) into the sparse per-topic eta boost
``{topic_index: (vocab_idx_array, weight_array)}`` that
``OnlinePCLDAEstimator.setEtaBoost`` consumes (plan D1-D4). DRIVER-OWNED and
DRIVER-SIDE ONLY: the mapping needs the bundle meta (concept -> condition
vocab index, mondo curie -> engine node id) that only the fit driver holds
(plan D4), and the built boost never reaches executors (GatedOnlineLDA
excludes it from pickles per ADR 0047) — so this module does not ride
--py-files. Pure logic, no Spark/BigQuery; unit-tested in
tests/scripts/test_profile_eta.py.

ORDER OF OPERATIONS, per credited node (normative; plan D1-D4):

  1. COVERAGE GATE (D4): ``min_coverage > 0`` and the node's coverage below it
     -> the node keeps a flat eta entirely (both polarities dropped).
  2. POSITIVE rows: map concept_id -> condition vocab index (unmapped concepts
     dropped and counted), then D3-NORMALIZE the positive weight vector to sum
     to ``strength * eta_base * v_condition`` added pseudo-mass. A vector
     summing to 0 (all idf-0) leaves the node with no positive boost (counted).
  3. NEGATIVE rows (D2's NOT downweight, applied HERE at fit-boost-build time,
     never at emit — the probe only carries the flag): AFTER the positives are
     normalized, each NOT concept's effective prior becomes
     ``max(0.1 * eta_base, 0.5 * (eta_base + pos_boost_at_concept))`` where
     pos_boost_at_concept is the normalized positive boost at that index (0 if
     none). The stored delta is ``effective - (eta_base + pos_boost)`` —
     negative. Normalize-positives-FIRST matters: the NOT rule is
     multiplicative in the (boosted) prior, so applying it before the D3 scale
     would downweight against un-normalized weights and the floor would bind
     at the wrong magnitude. The NOT row's own ``weight`` column is unused by
     design (D2: the downweight is a fixed multiplier, not weight-scaled — the
     flag is the information, which is why zero-weight neg rows are emitted).
  4. MERGE: one (idx, w) pair per topic, indices unique — a pos+neg concept's
     merged delta is ``effective - eta_base``; exact-zero merged deltas are
     dropped (the engine validator rejects zeros as caller bugs).
  5. TOPIC TARGETING (D1): the merged vector lands on the FIRST ``topics``
     topics of the node's block (default 1; the rest stay flat/free). The
     block layout is DagLayout's — the caller passes ``lay.block`` — which is
     the SAME sorted-engine-id node_order/block layout
     analysis/cloud/inspect_topics.py's ``topic_labels`` documents
     (test-pinned in tests/scripts/test_profile_eta.py).

EGRESS: the TSV's ``coverage`` column is train-derived and WORKSPACE-INTERNAL;
nothing here prints it row-wise, and the returned stats are counts of
nodes/concepts/rows only — safe for driver logs and the run manifest.
"""
from __future__ import annotations

import numpy as np


def not_effective_prior(eta_base: float, pos_boost_at: float, *,
                        mult: float = 0.5, floor_frac: float = 0.1) -> float:
    """Effective Dirichlet prior for a NOT (neg) concept (plan D2).

    ``max(floor_frac * eta_base, mult * (eta_base + pos_boost_at))``: a
    multiplicative downweight of the (possibly positively-boosted) prior,
    floored at a fraction of the FLAT eta so the parameter never approaches 0
    (Dirichlet params must stay strictly positive; "garnish, never zero").
    With the default ``mult=0.5`` and ``pos_boost_at >= 0`` the floor never
    binds (0.5 * (eta + p) >= 0.5 * eta > 0.1 * eta); it is the defensive
    guard that keeps any future multiplier below ``floor_frac`` — or a
    stacked adjustment — from emitting an improper prior, and the engine
    validator's eta + w > 0 check is sized against it.
    """
    if eta_base <= 0.0:
        raise ValueError(f"eta_base must be > 0, got {eta_base}")
    if pos_boost_at < 0.0:
        raise ValueError(
            f"pos_boost_at must be >= 0 (a normalized boost), got {pos_boost_at}")
    return max(floor_frac * eta_base, mult * (eta_base + pos_boost_at))


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


def build_profile_eta_boost(eta_df, *, eid_by_mondo, block_of,
                            vocab_index_by_concept, eta_base, v_condition,
                            strength=1.0, topics=1, min_coverage=0.0):
    """Build the sparse per-topic eta boost from the emitted prior table.

    Semantics and order of operations: see the module docstring (normative).

    Parameters
    ----------
    eta_df : pandas.DataFrame
        The ``--emit-eta`` table: columns mondo_id, concept_id, weight, neg
        (0/1), coverage. A concept may appear as BOTH a pos and a neg row for
        the same node; zero-weight rows exist and are legal.
    eid_by_mondo : dict[str, int]
        mondo_id (curie string) -> engine node id, for the nodes present in
        this run's label DAG (a node absent here is skipped and counted —
        unpowered/collapsed, expected).
    block_of : dict[int, list[int]]
        engine node id -> its topic-block indices, EXACTLY ``DagLayout.block``
        (sorted-engine-id order; the layout ``inspect_topics.topic_labels``
        documents). A credited node missing from it is a wiring bug and raises.
    vocab_index_by_concept : dict[int, int]
        OMOP standard concept_id -> CONDITION-domain (domain 0) vocab index —
        the bundle's ``vocab_maps[0]`` (or ``vocab_map`` single-domain).
        Membership is the survivorship test (leakage strip + min_df + cap).
    eta_base : float
        The fit's flat topic-word prior on the condition domain — the gated
        engine is constructed with ``eta = 1.0 / lay.K``
        (``_build_model_and_config``), so pass exactly that.
    v_condition : int
        The condition domain's vocabulary width (D3's V_condition).
    strength, topics, min_coverage :
        The three knobs (plan D3, D1, D4); defaults 1.0 / 1 / 0.0.

    Returns
    -------
    (boost, stats) : boost is ``{topic_index: (int64 idx array, float64 w
    array)}`` ready for ``setEtaBoost`` /
    ``gated_lda._resolve_eta_boost(..., eta=eta_base)``; stats is a dict of
    disclosure-safe COUNTS (nodes/concepts/rows — never coverage values, never
    patient counts) for the driver log and manifest.
    """
    eta_base = float(eta_base)
    if eta_base <= 0.0:
        raise ValueError(f"eta_base must be > 0, got {eta_base}")
    topics = int(topics)
    if topics < 1:
        raise ValueError(f"topics must be >= 1, got {topics}")
    target_mass = float(strength) * eta_base * int(v_condition)
    if target_mass <= 0.0:
        raise ValueError(
            f"strength * eta_base * v_condition must be > 0, got {target_mass}")

    mids = eta_df["mondo_id"].astype(str).to_numpy()
    cids = eta_df["concept_id"].astype("int64").to_numpy()
    weights = eta_df["weight"].astype("float64").to_numpy()
    negs = _neg_flags(eta_df["neg"].tolist())
    covs = eta_df["coverage"].astype("float64").to_numpy()

    stats = {
        "n_rows": int(len(eta_df)),
        "n_nodes_in_file": int(len(np.unique(mids))),
        "n_nodes_credited": 0,            # in the file AND in this run's DAG
        "n_nodes_skipped_not_in_dag": 0,
        "n_nodes_dropped_min_coverage": 0,
        "n_nodes_zero_positive": 0,       # mapped pos rows, all weight 0
        "n_nodes_boosted": 0,
        "n_node_concepts_unmapped": 0,    # (node, concept) pairs not in vocab
        "n_neg_concepts_applied": 0,
        "n_topics_boosted": 0,
        "nnz": 0,
        "target_mass_per_topic": target_mass,
    }

    boost: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mid in sorted(set(mids)):
        eid = eid_by_mondo.get(mid)
        if eid is None:
            stats["n_nodes_skipped_not_in_dag"] += 1
            continue
        stats["n_nodes_credited"] += 1
        block = block_of.get(int(eid))
        if block is None:
            raise ValueError(
                f"node {mid} (engine id {eid}) has no topic block in the "
                f"layout — eid_by_mondo and block_of disagree (wiring bug)")
        if topics > len(block):
            raise ValueError(
                f"topics={topics} exceeds the block width tpn={len(block)} "
                f"(node {mid}); the boost cannot target topics outside the "
                f"node's own block")
        rows = mids == mid
        # 1. coverage gate (D4). All of a node's rows carry the same coverage;
        #    max() is the defensive read. NEVER printed/returned per-node.
        if min_coverage > 0.0 and float(covs[rows].max()) < float(min_coverage):
            stats["n_nodes_dropped_min_coverage"] += 1
            continue

        # 2. positives: map -> vocab index, then D3-normalize.
        pos_w: dict[int, float] = {}
        for cid, w in zip(cids[rows & ~negs], weights[rows & ~negs]):
            idx = vocab_index_by_concept.get(int(cid))
            if idx is None:
                stats["n_node_concepts_unmapped"] += 1
                continue
            idx = int(idx)
            # Upstream already took max-over-terms per (node, concept,
            # polarity); a residual collision (two concepts on one index)
            # would be an assembler bug — keep the max defensively.
            if idx not in pos_w or w > pos_w[idx]:
                pos_w[idx] = float(w)
        pos_sum = float(sum(pos_w.values()))
        if pos_w and pos_sum <= 0.0:
            stats["n_nodes_zero_positive"] += 1
        pos_boost = ({i: w * (target_mass / pos_sum) for i, w in pos_w.items()
                      if w > 0.0}
                     if pos_sum > 0.0 else {})

        # 3. NOT downweights (multiplicative-with-floor, AFTER normalization;
        #    the neg row's weight column is unused by design — see docstring).
        merged = dict(pos_boost)
        for cid in np.unique(cids[rows & negs]):
            idx = vocab_index_by_concept.get(int(cid))
            if idx is None:
                stats["n_node_concepts_unmapped"] += 1
                continue
            idx = int(idx)
            effective = not_effective_prior(eta_base, pos_boost.get(idx, 0.0))
            merged[idx] = effective - eta_base   # = pos_delta + not_delta
            stats["n_neg_concepts_applied"] += 1

        # 4. merge/emit: unique indices; exact zeros dropped (validator
        #    rejects them — a zero delta IS "no boost here").
        merged = {i: w for i, w in merged.items() if w != 0.0}
        if not merged:
            continue
        idx_arr = np.array(sorted(merged), dtype=np.int64)
        w_arr = np.array([merged[i] for i in sorted(merged)], dtype=np.float64)

        # 5. D1: same vector on the block's first `topics` topics (fresh
        #    copies per topic — downstream owns its arrays).
        for t in block[:topics]:
            boost[int(t)] = (idx_arr.copy(), w_arr.copy())
        stats["n_nodes_boosted"] += 1
        stats["n_topics_boosted"] += topics
        stats["nnz"] += int(len(idx_arr)) * topics

    return boost, stats
