"""The int64 doc key: one id per document that yields the person back (exp 0111 WP-A1).

Every row the readout identifies is a DOCUMENT; until exp 0111 a document was a
person (`doc_id == person_id`, an int64). A multi-document corpus makes a person
carry several documents, so the readout keys rows by one synthesized int64 that is
unique per document, recovers the person by arithmetic, and leaves single-document
corpora on their existing ids (scaled order-preservingly). The generic readout
(`spark_vi.readout`) only needs an int64 row id; this module is the clinical
definition of it, used driver-side (`dag_gated_fit._doc_key_column` builds the
same map as a Spark Column).

Until ADR 0051 this file was `distributed_readout.py` and also held the readout's
partition kernels and Spark passes; those moved to `spark_vi.readout`.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# The int64 doc-key seam (exp 0111 WP-A1, spec R5.4).                          #
# --------------------------------------------------------------------------- #
# Every row the readout stack identifies is a DOCUMENT, and until exp 0111 a
# document was a person: `doc_id == person_id`, an int64, and the whole eval
# stack (`lean_eval_kernel`, `densify_lean_blocks`, the A/B alignment dict)
# leans on that int64-ness — string doc_ids raise the moment they reach
# `np.asarray(ids, dtype=np.int64)`. Exp 0111 makes a person carry up to a few
# episode documents (`EpisodeDocSpec`, `doc_id = "cohort:person:index_date"`, a
# STRING), so the stack needs one synthesized int64 that (a) is unique per
# document, (b) yields the person back by pure arithmetic wherever person-grain
# logic needs it, and (c) leaves the pre-0111 single-doc corpora on EXACTLY the
# ids they have today (up to an order-preserving scale), so nothing about the
# byte-identical single-doc path moves.
#
#   doc_key = person_id * DOC_KEY_RADIX + episode_no
#
# with `0 <= episode_no < DOC_KEY_RADIX` the per-person, WITHIN-CORPUS document
# index. RADIX=64 is a power of two so the inverse is a shift/mask, and 64 is
# generous: exp 0111 caps a person at <= 3 documents (cap-5 fallback), so the
# used range is <= 5 << 64. Because episode_no occupies the low bits and is
# bounded, sorting by doc_key sorts by (person_id, episode_no) — order-
# preserving — and the person blocks `[p*64, p*64+64)` are disjoint, so keys
# never collide across persons. For every corpus that exists TODAY there is one
# document per person, episode_no is 0, and doc_key = person_id * 64: the
# current ids, scaled, and `person_of(doc_key) == person_id` exactly.
#
# COMPATIBILITY WITH WP-D1 (verified, and it is NOT a free identity — read this).
# WP-D1's `episode_index.py` emits an `episode_no` that is the ORIGINAL
# chronological ordinal of a person's episode (1-based, NOT re-ranked after the
# per-person cap — deliberately, so R7.5's ordinal drop-rate table and the
# 66.2% first-episode-kill decomposition keep meaning "how many-th new-diagnosis
# cluster is this"). That ordinal is UNBOUNDED — a chronic patient's 70th
# episode reports `episode_no = 70` even if the cap keeps only three of them —
# so it is NOT the value that may occupy these low 6 bits. The `episode_no` this
# key wants is the bounded WITHIN-CORPUS document index (0 .. cap-1) of the
# documents actually assembled for a person; WP-D2, when it wires the episode
# corpus, must hand this seam that bounded index (or, equivalently, a dense
# per-person `row_number()-1` over the kept documents), never D1's raw ordinal.
# `synthesize_doc_key` enforces the bound (episode_no < RADIX raises), and the
# uniqueness assertion at densify time (`spark_vi.readout.kernels.
# assert_unique_ids`) is the second tripwire: a person carrying more than
# RADIX documents, or D1's raw ordinal leaking through, shows up as a collision
# there rather than as a silently mis-attributed row. EpisodeDocSpec itself keys
# its string `doc_id` on `index_date`, not on any ordinal, so nothing downstream
# of the doc spec assumes the two `episode_no`s are the same object.
DOC_KEY_RADIX = 64


# Overflow guard: doc_key must stay inside int64. The largest key is
# `(person_id) * 64 + 63`, so `person_id < 2**57` keeps it below 2**63 - 1 with
# the low 6 bits to spare. OMOP person_ids are far below this (int32-range in
# every corpus this repo has assembled), so the guard is a tripwire on a
# malformed id or a mis-scaled key, never a real-corpus constraint.
DOC_KEY_MAX_PERSON_ID = 1 << 57


def synthesize_doc_key(person_id, episode_no=0):
    """`(person_id, episode_no) -> doc_key = person_id * RADIX + episode_no` (int64).

    THE one definition of the synthesis (spec R5.4). Pure, and vectorized: a
    Python int in gives a Python int out; numpy arrays in give an int64 array
    out — the driver builds the (D,) key array for a whole split through the same
    call the scalar tests exercise, so the two can never drift.

    Guards both operands because a violation is silent otherwise: a `person_id >=
    2**57` overflows int64 after the multiply and wraps into some other person's
    block; an `episode_no >= RADIX` (e.g. WP-D1's raw chronological ordinal —
    see the module comment) carries into the next person's block. Either is a
    cross-person collision, so both raise HERE, at synthesis, rather than as a
    mystery row later.
    """
    p = np.asarray(person_id)
    e = np.asarray(episode_no)
    if np.any(p < 0) or np.any(p >= DOC_KEY_MAX_PERSON_ID):
        raise ValueError(
            f"doc_key: person_id out of [0, 2**57); the overflow guard "
            f"(DOC_KEY_MAX_PERSON_ID) tripped — max seen "
            f"{int(np.max(p))}")
    if np.any(e < 0) or np.any(e >= DOC_KEY_RADIX):
        raise ValueError(
            f"doc_key: episode_no out of [0, {DOC_KEY_RADIX}); this is the "
            f"bounded WITHIN-CORPUS document index, NOT WP-D1's unbounded "
            f"chronological ordinal — max seen {int(np.max(e))}")
    key = p.astype(np.int64) * np.int64(DOC_KEY_RADIX) + e.astype(np.int64)
    return int(key) if key.ndim == 0 else key


def person_of(doc_key):
    """`doc_key -> person_id` — the inverse's person half (`doc_key // RADIX`).

    THE one place person-grain logic recovers the person from a row's key
    (calibration splits, per-person dedup, A/B alignment when it wants persons
    rather than documents). Scalar- and array-safe like `synthesize_doc_key`, so
    a caller never open-codes `// 64` and can never disagree with the radix.
    """
    return np.asarray(doc_key).astype(np.int64) // np.int64(DOC_KEY_RADIX) \
        if np.ndim(doc_key) else int(doc_key) // DOC_KEY_RADIX


def episode_of(doc_key):
    """`doc_key -> episode_no` — the inverse's episode half (`doc_key % RADIX`)."""
    return np.asarray(doc_key).astype(np.int64) % np.int64(DOC_KEY_RADIX) \
        if np.ndim(doc_key) else int(doc_key) % DOC_KEY_RADIX


