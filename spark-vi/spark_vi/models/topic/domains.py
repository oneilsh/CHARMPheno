"""Shared multi-domain (multi-modality) vocabulary helpers.

A multi-domain topic model (MixEHR-style; Li, Nair, Lu et al. 2020, Nat. Commun.)
describes ONE concatenated vocabulary [domain 0 ; domain 1 ; ...] with two
interchangeable objects:

  * ``domains`` -- the per-domain vocabulary SIZES [V_0, V_1, ...], summing to V.
    What a model constructor takes, because a caller knows its per-domain sizes.
  * ``domain_bounds`` -- the cumulative OFFSETS [0, V_0, V_0+V_1, ..., V]. What
    every per-token computation takes, because token id w lives in domain
    ``searchsorted(bounds, w, "right") - 1``.

`domains_to_bounds` is the one conversion between them, and
`validate_domain_bounds` is the one validator: before it existed, only
`dag_placement.fit_gated` checked its bounds and every other consumer accepted a
malformed sequence and produced a SILENT WRONG ANSWER instead of an error --
`find_anchors(Q, 4, domain_bounds=[0, 3])` on a V=6 vocabulary returned 3 anchors
with columns 3..5 permanently ineligible, and `split_domains(beta, [0, 3])`
dropped columns 3..5 from the result.

Domain-agnostic: integer token ids and integer domain sizes only.
"""
from __future__ import annotations

import numpy as np


def domains_to_bounds(domains) -> np.ndarray:
    """Per-domain vocabulary sizes [V_0, V_1, ...] -> cumulative offsets
    [0, V_0, V_0+V_1, ..., V] as an int64 array.

    Every V_m must be a positive integer (an empty domain has no ids, so it can
    neither hold an anchor nor receive a lambda block; accepting one would make
    two adjacent bounds equal and break the strictly-increasing invariant the
    searchsorted domain lookup relies on).
    """
    sizes = np.asarray(list(domains))
    if sizes.ndim != 1 or sizes.size < 1:
        raise ValueError(
            f"domains must be a non-empty sequence of per-domain vocabulary "
            f"sizes, got {list(domains)!r}")
    if sizes.dtype.kind not in "iu" or np.any(sizes < 1):
        raise ValueError(
            f"every domain size must be a positive integer, got {sizes.tolist()}")
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


def validate_domain_bounds(domain_bounds, V: int, *,
                           name: str = "domain_bounds") -> np.ndarray:
    """Validate a cumulative domain-offset sequence against vocabulary size V.

    Returns an int64 array of the bounds. ``None`` means the single pooled
    domain and returns ``[0, V]`` -- not a special case: with one domain the
    per-domain quantities ARE the pooled quantities everywhere these bounds are
    consumed.

    Requires: strictly increasing, starting at 0, ending at exactly V -- i.e.
    the concatenated vocabulary is covered exactly once, with no gap, overlap or
    empty domain. Anything else is a caller error that otherwise degrades
    silently (columns past the last bound simply never participate).
    """
    raw = np.asarray([0, V] if domain_bounds is None else domain_bounds)

    def _reject():
        raise ValueError(
            f"{name} {raw.tolist()} must be strictly increasing offsets from 0 "
            f"to V={V} (the concatenated vocabulary must be covered exactly once)")

    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        _reject()
    if raw.dtype.kind == "f" and not (np.all(np.isfinite(raw))
                                      and np.all(raw == np.floor(raw))):
        _reject()
    bounds = raw.astype(np.int64)
    if bounds[0] != 0 or bounds[-1] != int(V) or np.any(np.diff(bounds) <= 0):
        _reject()
    return bounds
