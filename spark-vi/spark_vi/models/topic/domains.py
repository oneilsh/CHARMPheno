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

`resolve_per_domain` is the matching validator for a PER-DOMAIN HYPERPARAMETER
(eta_m, omega_m, beta_prior_m): scalar-broadcast-or-one-per-domain. It exists
once because the three hand-rolled copies did not agree -- two dispatched on
``np.isscalar``, which is False for a 0-d ndarray, so ``eta=np.array(0.02)``
raised "TypeError: iteration over a 0-d array" from inside a list comprehension
instead of being treated as the scalar it is.

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


def resolve_per_domain(value, n_domains: int, name: str, *,
                       allow_zero: bool = False) -> np.ndarray:
    """Resolve a scalar-or-per-domain hyperparameter to a length-n_domains array.

    A SCALAR broadcasts to every domain (which with one domain is exactly the
    scalar, so the single-domain path keeps the caller's own value); a SEQUENCE
    must give one value per domain, in domain order.

    The scalar/sequence dispatch is on the resolved array's ``ndim``, NOT
    ``np.isscalar`` (False for a 0-d ndarray, which would then take the sequence
    branch and fail with an opaque "iteration over a 0-d array" TypeError), and
    iterables are materialized first so a one-shot iterator is not consumed
    twice. The VALUE check runs BEFORE the shape dispatch so neither branch can
    bypass it: a negative weight slipping through the scalar branch corrupts
    downstream arithmetic with no error raised.

    ``allow_zero=False`` (the default) requires strictly positive values --
    every Dirichlet concentration (eta_m, beta_prior_m) must be > 0 or the
    prior is improper. ``allow_zero=True`` additionally admits 0.0, for a
    pseudo-likelihood weight where 0 legitimately means "drop this domain".
    """
    if isinstance(value, np.ndarray):
        raw = value
    elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raw = value
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a number or a length-{n_domains} (n_domains) "
            f"sequence of numbers, got {value!r}") from exc
    bad_value = (np.any(arr < 0.0) if allow_zero else np.any(arr <= 0.0))
    if not np.all(np.isfinite(arr)) or bad_value:
        raise ValueError(
            f"{name} components must be finite and "
            f"{'>= 0' if allow_zero else '> 0'}, got {arr.tolist()}")
    if arr.ndim == 0:
        return np.full(n_domains, float(arr), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n_domains:
        raise ValueError(
            f"{name} must be a scalar or a length-{n_domains} (n_domains) "
            f"sequence, got shape {arr.shape}")
    return arr.astype(np.float64, copy=True)
