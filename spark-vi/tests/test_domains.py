"""Unit tests for the shared multi-domain vocabulary helpers.

These three helpers replaced hand-rolled copies that disagreed with each other:
`domains_to_bounds` (two duplicated cumsum sites), `validate_domain_bounds` (one
validating caller, three silently-accepting ones) and `resolve_per_domain` (three
resolvers, only one of which handled a 0-d ndarray).
"""
import numpy as np
import pytest

from spark_vi.models.topic.domains import (
    domains_to_bounds, resolve_per_domain, validate_domain_bounds)


def test_domains_to_bounds_is_cumulative_offsets():
    np.testing.assert_array_equal(domains_to_bounds([4, 3, 2]), [0, 4, 7, 9])
    np.testing.assert_array_equal(domains_to_bounds([5]), [0, 5])
    assert domains_to_bounds([4, 3]).dtype == np.int64


def test_domains_to_bounds_rejects_empty_and_nonpositive_sizes():
    for bad in ([], [4, 0], [4, -1], [4.5, 3]):
        with pytest.raises(ValueError, match="domain"):
            domains_to_bounds(bad)


def test_validate_domain_bounds_none_is_the_single_pooled_domain():
    """None is not a special case: one domain spanning [0, V) makes every
    per-domain quantity equal its pooled counterpart."""
    np.testing.assert_array_equal(validate_domain_bounds(None, 6), [0, 6])
    np.testing.assert_array_equal(validate_domain_bounds([0, 6], 6), [0, 6])


def test_validate_domain_bounds_rejects_incomplete_or_disordered_cover():
    """The vocabulary must be covered exactly once. Each of these used to be
    accepted somewhere and produce a silent wrong answer instead of an error."""
    for bad in ([0, 3],                 # ends short of V: cols 3..5 uncovered
                [1, 6],                 # does not start at 0
                [0, 3, 3, 6],           # empty domain (not strictly increasing)
                [0, 4, 2, 6],           # decreasing
                [0, 6, 8],              # overshoots V
                [0],                    # fewer than two offsets
                [[0, 3], [3, 6]]):      # not 1-D
        with pytest.raises(ValueError, match="domain_bounds"):
            validate_domain_bounds(bad, 6)


def test_validate_domain_bounds_name_appears_in_the_error():
    with pytest.raises(ValueError, match="my_bounds"):
        validate_domain_bounds([0, 3], 6, name="my_bounds")


def test_resolve_per_domain_scalar_broadcasts_including_0d_ndarray():
    """np.isscalar(np.array(0.02)) is False, so the pre-consolidation resolvers
    took the SEQUENCE branch on a 0-d array and raised 'iteration over a 0-d
    array'. All scalar spellings must behave identically."""
    for val in (0.02, np.float64(0.02), np.array(0.02)):
        np.testing.assert_allclose(resolve_per_domain(val, 3, "eta"), [0.02] * 3)


def test_resolve_per_domain_sequence_forms_agree():
    for val in ([0.5, 0.2], (0.5, 0.2), np.array([0.5, 0.2]), iter([0.5, 0.2])):
        np.testing.assert_allclose(resolve_per_domain(val, 2, "eta"), [0.5, 0.2])


def test_resolve_per_domain_rejects_bad_values_in_every_input_form():
    """The value check runs BEFORE the scalar/sequence dispatch, so the scalar
    branch cannot bypass it."""
    for bad in (-0.5, np.array(-0.5), 0.0, float("nan"), float("inf"),
                [0.5, -0.5], (0.5, 0.0), np.array([0.5, float("nan")])):
        with pytest.raises(ValueError, match="eta"):
            resolve_per_domain(bad, 2, "eta")


def test_resolve_per_domain_allow_zero_admits_zero_but_not_negative():
    """A pseudo-likelihood weight may legitimately be 0 (drop the domain)."""
    np.testing.assert_allclose(
        resolve_per_domain(0.0, 2, "omega", allow_zero=True), [0.0, 0.0])
    np.testing.assert_allclose(
        resolve_per_domain([1.0, 0.0], 2, "omega", allow_zero=True), [1.0, 0.0])
    with pytest.raises(ValueError, match="omega"):
        resolve_per_domain([1.0, -1e-12], 2, "omega", allow_zero=True)


def test_resolve_per_domain_length_mismatch_and_rank_are_named_errors():
    with pytest.raises(ValueError, match="length-2"):
        resolve_per_domain([0.1, 0.2, 0.3], 2, "eta")
    with pytest.raises(ValueError, match="length-2"):
        resolve_per_domain(np.array([[0.1, 0.2]]), 2, "eta")
    with pytest.raises(ValueError, match="eta"):
        resolve_per_domain("nope", 2, "eta")


def test_resolve_per_domain_does_not_alias_a_caller_array():
    """The returned array is the model's own copy: mutating it must not write
    back through to the caller's array (or vice versa)."""
    src = np.array([0.5, 0.2])
    out = resolve_per_domain(src, 2, "eta")
    out[0] = 99.0
    assert src[0] == 0.5
