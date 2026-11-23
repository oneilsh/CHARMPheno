"""spark_vi.eval.topic.sharpness: the support fraction exp(H(β)) / V (spec R14.1)."""
import numpy as np

from spark_vi.eval.topic.sharpness import (lambda_domains, support_fraction,
                                           support_fraction_by_domain)


def test_sharp_row_reads_near_zero_and_a_flat_row_reads_one():
    V = 100
    sharp = np.full(V, 0.01)
    sharp[:3] = 50.0
    flat = np.full(V, 0.3)                       # the symmetric prior: uniform
    frac, mass = support_fraction(np.stack([sharp, flat]))
    assert frac[0] < 0.05
    np.testing.assert_allclose(frac[1], 1.0)
    np.testing.assert_allclose(mass, [sharp.sum(), flat.sum()])


def test_effective_support_counts_equally_weighted_terms():
    """A row uniform over m of V terms has exp(H) = m, so the fraction is m / V."""
    V, m = 40, 10
    row = np.zeros(V)
    row[:m] = 2.0
    frac, _ = support_fraction(row[None, :])
    np.testing.assert_allclose(frac[0], m / V)
    frac0, _ = support_fraction(np.zeros((1, V)))          # an all-zero row holds
    np.testing.assert_allclose(frac0[0], 1.0)              # nothing: flat


def test_the_sharpest_domain_decides():
    """A topic is flat only when flat in EVERY domain: a sharp row anywhere (even in
    its lighter domain) makes it sharp; a flat prior-level row never makes it flat."""
    a = np.full((3, 50), 1.0)                    # flat; 50 mass per topic
    b = np.full((3, 10), 0.5)                    # flat
    b[0, 0] = 500.0                              # topic 0: sharp and heavy in b
    b[1, 0] = 20.0                               # topic 1: sharp but LIGHT in b
    frac, mass = support_fraction({0: a, 1: b})
    assert frac[0] < 0.2 and frac[1] < 0.5       # the dominant rule called 1 flat
    np.testing.assert_allclose(frac[2], 1.0)     # flat everywhere
    np.testing.assert_allclose(mass, [554.5, 74.5, 55.0])


def test_by_domain_values_and_data_mass():
    a = np.full((2, 4), 0.25)                    # prior only (eta 0.25 * V 4 = 1)
    a[0, 0] += 9.0                               # 9 tokens on one term
    b = np.full((2, 2), 0.5)                     # prior only (eta 0.5 * V 2 = 1)
    frac, mass, data = support_fraction_by_domain({0: a, 1: b}, eta=[0.25, 0.5])
    assert frac.shape == mass.shape == data.shape == (2, 2)
    np.testing.assert_allclose(data, [[9.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(frac[1], [1.0, 1.0])
    assert frac[0, 0] < 0.5
    _, _, none = support_fraction_by_domain({0: a, 1: b})
    assert none is None
    _, _, d1 = support_fraction_by_domain(a, eta=0.25)     # scalar eta, one domain
    np.testing.assert_allclose(d1[:, 0], [9.0, 0.0])


def test_domain_keys_order_as_ints():
    doms = lambda_domains({"10": np.ones((1, 3)), "2": np.ones((1, 5))})
    assert [d.shape[1] for d in doms] == [5, 3]
