"""spark_vi.eval.topic.sharpness: the support fraction exp(H(β)) / V (spec R14.1)."""
import numpy as np

from spark_vi.eval.topic.sharpness import lambda_domains, support_fraction


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
    frac0, _ = support_fraction(np.zeros((1, V)))          # an all-zero row: 1 / V
    np.testing.assert_allclose(frac0[0], 1 / V)


def test_the_dominant_domain_decides():
    """A flat but light row in one domain does not make a sharp, heavy topic flat;
    a topic heavy in its flat domain is flat."""
    a = np.full((2, 50), 1.0)                    # flat; 50 mass per topic
    b = np.zeros((2, 10))
    b[0, 0] = 500.0                              # topic 0: sharp and heavy in b
    b[1, :] = 0.5                                # topic 1: flat and light in b
    frac, mass = support_fraction({0: a, 1: b})
    assert frac[0] < 0.2
    np.testing.assert_allclose(frac[1], 1.0)     # dominant = a (50 > 5), flat
    np.testing.assert_allclose(mass, [550.0, 55.0])


def test_domain_keys_order_as_ints():
    doms = lambda_domains({"10": np.ones((1, 3)), "2": np.ones((1, 5))})
    assert [d.shape[1] for d in doms] == [5, 3]
