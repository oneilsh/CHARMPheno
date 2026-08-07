"""Numerical guards for real-corpus degeneracies the toy-bars oracle never hits.

The pi-inference and loss_x divide/log by ``M_DV = pi @ topics``. On a real sparse
corpus a vocabulary word can have ~0 probability under every (softmax) topic, so
``M_DV`` underflows to 0 and, unguarded, produces ``x/0 = inf`` -> NaN (observed on
the first AoU run, 33k docs). These tests pin the floors that keep it finite.
"""
import numpy as np

from analysis.pc.slda_reference import (
    calc_loss__slda,
    make_convex_alpha_minus_1,
    nef_map_pi_DK,
)


def _dead_word_setup():
    """topics/X where word 2 has zero mass in every topic but a doc has that token."""
    # K=2 topics on V=3 words; word index 2 is dead (0 prob under both topics).
    topics_KV = np.array([[0.5, 0.5, 0.0],
                          [0.9, 0.1, 0.0]])
    # doc 0 normal; doc 1's ONLY token is the dead word -> M[1,2] = 0.
    X_DV = np.array([[3.0, 1.0, 0.0],
                     [0.0, 0.0, 2.0]])
    return topics_KV, X_DV


def test_nef_map_pi_finite_with_dead_word():
    topics_KV, X_DV = _dead_word_setup()
    Pi = nef_map_pi_DK(topics_KV, X_DV, make_convex_alpha_minus_1(1.1),
                       pi_iters=50, pi_step_size=0.005)
    assert np.all(np.isfinite(Pi)), "pi inference produced non-finite values"
    # rows still (approximately) normalized
    assert np.allclose(Pi.sum(axis=1), 1.0, atol=1e-6)


def test_loss_finite_with_dead_word():
    topics_KV, X_DV = _dead_word_setup()
    y_DC = np.array([[1.0], [0.0]])
    w_CK = np.zeros((1, 2))
    loss = calc_loss__slda(topics_KV, w_CK, X_DV, y_DC,
                           pi_iters=50, weight_y=1.0)
    assert np.isfinite(loss), "loss_x went non-finite on a dead-word doc"
