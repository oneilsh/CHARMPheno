"""Tests for the multi-domain (two-domain) cloud driver: the arg surface and the
pure dead-node init-quality read. The end-to-end BQ+fit run is the cluster
smoke (make multidomain-bq-smoke); only parse_args + dead_node_report are unit
tested here (no Spark session required)."""


def test_parse_args_requires_seed_and_per_domain_vocab_controls():
    from multidomain_cloud import parse_args
    import pytest
    with pytest.raises(SystemExit):            # --seed required
        parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x"])
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x",
                    "--seed", "7", "--drug-vocab-size", "500"])
    assert a.seed == 7 and a.drug_vocab_size == 500


def test_parse_args_parses_omega_and_eta_per_domain_to_float_lists():
    from multidomain_cloud import parse_args
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x",
                    "--seed", "0", "--omega", "1.0,0.5",
                    "--eta-per-domain", "0.1,0.2"])
    assert a.omega == [1.0, 0.5]
    assert a.eta_per_domain == [0.1, 0.2]
    # unset -> None (reaches the engine as the pre-multi-domain default)
    b = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x",
                    "--seed", "0"])
    assert b.omega is None and b.eta_per_domain is None


def test_dead_node_report_flags_a_node_stuck_at_the_prior():
    """A node whose per-domain topic never rose off the ~uniform prior is dead;
    a node with concentrated mass is not."""
    import numpy as np
    from multidomain_cloud import dead_node_report
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    V = 20
    # node 1's topic concentrated on 3 ids; node 2's flat (dead).
    lam = {0: np.full((lay.K, V), 1.0)}
    for k in lay.block[1]:
        lam[0][k] = 0.01
        lam[0][k, :3] = 100.0
    dead = dead_node_report({0: lam[0]}, lay, min_peak_ratio=5.0)
    assert 2 in dead and 1 not in dead
