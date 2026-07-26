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


def test_dead_node_report_spares_a_node_alive_in_only_one_domain():
    """dead_node_report's cross-domain check is an OR: a node concentrated in
    ANY domain is alive, even if flat in every other domain. This distinguishes
    the OR from a (wrong) AND requiring concentration in EVERY domain -- a
    single-domain lam_dict (as in the test above) can't tell them apart, since
    with only one domain OR and AND agree. Node 1 is flat in domain 0 but
    concentrated in domain 1 -> must be spared. Node 2 is flat in BOTH domains
    -> must be reported (also proves the report is non-empty / selective, not
    just "everything passes")."""
    import numpy as np
    from multidomain_cloud import dead_node_report
    from spark_vi.models.topic.dag_placement import DagLayout

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    V0, V1 = 20, 20
    lam0 = np.full((lay.K, V0), 1.0)   # flat everywhere in domain 0
    lam1 = np.full((lay.K, V1), 1.0)   # flat everywhere in domain 1 by default

    # Node 1: flat in domain 0 (peak/mean == 1), concentrated in domain 1.
    # Same magnitude/shape as the single-domain test above (peak/mean ~6.67).
    for k in lay.block[1]:
        lam1[k] = 0.01
        lam1[k, :3] = 100.0

    dead = dead_node_report({0: lam0, 1: lam1}, lay, min_peak_ratio=5.0)
    assert 1 not in dead   # alive in domain 1 -> spared by the cross-domain OR
    assert 2 in dead       # flat in both domains -> genuinely dead
