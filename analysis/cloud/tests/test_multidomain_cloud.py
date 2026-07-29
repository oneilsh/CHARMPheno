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


def test_parse_args_top_n_tokens_defaults_and_settable():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).top_n_tokens == 8            # default
    assert parse_args(base + ["--top-n-tokens", "0"]).top_n_tokens == 0  # disable


def test_topic_block_labels_bg_then_node_blocks():
    from multidomain_cloud import _topic_block_labels
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)         # K = 2 bg + 2 nodes*1 = 4
    labels = _topic_block_labels(lay, {1: "diabetes", 2: "eds"}, n_bg=2)
    assert labels[:2] == ["bg", "bg"]                    # background block
    # each node's topic block carries the node name; id fallback when unnamed.
    node_topic = {u: list(lay.block[u]) for u in lay.nodes}
    for k in node_topic[1]:
        assert labels[k] == "diabetes"
    for k in node_topic[2]:
        assert labels[k] == "eds"


def test_idx_to_name_inverts_vocab_and_falls_back_to_cid():
    from multidomain_cloud import _idx_to_name
    vocab = {201820: 0, 4048098: 1, 999: 2}              # {concept_id: index}
    names = {201820: "Diabetes", 4048098: "Neuropathy"}  # 999 has no name
    out = _idx_to_name(vocab, names)
    assert out == {0: "Diabetes", 1: "Neuropathy", 2: "999"}


def test_log_topics_orders_by_total_mass_and_maps_names(capsys):
    """_log_topics prints heaviest-first across both domains and resolves token
    indices to concept names. Topic 1 is the data-rich topic (big mass in domain
    0), so it must print before topic 0."""
    import numpy as np
    from multidomain_cloud import _log_topics
    K, Va, Vb = 2, 4, 3
    lam0 = np.full((K, Va), 0.01)
    lam0[1, 0] = 500.0                                   # topic 1 heavy in domain 0
    lam1 = np.full((K, Vb), 0.01)
    lam1[0, 2] = 5.0                                     # topic 0 mild in domain 1
    idx2name = {0: {0: "Metformin-cond", 1: "b", 2: "c", 3: "d"},
                1: {0: "e", 1: "f", 2: "Insulin-drug"}}
    order = _log_topics({0: lam0, 1: lam1}, idx2name, ["bg", "diabetes"], top_n=2,
                        domain_tags={0: "cond", 1: "drug"})
    assert order == [1, 0]                               # heaviest (topic 1) first
    out = capsys.readouterr().out
    assert "Metformin-cond" in out and "Insulin-drug" in out
    assert "cond:" in out and "drug:" in out
    assert "[          diabetes]" in out or "diabetes" in out


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


def test_parse_args_domains_defaults_to_drug_era_and_splits_a_list():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).domains == ["drug_era"]                     # default
    a = parse_args(base + ["--domains", "drug_era,observation"])
    assert a.domains == ["drug_era", "observation"]


def test_parse_args_svi_schedule_defaults_full_batch_and_settable():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    a = parse_args(base)
    assert a.mini_batch_fraction == 0.0         # default = full-batch
    assert a.learning_rate_tau0 == 1.0 and a.learning_rate_kappa == 0.7
    b = parse_args(base + ["--mini-batch-fraction", "0.1",
                           "--learning-rate-tau0", "10.0",
                           "--learning-rate-kappa", "0.7"])
    assert b.mini_batch_fraction == 0.1
    assert b.learning_rate_tau0 == 10.0 and b.learning_rate_kappa == 0.7


def test_parse_args_window_mode_and_lookback_knobs():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).window_mode == "forward"                   # default
    a = parse_args(base + ["--window-mode", "lookback",
                           "--lookback-days", "1825", "--label-window-days", "365"])
    assert a.window_mode == "lookback"
    assert a.lookback_days == 1825 and a.label_window_days == 365


def test_domain_vocab_spec_selects_the_right_arg_group():
    from multidomain_cloud import parse_args, _domain_vocab_spec
    a = parse_args(["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0",
                    "--cond-vocab-size", "5000", "--drug-vocab-size", "2000",
                    "--obs-vocab-size", "1500"])
    assert _domain_vocab_spec(a, "condition_era").vocab_size == 5000
    assert _domain_vocab_spec(a, "drug_era").vocab_size == 2000
    assert _domain_vocab_spec(a, "observation").vocab_size == 1500


def test_parse_args_rejects_unregistered_source_table_cond():
    """--source-table-cond condition_occurrence is a valid OMOP load source but
    is NOT in DOMAIN_REGISTRY (only condition_era is a registered condition
    source); main() does an unguarded DOMAIN_REGISTRY[cond_table] lookup, so
    parse_args must catch this with a clean p.error (SystemExit) instead of
    letting a raw KeyError surface deep in main(). A registered value
    (condition_era) must still be accepted."""
    from multidomain_cloud import parse_args
    import pytest
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--source-table-cond", "condition_occurrence"])
    a = parse_args(base + ["--source-table-cond", "condition_era"])
    assert a.source_table_cond == "condition_era"


def test_domain_registry_maps_source_tables_to_date_cols_and_names():
    from multidomain_cloud import DOMAIN_REGISTRY
    assert DOMAIN_REGISTRY["condition_era"]["date_col"] == "condition_era_start_date"
    assert DOMAIN_REGISTRY["drug_era"]["date_col"] == "drug_era_start_date"
    assert DOMAIN_REGISTRY["observation"]["date_col"] == "observation_date"
    assert DOMAIN_REGISTRY["observation"]["name"] == "observation"


def test_test_persist_cols_is_person_features_frontier():
    from multidomain_cloud import _test_persist_cols
    assert _test_persist_cols(["features_0", "features_1", "features_2"]) == \
        ["person_id", "features_0", "features_1", "features_2", "frontier"]


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


def test_vocab_vocabulary_tally_empty_vocab_is_empty_dict():
    # Empty vocab_map must short-circuit BEFORE touching spark (spark=None here
    # would blow up on any real spark.read call).
    from multidomain_cloud import _vocab_vocabulary_tally
    assert _vocab_vocabulary_tally(None, "p.d", "b", {}) == {}


def test_parse_args_obs_exclude_vocab_defaults_empty_and_parses_a_list():
    from multidomain_cloud import parse_args
    base = ["--cdr", "p.d", "--billing", "b", "--out-dir", "/x", "--seed", "0"]
    assert parse_args(base).obs_exclude_vocab == ()          # default = no strip
    a = parse_args(base + ["--obs-exclude-vocab", "PPI"])
    assert a.obs_exclude_vocab == ("PPI",)
    b = parse_args(base + ["--obs-exclude-vocab", "PPI,SNOMED"])
    assert b.obs_exclude_vocab == ("PPI", "SNOMED")
