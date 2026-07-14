import numpy as np
from spark_vi.models.topic.pg_stm_dag import dag_offset_ridge
from spark_vi.models.topic.pg_stm_dag_gibbs import dag_offset_ridge_draw, PGSTMDagGibbs


def test_offset_draw_mean_is_the_ridge_and_covariance_is_matrix_normal():
    rng = np.random.default_rng(0)
    Pw, Ksm1 = 4, 3
    W = rng.standard_normal((200, Pw))
    M = rng.standard_normal((200, Ksm1))
    WtW, WtM = W.T @ W, W.T @ M
    penalty = np.array([1e-6, 0.5, 0.5, 0.5])
    Sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    ridge_mean = dag_offset_ridge(WtW, WtM, penalty=penalty)

    draws = np.array([dag_offset_ridge_draw(WtW, WtM, Sigma, penalty=penalty, rng=rng)
                      for _ in range(4000)])           # (4000, Pw, Ksm1)
    # (a) Monte-Carlo mean == the ridge point
    assert np.abs(draws.mean(axis=0) - ridge_mean).max() < 0.05
    # (b) row-covariance of a fixed column c == A^{-1} * Sigma[c, c]
    Ainv = np.linalg.inv(WtW + np.diag(penalty))
    col = 1
    emp_cov = np.cov(draws[:, :, col].T)
    assert np.abs(emp_cov - Ainv * Sigma[col, col]).max() < 0.05


from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.pg_stm_dag import DagGate
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_partial_label_arm_hides_the_leaf_behind_a_candidate_set():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 80
    dag = DagGate([(), (0,), (0,), (1,), (1,)])          # A=1 has two leaves A1=3, A2=4
    rng = np.random.default_rng(1)
    node_offsets = {u: rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, doc_candidates = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={3: 20},
        partial_label_plan={1: 30}, sigma_true=2.0 * np.eye(K - 1),
        doc_len=60, seed=3)
    assert len(docs) == len(doc_nodes) == len(doc_candidates) == 50
    # the 30 partial-label docs carry a 2-candidate set over A's leaves {3,4}, weights sum to 1
    partial = [c for c in doc_candidates if len(c) > 1]
    assert len(partial) == 30
    for c in partial:
        assert abs(sum(p for p, _ in c) - 1.0) < 1e-9
        assert {min(nodes) for _, nodes in c} == {3, 4}


def test_softgate_membership_favors_the_generating_leaf():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 80
    dag = DagGate([(), (0,), (0,), (1,), (1,)])
    rng = np.random.default_rng(4)
    node_offsets = {u: 3.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, doc_candidates = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={},
        partial_label_plan={1: 40}, sigma_true=1.0 * np.eye(K - 1), doc_len=120, seed=5)

    from spark_vi.models.topic.pg_stm import stick_layout
    from spark_vi.models.topic.pg_stm_dag import _softgate_estep_doc, offset_penalty, dag_offset_ridge
    lay = stick_layout(part)
    # a rough warm-start Cf from the planted offsets (Gamma=0 intercept; B=node offsets)
    Cf = np.zeros((1 + dag.n_offset_nodes, K - 1))
    for u in (1, 2, 3, 4):
        Cf[u] = node_offsets[u]
    log_beta = np.log(beta)
    hits = 0
    for doc, dn, cand in zip(docs, doc_nodes, doc_candidates):
        cands = [(p, nodes, "A") for p, nodes in cand]         # all under anchor A
        weights, z_bar, _ = _softgate_estep_doc(
            doc, cands, lay["groups"], log_beta, Cf, np.eye(K - 1), dag,
            K=K, B=lay["B"], inner_rounds=8, inner_tol=1e-3)
        true_leaf = min(dn)
        pred_leaf = min(cand[int(np.argmax(weights))][1])
        hits += (pred_leaf == true_leaf)
    assert hits / len(docs) > 0.7        # membership recovers the generating leaf > chance (0.5)


from tests._stm_synth import planted_recovery


def test_gibbs_recovers_a_planted_identified_increment():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])           # A has TWO leaves -> A1 increment identified
    rng = np.random.default_rng(7)
    node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 200, 3: 200, 4: 200, 2: 150}, sigma_true=1.0 * np.eye(K - 1),
        doc_len=80, seed=8)
    eng = PGSTMDagGibbs(K=K, V=V, partition=part, dag=dag, P=1, n_iter=120, burn=60,
                        lam_base=0.25, seed=0)
    out = eng.run(docs, cand, beta_init=beta)
    draws = out["increment_draws"]                        # (n_kept, U, Ksm1)
    assert draws.shape[1] == dag.n_offset_nodes
    # the A1 (node 3) increment posterior mean correlates with the planted increment
    b3 = draws[:, 3 - 1, :].mean(axis=0)
    assert np.corrcoef(b3, node_offsets[3])[0, 1] > 0.5
    # Sigma stays PD
    assert np.linalg.eigvalsh(out["Sigma"]).min() > -1e-8
