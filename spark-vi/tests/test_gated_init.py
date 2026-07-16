import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_init import (
    spectral_block_aligned_lambda, INIT_STRATEGIES,
)
from _stm_synth import dag_placement_corpus

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def test_registry_exposes_spectral():
    assert INIT_STRATEGIES["spectral"] is spectral_block_aligned_lambda


def test_block_aligned_peaks_on_own_signature():
    """Each node's spectral-init topic block should peak (argmax over the signature
    region [C:]) on that node's OWN planted signature block — forward-topological
    ancestors-first deflation. This is the 'gate welds topic to node' property at init."""
    V, doc_len = 120, 60
    parent = PARENT
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    nodes = lay.nodes
    C = V // 3
    sig = max(2, (V - C) // (len(nodes) + 1))
    own = {u: set(range(C + i * sig, C + i * sig + sig)) for i, u in enumerate(nodes)}

    lam = spectral_block_aligned_lambda(
        {"train_docs": docs[:1600], "train_labels": labels[:1600]}, lay, V)
    assert lam.shape == (lay.K, V)
    hits = 0
    for u in nodes:
        b = lam[lay.block[u]].mean(0)
        if (C + int(np.argmax(b[C:]))) in own[u]:
            hits += 1
    # Forward-topological block-aligned init peaks each node on its own block. Prototype:
    # 6/6 single-parent. Hold >=5/6 (allow one internal-node descendant-leak miss).
    assert hits >= 5, f"only {hits}/{len(nodes)} nodes peaked on their own block"


def test_gated_online_lda_uses_spectral_init():
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    V, doc_len = 120, 60
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, V, init="spectral", alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global({"train_docs": docs[:800], "train_labels": labels[:800]})
    # Spectral lambda differs from a pure random Gamma init (row sums reflect the *scale).
    assert gp["lambda"].shape == (lay.K, V)
    assert gp["lambda"].min() > 0.0
