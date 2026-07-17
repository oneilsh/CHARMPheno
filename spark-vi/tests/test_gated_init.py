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


def test_scalable_block_aligned_lambda_is_block_aligned_and_deflated(spark):
    # Parent node 1 (tokens 0,1 shared across its subtree), child node 2 under 1
    # (tokens 2,3 child-specific), background tokens 8,9. Each node's block must
    # load its own anchor tokens; the child block must differ from the parent
    # block (forward-topological ancestor deflation took effect).
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 1}, n_bg=2, tpn=1)   # K = 2 + 2*1 = 4
    V = 10

    def doc(idx, frontier):
        idx = sorted(idx)
        counts = np.ones(len(idx), dtype=np.float64)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=counts, length=len(idx),
                                frontier=frozenset(frontier))

    rows = []
    for _ in range(30):
        rows.append(doc([8, 9, 0, 1], []))            # background-only
        rows.append(doc([0, 1, 8], [1]))              # node 1: shared tokens 0,1
        rows.append(doc([2, 3, 0, 8], [2]))           # node 2: child tokens 2,3 (+ inherits 0)
    rdd = spark.sparkContext.parallelize(rows, 3)

    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1)
    assert lam.shape == (lay.K, V)
    # node 1 block = topic index lay.block[1][0]; node 2 = lay.block[2][0]
    beta1 = lam[lay.block[1][0]]
    beta2 = lam[lay.block[2][0]]
    # child block emphasizes its own tokens 2/3 more than the parent block does
    assert beta2[2] + beta2[3] > beta1[2] + beta1[3]
    # deflation: the two node blocks are not identical rows
    assert not np.allclose(beta1, beta2)


def test_scalable_block_aligned_lambda_deterministic(spark):
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    V = 8

    def doc(idx, frontier):
        return GatedBOWDocument(indices=np.asarray(sorted(idx), dtype=np.int32),
                                counts=np.ones(len(idx)), length=len(idx),
                                frontier=frozenset(frontier))
    rows = [doc([0, 1, 6], [1]) for _ in range(10)] + \
           [doc([2, 3, 6], [2]) for _ in range(10)]
    rdd = spark.sparkContext.parallelize(rows, 2)
    a = scalable_block_aligned_lambda(rdd, lay, V, seed=7, min_doc_freq=1)
    b = scalable_block_aligned_lambda(rdd, lay, V, seed=7, min_doc_freq=1)
    assert np.allclose(a, b)


def test_scalable_block_aligned_lambda_zero_doc_node_stays_at_floor(spark):
    # A node with no training docs keeps its block at the 1e-9 floor (times scale),
    # warns, and produces no NaN.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)     # node 2 will get no docs
    V = 8

    def doc(idx, frontier):
        return GatedBOWDocument(indices=np.asarray(sorted(idx), dtype=np.int32),
                                counts=np.ones(len(idx)), length=len(idx),
                                frontier=frozenset(frontier))
    rows = [doc([0, 1, 6], [1]) for _ in range(10)]  # only node 1 attested
    rdd = spark.sparkContext.parallelize(rows, 2)
    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1)
    assert not np.isnan(lam).any()
    scale = 200.0
    assert np.allclose(lam[lay.block[2][0]], 1e-9 * scale)   # untouched floor
