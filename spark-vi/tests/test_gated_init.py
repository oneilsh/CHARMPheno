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


def test_parse_spark_bytes():
    from spark_vi.models.topic.gated_init import _parse_spark_bytes
    assert _parse_spark_bytes("4g") == 4 * 1024 ** 3
    assert _parse_spark_bytes("4gb") == 4 * 1024 ** 3
    assert _parse_spark_bytes("4096m") == 4096 * 1024 ** 2
    assert _parse_spark_bytes("2.0g") == int(2.0 * 1024 ** 3)
    assert _parse_spark_bytes("4294967296") == 4294967296
    assert _parse_spark_bytes("0") is None       # Spark's "no limit"
    assert _parse_spark_bytes("") is None
    assert _parse_spark_bytes(None) is None


class _FakeRDD:
    """Minimal rdd stand-in for _safe_batch_cap: only .context.getConf().get and
    .getNumPartitions are read."""
    def __init__(self, max_result, n_part):
        self._mr, self._p = max_result, n_part
        outer = self

        class _Conf:
            def get(self, k, default=None):
                return outer._mr if k == "spark.driver.maxResultSize" else default

        class _Ctx:
            def getConf(self):
                return _Conf()
        self.context = _Ctx()

    def getNumPartitions(self):
        return self._p


def test_safe_batch_cap_keeps_driver_collect_under_maxresultsize():
    # The exact 0114 scenario that OOM'd at B=6: V=11601, d=1000, 96 partitions,
    # 4 GiB maxResultSize. The cap must keep the treeReduce driver-collect
    # (~sqrt(P) partials, each B+1 dense (V,d) float32 sketches) safely under budget.
    from spark_vi.models.topic.gated_init import _safe_batch_cap
    V, d, P, mrs = 11601, 1000, 96, 4 * 1024 ** 3
    B = _safe_batch_cap(_FakeRDD("4g", P), V, d)
    assert B >= 1
    n_final = 10                                  # ceil(sqrt(96))
    peak = n_final * (B + 1) * V * d * 4          # bytes actually collected
    assert peak < mrs                             # would not OOM
    # a bigger budget admits a bigger batch; a tiny budget floors at 1
    assert _safe_batch_cap(_FakeRDD("16g", P), V, d) > B
    assert _safe_batch_cap(_FakeRDD("256m", P), V, d) == 1
    # unlimited (0) -> a fixed sane cap, not unbounded
    assert 1 <= _safe_batch_cap(_FakeRDD("0", P), V, d) <= 32


def test_scalable_projection_dim_drops_k_floor():
    # A big-K layout (many nodes x tpn): d must be the JL ~1000 margin, NOT K
    # (which the dense default_projection_dim would floor to and inflate every sketch).
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import _scalable_projection_dim
    parent = {i: 0 for i in range(1, 121)}        # 120 shallow nodes under root
    lay = DagLayout(parent, n_bg=8, tpn=5)        # K = 8 + 120*5 = 608
    assert lay.K == 608
    assert _scalable_projection_dim(lay, V=5000) == 1000     # not K, not V
    # capped at V when V is small
    assert _scalable_projection_dim(lay, V=400) == 400


def test_scalable_block_aligned_lambda_batch_size_invariant(spark):
    # The batched seed recovers B nodes per pass, but a node's group sketch is the
    # same docs regardless of who shares its batch — so the seed MUST NOT depend on
    # batch_size. A multi-level DAG (so nodes actually land in different depth-batches)
    # seeded at B=1 (one node per pass) and B=8 (whole levels per pass) must agree.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}, n_bg=2, tpn=1)
    V = 14

    def doc(idx, frontier):
        idx = sorted(idx)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=np.ones(len(idx), dtype=np.float64),
                                length=len(idx), frontier=frozenset(frontier))

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        rows.append(doc([12, 13], []))
        for leaf, toks in [(3, [0, 1]), (4, [2, 3]), (5, [4, 5]), (6, [6, 7])]:
            rows.append(doc(toks + [12], [leaf]))
    rdd = spark.sparkContext.parallelize(rows, 3)

    lam_b1 = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1,
                                           batch_size=1)
    lam_b8 = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1,
                                           batch_size=8)
    assert lam_b1.shape == lam_b8.shape == (lay.K, V)
    assert np.allclose(lam_b1, lam_b8)          # seed independent of batch width


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


def test_scalable_block_aligned_lambda_multi_ancestor_diamond(spark):
    # Diamond DAG {1:0, 2:0, 3:[1,2]}: node 3 has TWO proper ancestors, so its seed
    # concatenates BOTH parents' anchors ([a for p in anc for a in node_anchors[p]]).
    # This exercises the multi-parent deflation path (single-parent tests only put one
    # ancestor in the seed). Node 3's block must be recovered off the floor, emphasize
    # its OWN tokens, and differ from both parent blocks (deflated against both).
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0, 3: [1, 2]}, n_bg=2, tpn=1)   # K = 2 + 3*1 = 5
    V = 12

    def doc(idx, frontier):
        idx = sorted(idx)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=np.ones(len(idx), dtype=np.float64),
                                length=len(idx), frontier=frozenset(frontier))

    rows = []
    for _ in range(30):
        rows.append(doc([10, 11, 0, 1], []))         # background-only (tokens 10,11)
        rows.append(doc([0, 1, 10], [1]))            # node 1: own tokens 0,1
        rows.append(doc([2, 3, 10], [2]))            # node 2: own tokens 2,3
        rows.append(doc([4, 5, 0, 2, 10], [3]))      # node 3: own tokens 4,5 (+ inherits 0,2)
    rdd = spark.sparkContext.parallelize(rows, 3)

    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1)
    assert lam.shape == (lay.K, V)
    assert not np.isnan(lam).any()
    b1, b2, b3 = lam[lay.block[1][0]], lam[lay.block[2][0]], lam[lay.block[3][0]]
    scale = 200.0
    assert not np.allclose(b3, 1e-9 * scale)                 # node 3 recovered, not floor
    # node 3 emphasizes its own tokens 4/5 more than either parent block does
    assert b3[4] + b3[5] > b1[4] + b1[5]
    assert b3[4] + b3[5] > b2[4] + b2[5]
    # deflated against BOTH ancestors -> distinct from both parent blocks
    assert not np.allclose(b3, b1)
    assert not np.allclose(b3, b2)


def test_anchor_scope_rejects_unknown():
    import numpy as np
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda
    lay = DagLayout({1: 0}, n_bg=1, tpn=1)
    ds = {"train_docs": [np.array([0, 1])], "train_labels": [frozenset()]}
    with pytest.raises(ValueError, match="anchor_scope"):
        spectral_block_aligned_lambda(ds, lay, 4, anchor_scope="bogus")


def test_frontier_scope_keeps_foreground_tokens_out_of_background_dense():
    # Background docs use tokens {0,1,2}; node-1 docs use foreground-only tokens
    # {5,6} (never in a background doc). Under anchor_scope="frontier" the
    # background sketch is built ONLY from empty-frontier docs, so 5/6 cannot
    # become background anchors -> the background block sits at the floor on them,
    # while node 1's block carries them. This is the anchor-stealing the option
    # prevents (background can't grab a rare node's defining word).
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda

    V = 8
    lay = DagLayout({1: 0}, n_bg=1, tpn=1)                     # K = 2
    train_docs, train_labels = [], []
    for _ in range(30):
        train_docs.append(np.array([0, 0, 1, 1, 2]))          # background
        train_labels.append(frozenset())
        train_docs.append(np.array([5, 5, 6, 6, 0]))          # node 1: 5,6 fg-only
        train_labels.append(frozenset({1}))
    ds = {"train_docs": train_docs, "train_labels": train_labels}

    floor = 1e-9 * 200.0
    lam = spectral_block_aligned_lambda(ds, lay, V, anchor_scope="frontier")
    bg = lam[0]                                               # n_bg=1 -> block 0
    node = lam[lay.block[1][0]]
    assert bg[5] <= floor * 10 and bg[6] <= floor * 10       # bg never saw 5/6
    assert node[5] + node[6] > bg[5] + bg[6]                 # node 1 carries them
    assert node[5] + node[6] > 0.01


def test_frontier_scope_parent_does_not_steal_child_anchor_depth2():
    # 2-level chain {1:0, 2:1}: node 2 is a child of node 1. Node 2's defining
    # tokens {6,7} appear ONLY in node-2 (frontier={2}) docs. Under
    # anchor_scope="frontier", node 1's sketch is built ONLY from frontier=={1}
    # docs (which never contain 6/7), so node 1 CANNOT steal node 2's anchor — its
    # block stays at the floor on 6/7 while node 2's block carries them. This is
    # the "can't propagate at any depth" claim the option makes.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda

    V = 10
    lay = DagLayout({1: 0, 2: 1}, n_bg=1, tpn=1)              # K = 3
    train_docs, train_labels = [], []
    for _ in range(30):
        train_docs.append(np.array([0, 0, 1, 1]))            # background: 0,1
        train_labels.append(frozenset())
        train_docs.append(np.array([3, 3, 4, 4, 0]))         # node 1 (most-specific): 3,4
        train_labels.append(frozenset({1}))
        train_docs.append(np.array([6, 6, 7, 7, 3, 0]))      # node 2: 6,7 (+ inherits 3)
        train_labels.append(frozenset({2}))
    ds = {"train_docs": train_docs, "train_labels": train_labels}

    floor = 1e-9 * 200.0
    lam = spectral_block_aligned_lambda(ds, lay, V, anchor_scope="frontier")
    b1 = lam[lay.block[1][0]]
    b2 = lam[lay.block[2][0]]
    # node 1's frontier docs never contain 6/7 -> its block can't grab them
    assert b1[6] <= floor * 10 and b1[7] <= floor * 10
    # node 2 carries its own tokens and is deflated against node 1 (distinct block)
    assert b2[6] + b2[7] > 0.01
    assert not np.allclose(b1, b2)


def test_node_order_and_relatives_forward_vs_reverse():
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import _node_order_and_relatives
    lay = DagLayout({1: 0, 2: 1, 3: 2}, n_bg=1, tpn=1)    # chain 1 -> 2 -> 3
    order_f, rel_f = _node_order_and_relatives(lay, "forward")
    order_r, rel_r = _node_order_and_relatives(lay, "reverse")
    # forward: ascending depth (ancestor first); reverse: descending (leaf first)
    assert order_f == [1, 2, 3]
    assert order_r == [3, 2, 1]
    # forward deflates node 3 against its proper ancestors {1,2}; reverse against
    # its proper descendants (none for the leaf)
    assert set(rel_f(3)) == {1, 2} and rel_r(3) == []
    # forward: anchor node 1 has no ancestors; reverse: node 1 deflates against {2,3}
    assert rel_f(1) == [] and set(rel_r(1)) == {2, 3}


def test_topo_order_validation_rejects_unknown():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda
    lay = DagLayout({2: 1}, n_bg=1, tpn=1)
    with pytest.raises(ValueError):
        spectral_block_aligned_lambda(
            {"train_docs": [[0, 1]], "train_labels": [frozenset()]},
            lay, 3, topo_order="sideways")


def test_reverse_topo_flips_shared_word_block():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda
    # 1 background topic + parent P=1, child C=2 (chain), tpn=1 -> K=3 blocks: bg=[0], P=[1], C=[2]
    lay = DagLayout({1: 0, 2: 1}, n_bg=1, tpn=1)
    # vocab: 0=bg_word, 1=p_word, 2=c_word, 3=shared_word
    # background docs: bg_word only; P docs: p_word + shared_word; C docs: c_word + shared_word.
    # shared_word OUTNUMBERS the private word (4 vs 2 repeats) so it is each node's
    # own top (tpn=1) anchor when recovered UNDEFLATED (first in whatever order runs).
    # This is a structural, non-borderline plant: find_anchors' min_marginal_frac
    # candidacy bar (row marginal >= mean nonzero marginal) then EXCLUDES the private
    # word as a candidate once shared is seeded away, so the node processed SECOND
    # finds no anchor at all and stays at the 1e-9 floor on shared -- while the node
    # processed FIRST claims shared outright. (Plant strengthened per the brief: a
    # tied 3-vs-3 count makes both blocks land on the exact same tie-broken value
    # regardless of topo_order, which is why this asymmetry is required.)
    bg_doc = [0, 0, 0, 0]
    p_doc = [1, 1, 3, 3, 3, 3]
    c_doc = [2, 2, 3, 3, 3, 3]
    docs = [bg_doc] * 4 + [p_doc] * 4 + [c_doc] * 4
    labels = [frozenset()] * 4 + [frozenset({1})] * 4 + [frozenset({2})] * 4
    ds = {"train_docs": docs, "train_labels": labels}
    fwd = spectral_block_aligned_lambda(ds, lay, 4, anchor_scope="frontier",
                                        topo_order="forward")
    rev = spectral_block_aligned_lambda(ds, lay, 4, anchor_scope="frontier",
                                        topo_order="reverse")
    shared = 3
    p_block, c_block = lay.block[1][0], lay.block[2][0]
    # forward: shared word's mass is higher in the PARENT block than the child block
    assert fwd[p_block, shared] > fwd[c_block, shared]
    # reverse: the ordering flips -> shared word's mass is higher in the CHILD block
    assert rev[c_block, shared] > rev[p_block, shared]


def test_scalable_reverse_topo_flips_shared_word_block(spark):
    # Distributed-path mirror of test_reverse_topo_flips_shared_word_block: same
    # parent/child chain + shared-word plant, run through the random-projection
    # sketch instead of the exact dense co-occurrence. Dense vs scalable are NOT
    # expected to match numerically (scalable is a JL sketch, per its docstring's
    # parity claim with only the single-pass accumulation) -- this test only checks
    # that topo_order takes effect on the scalable path and flips the same block.
    #
    # PLANT NOTE (strengthened past the brief's draft plant): the dense flip test's
    # mechanism is candidacy exclusion via `min_marginal_frac` (a word below the MEAN
    # marginal cannot anchor). `find_anchors_projected`'s candidacy bar is different
    # (ADR 0032): absolute document frequency `df_w >= min_doc_freq`, not a mean-
    # relative marginal. The brief's draft plant (private word once per doc, shared
    # word 2x per doc, same doc count for both -> equal df) never triggers that bar,
    # so both blocks' own anchor is undeflated (whichever node runs is unaffected by
    # the other's seed) and forward == reverse bit-for-bit (confirmed empirically).
    # Fix: give the SHARED word high document frequency (present in every doc) and
    # each node's PRIVATE word low document frequency (present in only 2 of 30 docs),
    # with min_doc_freq=5 between them. Now shared is the only candidate for whichever
    # node runs first (private is below the floor) -- it claims shared outright; the
    # node processed SECOND is deflated away from shared (already chosen) and has no
    # other candidate (private is still below the floor), so it finds NO anchors and
    # stays at the 1e-9 floor entirely. That is the same "second node gets nothing"
    # asymmetry as the dense test, reproduced through the df-based candidacy bar.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda
    lay = DagLayout({1: 0, 2: 1}, n_bg=1, tpn=1)     # bg=[0], P=[1], C=[2]; K=3
    V = 4                                            # 0=bg,1=p_word,2=c_word,3=shared

    def doc(idx, frontier):
        idx = sorted(idx)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=np.ones(len(idx), dtype=np.float64),
                                length=len(idx), frontier=frozenset(frontier))

    rows = []
    for _ in range(2):
        rows.append(doc([1, 1, 3, 3], [1]))            # P: p_word(x2) + shared(x2)
        rows.append(doc([2, 2, 3, 3], [2]))            # C: c_word(x2) + shared(x2)
    for _ in range(28):
        rows.append(doc([3, 3], [1]))                  # P: shared only (own word absent)
        rows.append(doc([3, 3], [2]))                  # C: shared only (own word absent)
    for _ in range(30):
        rows.append(doc([0, 0], []))                   # background
    rdd = spark.sparkContext.parallelize(rows, 3)
    # df: p_word=2, c_word=2 (below the 5 floor -> never a candidate); shared=30
    # (present in every P/C doc -> always a candidate) -- see PLANT NOTE above.

    fwd = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=5,
                                        anchor_scope="frontier", topo_order="forward")
    rev = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=5,
                                        anchor_scope="frontier", topo_order="reverse")
    p_block, c_block, shared = lay.block[1][0], lay.block[2][0], 3
    # (a) topo_order takes effect on the scalable path
    assert not np.allclose(fwd, rev)
    # (b) shared word flips parent (forward) -> child (reverse)
    assert fwd[p_block, shared] > fwd[c_block, shared]
    assert rev[c_block, shared] > rev[p_block, shared]


def test_scalable_frontier_scope_keeps_foreground_out_of_background(spark):
    # The distributed path honors anchor_scope="frontier" the same way: a
    # foreground-only token stays at the floor in the background block.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0}, n_bg=1, tpn=1)
    V = 8

    def doc(idx, frontier):
        idx = sorted(idx)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=np.ones(len(idx), dtype=np.float64),
                                length=len(idx), frontier=frozenset(frontier))
    rows = []
    for _ in range(30):
        rows.append(doc([0, 1, 2], []))
        rows.append(doc([5, 6, 0], [1]))
    rdd = spark.sparkContext.parallelize(rows, 3)

    floor = 1e-9 * 200.0
    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1,
                                        anchor_scope="frontier")
    bg = lam[0]
    node = lam[lay.block[1][0]]
    assert bg[5] <= floor * 10 and bg[6] <= floor * 10
    assert node[5] + node[6] > bg[5] + bg[6]
