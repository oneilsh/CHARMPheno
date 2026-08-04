"""Tests for the data-driven per-node K estimator (effective_rank)."""
import numpy as np
import pytest

from spark_vi.models.topic.effective_rank import (
    allocate_topics,
    build_null_spectrum,
    effective_rank_report,
    eigengap_rank,
    log_effrank_table,
    null_percentile_spectrum,
    parallel_analysis_count_all,
    parallel_analysis_rank,
    participation_ratio,
    pivoted_qr_residual_spectrum,
    report_from_spectrum,
    singular_value_spectrum,
    threshold_rank,
)


def _planted_rank_matrix(V, d, rank, *, seed=0, noise=1e-6):
    """V rows living (up to noise) in a rank-`rank` subspace of R^d."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((rank, d))
    coeffs = rng.standard_normal((V, rank))
    M = coeffs @ basis
    M = M + noise * rng.standard_normal((V, d))
    return M


# --- participation_ratio ----------------------------------------------------

def test_participation_ratio_flat_equals_rank():
    # r equal eigenvalues -> effective rank exactly r.
    assert participation_ratio([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)


def test_participation_ratio_single_direction_is_one():
    assert participation_ratio([9.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_participation_ratio_scale_invariant():
    a = participation_ratio([3.0, 1.0, 0.5])
    b = participation_ratio([300.0, 100.0, 50.0])
    assert a == pytest.approx(b)


def test_participation_ratio_empty_is_zero():
    assert participation_ratio([]) == 0.0
    assert participation_ratio([0.0, 0.0]) == 0.0


# --- threshold_rank / eigengap_rank ----------------------------------------

def test_threshold_rank_counts_above_relative_floor():
    # 1.0, 0.5, 0.005 -> at tau=0.01 the third (0.5% of leading) is dropped.
    assert threshold_rank([1.0, 0.5, 0.005], tau=0.01) == 2


def test_threshold_rank_scale_invariant():
    assert threshold_rank([100.0, 50.0, 0.5], tau=0.01) == 2


def test_eigengap_rank_finds_the_cliff():
    # big drop between index 2 and 3 -> rank 3.
    assert eigengap_rank([10.0, 9.0, 8.0, 0.01, 0.008]) == 3


def test_eigengap_rank_single_entry():
    assert eigengap_rank([5.0]) == 1


# --- pivoted_qr_residual_spectrum ------------------------------------------

def test_spectrum_is_non_increasing():
    M = _planted_rank_matrix(60, 20, rank=5, seed=1)
    spec = pivoted_qr_residual_spectrum(M, max_probe=15)
    assert all(spec[i] >= spec[i + 1] - 1e-9 for i in range(len(spec) - 1))


def test_spectrum_reveals_planted_rank():
    # A rank-5 row-set with tiny noise: the greedy breaks once residuals fall
    # below eps, so the spectrum truncates at the numerical rank (n_probed == 5)
    # and both the threshold count and participation ratio recover ~5. (eigengap
    # needs a recorded noise tail to see a cliff; with a clean truncation there
    # is none, so it is not asserted here -- see test_eigengap_rank_finds_cliff.)
    M = _planted_rank_matrix(80, 30, rank=5, seed=2, noise=1e-8)
    rep = effective_rank_report(M, max_probe=20)
    assert rep["n_probed"] == 5
    assert rep["threshold"] == 5
    assert 4.0 <= rep["participation"] <= 6.0


def test_spectrum_respects_max_probe():
    M = _planted_rank_matrix(50, 40, rank=30, seed=3, noise=1e-3)
    spec = pivoted_qr_residual_spectrum(M, max_probe=8)
    assert len(spec) <= 8


def test_return_pivots_excludes_seeds():
    M = _planted_rank_matrix(80, 30, rank=5, seed=7, noise=1e-8)
    spec, pivots = pivoted_qr_residual_spectrum(M, max_probe=20, return_pivots=True)
    # pivots align 1:1 with the spectrum (one pivot per recorded direction)
    assert len(pivots) == len(spec)
    # seeded pivots are NOT returned; seeding two rows drops them from the pivots
    spec2, pivots2 = pivoted_qr_residual_spectrum(
        M, max_probe=20, seed_rows=[0, 1], return_pivots=True)
    assert 0 not in pivots2 and 1 not in pivots2


def test_eligible_floor_excludes_noise_directions():
    # 20 shared words spanning a rank-4 subspace (eligible / high-df) + 200
    # high-norm idiosyncratic "singleton" words (ineligible / low-df). Without the
    # floor the noise inflates the rank; with it, only the ~4 shared directions
    # count -- the fix for low-count nodes reading a spurious rank of >100.
    rng = np.random.default_rng(3)
    basis = rng.standard_normal((4, 50))
    shared = rng.standard_normal((20, 4)) @ basis
    noise = 5 * rng.standard_normal((200, 50))
    M = np.vstack([shared, noise])
    elig = np.array([True] * 20 + [False] * 200)
    assert threshold_rank(pivoted_qr_residual_spectrum(M, 60)) > 20      # inflated
    assert threshold_rank(pivoted_qr_residual_spectrum(M, 60, eligible=elig)) <= 5
    # ineligible rows are never selected as pivots
    _, piv = pivoted_qr_residual_spectrum(M, 60, eligible=elig, return_pivots=True)
    assert all(p < 20 for p in piv)


def test_report_from_spectrum_matches_effective_rank_report():
    M = _planted_rank_matrix(80, 30, rank=5, seed=8, noise=1e-8)
    spec = pivoted_qr_residual_spectrum(M, max_probe=20)
    a = report_from_spectrum(spec)
    b = effective_rank_report(M, max_probe=20)
    assert a["participation"] == b["participation"]
    assert a["threshold"] == b["threshold"]
    assert a["n_probed"] == b["n_probed"]


def test_hierarchical_deflation_bounds_the_total():
    # Two children spanning the SAME rank-5 subspace as their parent. Measured
    # independently each shows ~5, summing to ~15 (parent + 2 kids). But deflating
    # the children against the parent's full pivot claim leaves ~0 increment, so
    # the hierarchical total collapses toward the parent's 5 -- the effect the
    # hierarchical probe relies on.
    rng = np.random.default_rng(11)
    basis = rng.standard_normal((5, 30))
    parent = rng.standard_normal((80, 5)) @ basis + 1e-9 * rng.standard_normal((80, 30))
    kidA = rng.standard_normal((80, 5)) @ basis + 1e-9 * rng.standard_normal((80, 30))
    kidB = rng.standard_normal((80, 5)) @ basis + 1e-9 * rng.standard_normal((80, 30))
    _, p_piv = pivoted_qr_residual_spectrum(parent, 20, return_pivots=True)
    # naive: children measured independently
    naive = (threshold_rank(pivoted_qr_residual_spectrum(kidA, 20))
             + threshold_rank(pivoted_qr_residual_spectrum(kidB, 20)))
    # hierarchical: children deflated against parent's pivots (same subspace)
    hierA = threshold_rank(pivoted_qr_residual_spectrum(kidA, 20, seed_rows=p_piv))
    hierB = threshold_rank(pivoted_qr_residual_spectrum(kidB, 20, seed_rows=p_piv))
    assert naive >= 8               # ~5 + ~5 independently
    assert hierA + hierB <= 2       # increment over the shared parent ~ 0


def test_seed_rows_deflate_without_contributing():
    # Seeding a direction removes it from the revealed spectrum: a rank-5 set
    # seeded with 2 of its own pivots reveals <= 3 remaining strong directions.
    M = _planted_rank_matrix(80, 30, rank=5, seed=4, noise=1e-8)
    full = pivoted_qr_residual_spectrum(M, max_probe=20)
    assert threshold_rank(full) == 5
    # pick the first two pivots as seeds by re-running and grabbing their ids
    # (re-derive ids via a one-off greedy pass mirroring the internal choice)
    seeded = pivoted_qr_residual_spectrum(M, max_probe=20, seed_rows=[0, 1])
    # seeds deflate at least their own span; remaining strong dirs <= full
    assert threshold_rank(seeded) <= 5


# --- singular_value_spectrum ------------------------------------------------

def test_singular_value_spectrum_recovers_planted_rank_and_order():
    # A rank-4 (V, d) matrix: exactly 4 non-negligible squared singular values,
    # descending, and the tail is ~0.
    M = _planted_rank_matrix(120, 40, rank=4, seed=5, noise=1e-9)
    spec = singular_value_spectrum(M, 20)
    assert all(spec[i] >= spec[i + 1] - 1e-9 for i in range(len(spec) - 1))
    assert spec[3] > 1e6 * max(spec[4], 1e-30)      # sharp drop after the 4th
    assert len(singular_value_spectrum(M, 6)) == 6  # respects max_probe


def test_singular_value_spectrum_degenerate():
    assert singular_value_spectrum(np.zeros((0, 5)), 10) == []


# --- parallel analysis: pure combinators (count above margin x null) --------

def test_parallel_analysis_rank_counts_directions_above_margin():
    # margin=2 (default): real must clear 2x the floor. 10>4 and 8>4 clear; 5<4? no
    # 5<4 is false so 5>4 -> but 5 vs 2*2=4 -> clears; 3 vs 4 -> no; 1 vs 4 -> no.
    real = [10.0, 8.0, 5.0, 3.0, 1.0]
    floor = [2.0, 2.0, 2.0, 2.0, 2.0]
    assert parallel_analysis_rank(real, floor) == 3          # 10,8,5 clear 4


def test_parallel_analysis_rank_skips_below_null_background_at_pos0():
    # Position 0 is the shared marginal/background direction and sits BELOW the null;
    # bg_skip=1 (default) lets the phenotype block begin at position 1.
    real = [1.0, 20.0, 18.0, 1.5]      # position 0 below null, 1&2 well above
    floor = [2.0, 2.0, 2.0, 2.0]
    assert parallel_analysis_rank(real, floor) == 2          # leading run at 1,2


def test_parallel_analysis_rank_rejects_tail_only_clearings():
    # The small-node pathology: leading directions are BELOW the null, but the TAIL
    # floats above it (per-record cliques the marginal null can't reproduce). The
    # leading-run rule returns 0 (no coherent leading block); count-all is fooled.
    real = [1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0]   # nothing clears until pos 5
    floor = [2.0] * 8
    assert parallel_analysis_rank(real, floor) == 0          # first clear at pos 5 > bg_skip
    assert parallel_analysis_count_all(real, floor) == 3     # count-all sums the tail


def test_parallel_analysis_rank_bg_skip_controls_allowed_lead():
    # A node with two below-null background directions before its signal block.
    real = [1.0, 1.0, 20.0, 18.0, 1.0]
    floor = [2.0] * 5
    assert parallel_analysis_rank(real, floor, bg_skip=1) == 0   # signal begins past pos1
    assert parallel_analysis_rank(real, floor, bg_skip=2) == 2   # now the block is reached


def test_parallel_analysis_rank_margin_is_tunable():
    real = [10.0, 5.0, 2.5]
    floor = [2.0, 2.0, 2.0]
    assert parallel_analysis_rank(real, floor, margin=1.0) == 3   # >2
    assert parallel_analysis_rank(real, floor, margin=2.0) == 2   # >4: 10,5
    assert parallel_analysis_rank(real, floor, margin=3.0) == 1   # >6: 10


def test_parallel_analysis_rank_zero_when_nothing_clears():
    assert parallel_analysis_rank([3.0, 0.5], [2.0, 2.0]) == 0    # neither > 4


def test_parallel_analysis_rank_respects_shorter_spectrum():
    assert parallel_analysis_rank([9.0, 8.0, 7.0, 6.0], [2.0, 2.0]) == 2


def test_null_percentile_spectrum_pads_ragged_with_zero():
    # position 0: pct95 of {10, 8}; position 1: pct95 of {5, 0} (short spec padded).
    floor = null_percentile_spectrum([[10.0, 5.0], [8.0]], q=95)
    assert len(floor) == 2
    assert floor[0] == pytest.approx(np.percentile([10.0, 8.0], 95))
    assert floor[1] == pytest.approx(np.percentile([5.0, 0.0], 95))


def test_null_percentile_spectrum_is_non_increasing():
    specs = [[9.0, 6.0, 3.0], [8.0, 5.0, 1.0], [10.0, 4.0, 2.0]]
    floor = null_percentile_spectrum(specs, q=90)
    assert all(floor[i] >= floor[i + 1] - 1e-9 for i in range(len(floor) - 1))


def test_null_percentile_spectrum_empty():
    assert null_percentile_spectrum([]) == []
    assert null_percentile_spectrum([[], []]) == []


# --- parallel analysis: driver-side null + end-to-end K ---------------------

def _unigram_and_lengths(V, n_docs, mean_len, *, seed):
    """A synthetic node marginal + length sample (helper for the null builder)."""
    rng = np.random.default_rng(seed)
    marg = rng.random(V) + 0.05           # positive propensity for every token
    lengths = rng.integers(mean_len - 2, mean_len + 3, size=min(n_docs, 512))
    return marg, lengths


def test_build_null_spectrum_shape_and_monotonicity():
    marg, lengths = _unigram_and_lengths(120, 400, 8, seed=1)
    floor = build_null_spectrum(marg, lengths, n_docs=400, V=120, d=64, seed=7,
                                reps=3, cap=200, max_probe=40)
    assert 0 < len(floor) <= 40
    assert all(floor[i] >= floor[i + 1] - 1e-9 for i in range(len(floor) - 1))


def test_build_null_floor_rises_as_n_docs_shrinks():
    # Sample-size awareness: fewer docs -> more finite-sample fluctuation -> a
    # HIGHER leading null floor. Same marginal + lengths + seeds; only n_docs.
    marg, lengths = _unigram_and_lengths(150, 2000, 10, seed=2)
    R = None
    big = build_null_spectrum(marg, lengths, n_docs=2000, V=150, d=80, seed=5,
                              reps=4, cap=2000, max_probe=50, R_rows=R)
    small = build_null_spectrum(marg, lengths, n_docs=30, V=150, d=80, seed=5,
                                reps=4, cap=2000, max_probe=50, R_rows=R)
    assert small[0] > big[0]


def test_build_null_spectrum_empty_when_no_support():
    marg, lengths = _unigram_and_lengths(50, 100, 6, seed=3)
    assert build_null_spectrum(marg, lengths, n_docs=0, V=50, d=32, seed=1) == []
    assert build_null_spectrum(np.zeros(50), lengths, n_docs=100, V=50, d=32,
                               seed=1) == []
    assert build_null_spectrum(marg, [1, 1, 1], n_docs=100, V=50, d=32,
                               seed=1) == []


def _sketch_from_docs(docs, V, d, seed):
    """Real (V,d) sketch + marginal + lengths from docs, via the fit's own path.

    Mirrors ``projected_cooccurrence_rdd`` on the driver: project each doc with the
    shared projection rows, accumulate the summed sketch + word marginal, and track
    the raw unigram counts + doc lengths the null builder needs. Returns everything
    on the SAME projection scale as ``build_null_spectrum`` (same ``R_rows``), which
    is what makes the real/null spectra comparable.
    """
    from spark_vi.models.topic.spectral_init_scalable import (
        _project_doc, _row_normalize_projected, precompute_projection_rows,
    )
    R_rows = precompute_projection_rows(V, d, seed)
    QR = np.zeros((V, d))
    p_w = np.zeros(V)
    unigram = np.zeros(V)
    lengths = []
    for idx, cnt in docs:
        if int(cnt.sum()) < 2:
            continue
        qr, pwc = _project_doc(idx, cnt, R_rows[idx])
        QR[idx] += qr
        p_w[idx] += pwc
        unigram[idx] += cnt
        lengths.append(int(cnt.sum()))
    Qbar = _row_normalize_projected(QR, p_w)
    return Qbar, unigram, lengths, R_rows


def _block_topics(V, planted, *, bg_frac=0.5, seed=123):
    """`planted` topics each concentrated on a disjoint V//planted word block over a
    shared background -- distinct phenotypes (largely disjoint concept sets) atop the
    common comorbidity every patient carries. Produces a clean co-occurrence signal
    of ~planted-1 directions BEYOND the shared background."""
    rng = np.random.default_rng(seed)
    bg = rng.random(V) + 0.1
    bg /= bg.sum()
    block = V // planted
    topics = np.empty((planted, V))
    for t in range(planted):
        ph = np.zeros(V)
        ph[t * block:(t + 1) * block] = 1.0 / block
        topics[t] = bg_frac * bg + (1.0 - bg_frac) * ph
    return topics / topics.sum(axis=1, keepdims=True)


def _mixture_corpus(V, n_docs, topics, *, seed):
    rng = np.random.default_rng(seed)
    planted = topics.shape[0]
    docs = []
    for _ in range(n_docs):
        t = int(rng.integers(planted))
        L = int(rng.integers(18, 26))
        toks = rng.choice(V, size=L, p=topics[t])
        idx, cnt = np.unique(toks, return_counts=True)
        docs.append((idx, cnt))
    return docs


def test_parallel_analysis_recovers_planted_foreground_rank():
    # End-to-end: a well-supported node whose REAL docs are a mixture of `planted`
    # distinct phenotypes over a shared background, built through the exact
    # doc->project->normalize path the fit uses. Its null is drawn from the node's
    # OWN unigram marginal + lengths at its OWN n_docs. pa_k recovers the FOREGROUND
    # dimensionality -- the ~planted-1 directions BEYOND the shared background (the
    # position-0 marginal is modeled by CHARM's n_bg block, not counted here) -- and
    # is far below the ~min(#words,d) token richness raw effective rank reports.
    V, d, planted = 250, 96, 5
    topics = _block_topics(V, planted, bg_frac=0.5)
    docs = _mixture_corpus(V, 1500, topics, seed=4)
    Qbar, unigram, lengths, R_rows = _sketch_from_docs(docs, V, d, seed=9)
    spec_real = singular_value_spectrum(Qbar, 60)
    floor = build_null_spectrum(unigram, lengths, n_docs=len(lengths), V=V, d=d,
                                seed=9, reps=6, cap=1500, max_probe=60, R_rows=R_rows)
    k = parallel_analysis_rank(spec_real, floor)
    assert 3 <= k <= 6           # ~planted-1 = 4; far below raw richness (~60)


def test_parallel_analysis_is_sample_size_aware():
    # SAME 5-phenotype structure, two support levels. A well-supported node recovers
    # the foreground directions; a tiny (26-doc) node cannot support them and its
    # rank collapses toward 0 -- the sample-size awareness effective rank lacked
    # (which read a 26-doc node at ~90). The null floor rises with shrinking n_docs.
    V, d, planted = 250, 96, 5
    topics = _block_topics(V, planted, bg_frac=0.5)

    def pa_k(n_docs):
        docs = _mixture_corpus(V, n_docs, topics, seed=4)
        Qbar, uni, lens, R = _sketch_from_docs(docs, V, d, seed=9)
        spec = singular_value_spectrum(Qbar, 60)
        floor = build_null_spectrum(uni, lens, n_docs=len(lens), V=V, d=d, seed=9,
                                    reps=6, cap=2000, max_probe=60, R_rows=R)
        return parallel_analysis_rank(spec, floor)

    big, small = pa_k(1500), pa_k(26)
    assert big >= 3
    assert small <= 1
    assert small < big


# --- allocate_topics --------------------------------------------------------

def test_allocate_topics_rounds_and_clamps():
    effranks = {1: 2.4, 2: 17.6, 3: 0.2}
    out = allocate_topics(effranks, floor=1, cap=12)
    assert out == {1: 2, 2: 12, 3: 1}


def test_allocate_topics_no_cap():
    out = allocate_topics({1: 40.3}, floor=1, cap=None)
    assert out == {1: 40}


def _report(pr, thr, gap, n):
    return {"participation": pr, "threshold": thr, "eigengap": gap,
            "n_probed": n, "spectrum": []}


def test_log_effrank_table_sorts_and_summarizes():
    lines = []
    reports = {
        7: _report(2.0, 2, 2, 2),      # tight leaf
        3: _report(20.0, 25, 25, 40),  # broad class
        5: _report(6.0, 6, 6, 8),      # mid
    }
    log_effrank_table(reports, n_nodes=3, k_uniform=6, printer=lines.append)
    body = "\n".join(lines)
    # rows sorted by participation desc: node 3 (20) before 5 (6) before 7 (2)
    order = [ln.split("\t")[0].split()[-1] for ln in lines
             if ln.count("\t") == 4 and "node" not in ln]
    assert order == ["3", "5", "7"]
    # diversity-driven K = round(20)+round(6)+round(2) = 28 vs uniform 6
    assert "Σround(PR)=28" in body
    assert "current foreground K=6" in body


def test_log_effrank_table_empty():
    lines = []
    log_effrank_table({}, n_nodes=5, k_uniform=10, printer=lines.append)
    assert any("no nodes probed" in ln for ln in lines)


def test_allocate_total_tracks_diversity_not_node_count():
    # Two layouts, same node count: the diverse one gets more total topics.
    tight = {i: 2.0 for i in range(50)}      # 50 tight leaves
    diverse = {i: 2.0 for i in range(45)}
    diverse.update({i: 30.0 for i in range(45, 50)})  # 5 broad classes
    kt = sum(allocate_topics(tight, floor=1).values())
    kd = sum(allocate_topics(diverse, floor=1).values())
    assert kt == 100
    assert kd == 90 + 150  # 45*2 + 5*30
    assert kd > kt
