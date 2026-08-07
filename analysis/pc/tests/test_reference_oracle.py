"""Reference-oracle validation of the FAITHFUL PC model against the authors' own
toy dataset and their published known-good parameter dumps (plan Task A3).

Dataset: ``datasets/toy_bars_3x3`` from
``github.com/dtak/prediction-constrained-topic-models`` — a 3x3 "bars" corpus
(V=9, the vocab laid out on a 3x3 grid; the "true" topics are the 6 row/column
bars) with a single binary label ``has_needle`` (whether the top-left grid word,
"needle", is present). The Hughes headline on this toy is that an *unsupervised*
topic model spends its topics on the dominant bar co-occurrence structure and
ignores the rare label direction, so a two-stage baseline predicts at chance,
while PC training reshapes the global topics to also carry the label — recovering
interpretable bars AND predicting the label well.

Two checks here:

  1. **From-scratch reproduction** (primary oracle): fit the faithful model with
     ``K=4`` and ``weight_y>0`` from a seeded init; assert it recovers
     interpretable bar-like topics AND clears heldout ROC AUC well above chance,
     and beats the unsupervised two-stage baseline by a wide margin.

  2. **Known-good-parameter loss ranking** (stretch, guarded): load the authors'
     Py2 zlib+joblib parameter dumps and confirm our faithful loss ranks them
     exactly as their names imply (``good_loss_x`` minimizes generative loss,
     ``good_loss_pc`` trades a little generative loss for near-perfect prediction,
     ``good_loss_y`` overfits prediction and destroys the topics). Skipped
     gracefully if the legacy pickle cannot be read in this environment.

Both fits are deterministic given the seed. All autograd/numpy/scipy in the core;
sklearn only here in the test.
"""
from __future__ import annotations

import io
import os
import zlib

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.pc.model import PCTopicModel
from analysis.pc.slda_reference import calc_loss__slda

# The toy_bars_3x3 fixture is vendored into the repo (MIT, see
# data/toy_bars_3x3/ATTRIBUTION.md) so this oracle is reproducible anywhere and
# does not depend on the ephemeral scratchpad clone. PC_REPO_DIR still overrides
# it (pointing at a full upstream checkout's datasets/toy_bars_3x3) if set.
_VENDORED = os.path.join(os.path.dirname(__file__), "data", "toy_bars_3x3")
_DATA_DIR = (
    os.path.join(os.environ["PC_REPO_DIR"], "datasets", "toy_bars_3x3")
    if os.environ.get("PC_REPO_DIR")
    else _VENDORED
)

# The 6 "true" bars of the 3x3 grid (row sets then column sets), as word-id sets.
_BARS = [
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},   # rows
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},   # columns
]

# Fit controls (see module docstring): faithful defaults, deterministic seed.
_K = 4
_WEIGHT_Y = 10.0
_ALPHA = 1.1
_PI_ITERS = 100
_MAX_ITER = 150

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DATA_DIR),
    reason=f"toy_bars_3x3 reference dataset not found at {_DATA_DIR}",
)


def _load_split(split):
    """Rebuild a dense count matrix + label array for one toy_bars split."""
    z = np.load(os.path.join(_DATA_DIR, f"X_csr_{split}.npz"))
    X = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    ).toarray().astype(np.float64)
    y = np.load(os.path.join(_DATA_DIR, f"Y_{split}.npy")).astype(np.float64)
    return X, y


def _bar_topics_and_concentration(topics_KV):
    """Return (n_exact_bar_topics, min_top3_concentration) for learned topics.

    A topic is an "exact bar" if its top-3 words are exactly one of the 6 grid
    bars; concentration is the fraction of a topic's mass on its top-3 words (a
    peaked, interpretable topic has high concentration).
    """
    n_bar = 0
    concs = []
    for k in range(topics_KV.shape[0]):
        top3 = set(np.argsort(-topics_KV[k])[:3].tolist())
        concs.append(float(topics_KV[k][list(top3)].sum()))
        n_bar += top3 in _BARS
    return n_bar, min(concs)


def test_toy_bars_from_scratch_recovers_bars_and_predicts(capsys):
    """Primary oracle: faithful PC recovers bar-like topics and predicts the
    heldout label well above chance, beating the unsupervised two-stage."""
    Xtr, Ytr = _load_split("train")
    Xte, Yte = _load_split("test")

    # --- PC (supervised, constrained) ---
    pc = PCTopicModel(
        K=_K, C=1, weight_y=_WEIGHT_Y, alpha=_ALPHA, tau=1.1,
        pi_iters=_PI_ITERS, max_iter=_MAX_ITER, seed=0,
    ).fit(Xtr, Ytr)
    pc_auc = roc_auc_score(Yte[:, 0], pc.predict_proba(Xte)[:, 0])
    n_bar, min_conc = _bar_topics_and_concentration(pc.topics_)

    # --- Unsupervised (weight_y=0) -> two-stage logistic regression ---
    unsup = PCTopicModel(
        K=_K, C=1, weight_y=0.0, alpha=_ALPHA, tau=1.1,
        pi_iters=_PI_ITERS, max_iter=_MAX_ITER, seed=0,
    ).fit(Xtr, Ytr)
    lr = LogisticRegression(max_iter=1000).fit(unsup.Pi_, Ytr[:, 0])
    two_stage_auc = roc_auc_score(
        Yte[:, 0], lr.predict_proba(unsup.transform(Xte))[:, 1]
    )

    with capsys.disabled():
        np.set_printoptions(precision=3, suppress=True)
        print("\n[toy_bars_3x3 / faithful PC] learned topics (K=4, rows=topics):")
        print(pc.topics_)
        for k in range(_K):
            top3 = sorted(np.argsort(-pc.topics_[k])[:3].tolist())
            print(
                f"  topic {k}: top3 words {top3} "
                f"(mass {pc.topics_[k][top3].sum():.3f})"
                f"{'  <- exact bar' if set(top3) in _BARS else ''}"
            )
        print(
            f"[toy_bars_3x3] heldout ROC AUC — PC={pc_auc:.4f}  "
            f"two-stage(unsup+LR)={two_stage_auc:.4f}  "
            f"(exact-bar topics={n_bar}/{_K}, min top3 conc={min_conc:.3f}, "
            f"pos rate te={Yte.mean():.2f})"
        )

    # (a) recovers the paper's interpretable bar-like topics
    assert n_bar >= 2, f"only {n_bar} learned topics form an exact grid bar"
    assert min_conc > 0.40, (
        f"topics not peaked/interpretable (min top3 concentration {min_conc:.3f})"
    )
    # (b) predicts the heldout label clearly above chance
    assert pc_auc > 0.70, f"PC heldout AUC {pc_auc:.3f} not above chance gate"
    # PC-vs-unsupervised contrast (the Hughes headline)
    assert pc_auc > two_stage_auc + 0.10, (
        f"PC {pc_auc:.3f} did not beat two-stage {two_stage_auc:.3f} by margin"
    )


# --- Stretch: our loss at the authors' known-good parameter dumps ----------

def _load_legacy_param_dump(filename):
    """Load one Py2 zlib+joblib parameter dump, shimming the old
    ``sklearn.externals.joblib`` module path. Returns the param dict.

    Raises on any failure (caller turns that into a skip)."""
    import sys
    import types

    import joblib
    from joblib import numpy_pickle as jnp

    # Old dumps reference sklearn.externals.joblib(.numpy_pickle); alias to the
    # standalone joblib whose NumpyArrayWrapper/NumpyUnpickler are compatible.
    m1 = types.ModuleType("sklearn.externals.joblib")
    m1.__dict__.update(joblib.__dict__)
    m2 = types.ModuleType("sklearn.externals.joblib.numpy_pickle")
    m2.__dict__.update(jnp.__dict__)
    m1.numpy_pickle = m2
    sys.modules.setdefault("sklearn.externals.joblib", m1)
    sys.modules.setdefault("sklearn.externals.joblib.numpy_pickle", m2)

    raw = open(os.path.join(_DATA_DIR, filename), "rb").read()
    dec = zlib.decompress(raw)  # -> protocol-2 pickle stream
    unpickler = jnp.NumpyUnpickler(
        "mem", io.BytesIO(dec), ensure_native_byte_order=True
    )
    return unpickler.load()


def test_loss_ranks_authors_known_good_params(capsys):
    """Stretch (non-blocking): our faithful loss ranks the authors' provided
    known-good parameter dumps consistently with their names."""
    try:
        dumps = {
            name: _load_legacy_param_dump(f"good_loss_{name}_K4_param_dict.dump")
            for name in ["x", "pc", "y"]
        }
    except Exception as exc:  # legacy pickle / joblib incompatibility
        pytest.skip(f"could not load legacy param dumps: {exc!r}")

    Xtr, Ytr = _load_split("train")
    results = {}
    for name, GP in dumps.items():
        d = calc_loss__slda(
            GP["topics_KV"], GP["w_CK"], Xtr, Ytr, None,
            alpha=_ALPHA, tau=1.1, weight_y=1.0, pi_iters=_PI_ITERS,
            return_dict=True,
        )
        d["auc"] = roc_auc_score(Ytr[:, 0], d["y_proba_DC"][:, 0])
        results[name] = d

    with capsys.disabled():
        print("\n[toy_bars_3x3] our loss at authors' known-good params:")
        for name in ["x", "pc", "y"]:
            d = results[name]
            print(
                f"  good_loss_{name:<2}: ttl={d['loss_ttl']:.4f} "
                f"loss_x={d['loss_x']:.4f} loss_y={d['loss_y']:.4f} "
                f"AUC(train)={d['auc']:.3f}"
            )

    # good_loss_x achieves the best (lowest) generative loss of the three.
    assert results["x"]["loss_x"] <= results["pc"]["loss_x"] + 1e-9
    assert results["x"]["loss_x"] < results["y"]["loss_x"]
    # good_loss_pc predicts near-perfectly while keeping generative loss close to
    # the x-optimum (the PC tradeoff), unlike good_loss_y which wrecks the topics.
    assert results["pc"]["auc"] > 0.95
    assert results["pc"]["loss_x"] < 0.5 * results["y"]["loss_x"]
    # PC's prediction loss is far better than the purely generative x-fit's.
    assert results["pc"]["loss_y"] < results["x"]["loss_y"]
