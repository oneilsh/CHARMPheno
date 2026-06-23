"""StreamingSTM: MLlib-shim estimator for OnlineSTM.

Two input paths:
  (A) Caller supplies a pre-built `covariates` DenseVector column and
      a list of covariate names. No formulaic dependency required.
  (B) Caller supplies a `covariate_formula` string + a covariate
      DataFrame. Requires the `formula` extra: pip install spark-vi[formula].

Path B is implemented via `covariate_formula`; see `_resolve_model_spec_from_pandas`
and `_formula.fit_model_spec`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from spark_vi.models.topic.stm import prior_topic_proportions


def corpus_mean_topic_proportions_rdd(cov_rdd, Gamma: np.ndarray, depth: int = 2):
    """Distributed α-equivalent: (1/D) Σ_d softmax(Γᵀ x_d) over an RDD.

    ``cov_rdd`` is an RDD of length-P covariate vectors (bare numpy arrays —
    no person_id, honoring the spark-vi layering rule). Mirrors the engine's
    mapPartitions+treeReduce idiom (see ``core/runner.py``): each partition
    accumulates a (K-vector sum, count) locally and the tree-reduce combines
    them, so only a K-vector and a scalar ever reach the driver. Scales to any
    D and any covariate cardinality — continuous covariates included.

    Γ is broadcast via the Spark-safe default-arg closure convention. Returns
    a length-K probability vector.
    """
    sc = cov_rdd.context
    bcast = sc.broadcast(Gamma)

    def _local(rows, _bcast=bcast):
        G = _bcast.value
        acc = np.zeros(G.shape[1], dtype=np.float64)
        n = 0
        for x in rows:
            acc += prior_topic_proportions(G, np.asarray(x, dtype=np.float64))
            n += 1
        return [(acc, n)]

    def _combine(a, b):
        return a[0] + b[0], a[1] + b[1]

    sum_vec, count = cov_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth)
    if count == 0:
        raise ValueError("corpus_mean_topic_proportions_rdd: empty covariate RDD")
    return sum_vec / count


class StreamingSTM:
    """Streaming-VI estimator for OnlineSTM with DataFrame input.

    Constructor enforces that the caller supplies enough information
    to determine P (covariate dimension) — either via covariate_names
    (Path A) or covariate_formula (Path B).
    """

    def __init__(
        self,
        K: int,
        features_col: str = "features",
        covariates_col: str | None = None,
        covariate_names: list[str] | None = None,
        covariate_formula: str | None = None,
        covariate_df: Any | None = None,
        join_key: str | None = None,
        max_levels: int = 10_000,
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        random_seed: int | None = None,
    ) -> None:
        # Path A vs B validation.
        path_a = covariates_col is not None and covariate_names is not None
        path_b = covariate_formula is not None
        if not (path_a or path_b):
            raise ValueError(
                "StreamingSTM requires either (covariates_col + covariate_names) "
                "for Path A, or covariate_formula for Path B."
            )
        if path_a and path_b:
            raise ValueError("Use either Path A or Path B, not both.")

        self.K = int(K)
        self.features_col = features_col

        if path_a:
            if not covariate_names:
                raise ValueError("covariate_names must be non-empty for Path A.")
            self.covariates_col = covariates_col
            self.covariate_names = list(covariate_names)
            self.P = len(self.covariate_names)
            self.covariate_formula = None
        else:
            # Path B: uses formulaic ModelSpec for covariate resolution.
            self.covariates_col = "covariates"
            self.covariate_formula = covariate_formula
            self.covariate_df = covariate_df
            self.join_key = join_key
            self.max_levels = max_levels
            self.covariate_names = None       # set during fit
            self.P = None                     # set during fit

        self.sigma_init = sigma_init
        self.sigma_ridge = sigma_ridge
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        self.random_seed = random_seed

    def fit(
        self,
        dataset,
        *,
        max_iter: int = 20,
        subsampling_rate: float = 0.2,
        tau0: float = 64.0,
        kappa: float = 0.7,
        save_interval: int | None = None,
        checkpoint_dir: str | None = None,
        on_iteration=None,
        resume_from: str | None = None,
    ) -> "STMModel":
        """Fit OnlineSTM via VIRunner on a DataFrame with features + covariates columns.

        The input DataFrame must have the configured `features_col` (SparseVector)
        and `covariates_col` (DenseVector). Vocab size is discovered from the
        first features row.

        Parameters:
            dataset: Spark DataFrame with `features_col` and `covariates_col`.
            max_iter: maximum number of SVI iterations.
            subsampling_rate: fraction of documents per mini-batch (maps to
                VIConfig.mini_batch_fraction).
            tau0: Robbins-Monro delay parameter (maps to
                VIConfig.learning_rate_tau0).
            kappa: Robbins-Monro decay exponent (maps to
                VIConfig.learning_rate_kappa).
            save_interval: if set, checkpoint every N iterations
                (requires checkpoint_dir).
            checkpoint_dir: directory for periodic checkpoints.
            on_iteration: optional per-iteration callback
                fn(iter_num, global_params, elbo_trace).
            resume_from: optional path to a previously-saved STMModel dir. When
                set, VIRunner loads its global_params + n_iterations and
                continues the Robbins-Monro schedule from there; max_iter is
                then ADDITIONAL iterations on top of the loaded count. The
                resumed corpus/covariate shapes (V, P) must match the loaded
                params, so resume only with the same corpus + formula.
        """
        from pyspark import StorageLevel

        from spark_vi.core.config import VIConfig
        from spark_vi.core.runner import VIRunner
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        from spark_vi.models.topic.stm import OnlineSTM

        if self.covariate_names is None:
            raise ValueError(
                "StreamingSTM.fit requires covariate_names to be set. For Path A "
                "supply covariate_names at construction; for Path B call "
                "_resolve_model_spec_from_pandas first."
            )

        first = dataset.select(self.features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first[0][0].size

        model = OnlineSTM(
            K=self.K,
            vocab_size=vocab_size,
            P=self.P,
            sigma_init=self.sigma_init,
            sigma_ridge=self.sigma_ridge,
            lbfgs_max_iter=self.lbfgs_max_iter,
            lbfgs_tol=self.lbfgs_tol,
            random_seed=self.random_seed,
        )

        # VIConfig uses learning_rate_tau0/kappa and mini_batch_fraction;
        # checkpoint_interval + checkpoint_dir must be both set or both None.
        checkpoint_kwargs: dict = {}
        if save_interval is not None and checkpoint_dir is not None:
            checkpoint_kwargs = {
                "checkpoint_interval": save_interval,
                "checkpoint_dir": checkpoint_dir,
            }

        config = VIConfig(
            max_iterations=max_iter,
            learning_rate_tau0=tau0,
            learning_rate_kappa=kappa,
            mini_batch_fraction=subsampling_rate if subsampling_rate < 1.0 else None,
            **checkpoint_kwargs,
        )

        features_col = self.features_col
        covariates_col = self.covariates_col
        rdd = (
            dataset.select(features_col, covariates_col).rdd
            .map(lambda row: _vector_to_stm_document(
                {features_col: row[0], covariates_col: row[1]},
                features_col=features_col,
                covariates_col=covariates_col,
            ))
        )
        rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count()

        runner = VIRunner(model, config=config)
        try:
            result = runner.fit(
                rdd, on_iteration=on_iteration, resume_from=resume_from,
            )
        finally:
            rdd.unpersist(blocking=False)

        return STMModel(
            global_params=result.global_params,
            metadata=dict(result.metadata),
            model_spec=getattr(self, "model_spec", None),
            covariate_names=list(self.covariate_names),
            n_iterations=result.n_iterations,
            elbo_trace=list(result.elbo_trace),
            converged=result.converged,
            diagnostic_traces=dict(result.diagnostic_traces),
        )

    def _resolve_model_spec_from_pandas(self, covariate_pdf):
        """Resolve P and covariate_names from a pre-collected pandas covariate DataFrame.

        Used by tests and by the in-memory Path-B construction. Production
        .fit() invocations against Spark DataFrames will use the
        schema-frame Spark discovery path instead (see _formula.fit_model_spec_from_spark and ADR 0024).
        """
        from spark_vi.mllib.topic._formula import fit_model_spec
        spec, names = fit_model_spec(self.covariate_formula, covariate_pdf)
        self.model_spec = spec
        self.covariate_names = names
        self.P = len(names)


class STMModel:
    """Fitted MLlib-shim STM model. Wraps OnlineSTM's global params + ModelSpec.

    Persistence layout under <model_dir> (VIResult-compatible):
        manifest.json           # metadata + elbo_trace (written by save_result)
        params/<name>.npy       # one file per global_param key
        model_spec.pkl          # formulaic ModelSpec (pickle sidecar)
        covariate_names.json    # list of covariate name strings (sidecar)
    """

    def __init__(
        self,
        global_params: dict[str, np.ndarray],
        metadata: dict[str, Any],
        model_spec: Any,
        covariate_names: list[str],
        n_iterations: int = 0,
        elbo_trace: list[float] | None = None,
        converged: bool = False,
        diagnostic_traces: dict | None = None,
    ) -> None:
        self.global_params = global_params
        self.metadata = metadata
        self.model_spec = model_spec
        self.covariate_names = covariate_names
        # Resume state: n_iterations + elbo_trace are persisted so a later
        # fit(resume_from=...) continues the Robbins-Monro counter (rho_t
        # depends on the loaded iteration count) instead of restarting at t=0.
        self.n_iterations = n_iterations
        self.elbo_trace = list(elbo_trace) if elbo_trace is not None else []
        self.converged = converged
        self.diagnostic_traces = (
            dict(diagnostic_traces) if diagnostic_traces is not None else {}
        )

    def save(self, out_dir: Path) -> None:
        from spark_vi.core.result import VIResult
        from spark_vi.io.export import save_result

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wrap state as a VIResult so the canonical saver handles the
        # standard layout (manifest.json, params/*.npy, traces/).
        # load_result in build_dashboard_cloud.py expects this layout.
        result = VIResult(
            global_params=self.global_params,
            metadata=dict(self.metadata),
            elbo_trace=list(self.elbo_trace),
            n_iterations=self.n_iterations,
            converged=self.converged,
            # Not persisted: STM's per-iteration diagnostics include 2-D Gamma
            # snapshots that save_result rejects (it handles scalar/1-D traces
            # only), and resume needs only global_params + n_iterations +
            # elbo_trace. Dropping them here keeps the on-disk layout valid.
            diagnostic_traces={},
        )
        save_result(result, out_dir)
        # Sidecars: formulaic ModelSpec + covariate names list.
        with (out_dir / "model_spec.pkl").open("wb") as f:
            pickle.dump(self.model_spec, f)
        (out_dir / "covariate_names.json").write_text(
            json.dumps(self.covariate_names)
        )

    @classmethod
    def load(cls, in_dir: Path) -> "STMModel":
        from spark_vi.io.export import load_result

        in_dir = Path(in_dir)
        result = load_result(in_dir)
        with (in_dir / "model_spec.pkl").open("rb") as f:
            spec = pickle.load(f)
        covariate_names = json.loads(
            (in_dir / "covariate_names.json").read_text()
        )
        return cls(
            global_params=result.global_params,
            metadata=dict(result.metadata),
            model_spec=spec,
            covariate_names=covariate_names,
            n_iterations=result.n_iterations,
            elbo_trace=list(result.elbo_trace),
            converged=result.converged,
            diagnostic_traces=dict(result.diagnostic_traces),
        )
