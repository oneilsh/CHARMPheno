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
            result = runner.fit(rdd, on_iteration=on_iteration)
        finally:
            rdd.unpersist(blocking=False)

        return STMModel(
            global_params=result.global_params,
            metadata=dict(result.metadata),
            model_spec=getattr(self, "model_spec", None),
            covariate_names=list(self.covariate_names),
        )

    def _resolve_model_spec_from_pandas(self, covariate_pdf):
        """Resolve P and covariate_names from a pre-collected pandas covariate DataFrame.

        Used by tests and by the in-memory Path-B construction. Production
        .fit() invocations against Spark DataFrames will use the
        schema-frame Spark discovery path instead (Task 13).
        """
        from spark_vi.mllib.topic._formula import fit_model_spec
        spec, names = fit_model_spec(self.covariate_formula, covariate_pdf)
        self.model_spec = spec
        self.covariate_names = names
        self.P = len(names)


class STMModel:
    """Fitted MLlib-shim STM model. Wraps OnlineSTM's global params + ModelSpec.

    Persistence layout under <model_dir>:
        global_params.npz   # lambda, Gamma, Sigma, eta (numpy arrays)
        metadata.json       # K, V, P, covariate_names
        model_spec.pkl      # formulaic ModelSpec (pickle)
    """

    def __init__(
        self,
        global_params: dict[str, np.ndarray],
        metadata: dict[str, Any],
        model_spec: Any,
        covariate_names: list[str],
    ) -> None:
        self.global_params = global_params
        self.metadata = metadata
        self.model_spec = model_spec
        self.covariate_names = covariate_names

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "global_params.npz",
            **{k: np.asarray(v) for k, v in self.global_params.items()},
        )
        (out_dir / "metadata.json").write_text(json.dumps({
            **self.metadata,
            "covariate_names": self.covariate_names,
        }))
        with (out_dir / "model_spec.pkl").open("wb") as f:
            pickle.dump(self.model_spec, f)

    @classmethod
    def load(cls, in_dir: Path) -> "STMModel":
        in_dir = Path(in_dir)
        npz = np.load(in_dir / "global_params.npz")
        global_params = {k: npz[k] for k in npz.files}
        md = json.loads((in_dir / "metadata.json").read_text())
        covariate_names = md.pop("covariate_names", [])
        with (in_dir / "model_spec.pkl").open("rb") as f:
            spec = pickle.load(f)
        return cls(
            global_params=global_params,
            metadata=md,
            model_spec=spec,
            covariate_names=covariate_names,
        )
