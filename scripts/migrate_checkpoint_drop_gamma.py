"""One-shot checkpoint migration: backfill θ aggregates / corpus_prevalence.

Background
----------
OnlineLDA's shim (spark_vi.mllib.topic.lda) does NOT persist γ (the per-doc
Dirichlet posterior) in global_params — it only stores lambda, alpha, eta.
γ is computed on-demand via CAVI inside _transform().  As a result, θ
aggregates (theta_histogram, theta_percentiles, corpus_prevalence, n_patients)
are computed at fit time by calling model.transform(fit_df) inside the fit
drivers, not by inspecting global_params.  No γ is ever written to disk.

LDA branch
----------
Because γ is never persisted, this script's LDA branch is minimal:

  - If the checkpoint already has theta_histogram in metadata AND no γ in
    global_params: no-op (idempotent — aggregates were already computed at
    fit time by the driver).
  - If neither γ nor aggregates are present: raises ValueError — re-fit is
    required (there is no way to reconstruct θ aggregates post-hoc without
    γ or the original documents + model).

The "γ present" branches from older designs are dead code; they are omitted.

HDP branch
----------
For an HDP checkpoint (no γ):
  - If corpus_prevalence is missing from metadata, compute and write it from
    the GEM stick masses (u, v).
  - No histogram, no percentiles (HDP has no per-doc θ).
  - If corpus_prevalence is already present: no-op.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np



def _hdp_corpus_prevalence(u: np.ndarray, v: np.ndarray) -> list[float]:
    """Compute full-length stick-breaking corpus prevalence from Beta params u, v.

    Mirrors the formula used in charmpheno.export.model_adapter.adapt_hdp,
    but writes the full T-length list (not sliced to top-K — the adapter
    slices by `order` at display time).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    stick_means = u / (u + v)
    remainder = np.cumprod(np.concatenate([[1.0], 1 - stick_means[:-1]]))
    e_beta = stick_means * remainder  # corpus-level mass per truncation index
    return e_beta.tolist()


def migrate(
    checkpoint_path: str | Path,
    out_path: str | Path | None = None,
    *,
    n_bins: int = 50,
    min_count: int = 20,
) -> dict:
    """Migrate one checkpoint. Returns a small status dict.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the existing checkpoint directory (must contain manifest.json).
    out_path : str | Path | None
        Destination directory for the migrated checkpoint. If None, the
        migration is performed in-place (checkpoint_path is overwritten).
        When out_path differs from checkpoint_path, the source is NOT
        modified.
    n_bins : int
        Number of histogram bins; passed to compute_theta_aggregates.
    min_count : int
        Small-cell suppression threshold; passed to compute_theta_aggregates.

    Returns
    -------
    dict with keys:
        'kind'      : 'lda' | 'hdp'
        'action'    : 'migrated' | 'noop' | 'recomputed'
        'n_patients': int | None  (None for HDP)
    """
    # Late imports: spark_vi and charmpheno are on the path at runtime but not
    # necessarily during a bare import of this module.
    from spark_vi.core.result import VIResult
    from spark_vi.io import load_result, save_result
    from charmpheno.export.theta_aggregates import compute_theta_aggregates
    # Use the canonical alias sets from model_adapter to avoid drift.
    from charmpheno.export.model_adapter import _HDP_ALIASES as HDP_ALIASES, _LDA_ALIASES as LDA_ALIASES

    src = Path(checkpoint_path)
    in_place = out_path is None
    dst = src if in_place else Path(out_path)

    print(f"[migrate] loading checkpoint from {src}", flush=True)
    result = load_result(src)

    model_class = str(result.metadata.get("model_class", "lda")).lower()

    # ------------------------------------------------------------------
    # HDP branch
    # ------------------------------------------------------------------
    if model_class in HDP_ALIASES:
        print(f"[migrate] detected HDP checkpoint (model_class={model_class!r})", flush=True)
        if "corpus_prevalence" in result.metadata:
            print("[migrate] hdp: corpus_prevalence already present — noop", flush=True)
            return {"kind": "hdp", "action": "noop", "n_patients": None}

        gp = result.global_params
        if "u" not in gp or "v" not in gp:
            raise ValueError(
                "HDP checkpoint missing 'u' or 'v' in global_params; "
                "cannot compute corpus_prevalence without stick parameters."
            )
        u = np.asarray(gp["u"], dtype=np.float64)
        v = np.asarray(gp["v"], dtype=np.float64)
        corpus_prevalence = _hdp_corpus_prevalence(u, v)

        new_metadata = dict(result.metadata)
        new_metadata["corpus_prevalence"] = corpus_prevalence

        new_result = VIResult(
            global_params=result.global_params,
            elbo_trace=result.elbo_trace,
            n_iterations=result.n_iterations,
            converged=result.converged,
            metadata=new_metadata,
            diagnostic_traces=result.diagnostic_traces,
        )
        print(f"[migrate] hdp: writing corpus_prevalence (T={len(corpus_prevalence)}) to {dst}", flush=True)
        save_result(new_result, dst)
        print(f"[migrate] hdp: migrated", flush=True)
        return {"kind": "hdp", "action": "migrated", "n_patients": None}

    # ------------------------------------------------------------------
    # LDA branch
    # ------------------------------------------------------------------
    if model_class not in LDA_ALIASES:
        raise ValueError(f"unsupported model class: {model_class}")
    print(f"[migrate] detected LDA checkpoint (model_class={model_class!r})", flush=True)
    has_aggregates = "theta_histogram" in result.metadata

    # --- idempotent no-op: aggregates already present (written at fit time) ---
    if has_aggregates:
        print("[migrate] lda: aggregates already present — noop", flush=True)
        return {"kind": "lda", "action": "noop", "n_patients": result.metadata.get("n_patients")}

    # --- cannot migrate: γ was never persisted by OnlineLDA; re-fit needed ---
    raise ValueError(
        "LDA checkpoint has no 'theta_histogram' in metadata, and γ is never "
        "persisted by OnlineLDA's shim — post-hoc migration is not possible. "
        "Re-fit the model to produce a checkpoint with theta aggregates."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate an LDA/HDP checkpoint: compute θ aggregates and drop γ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the checkpoint directory to migrate.",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path for the migrated checkpoint. Defaults to in-place.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of histogram bins for theta_histogram.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Small-cell suppression threshold.",
    )

    args = parser.parse_args(argv)

    try:
        status = migrate(
            checkpoint_path=args.checkpoint,
            out_path=args.out,
            n_bins=args.n_bins,
            min_count=args.min_count,
        )
        print(f"[migrate] done: {status}", flush=True)
        return 0
    except Exception as exc:
        print(f"[migrate] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
