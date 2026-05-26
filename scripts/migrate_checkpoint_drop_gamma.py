"""One-shot checkpoint migration: compute θ aggregates, drop γ.

For an LDA checkpoint with γ in global_params:
  - Compute theta_histogram (K × n_bins, fractions, suppressed at count<min_count),
    theta_percentiles (K × 5 dicts), corpus_prevalence (K floats), n_patients (int)
    via charmpheno.export.theta_aggregates.compute_theta_aggregates(gamma).
  - Write all four into result.metadata.
  - Drop "gamma" from result.global_params.
  - Re-save the checkpoint and remove the orphan params/gamma.npy file when
    migrating in-place.

For an HDP checkpoint (no γ):
  - If corpus_prevalence is missing from metadata, write it from the
    GEM stick masses (u, v).
  - No histogram, no percentiles (HDP has no per-doc θ).

Idempotency:
  - If the LDA checkpoint already has theta_histogram in metadata AND no γ
    in global_params, the script is a no-op and exits cleanly.
  - If γ AND aggregates are both present (unusual — possibly a re-fit
    collision), the script warns, recomputes aggregates from γ, and drops γ.
  - If neither γ nor aggregates are present, the script raises ValueError
    (cannot migrate; re-fit needed).

After γ is dropped, the empirical θ distribution is gone. The histogram
cannot be re-computed without re-running the fit. Bin edges and small-cell
threshold are baked into the migrated metadata at this step.
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
    gp = result.global_params
    has_gamma = "gamma" in gp
    has_aggregates = "theta_histogram" in result.metadata

    # --- idempotent no-op ---
    if not has_gamma and has_aggregates:
        print("[migrate] lda: already migrated (no gamma, aggregates present) — noop", flush=True)
        return {"kind": "lda", "action": "noop", "n_patients": result.metadata.get("n_patients")}

    # --- cannot migrate: re-fit needed ---
    if not has_gamma and not has_aggregates:
        raise ValueError(
            "LDA checkpoint has neither 'gamma' in global_params nor "
            "'theta_histogram' in metadata. Cannot compute theta aggregates "
            "without gamma. re-fit the model to produce a migratable checkpoint."
        )

    # --- recompute if both gamma and aggregates are present (warn) ---
    if has_gamma and has_aggregates:
        print(
            "[migrate] lda: WARNING — gamma AND aggregates both present "
            "(possible re-fit collision). Recomputing aggregates from gamma and dropping gamma.",
            flush=True,
        )
        action = "recomputed"
    else:
        # Normal path: gamma present, no aggregates
        action = "migrated"

    gamma = np.asarray(gp["gamma"], dtype=np.float64)
    print(f"[migrate] lda: computing theta aggregates (N={gamma.shape[0]}, K={gamma.shape[1]}, n_bins={n_bins}, min_count={min_count})", flush=True)
    aggregates = compute_theta_aggregates(gamma, n_bins=n_bins, min_count=min_count)

    new_metadata = dict(result.metadata)
    new_metadata.update(aggregates)

    new_global_params = {k: v for k, v in gp.items() if k != "gamma"}

    new_result = VIResult(
        global_params=new_global_params,
        elbo_trace=result.elbo_trace,
        n_iterations=result.n_iterations,
        converged=result.converged,
        metadata=new_metadata,
        diagnostic_traces=result.diagnostic_traces,
    )

    print(f"[migrate] lda: saving migrated checkpoint to {dst}", flush=True)
    save_result(new_result, dst)

    # Orphan cleanup: save_result only writes the params it's given.
    # If migrating in-place, the old params/gamma.npy may still exist
    # because save_result does not delete files it is not writing.
    # We only clean up AFTER a successful save.
    if in_place:
        orphan = dst / "params" / "gamma.npy"
        if orphan.exists():
            orphan.unlink()
            print(f"[migrate] lda: removed orphan {orphan}", flush=True)

    n_patients = aggregates["n_patients"]
    print(f"[migrate] lda: {action} (N={n_patients}, K={gamma.shape[1]})", flush=True)
    return {"kind": "lda", "action": action, "n_patients": n_patients}


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
