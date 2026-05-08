"""Head-to-head LDA comparison driver.

Runs OnlineLDA and pyspark.ml.clustering.LDA on the same OMOP parquet,
recovers ground truth from the simulator's true_topic_id oracle, and
produces a three-panel JS-similarity biplot:

    [ ours vs truth ] [ mllib vs truth ] [ ours vs mllib ]

Each panel is prevalence-ordered. Diagonal-dominance after ordering = topic
agreement; off-diagonal smear localizes split/merge failures.

Usage:
    poetry run python analysis/local/compare_lda_local.py \\
        --input data/simulated/omop_N1000_seed42.parquet \\
        --K 10 \\
        --max-iterations 200 \\
        --K-true 10 \\
        --output data/runs/compare_<timestamp>
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless figure rendering
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession

from charmpheno.evaluate.lda_compare import run_mllib, run_ours
from charmpheno.evaluate.topic_alignment import (
    alignment_biplot_data,
    ground_truth_from_oracle,
)
from charmpheno.omop import load_omop_parquet, to_bow_dataframe
from spark_vi.core import BOWDocument, VIConfig

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("compare_lda_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def _render_three_panel_biplot(
    ours_vs_truth: dict, mllib_vs_truth: dict, ours_vs_mllib: dict,
    output_path: Path,
) -> None:
    """Render the three-panel JS biplot figure to PNG.

    All panels share the JS color scale; each axis is prevalence-sorted in
    its own implementation's frame.
    """
    js_max = max(
        ours_vs_truth["js_matrix"].max(),
        mllib_vs_truth["js_matrix"].max(),
        ours_vs_mllib["js_matrix"].max(),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("ours vs truth", ours_vs_truth, "ours (prev. desc)", "truth (prev. desc)"),
        ("mllib vs truth", mllib_vs_truth, "mllib (prev. desc)", "truth (prev. desc)"),
        ("ours vs mllib", ours_vs_mllib, "ours (prev. desc)", "mllib (prev. desc)"),
    ]
    for ax, (title, data, ylab, xlab) in zip(axes, panels):
        im = ax.imshow(data["js_matrix"], vmin=0.0, vmax=js_max,
                        cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                  label="JS divergence (nats)")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_perf_table(
    ours, mllib, csv_path: Path,
) -> None:
    """Write a single-row CSV summarizing per-iteration time and totals."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "wall_time_seconds", "mean_per_iter_seconds",
                    "n_iter", "final_elbo_or_loglik"])
        w.writerow([
            "ours", ours.wall_time_seconds,
            float(np.mean(ours.per_iter_seconds)),
            len(ours.per_iter_seconds),
            ours.elbo_trace[-1] if ours.elbo_trace else "",
        ])
        w.writerow([
            "mllib", mllib.wall_time_seconds,
            float(np.mean(mllib.per_iter_seconds)),
            len(mllib.per_iter_seconds),
            mllib.final_log_likelihood,
        ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--K-true", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--mini-batch-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    spark = _build_spark()
    try:
        df_raw = load_omop_parquet(str(args.input), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(df_raw)
        bow_df.persist()
        rdd = bow_df.rdd.map(BOWDocument.from_spark_row).persist()

        cfg = VIConfig(
            max_iterations=args.max_iterations,
            mini_batch_fraction=args.mini_batch_fraction,
            random_seed=args.seed,
            convergence_tol=1e-6,
            # Match MLlib's pyspark.ml.clustering.LDA defaults so the head-
            # to-head is apples-to-apples on the learning-rate schedule.
            # Our framework default (tau0=1.0, kappa=0.7) follows Hoffman
            # 2013's general SVI recommendation; MLlib uses Hoffman 2010's
            # LDA-tuned settings (tau0=1024, kappa=0.51) which are gentler
            # in the early iterations.
            #
            # Empirical note: on long-tailed asymmetric-prior corpora the
            # tau0=1024 schedule recovers fewer rare topics than tau0=1.0
            # — the gentle warmup lets dominant topics consume the rare
            # ones' evidence before they differentiate. The fair-comparison
            # contract here trumps the recovery-quality knob; tune tau0/
            # kappa per workload at the call site if you need it.
            learning_rate_tau0=1024.0,
            learning_rate_kappa=0.51,
        )
        log.info("Running OnlineLDA...")
        ours = run_ours(rdd, vocab_size=len(vocab_map), K=args.K, config=cfg)
        log.info("Running MLlib LDA...")
        mllib = run_mllib(df=bow_df, vocab_size=len(vocab_map), K=args.K,
                          max_iter=args.max_iterations,
                          subsampling_rate=args.mini_batch_fraction,
                          seed=args.seed,
                          optimize_doc_concentration=False)
        log.info("Recovering ground truth from oracle...")
        true_beta, true_prev = ground_truth_from_oracle(
            df_raw, vocab_map, K_true=args.K_true,
        )

        log.info("Computing biplot data...")
        ours_vs_truth = alignment_biplot_data(
            ours.topics_matrix, ours.topic_prevalence, true_beta, true_prev,
        )
        mllib_vs_truth = alignment_biplot_data(
            mllib.topics_matrix, mllib.topic_prevalence, true_beta, true_prev,
        )
        ours_vs_mllib = alignment_biplot_data(
            ours.topics_matrix, ours.topic_prevalence,
            mllib.topics_matrix, mllib.topic_prevalence,
        )

        _render_three_panel_biplot(
            ours_vs_truth, mllib_vs_truth, ours_vs_mllib,
            output_path=args.output / "biplots.png",
        )
        _write_perf_table(ours, mllib, csv_path=args.output / "perf_table.csv")

        np.savez(
            args.output / "artifacts.npz",
            ours_topics=ours.topics_matrix,
            ours_prevalence=ours.topic_prevalence,
            ours_elbo=np.asarray(ours.elbo_trace) if ours.elbo_trace else np.empty(0),
            mllib_topics=mllib.topics_matrix,
            mllib_prevalence=mllib.topic_prevalence,
            true_beta=true_beta,
            true_prevalence=true_prev,
        )

        log.info("Wrote biplots.png, perf_table.csv, artifacts.npz to %s",
                 args.output)
        rdd.unpersist()
        bow_df.unpersist()
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
