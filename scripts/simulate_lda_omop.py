"""Generate synthetic OMOP-shaped data using a fixed LDA β distribution
and the standard LDA generative process.

For each patient:
    θ_p ~ Dirichlet(α · 1)                    (topic mixture weights)
    N_v ~ Poisson(visits_per_patient_mean)    (visits for this patient)
    for each of N_v visits:
        N_c ~ Poisson(codes_per_visit_mean)   (codes for this visit)
        for each of N_c codes:
            z   ~ Categorical(θ_p)            (pick a topic)
            w   ~ Categorical(β[z, :])        (pick a concept from topic z)
            emit (person_id, visit_occurrence_id, w.concept_id, w.concept_name, z)

Output columns:
    person_id:int, visit_occurrence_id:int, concept_id:int,
    concept_name:str, true_topic_id:int

`true_topic_id` is oracle metadata for evaluation; training code must not read it.

Default output: data/simulated/omop_N<n>_seed<seed>.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BETA_PATH = Path("data/cache/lda_beta_topk.parquet")
DEFAULT_OUTPUT_DIR = Path("data/simulated")


def _beta_as_matrix(beta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Pivot the long-form beta DataFrame to (K × V) matrix form.

    Returns:
        beta_mat: float array shape (K, V), rows sum to 1.
        concept_ids: int array length V (column index → concept_id).
        concept_names: dict mapping concept_id to concept_name.
    """
    concept_ids = np.sort(beta["concept_id"].unique())
    topic_ids = np.sort(beta["topic_id"].unique())
    cid_to_col = {cid: i for i, cid in enumerate(concept_ids)}
    tid_to_row = {tid: i for i, tid in enumerate(topic_ids)}

    mat = np.zeros((len(topic_ids), len(concept_ids)), dtype=np.float64)
    for row in beta.itertuples(index=False):
        mat[tid_to_row[row.topic_id], cid_to_col[row.concept_id]] = row.weight

    # Defensive renorm — the filter script already does this but if the user
    # passes a hand-constructed beta, we keep the math sane.
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.where(row_sums > 0, row_sums, 1.0)

    names = dict(zip(beta["concept_id"], beta["concept_name"]))
    return mat, concept_ids, names


def simulate(
    beta: pd.DataFrame,
    n_patients: int,
    theta_alpha: float,
    visits_per_patient_mean: float,
    codes_per_visit_mean: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a synthetic OMOP-shaped DataFrame from a fixed β.

    Returns a DataFrame with columns person_id, visit_occurrence_id,
    concept_id, concept_name, true_topic_id.
    """
    rng = np.random.default_rng(seed)
    beta_mat, concept_ids, concept_names = _beta_as_matrix(beta)
    K, V = beta_mat.shape

    # Patient-level θ_p ~ Dirichlet(α · 1_K)
    alpha_vec = np.full(K, theta_alpha, dtype=np.float64)
    theta = rng.dirichlet(alpha_vec, size=n_patients)  # (n_patients, K)

    rows: list[tuple[int, int, int, str, int]] = []
    visit_counter = 0
    for p in range(n_patients):
        n_visits = max(1, int(rng.poisson(visits_per_patient_mean)))
        for _ in range(n_visits):
            visit_counter += 1
            n_codes = max(1, int(rng.poisson(codes_per_visit_mean)))
            z = rng.choice(K, size=n_codes, p=theta[p])
            for zi in z:
                w_col = rng.choice(V, p=beta_mat[zi])
                cid = int(concept_ids[w_col])
                rows.append((p, visit_counter, cid, concept_names[cid], int(zi)))

    return pd.DataFrame(
        rows,
        columns=["person_id", "visit_occurrence_id", "concept_id",
                 "concept_name", "true_topic_id"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=Path, default=DEFAULT_BETA_PATH,
                        help=f"Input filtered-beta parquet (default {DEFAULT_BETA_PATH})")
    parser.add_argument("--n-patients", type=int, default=10_000)
    parser.add_argument("--theta-alpha", type=float, default=0.1,
                        help="Symmetric Dirichlet concentration on θ (default 0.1)")
    parser.add_argument("--visits-per-patient-mean", type=float, default=3.0)
    parser.add_argument("--codes-per-visit-mean", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Reading beta from %s", args.beta)
    beta = pd.read_parquet(args.beta)

    df = simulate(
        beta=beta,
        n_patients=args.n_patients,
        theta_alpha=args.theta_alpha,
        visits_per_patient_mean=args.visits_per_patient_mean,
        codes_per_visit_mean=args.codes_per_visit_mean,
        seed=args.seed,
    )
    log.info("Generated %d rows for %d patients", len(df), args.n_patients)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"omop_N{args.n_patients}_seed{args.seed}.parquet"
    df.to_parquet(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "n_patients": args.n_patients,
        "theta_alpha": args.theta_alpha,
        "visits_per_patient_mean": args.visits_per_patient_mean,
        "codes_per_visit_mean": args.codes_per_visit_mean,
        "seed": args.seed,
        "beta_path": str(args.beta),
    }, indent=2))
    log.info("Wrote %s (and %s)", out_path, meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
