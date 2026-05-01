"""Generate synthetic OMOP-shaped data using a fixed LDA β distribution
and the standard LDA generative process.

For each patient:
    θ_p ~ Dirichlet(α)                        (topic mixture weights)
    N_v ~ Poisson(visits_per_patient_mean)    (visits for this patient)
    for each of N_v visits:
        N_c ~ Poisson(codes_per_visit_mean)   (codes for this visit)
        for each of N_c codes:
            z   ~ Categorical(θ_p)            (pick a topic)
            w   ~ Categorical(β[z, :])        (pick a concept from topic z)
            emit (person_id, visit_occurrence_id, w.concept_id, w.concept_name, z)

Two prior shapes for α
----------------------
* Symmetric (default): α_k = theta_alpha for all k. E[θ_k] is uniform.
* Asymmetric (when --topic-metadata is provided): α_k =
  K * theta_alpha * Ũ_k where Ũ_k = U_k / Σ U_k' is the upstream topic-
  usage fraction renormalized over the topics actually present in β.
  Mean E[θ_k] = Ũ_k, total concentration α_0 = K * theta_alpha — the
  same total concentration as the symmetric case, so theta_alpha keeps
  its old "per-topic" interpretation. The choice of an asymmetric prior
  on θ (paired with a symmetric prior on β) is broadly motivated by
  Wallach, Mimno, McCallum 2009, "Rethinking LDA: Why Priors Matter";
  note that paper learns α from data via empirical Bayes, while we feed
  in U from external metadata as a fixed base measure — different
  mechanism, same family of asymmetry.

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


def _asymmetric_alpha(
    beta: pd.DataFrame,
    topic_metadata: pd.DataFrame,
    theta_alpha: float,
) -> np.ndarray:
    """Build α_k = K * theta_alpha * Ũ_k from a topic-usage sidecar.

    Ũ is U_k renormalized over topics present in `beta` (so subsetting
    β to a topic slice keeps mean(θ) on the simplex). Total
    concentration α_0 = K * theta_alpha matches the symmetric case.
    """
    topic_ids = np.sort(beta["topic_id"].unique())
    meta = topic_metadata.set_index("topic_id")
    missing = [tid for tid in topic_ids if tid not in meta.index]
    if missing:
        raise ValueError(
            f"topic_metadata is missing usage_pct for topic_id(s): {missing}"
        )
    u = meta.loc[topic_ids, "usage_pct"].to_numpy(dtype=np.float64)
    if (u < 0).any():
        raise ValueError("usage_pct values must be non-negative")
    if u.sum() <= 0:
        raise ValueError("usage_pct values sum to zero — no signal to use")
    u_tilde = u / u.sum()
    K = len(topic_ids)
    return K * theta_alpha * u_tilde


def simulate(
    beta: pd.DataFrame,
    n_patients: int,
    theta_alpha: float,
    visits_per_patient_mean: float,
    codes_per_visit_mean: float,
    seed: int,
    topic_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a synthetic OMOP-shaped DataFrame from a fixed β.

    Returns a DataFrame with columns person_id, visit_occurrence_id,
    concept_id, concept_name, true_topic_id.

    If `topic_metadata` is provided (columns: topic_id, usage_pct, ...)
    the per-doc topic prior is asymmetric: α_k = K * theta_alpha * Ũ_k
    with Ũ renormalized over the topic_ids in β. Otherwise the prior is
    symmetric: α_k = theta_alpha for all k.
    """
    rng = np.random.default_rng(seed)
    beta_mat, concept_ids, concept_names = _beta_as_matrix(beta)
    K, V = beta_mat.shape

    if topic_metadata is None:
        alpha_vec = np.full(K, theta_alpha, dtype=np.float64)
    else:
        alpha_vec = _asymmetric_alpha(beta, topic_metadata, theta_alpha)
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
    parser.add_argument("--topic-metadata", type=Path, default=None,
                        help="Optional per-topic metadata parquet "
                             "(columns: topic_id, usage_pct, ...). "
                             "When given, θ uses an asymmetric Dirichlet "
                             "with mean = renormalized usage_pct.")
    parser.add_argument("--n-patients", type=int, default=10_000)
    parser.add_argument("--theta-alpha", type=float, default=0.1,
                        help="Per-topic Dirichlet concentration on θ "
                             "(default 0.1). Total α_0 = K * theta_alpha "
                             "regardless of symmetric vs asymmetric prior.")
    parser.add_argument("--visits-per-patient-mean", type=float, default=3.0)
    parser.add_argument("--codes-per-visit-mean", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Reading beta from %s", args.beta)
    beta = pd.read_parquet(args.beta)

    topic_metadata = None
    if args.topic_metadata is not None:
        log.info("Reading topic metadata from %s", args.topic_metadata)
        topic_metadata = pd.read_parquet(args.topic_metadata)

    df = simulate(
        beta=beta,
        n_patients=args.n_patients,
        theta_alpha=args.theta_alpha,
        visits_per_patient_mean=args.visits_per_patient_mean,
        codes_per_visit_mean=args.codes_per_visit_mean,
        seed=args.seed,
        topic_metadata=topic_metadata,
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
        "topic_metadata_path": (
            str(args.topic_metadata) if args.topic_metadata is not None else None
        ),
        "theta_prior": "asymmetric" if topic_metadata is not None else "symmetric",
    }, indent=2))
    log.info("Wrote %s (and %s)", out_path, meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
