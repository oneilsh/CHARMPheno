"""End-to-end local: gated sim parquet -> OnlineSTM via the shim -> checkpoint.

Sibling of fit_lda_local.py for gated STM. Builds a local SparkSession, loads
the gated OMOP + person parquets, builds the patient_cohort BOW (doc_id =
"source_cohort:person_id") and the covariate DataFrame (~ C(sex) + age, keyed
per person), and fits StreamingSTM with a TopicBlockPartition + doc_group_col.
Saves the STMModel checkpoint with full metadata + a covariates.parquet so the
local dashboard can compute masked prevalence offline.

source_cohort is the gating group label, NOT a covariate (it must not be in the
formula); it is materialized from doc_id, exactly as the cloud driver does.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from charmpheno.omop import doc_spec_from_cli, load_omop_parquet, to_bow_dataframe
from charmpheno.omop.covariates import build_patient_covariate_df
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.mllib.topic.stm import StreamingSTM

log = logging.getLogger(__name__)


def _extract_categorical_levels(
    model_spec,
    categorical_cols: list[str],
) -> dict[str, dict]:
    """Extract {var: {"levels": [...], "reference": "..."}} from a formulaic ModelSpec.

    Uses encoder_state (formulaic >= 0.5) to read the full category list and
    derive the reference (dropped) level for treatment-coded contrasts.  With
    TreatmentContrasts the reference is the first sorted category when the base
    is UNSET, or the explicit base value otherwise.

    Returns an empty dict for any variable whose info cannot be recovered, so
    the caller degrades gracefully rather than raising.
    """
    result: dict[str, dict] = {}
    if model_spec is None:
        return result

    encoder_state = getattr(model_spec, "encoder_state", None) or {}
    for factor_key, state_tuple in encoder_state.items():
        if not (isinstance(state_tuple, tuple) and len(state_tuple) == 2):
            continue
        kind, state_dict = state_tuple
        # Only process categorical factors.
        if not (hasattr(kind, "value") and kind.value == "categorical"):
            continue
        categories = state_dict.get("categories") if isinstance(state_dict, dict) else None
        if not categories:
            continue
        categories = list(categories)

        # Derive the variable name: strip the C(...) wrapper if present.
        import re as _re
        m = _re.match(r"^C\(\s*([^,)\s]+)", factor_key)
        var = m.group(1) if m else factor_key

        # Derive the reference (dropped) level from the contrasts state.
        reference = ""
        contrasts_state = state_dict.get("contrasts") if isinstance(state_dict, dict) else None
        if contrasts_state is not None:
            try:
                c = contrasts_state.contrasts
                base = getattr(c, "base", None)
                # formulaic represents an unset base as a Sentinel with value
                # "UNSET". When unset, treatment coding drops the first sorted
                # category, which is categories[0].
                if base is None or "UNSET" in str(base):
                    reference = categories[0]
                else:
                    reference = str(base)
            except Exception:
                # Fall back to first category if introspection fails.
                reference = categories[0] if categories else ""

        result[var] = {"levels": categories, "reference": reference}

    return result


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("fit_stm_local")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def _parse_foreground(s):
    out = []
    for piece in s.split(","):
        g, _, k = piece.partition(":")
        out.append((g.strip(), int(k)))
    return tuple(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omop", type=Path, required=True)
    p.add_argument("--person", type=Path, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--background-k", type=int, required=True)
    p.add_argument("--foreground", required=True, help="'g:K,g:K'")
    p.add_argument("--group-var", default="source_cohort")
    p.add_argument("--covariate-formula", default="~ C(sex) + age")
    p.add_argument("--categorical-cols", default="sex")
    p.add_argument("--continuous-cols", default="age")
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--subsampling-rate", type=float, default=1.0)
    p.add_argument("--tau0", type=float, default=64.0)
    p.add_argument("--kappa", type=float, default=0.7)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-prior-scale", type=float, default=None,
                   help="Inverse-gamma Sigma-prior scale s0 (off when unset).")
    p.add_argument("--sigma-prior-count", type=float, default=0.0,
                   help="Inverse-gamma Sigma-prior pseudo-count c0 (default 0).")
    p.add_argument("--reference-topic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="K-1 reference parameterization (default on, insight 0030; "
                        "--no-reference-topic to disable).")
    p.add_argument("--spectral-init", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Anchor-word spectral beta seed (default on, insight 0030; "
                        "--no-spectral-init to disable).")
    p.add_argument("--spectral-method", choices=["dense", "scalable"], default="dense",
                   help="Spectral-init co-occurrence method: 'dense' (exact V×V on the "
                        "driver, the validated default) or 'scalable' (random-projection "
                        "sketch for large vocabularies).")
    p.add_argument("--spectral-d", type=int, default=None,
                   help="Scalable projection dimension (default ~1000 per Mimno/Arora).")
    p.add_argument("--spectral-min-doc-freq", type=int, default=5,
                   help="Scalable absolute document-frequency floor for anchor candidates.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    foreground = _parse_foreground(args.foreground)
    partition = TopicBlockPartition(group_var=args.group_var,
                                    background_k=args.background_k,
                                    foreground=foreground)
    if partition.K != args.K:
        raise SystemExit(f"partition K {partition.K} != --K {args.K}")
    cat_cols = [c for c in args.categorical_cols.split(",") if c]
    cont_cols = [c for c in args.continuous_cols.split(",") if c]
    doc_spec = doc_spec_from_cli("patient_cohort", min_doc_length=None)

    spark = _build_spark()
    try:
        omop = load_omop_parquet(str(args.omop), spark=spark)
        bow_df, vocab_map = to_bow_dataframe(omop, doc_spec=doc_spec)
        # source_cohort from doc_id (gating label; not a covariate).
        bow_df = bow_df.withColumn(
            "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))

        person_df = spark.read.parquet(str(args.person))
        cov_df, model_spec, covariate_names = build_patient_covariate_df(
            person_df, covariate_formula=args.covariate_formula,
            categorical_cols=cat_cols, continuous_cols=cont_cols,
            key_cols=("person_id",))
        joined = bow_df.join(F.broadcast(cov_df), on="person_id", how="inner")

        est = StreamingSTM(
            K=args.K, features_col="features", covariates_col="covariates",
            covariate_names=covariate_names, topic_blocks=partition,
            doc_group_col="source_cohort", random_seed=args.seed,
            sigma_prior_scale=args.sigma_prior_scale,
            sigma_prior_count=args.sigma_prior_count,
            reference_topic=args.reference_topic,
            spectral_init=args.spectral_init,
            spectral_method=args.spectral_method,
            spectral_d=args.spectral_d,
            spectral_min_doc_freq=args.spectral_min_doc_freq)
        model = est.fit(joined, max_iter=args.max_iter,
                        subsampling_rate=args.subsampling_rate,
                        tau0=args.tau0, kappa=args.kappa)

        # Concept names from the simulator's concept_name column.
        name_rows = (omop.select("concept_id", "concept_name")
                     .dropDuplicates(["concept_id"]).collect())
        name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}
        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid

        model.metadata["corpus_manifest"] = {
            "cdr": "local", "source_table": "condition_occurrence",
            "cohort": "gated_sim", "prior_obs_days": 0, "person_mod": 1,
            "doc_spec": doc_spec.manifest(), "vocab_size": len(vocab_map),
            "vocab": vocab_list, "name_by_id": name_by_id,
            "min_patient_count": args.min_patient_count,
            "topic_block_spec": partition.to_dict(),
        }
        model.metadata["vocab"] = vocab_list
        model.metadata["name_by_id"] = name_by_id
        model.metadata["covariate_manifest"] = {
            "covariate_formula": args.covariate_formula,
            "categorical_cols": cat_cols, "continuous_cols": cont_cols,
            "covariate_names": covariate_names,
            "categorical_levels": _extract_categorical_levels(
                model_spec, cat_cols,
            ),
        }
        model.metadata["model_class"] = "stm"
        model.metadata["concept_names"] = {str(k): v for k, v in name_by_id.items()}
        model.metadata["concept_domains"] = {str(k): "condition" for k in name_by_id}

        args.out_dir.mkdir(parents=True, exist_ok=True)
        model.save(args.out_dir)

        # Persist per-doc covariate vectors + group label for offline masked
        # prevalence (one row per joined doc).
        from pyspark.ml.functions import vector_to_array
        (joined.select("person_id", "source_cohort",
                       vector_to_array("covariates").alias("covariates"))
         .toPandas().to_parquet(args.out_dir / "covariates.parquet", index=False))

        log.info("wrote gated STM checkpoint to %s (K=%d, V=%d)",
                 args.out_dir, args.K, len(vocab_map))
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
