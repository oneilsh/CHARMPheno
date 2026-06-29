# analysis/cloud/tests/test_stm_driver_partition.py  (new; mirror the dir of other cloud tests)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/cloud on path

from stm_bigquery_cloud import build_topic_block_partition


def test_build_partition_from_cli():
    p = build_topic_block_partition(
        group_var="source_cohort", background_k=30,
        foreground_arg="cancer:10,dementia:10", K=50)
    assert p.K == 50
    assert p.groups == ("cancer", "dementia")


def test_build_partition_none_when_unset():
    assert build_topic_block_partition(
        group_var="source_cohort", background_k=None,
        foreground_arg=None, K=40) is None


def test_build_partition_k_mismatch_raises():
    import pytest
    with pytest.raises(ValueError, match="K"):
        build_topic_block_partition(
            group_var="source_cohort", background_k=30,
            foreground_arg="cancer:10", K=50)  # 30+10 != 50


_REQUIRED_ARGS = [
    "--cdr", "p.d", "--billing", "b",
    "--covariate-formula", "~ sex",
    "--categorical-cols", "sex",
    "--continuous-cols", "age",
    "--out-dir", "/tmp/out",
]


def test_parse_args_hardening_flags_default_on():
    from stm_bigquery_cloud import parse_args
    args = parse_args(_REQUIRED_ARGS)
    assert args.reference_topic is True
    assert args.sigma_prior_scale is None
    assert args.sigma_prior_count == 0.0
    assert args.spectral_init is True


def test_parse_args_hardening_flags_set():
    from stm_bigquery_cloud import parse_args
    args = parse_args(_REQUIRED_ARGS + [
        "--reference-topic",
        "--sigma-prior-scale", "2.0",
        "--sigma-prior-count", "500.0",
        "--spectral-init",
    ])
    assert args.reference_topic is True
    assert args.sigma_prior_scale == 2.0
    assert args.sigma_prior_count == 500.0
    assert args.spectral_init is True


def test_parse_args_hardening_flags_disabled():
    from stm_bigquery_cloud import parse_args
    args = parse_args(_REQUIRED_ARGS + [
        "--no-reference-topic",
        "--no-spectral-init",
    ])
    assert args.reference_topic is False
    assert args.spectral_init is False
