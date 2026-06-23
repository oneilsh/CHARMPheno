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
