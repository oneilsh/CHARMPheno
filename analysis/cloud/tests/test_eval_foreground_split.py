# analysis/cloud/tests/test_eval_foreground_split.py  (new)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_coherence_cloud import foreground_reference_groups


def test_maps_topics_to_groups_or_background():
    spec = {"group_var": "source_cohort", "background_k": 2,
            "foreground": [["cancer", 2], ["dementia", 1]]}
    assert foreground_reference_groups(spec) == {
        0: None, 1: None, 2: "cancer", 3: "cancer", 4: "dementia"}


def test_none_spec_yields_empty_map():
    assert foreground_reference_groups(None) == {}
