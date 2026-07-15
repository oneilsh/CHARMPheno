# analysis/cloud/tests/test_eval_foreground_split.py  (new)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_coherence_cloud import foreground_reference_groups, main


def test_main_does_not_shadow_module_level_F():
    """Regression guard: `F` (pyspark.sql.functions) must stay a module-global
    inside main(), never a function-local.

    A stray `from pyspark.sql import functions as F` inside any branch of
    main() makes `F` a local for the *entire* function (Python decides scope
    at compile time), so the first use of `F` — which runs long before that
    branch — raises UnboundLocalError. That is exactly the bug that crashed
    the gated-STM eval. Python records function-local names in co_varnames and
    globals/attrs in co_names, so the guard is: `F` is referenced but not
    local.
    """
    assert "F" in main.__code__.co_names, "main() no longer references F"
    assert "F" not in main.__code__.co_varnames, (
        "main() rebinds F as a function-local (e.g. a nested "
        "`from pyspark.sql import functions as F`); this shadows the "
        "module-level import and raises UnboundLocalError on F's first use."
    )


def test_maps_topics_to_groups_or_background():
    spec = {"group_var": "source_cohort", "background_k": 2,
            "foreground": [["cancer", 2], ["dementia", 1]]}
    assert foreground_reference_groups(spec) == {
        0: None, 1: None, 2: "cancer", 3: "cancer", 4: "dementia"}


def test_none_spec_yields_empty_map():
    assert foreground_reference_groups(None) == {}
