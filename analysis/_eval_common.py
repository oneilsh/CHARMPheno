"""Shared eval-driver utilities for both local and cloud NPMI coherence drivers.

Currently houses `verify_split_contract`, factored out of
`analysis/local/eval_coherence.py` so the cloud eval driver
(`analysis/cloud/eval_coherence_cloud.py`) can reuse the exact same
fit/eval split-provenance check without duplicating logic.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def verify_split_contract(result, *, holdout_fraction: float, seed: int) -> None:
    """Verify the eval CLI args match the split provenance stamped at fit time.

    The fit driver (fit_lda_local.py / fit_hdp_local.py / lda_bigquery_cloud.py)
    stamps split parameters under VIResult.metadata['split']. If absent, the
    model was fit on the full corpus and the eval is optimistically biased
    (the held-out patients were seen during training); we warn loudly. If
    present but mismatched, we abort.
    """
    split_meta = result.metadata.get("split")
    if split_meta is None or not split_meta.get("applied", False):
        log.warning(
            "checkpoint has no split provenance; the fit driver was likely run "
            "without --holdout-fraction. NPMI on the hashed holdout will be "
            "OPTIMISTICALLY BIASED because the model saw those patients during "
            "fitting. Re-fit with matching --holdout-fraction and --holdout-seed "
            "for an honest benchmark."
        )
        return
    fit_frac = float(split_meta.get("holdout_fraction", -1.0))
    fit_seed = int(split_meta.get("holdout_seed", -1))
    if (abs(fit_frac - holdout_fraction) > 1e-9) or (fit_seed != seed):
        raise SystemExit(
            "split mismatch: checkpoint was fit with "
            f"holdout_fraction={fit_frac}, seed={fit_seed} but eval was invoked "
            f"with holdout_fraction={holdout_fraction}, seed={seed}. "
            "Re-run with matching values (the eval holdout must be the held-out "
            "portion the model did NOT see)."
        )
