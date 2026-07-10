"""Per-document concentration-HETEROGENEITY + burstiness diagnostic.

Companion to spark_vi.eval.topic.concentration: that module summarizes ONE
theta matrix's per-document concentration (top_mass, eff_topics -- the
inverse-Simpson / Hill-number-of-order-2 diversity index; Hill 1973,
"Diversity and Evenness: A Unifying Notation and Its Consequences", Ecology
54(2); Jost 2006, "Entropy and diversity", Oikos 113(2)). This module asks a
different question: across a corpus, how much of the SPREAD in per-document
concentration is genuine cross-topic structure versus an artifact of
within-document token burstiness (a handful of repeated tokens making a
document look more topically peaked than its underlying topic mixture
actually is)?

The approach: re-run the model's own theta inference twice per document --
once on the document's raw token counts, once on a "deduped" version where
every observed token type is capped at count 1 (so repeated tokens no
longer inflate any topic's evidence) -- and compare the resulting top_mass
/ eff_topics distributions. Because deduping always shortens the document
(a length confound baked into any count-based inference), the MEAN
concentration is expected to drop under dedup regardless of burstiness;
that shift is not informative on its own. What is informative:

  - spread_ratio_top_mass: does the cross-document SPREAD of top_mass
    collapse under dedup (std_dedup << std_raw -- consistent with
    burstiness manufacturing apparent heterogeneity) or hold up (consistent
    with genuine cross-topic structure)?
  - rank_corr_top_mass: do documents keep their relative ORDERING of
    concentration under dedup (high Spearman rank correlation -- genuine
    structure) or does the ordering scramble (burstiness artifact)?
  - burstiness_corr_top_mass: is raw top_mass itself correlated with how
    repetitive the document's tokens are (repeat_fraction), i.e. is the
    apparent peakiness actually explained by burstiness rather than topic
    content?

This module is pure numpy/scipy (no Spark, no model imports): theta
inference is injected via a caller-supplied `infer_theta(indices, counts)
-> theta` callable, so the diagnostic is agnostic to which topic model
(STM, LDA, HDP, ...) produced it. It reports numbers only -- it does not
emit a verdict, threshold, or recommendation; interpreting spread_ratio /
rank_corr / burstiness_corr to decide between a per-document-scale prior
fix and a burstiness-aware emission fix is a caller-level judgment call.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from spark_vi.eval.topic.concentration import doc_concentration


def doc_burstiness(indices, counts) -> dict:
    """Pure count statistics for one document's token bag -- NO model
    inference, just arithmetic over the (indices, counts) bag-of-words
    representation (see spark_vi.models.topic.types.BOWDocument/STMDocument
    for the convention: counts is one positive count per distinct token in
    indices).

    total = counts.sum() -- total token occurrences in the document.
    unique = len(indices) -- number of distinct token types.
    repeat_fraction = 1 - unique / total: 0 when every token is distinct
        (no repeats), approaching 1 as a document is dominated by a few
        tokens repeated many times.
    max_token_share = counts.max() / total: the single most-repeated
        token's share of all occurrences.

    Returns {"total", "unique", "repeat_fraction", "max_token_share"}.
    """
    counts = np.asarray(counts, dtype=np.float64)
    indices = np.asarray(indices)
    total = float(counts.sum())
    unique = int(indices.shape[0])
    if total > 0.0:
        repeat_fraction = 1.0 - unique / total
        max_token_share = float(counts.max()) / total
    else:
        repeat_fraction = float("nan")
        max_token_share = float("nan")
    return {
        "total": total,
        "unique": unique,
        "repeat_fraction": repeat_fraction,
        "max_token_share": max_token_share,
    }


def dedup_counts(counts) -> np.ndarray:
    """Cap each token's count at 1.0 -- i.e. treat the document as if every
    observed token type occurred exactly once, discarding within-document
    repetition. Same index set as the input; only the counts shrink."""
    return np.minimum(np.asarray(counts, dtype=np.float64), 1.0)


def _summary_block(values: np.ndarray) -> dict:
    """p10/p50/p90/mean/std over a 1-D array. None fields (not NaN) when
    empty, so the block is still valid JSON with no computed division."""
    if values.size == 0:
        return {"p10": None, "p50": None, "p90": None, "mean": None, "std": None}
    return {
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summarize_concentration_heterogeneity(
    *,
    top_mass_raw: np.ndarray,
    top_mass_dedup: np.ndarray,
    eff_topics_raw: np.ndarray,
    eff_topics_dedup: np.ndarray,
    repeat_fraction: np.ndarray,
    n_skipped: int = 0,
) -> dict:
    """Aggregate ALREADY-COMPUTED per-doc arrays into the raw-vs-dedup
    concentration-heterogeneity summary.

    This is the aggregation half of `concentration_raw_vs_dedup`, extracted
    so a caller that computes the five per-doc arrays some other way (e.g. a
    distributed per-doc pass that only ships small per-doc scalars back to
    the driver, never the documents or theta vectors themselves -- see
    `spark_vi.mllib.topic.stm.corpus_concentration_heterogeneity_rdd`) can
    reuse the identical aggregation instead of reimplementing it.

    Inputs are the five same-length 1-D arrays `concentration_raw_vs_dedup`
    computes per surviving (non-skipped) document: "top_mass_raw",
    "top_mass_dedup", "eff_topics_raw", "eff_topics_dedup",
    "repeat_fraction". `n_skipped` is the caller's own count of documents
    skipped by its guard (see `concentration_raw_vs_dedup`'s docstring for
    the guard definition) -- it is not recomputed here since this function
    never sees the skipped documents.

    Returns the SAME dict shape as `concentration_raw_vs_dedup` (see that
    function's docstring for the full field list): the five per-doc arrays
    echoed back, their "<name>_summary" blocks, "spread_ratio_top_mass",
    "rank_corr_top_mass", "burstiness_corr_top_mass", "n_docs" (derived from
    the input arrays' length), and "n_skipped" (passed through).
    """
    top_mass_raw_arr = np.asarray(top_mass_raw, dtype=np.float64)
    top_mass_dedup_arr = np.asarray(top_mass_dedup, dtype=np.float64)
    eff_topics_raw_arr = np.asarray(eff_topics_raw, dtype=np.float64)
    eff_topics_dedup_arr = np.asarray(eff_topics_dedup, dtype=np.float64)
    repeat_fraction_arr = np.asarray(repeat_fraction, dtype=np.float64)

    n_docs = int(top_mass_raw_arr.shape[0])

    if n_docs >= 2:
        std_raw = float(np.std(top_mass_raw_arr))
        spread_ratio_top_mass = (
            float(np.std(top_mass_dedup_arr)) / std_raw if std_raw > 0.0 else float("nan")
        )
        rank_corr_top_mass = float(
            spearmanr(top_mass_raw_arr, top_mass_dedup_arr).statistic
        )
        burstiness_corr_top_mass = float(
            pearsonr(top_mass_raw_arr, repeat_fraction_arr).statistic
        )
    else:
        spread_ratio_top_mass = float("nan")
        rank_corr_top_mass = float("nan")
        burstiness_corr_top_mass = float("nan")

    result = {
        "top_mass_raw": top_mass_raw_arr,
        "top_mass_dedup": top_mass_dedup_arr,
        "eff_topics_raw": eff_topics_raw_arr,
        "eff_topics_dedup": eff_topics_dedup_arr,
        "repeat_fraction": repeat_fraction_arr,
        "spread_ratio_top_mass": spread_ratio_top_mass,
        "rank_corr_top_mass": rank_corr_top_mass,
        "burstiness_corr_top_mass": burstiness_corr_top_mass,
        "n_docs": n_docs,
        "n_skipped": int(n_skipped),
    }
    for name, arr in (
        ("top_mass_raw", top_mass_raw_arr),
        ("top_mass_dedup", top_mass_dedup_arr),
        ("eff_topics_raw", eff_topics_raw_arr),
        ("eff_topics_dedup", eff_topics_dedup_arr),
        ("repeat_fraction", repeat_fraction_arr),
    ):
        result[f"{name}_summary"] = _summary_block(arr)
    return result


def concentration_raw_vs_dedup(
    docs: Sequence, infer_theta: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> dict:
    """Per-document raw-vs-dedup concentration comparison over a corpus.

    For each document (an object exposing `.indices` and `.counts`, e.g.
    BOWDocument/STMDocument), infers theta twice via the caller-supplied
    `infer_theta(indices, counts) -> theta` -- once on the raw counts, once
    on `dedup_counts(counts)` -- and computes top_mass/eff_topics
    (spark_vi.eval.topic.concentration.doc_concentration) under each, plus
    the document's burstiness (doc_burstiness). `infer_theta` is the
    model's own inference, injected by the caller, so this function never
    assumes a particular model class.

    Guard: documents with total token count < 2 or a single unique token
    are skipped (dedup is degenerate/undefined-signal for them: dedup
    would strip all repetition information from a 1-type document,
    collapsing repeat_fraction and top_mass together trivially). Skipped
    docs are counted in "n_skipped", not included in "n_docs" or any
    per-doc array.

    The mean concentration shift between raw and dedup is a length
    confound (dedup always shortens the document), not evidence either
    way -- see the module docstring. This function does not compute or
    emit a verdict; it reports the spread/rank/correlation numbers a
    caller needs to make that judgment.

    Returns a dict with:
      - per-doc arrays (length n_docs each): "top_mass_raw",
        "top_mass_dedup", "eff_topics_raw", "eff_topics_dedup",
        "repeat_fraction"
      - "<name>_summary" (p10/p50/p90/mean/std) for each of the above five
        arrays, e.g. "top_mass_raw_summary"
      - "spread_ratio_top_mass": std(top_mass_dedup) / std(top_mass_raw)
      - "rank_corr_top_mass": Spearman correlation of top_mass_raw vs
        top_mass_dedup
      - "burstiness_corr_top_mass": Pearson correlation of top_mass_raw vs
        repeat_fraction
      - "n_docs": number of documents included (post-skip)
      - "n_skipped": number of documents skipped by the guard above

    spread_ratio_top_mass and the two correlations are NaN when fewer than
    2 documents survive the guard (std/correlation are undefined).

    The per-doc loop lives here (it needs the caller's `infer_theta` and the
    actual documents); the aggregation is `summarize_concentration_heterogeneity`,
    reused as-is so a distributed caller that computes the same five arrays
    on workers gets byte-identical aggregation.
    """
    top_mass_raw: list[float] = []
    top_mass_dedup: list[float] = []
    eff_topics_raw: list[float] = []
    eff_topics_dedup: list[float] = []
    repeat_fraction: list[float] = []
    n_skipped = 0

    for doc in docs:
        indices = np.asarray(doc.indices)
        counts = np.asarray(doc.counts, dtype=np.float64)
        burst = doc_burstiness(indices, counts)
        if burst["total"] < 2.0 or burst["unique"] <= 1:
            n_skipped += 1
            continue

        theta_raw = np.asarray(infer_theta(indices, counts), dtype=np.float64)
        theta_dedup = np.asarray(
            infer_theta(indices, dedup_counts(counts)), dtype=np.float64
        )
        top_raw, eff_raw = doc_concentration(theta_raw)
        top_dedup, eff_dedup = doc_concentration(theta_dedup)

        top_mass_raw.append(top_raw)
        top_mass_dedup.append(top_dedup)
        eff_topics_raw.append(eff_raw)
        eff_topics_dedup.append(eff_dedup)
        repeat_fraction.append(burst["repeat_fraction"])

    return summarize_concentration_heterogeneity(
        top_mass_raw=np.array(top_mass_raw, dtype=np.float64),
        top_mass_dedup=np.array(top_mass_dedup, dtype=np.float64),
        eff_topics_raw=np.array(eff_topics_raw, dtype=np.float64),
        eff_topics_dedup=np.array(eff_topics_dedup, dtype=np.float64),
        repeat_fraction=np.array(repeat_fraction, dtype=np.float64),
        n_skipped=n_skipped,
    )
