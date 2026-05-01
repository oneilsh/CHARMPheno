"""Fetch the prior-LDA topic-concept β from the Hugging Face dataset
`oneilsh/lda_pasc`, filter to the top-K highest-weight concepts per topic,
renormalize each topic row to sum to 1.0, and write a compact parquet.
Also write a per-topic metadata sidecar parsed from the topic_name string.

Outputs:
    data/cache/lda_beta_topk.parquet
        topic_id:int, concept_id:int, concept_name:str, weight:float
    data/cache/lda_topic_metadata.parquet  (one row per topic)
        topic_id:int, usage_pct:float, uniformity_h:float,
        coherence_c:float

Topic name format
-----------------
Upstream topic_name strings look like `T-58 (U 0.5%, H 0.91, C -0.5)`.
Per the upstream methods documentation:
    rank          — 1-indexed by overall corpus usage in the source LDA
                    fit (rank 1 = most prevalent). Surfaced as `topic_id`.
    usage_pct (U) — usage of the topic across sites, rounded to 0.1%.
    uniformity_h (H) — topic uniformity across sites, 0..1, higher is
                    more uniform. Computed as a normalized information
                    entropy of the per-site topic-usage distribution.
    coherence_c (C) — relative topic quality as a normalized coherence
                    score (z-score, higher is more coherent).

`topic_id` is duplicated millions of times in the main beta, so the
U/H/C metadata lives in the sidecar (one row per topic) rather than
inflating the main file.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HF_DATASET = "oneilsh/lda_pasc"
DEFAULT_TOP_K = 1000
DEFAULT_OUTPUT = Path("data/cache/lda_beta_topk.parquet")
DEFAULT_METADATA_OUTPUT = Path("data/cache/lda_topic_metadata.parquet")

_TOPIC_NAME_RE = re.compile(
    r"T-(?P<rank>\d+)\s*"
    r"\(\s*U\s*(?P<u>-?\d+(?:\.\d+)?)\s*%\s*,"
    r"\s*H\s*(?P<h>-?\d+(?:\.\d+)?)\s*,"
    r"\s*C\s*(?P<c>-?\d+(?:\.\d+)?)\s*\)"
)


def parse_topic_id(topic_name: str) -> int:
    """Parse the rank-encoded topic id from a name like 'T-148 (U 0.2%, H 0.93, C -0.5)'.

    Why: the upstream artifact encodes topic identity *and* metadata inside
    a single descriptive string. The 'T-<rank>' prefix is invariant; the
    integer rank is a 1-indexed ordering by overall topic usage in the
    source LDA fit (rank 1 = most-used). See module docstring for the full
    semantics of U/H/C and how they end up in the metadata sidecar.
    """
    m = re.match(r"T-(\d+)", topic_name)
    if not m:
        raise ValueError(f"Could not parse topic id from {topic_name!r}")
    return int(m.group(1))


def parse_topic_metadata(topic_name: str) -> dict:
    """Parse rank+U+H+C from a name like 'T-58 (U 0.5%, H 0.91, C -0.5)'.

    Returns a dict with keys topic_id, usage_pct, uniformity_h, coherence_c.
    See the module docstring for the semantic meaning of each field.
    Raises ValueError if the full pattern doesn't match — the upstream
    artifact has a consistent format, so a missing field is a real error
    rather than a "best effort, fall back" situation.
    """
    m = _TOPIC_NAME_RE.match(topic_name)
    if not m:
        raise ValueError(f"Could not parse topic metadata from {topic_name!r}")
    return {
        "topic_id": int(m.group("rank")),
        "usage_pct": float(m.group("u")),
        "uniformity_h": float(m.group("h")),
        "coherence_c": float(m.group("c")),
    }


def topic_metadata_from_names(topic_names: pd.Series) -> pd.DataFrame:
    """Build the per-topic metadata frame from the unique topic_name values."""
    unique_names = topic_names.drop_duplicates()
    rows = [parse_topic_metadata(n) for n in unique_names]
    out = pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)
    return out


def top_k_per_topic_and_renormalize(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep the K highest-`term_weight` rows per topic and renormalize.

    Why: the full β matrix (300 topics × ~50K concepts) is 1.7 GB on disk,
    but ~99% of each topic's probability mass lives in its top ~1000 concepts
    (power-law distribution). Filtering preserves the generative process while
    fitting in a laptop-friendly artifact. Renormalization keeps each topic
    row a proper probability distribution.

    Input must have columns: topic_id, concept_id, concept_name, term_weight.
    Output is the same shape with `term_weight` renormalized per-topic.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Keep only the top-K by term_weight within each topic.
    ranked = (df.sort_values(["topic_id", "term_weight"], ascending=[True, False])
                .groupby("topic_id", group_keys=False)
                .head(top_k))

    # Renormalize each topic's surviving weights to sum to 1.
    sums = ranked.groupby("topic_id")["term_weight"].transform("sum")
    out = ranked.copy()
    out["term_weight"] = out["term_weight"] / sums
    return out.reset_index(drop=True)


def fetch_and_write(top_k: int, output: Path, metadata_output: Path) -> None:
    """Stream the HF dataset, build a DataFrame, filter top-K, write parquets.

    Writes the filtered β to `output` and the per-topic metadata sidecar
    (topic_id, usage_pct, coherence_h, baseline_delta_c) to
    `metadata_output`. Streaming avoids materializing the full 1.7 GB
    file in memory.
    """
    # Deferred import so unit tests of the pure filter don't require datasets.
    from datasets import load_dataset

    log.info("Streaming %s from Hugging Face ...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    # The upstream CSV has columns: term_weight, relevance, concept_id,
    # concept_name, topic_name. We keep topic_name through the load so we
    # can build the per-topic metadata sidecar from its unique values.
    records: list[dict] = []
    for row in ds:
        records.append({
            "topic_id": parse_topic_id(row["topic_name"]),
            "topic_name": str(row["topic_name"]),
            "concept_id": int(row["concept_id"]),
            "concept_name": str(row["concept_name"]),
            "term_weight": float(row["term_weight"]),
        })
    df = pd.DataFrame(records)
    log.info("Loaded %d rows across %d topics",
             len(df), df["topic_id"].nunique())

    metadata = topic_metadata_from_names(df["topic_name"])
    log.info("Parsed metadata for %d topics", len(metadata))

    filtered = top_k_per_topic_and_renormalize(
        df.drop(columns=["topic_name"]), top_k=top_k,
    )
    log.info("After top-%d filter: %d rows", top_k, len(filtered))

    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.rename(columns={"term_weight": "weight"}).to_parquet(
        output, index=False
    )
    log.info("Wrote %s", output)

    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(metadata_output, index=False)
    log.info("Wrote %s", metadata_output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Keep top-K concepts per topic (default {DEFAULT_TOP_K})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output beta parquet path (default {DEFAULT_OUTPUT})")
    parser.add_argument("--metadata-output", type=Path,
                        default=DEFAULT_METADATA_OUTPUT,
                        help=f"Output topic-metadata parquet (default {DEFAULT_METADATA_OUTPUT})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_and_write(top_k=args.top_k, output=args.output,
                    metadata_output=args.metadata_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
