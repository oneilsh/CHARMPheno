"""Fetch the prior-LDA topic-concept β from the Hugging Face dataset
`oneilsh/lda_pasc`, filter to the top-K highest-weight concepts per topic,
renormalize each topic row to sum to 1.0, and write a compact parquet.

Output columns: topic_id:int, concept_id:int, concept_name:str, weight:float.

Default output: data/cache/lda_beta_topk.parquet
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


def parse_topic_id(topic_name: str) -> int:
    """Parse numeric id out of a topic_name string like 'T-148 (U 0.2%, H 0.93, ...)'.

    Why: the source file encodes the topic id inside a descriptive string; we
    need a stable integer key. The 'T-<digits>' prefix is invariant in the
    upstream artifact.
    """
    m = re.match(r"T-(\d+)", topic_name)
    if not m:
        raise ValueError(f"Could not parse topic id from {topic_name!r}")
    return int(m.group(1))


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


def fetch_and_write(top_k: int, output: Path) -> None:
    """Stream the HF dataset, build a DataFrame, filter top-K, write parquet.

    Streaming avoids materializing the full 1.7 GB file in memory.
    """
    # Deferred import so unit tests of the pure filter don't require datasets.
    from datasets import load_dataset

    log.info("Streaming %s from Hugging Face ...", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    # The upstream CSV has columns: term_weight, relevance, concept_id,
    # concept_name, topic_name. We only need four of the five.
    records: list[dict] = []
    for row in ds:
        records.append({
            "topic_id": parse_topic_id(row["topic_name"]),
            "concept_id": int(row["concept_id"]),
            "concept_name": str(row["concept_name"]),
            "term_weight": float(row["term_weight"]),
        })
    df = pd.DataFrame(records)
    log.info("Loaded %d rows across %d topics",
             len(df), df["topic_id"].nunique())

    filtered = top_k_per_topic_and_renormalize(df, top_k=top_k)
    log.info("After top-%d filter: %d rows", top_k, len(filtered))

    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.rename(columns={"term_weight": "weight"}).to_parquet(
        output, index=False
    )
    log.info("Wrote %s", output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Keep top-K concepts per topic (default {DEFAULT_TOP_K})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output parquet path (default {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_and_write(top_k=args.top_k, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
