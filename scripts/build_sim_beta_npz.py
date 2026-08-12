"""Build a compact real-EHR beta bundle for SIMULATION from the Hugging Face dataset
`oneilsh/lda_pasc` (the cross-site prior-LDA fit: 300 topics x ~50K OMOP concepts).

Memory-safe: streams the long-format rows once, keeping only a size-K max-heap of the
highest-weight concepts PER TOPIC (<= n_topics * top_k entries resident, not the full
~15M rows). Also parses each topic's usage% from its `T-<rank> (U .., H .., C ..)` name
so simulations can weight background topic prevalence realistically (Zipf-like).

Output (data/cache/sim_beta.npz), consumable by tests/_stm_synth.real_beta_from(source=):
    beta        : (K, V) float64, each row a renormalized topic-word distribution
    concept_ids : (V,) int64     OMOP concept_id for each vocab column
    topic_rank  : (K,) int64     upstream 1-indexed usage rank (1 = most prevalent)
    usage_pct   : (K,) float64   per-topic usage % (background-prevalence weight)

Run (in a network-enabled env with `datasets` installed):
    <venv>/bin/python scripts/build_sim_beta_npz.py --top-k 400
"""
from __future__ import annotations

import argparse
import heapq
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HF_DATASET = "oneilsh/lda_pasc"
DEFAULT_OUTPUT = Path("data/cache/sim_beta.npz")
_NAME_RE = re.compile(r"T-(?P<rank>\d+)\s*\(\s*U\s*(?P<u>-?\d+(?:\.\d+)?)\s*%")


def _parse(topic_name: str) -> tuple[int, float]:
    m = _NAME_RE.match(topic_name)
    if not m:
        raise ValueError(f"unparseable topic_name {topic_name!r}")
    return int(m.group("rank")), float(m.group("u"))


def build(top_k: int, output: Path, limit: int | None = None) -> None:
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    heaps: dict[int, list] = {}          # topic_rank -> min-heap of (weight, concept_id)
    usage: dict[int, float] = {}
    n = 0
    for row in ds:
        rank, u = _parse(row["topic_name"])
        w = float(row["term_weight"])
        cid = int(row["concept_id"])
        usage.setdefault(rank, u)
        h = heaps.setdefault(rank, [])
        if len(h) < top_k:
            heapq.heappush(h, (w, cid))
        elif w > h[0][0]:
            heapq.heapreplace(h, (w, cid))
        n += 1
        if n % 2_000_000 == 0:
            log.info("streamed %d rows; %d topics seen", n, len(heaps))
        if limit is not None and n >= limit:
            break
    log.info("done streaming %d rows across %d topics", n, len(heaps))

    ranks = sorted(heaps)
    vocab = sorted({cid for h in heaps.values() for _, cid in h})
    col = {cid: j for j, cid in enumerate(vocab)}
    K, V = len(ranks), len(vocab)
    beta = np.zeros((K, V), dtype=np.float64)
    for i, r in enumerate(ranks):
        for w, cid in heaps[r]:
            beta[i, col[cid]] = w
    beta /= beta.sum(axis=1, keepdims=True)      # renormalize each topic row
    out = dict(beta=beta, concept_ids=np.array(vocab, dtype=np.int64),
               topic_rank=np.array(ranks, dtype=np.int64),
               usage_pct=np.array([usage[r] for r in ranks], dtype=np.float64))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    log.info("wrote %s  beta=(%d,%d)  vocab=%d", output, K, V, V)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=400)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (debug)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(top_k=args.top_k, output=args.output, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
