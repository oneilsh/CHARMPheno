"""Build a compact real-EHR beta bundle for SIMULATION from the Hugging Face dataset
`oneilsh/lda_pasc` (the cross-site prior-LDA fit: 300 topics x ~50K OMOP concepts).

Downloads the single long-format CSV (`all_topic_descriptions.csv`, ~1.7 GB) once and
does a VECTORIZED pandas top-K per topic (seconds), rather than the datasets streaming
iterator (which parses ~15M rows in Python, ~100x slower). Only the three needed columns
are read; topic_name is categorical. Parses each topic's usage% from its
`T-<rank> (U .., H .., C ..)` name so simulations can weight background topic prevalence
realistically (Zipf-like).

Output (data/cache/sim_beta.npz), consumable by tests/_stm_synth.real_beta_from(source=):
    beta        : (K, V) float64, each row a renormalized topic-word distribution
    concept_ids : (V,) int64     OMOP concept_id for each vocab column
    topic_rank  : (K,) int64     upstream 1-indexed usage rank (1 = most prevalent)
    usage_pct   : (K,) float64   per-topic usage % (background-prevalence weight)

Run (network-enabled env with `datasets`/`huggingface_hub` + `pandas`):
    <venv>/bin/python scripts/build_sim_beta_npz.py --top-k 400
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HF_DATASET = "oneilsh/lda_pasc"
CSV_FILE = "all_topic_descriptions.csv"
DEFAULT_OUTPUT = Path("data/cache/sim_beta.npz")
_NAME_RE = re.compile(r"T-(?P<rank>\d+)\s*\(\s*U\s*(?P<u>-?\d+(?:\.\d+)?)\s*%")


def _parse(topic_name: str) -> tuple[int, float]:
    m = _NAME_RE.match(topic_name)
    if not m:
        raise ValueError(f"unparseable topic_name {topic_name!r}")
    return int(m.group("rank")), float(m.group("u"))


def build(top_k: int, output: Path, limit: int | None = None) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATASET, CSV_FILE, repo_type="dataset")
    log.info("downloaded %s", path)
    df = pd.read_csv(
        path, usecols=["term_weight", "concept_id", "topic_name"],
        dtype={"term_weight": "float64", "concept_id": "int64", "topic_name": "category"},
        nrows=limit)
    log.info("read %d rows, %d topics", len(df), df["topic_name"].cat.categories.size)

    cats = list(df["topic_name"].cat.categories)
    parsed = [_parse(n) for n in cats]
    rank_of = np.array([p[0] for p in parsed], dtype=np.int64)
    usage_of = np.array([p[1] for p in parsed], dtype=np.float64)
    codes = df["topic_name"].cat.codes.to_numpy()
    df = df.assign(rank=rank_of[codes]).drop(columns=["topic_name"])

    # top-K concepts per topic by weight (vectorized), then renormalize.
    top = (df.sort_values(["rank", "term_weight"], ascending=[True, False])
             .groupby("rank", sort=True).head(top_k))
    ranks = np.sort(top["rank"].unique())
    vocab = np.sort(top["concept_id"].unique())
    col = {int(c): j for j, c in enumerate(vocab)}
    row_of = {int(r): i for i, r in enumerate(ranks)}
    K, V = len(ranks), len(vocab)
    beta = np.zeros((K, V), dtype=np.float64)
    r = top["rank"].to_numpy(); c = top["concept_id"].to_numpy(); w = top["term_weight"].to_numpy()
    for ri, ci, wi in zip(r, c, w):
        beta[row_of[int(ri)], col[int(ci)]] = wi
    beta /= beta.sum(axis=1, keepdims=True)

    rank_to_usage = {int(rk): float(u) for rk, u in zip(rank_of, usage_of)}
    out = dict(beta=beta, concept_ids=vocab.astype(np.int64),
               topic_rank=ranks.astype(np.int64),
               usage_pct=np.array([rank_to_usage[int(rk)] for rk in ranks], dtype=np.float64))
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
