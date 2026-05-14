"""Produce a tiny, schema-conformant dashboard bundle without Spark or a
real checkpoint. Use for dashboard development when the real export path
is too slow or unavailable.

Usage:
    python scripts/make_dev_bundle.py --out-dir dashboard/public/data \\
        --k 10 --v 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _round(x, d: int = 6):
    return np.round(np.asarray(x, dtype=np.float64), d).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--v", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Synthetic β: each topic has a "peak" cluster of vocab indices it favors.
    beta = np.full((args.k, args.v), 0.005)
    for k in range(args.k):
        peak = rng.choice(args.v, size=max(3, args.v // (args.k * 2)), replace=False)
        beta[k, peak] += rng.uniform(0.5, 2.0, size=peak.shape)
    beta = beta / beta.sum(axis=1, keepdims=True)
    alpha = np.full(args.k, 0.1)

    # marginals: a Dirichlet-distributed marginal, decreasing for top-N realism
    raw = rng.gamma(1.0, 1.0, size=args.v) * np.linspace(2.0, 0.5, args.v)
    marginals = raw / raw.sum()

    # model.json (no trimming here — dev bundle ships full V)
    (args.out_dir / "model.json").write_text(json.dumps({
        "K": args.k, "V": args.v, "alpha": _round(alpha), "beta": _round(beta),
    }))

    # vocab.json
    domains_pool = ["condition", "drug", "procedure", "measurement", "observation"]
    codes = []
    for i in range(args.v):
        codes.append({
            "id": i,
            "code": f"DEV{i:04d}",
            "description": f"Synthetic code {i}",
            "domain": domains_pool[i % len(domains_pool)],
            "corpus_freq": float(marginals[i]),
        })
    (args.out_dir / "vocab.json").write_text(json.dumps({"codes": codes}))

    # phenotypes.json — fake NPMI / pair_coverage / quality so the
    # dashboard's simple/advanced modes both have something to render.
    npmi = rng.normal(0.15, 0.08, size=args.k)
    pair_cov = rng.uniform(0.4, 1.0, size=args.k)
    corpus_prev = rng.dirichlet(alpha=np.full(args.k, 2.0))
    qualities = ["phenotype"] * args.k
    if args.k >= 1:
        qualities[-1] = "dead"  # one dead so the simple-mode filter has work
    if args.k >= 2:
        qualities[-2] = "mixed"
    if args.k >= 3:
        qualities[0] = "background"
    (args.out_dir / "phenotypes.json").write_text(json.dumps({
        "phenotypes": [
            {
                "id": k,
                "label": "",
                "description": "",
                "quality": qualities[k],
                "npmi": float(npmi[k]),
                "pair_coverage": float(pair_cov[k]),
                "corpus_prevalence": float(corpus_prev[k]),
                "original_topic_id": k,
            }
            for k in range(args.k)
        ],
    }))

    # corpus_stats.json
    (args.out_dir / "corpus_stats.json").write_text(json.dumps({
        "corpus_size_docs": 50000,
        "mean_codes_per_doc": 18.0,
        "k": args.k,
        "v": args.v,
        "v_full": args.v,
    }))
    print(f"wrote dev bundle to {args.out_dir} (K={args.k}, V={args.v})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
