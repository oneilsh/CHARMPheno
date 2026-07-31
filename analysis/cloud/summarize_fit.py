"""Compact post-fit summary ("fit card") for a multidomain gated run.

Reads only the self-describing run-dir artifacts (manifest.json + params/
lambda_<m>.npy) — no Spark, no BigQuery, no spark_vi import — so it runs anywhere
with numpy. Emits a small, pasteable Markdown card: fit structure + health + a
node-level topic sample (top tokens per domain for a capped, evenly-spread set of
disease anchors), and upserts it into the run's summary.md between markers so
`make summary ID=N` shows it.

The card is deliberately small: with N domains and hundreds of topics the full
topic dump is unwieldy to relay, but a node-level slice across a few anchors is
enough to sanity-check that topics look disease-specific.

Run: `make -C analysis/cloud summarize-exp ID=N` (or directly with --run-dir).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

CARD_START = "<!-- FIT-CARD START -->"
CARD_END = "<!-- FIT-CARD END -->"


# --- pure layout (replicates spark_vi DagLayout block assignment) ---

def layout(parent_int: dict[int, list[int]], n_bg: int, tpn: int) -> dict:
    """Topic-block layout: nodes = sorted non-root ids; each node u (index i) owns
    topics [n_bg+i*tpn, n_bg+(i+1)*tpn); anchors = children of root 0."""
    nodes = sorted(parent_int)
    block = {u: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
             for i, u in enumerate(nodes)}
    K = n_bg + len(nodes) * tpn
    anchors = sorted(u for u, ps in parent_int.items() if 0 in ps)
    return {"nodes": nodes, "block": block, "K": K, "anchors": anchors}


def select_sample_nodes(anchors: list[int], sample: int) -> list[int]:
    """Deterministic, evenly-spread subset of anchors (diversity across the DAG)."""
    if sample <= 0 or len(anchors) <= sample:
        return list(anchors)
    step = len(anchors) / sample
    return [anchors[int(i * step)] for i in range(sample)]


def top_tokens(row: np.ndarray, idx2name: dict[int, str], top_n: int) -> list[str]:
    """Top-``top_n`` token display names for a (non-negative) lambda row."""
    if row.size == 0:
        return []
    order = np.argsort(row)[::-1][:top_n]
    return [idx2name.get(int(j), str(int(j))) for j in order if row[int(j)] > 0]


# --- pure rendering ---

def render_card(*, meta: dict, health: list[str], topic_lines: list[str]) -> str:
    lines = [
        CARD_START,
        f"## Fit card — {meta.get('disease', '?')}",
        "",
        meta.get("structure", ""),
        meta.get("domains", ""),
        meta.get("config", ""),
        "",
        "**Health:** " + " · ".join(health),
        "",
        meta.get("corpus", ""),
        "",
        meta.get("sample_header", "### Node topics"),
        *topic_lines,
        "",
        "_Full topic dump + convergence are in the run stdout above._",
        CARD_END,
        "",
    ]
    return "\n".join(l for l in lines if l is not None)


def upsert_summary_card(summary_path: Path, card: str) -> None:
    """Append the card to summary.md, replacing any prior card region (idempotent)."""
    existing = summary_path.read_text() if summary_path.exists() else ""
    if CARD_START in existing and CARD_END in existing:
        pre = existing.split(CARD_START, 1)[0].rstrip()
        post = existing.split(CARD_END, 1)[1].lstrip("\n")
        existing = (pre + ("\n\n" if post else "\n")) + post
    sep = "" if existing.endswith("\n") or not existing else "\n\n"
    summary_path.write_text(existing + sep + card)


# --- artifact loading (numpy + json only) ---

def _load_lambda(run_dir: Path, n_domains: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for m in range(n_domains):
        p = run_dir / "params" / f"lambda_{m}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing lambda sidecar {p}")
        out[m] = np.asarray(np.load(p), dtype=float)
    return out


def _idx2name(corpus_manifest: dict, domain: str) -> dict[int, str]:
    """token-index -> concept name for a domain, from vocab_<d> ({cid:idx}) +
    vocab_names_<d> ({cid:name})."""
    vocab = corpus_manifest.get(f"vocab_{domain}", {})
    names = corpus_manifest.get(f"vocab_names_{domain}", {})
    return {int(idx): names.get(cid, str(cid)) for cid, idx in vocab.items()}


def _node_display_names(corpus_manifest: dict) -> dict[int, str]:
    int2cid = {int(e): int(c) for e, c in corpus_manifest.get("int2cid", {}).items()}
    name_by_id = {int(c): str(n) for c, n in corpus_manifest.get("name_by_id", {}).items()}
    return {e: name_by_id.get(c, str(c)) for e, c in int2cid.items()}


def _starved_engine_ids(starved) -> set[int]:
    """Best-effort set of engine topic-ids flagged starved (structure-tolerant)."""
    ids: set[int] = set()
    if isinstance(starved, dict):
        starved = starved.get("topics", starved.get("starved", []))
    if isinstance(starved, list):
        for e in starved:
            if isinstance(e, int):
                ids.add(e)
            elif isinstance(e, dict):
                for key in ("topic", "k", "index"):
                    if isinstance(e.get(key), int):
                        ids.add(e[key])
    return ids


def build_card(manifest: dict, lam: dict[int, np.ndarray], *, top_n: int, sample: int) -> str:
    cm = manifest.get("corpus_manifest", {})
    domains = list(manifest.get("domains", []))
    parent_int = {int(n): [int(p) for p in ps]
                  for n, ps in cm.get("parent_int", {}).items()}
    lay = layout(parent_int, int(manifest["n_bg"]), int(manifest["tpn"]))
    node_names = _node_display_names(cm)
    idx2name = {m: _idx2name(cm, d) for m, d in enumerate(domains)}
    starved_ids = _starved_engine_ids(manifest.get("starved_topics"))

    def domain_cell(row_indices: list[int]) -> str:
        parts = []
        for m, d in enumerate(domains):
            node_row = lam[m][row_indices].sum(axis=0)
            toks = ", ".join(top_tokens(node_row, idx2name[m], top_n)) or "—"
            parts.append(f"{d}: {toks}")
        return " | ".join(parts)

    # background: show the two most concentrated bg topics
    n_bg = int(manifest["n_bg"])
    bg_peak = [(k, max(float(lam[m][k].max()) for m in range(len(domains))))
               for k in range(n_bg)]
    bg_show = [k for k, _ in sorted(bg_peak, key=lambda t: -t[1])[:2]]
    topic_lines = [f"- **bg#{k}**: {domain_cell([k])}" for k in bg_show]

    shown = select_sample_nodes(lay["anchors"], sample)
    for u in shown:
        block = lay["block"][u]
        n_starved = sum(1 for k in block if k in starved_ids)
        tag = f", starved {n_starved}/{len(block)}" if n_starved else ""
        name = node_names.get(u, str(u))
        topic_lines.append(f"- **{name}** (n{u}{tag}): {domain_cell(block)}")

    dead = manifest.get("dead_nodes") or []
    starved = manifest.get("starved_topics") or []
    n_dead = len(dead) if isinstance(dead, (list, dict)) else 0
    n_starved = len(starved) if isinstance(starved, (list, dict)) else 0
    health = [
        f"dead nodes {n_dead}/{len(lay['nodes'])}",
        f"starved topics {n_starved}/{lay['K']}",
    ]

    cs = manifest.get("corpus_stats") or {}
    corpus = ""
    if isinstance(cs, dict) and cs:
        bits = []
        for key in ("n_train", "n_test", "n_docs"):
            if key in cs:
                bits.append(f"{key}={cs[key]}")
        by = cs.get("by_source_cohort")
        if isinstance(by, dict):
            bits.append("by_source_cohort=" + json.dumps(by, separators=(",", ":")))
        corpus = "**Corpus:** " + " · ".join(bits) if bits else ""

    dom_line = " · ".join(f"{d} V={lam[m].shape[1]}" for m, d in enumerate(domains))
    meta = {
        "disease": manifest.get("disease", "?"),
        "structure": (f"**Structure:** K={lay['K']} = {n_bg} bg + "
                      f"{len(lay['nodes'])} nodes × {manifest.get('tpn')} tpn · "
                      f"{len(lay['anchors'])} anchors"),
        "domains": f"**Domains:** {dom_line}",
        "config": (f"**Config:** init={manifest.get('init')} · "
                   f"mini_batch_fraction={manifest.get('mini_batch_fraction')} · "
                   f"seed={manifest.get('seed')}"),
        "corpus": corpus,
        "sample_header": (f"### Node topics (node-level λ, top-{top_n} tokens/domain; "
                          f"{len(shown)}/{len(lay['anchors'])} anchors)"),
    }
    return render_card(meta=meta, health=health, topic_lines=topic_lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Compact post-fit summary (fit card).")
    p.add_argument("--run-dir", required=True, help="the fit's run dir (has manifest.json)")
    p.add_argument("--top-n", type=int, default=6, help="top tokens per domain per node")
    p.add_argument("--sample-nodes", type=int, default=8, help="anchor nodes to show (0=all)")
    p.add_argument("--no-append", action="store_true", help="do not touch summary.md")
    args = p.parse_args(argv)

    # tolerate a shell glob that expanded to the real dir
    matches = glob.glob(args.run_dir)
    run_dir = Path(matches[0] if matches else args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    lam = _load_lambda(run_dir, len(manifest.get("domains", [])))

    card = build_card(manifest, lam, top_n=args.top_n, sample=args.sample_nodes)
    (run_dir / "fit_card.md").write_text(card + "\n")
    sys.stdout.write(card + "\n")
    if not args.no_append:
        upsert_summary_card(run_dir / "summary.md", card)
        sys.stderr.write(f"[summarize] wrote fit_card.md and upserted {run_dir/'summary.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
