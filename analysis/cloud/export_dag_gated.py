#!/usr/bin/env python
"""export_dag_gated.py -- the browsable export of a DAG-gated run (`dag-gated-export/1`).

Spec: docs/superpowers/specs/2026-11-11-dag-gated-dashboard-export.md. Reads a fitted
run dir and writes `<run>/dag_export/{manifest,nodes,vocab,topics}.json` (minified,
columnar; the browser in `dag-dashboard/` reads them), then prints the raw and gzip
size of each file and the total (also in `manifest.sizes`). Pure Python + numpy: no
Spark, no YARN, safe on the master mid-anything. The pure work is
`charmpheno.export.dag_gated`; this file only gathers the inputs:

- `<run>/model/result` (spark_vi.io.export.load_result): λ per domain, and the node set
  (`metadata["dag_node_set"]`, with n_bg and tpn) -- the cleaned model when the fit
  dropped unsupported nodes;
- `<run>/label_dag.json`: the DAG the fit was GIVEN (every node the browser shows:
  kept, pruned, spliced, unsupported);
- `<run>/bundle_meta.json`: the per-domain vocab maps (term code -> index);
- `<run>/manifest.json`: domain names, cdr, mondo_version / mondo_cache_dir, and the
  node-statistics file name;
- `<run>/node_statistics.tsv` (the model AS FITTED): the support of the nodes dropped as
  unsupported, which the cleaned node set no longer carries (banded on export);
- Mondo labels for every given node from the Mondo release's `mondo_nodes.tsv` in the
  fit's Mondo cache (`<mondo_cache_dir>/<mondo_version>_mondo_nodes.tsv`, fetched once
  if absent, as the fit fetches it); fallback: the bundle meta's names (kept nodes
  only), then the id;
- term names: `--names <csv>` (resolve_vocab_names / inspect_topics format), or
  `--resolve-names` (runs resolve_vocab_names.py: a targeted `bq query`, off-YARN), or
  `<run>/concept_names.csv` when present.

The export holds model parameters and banded node support only: no per-document or
patient data (AGENTS.md, egress floor). It refuses a run whose vocabulary floor
(`corpus_manifest.min_patient_count`) or node-set floor (`min_docs_per_node`) is below
the egress floor, and ships the topic NPMI only when the result's `min_pair_count` is at
least the floor (readout `--npmi --npmi-min-pair-count 20`). Still: copy it out of the
workspace only per the workspace's egress rules.

  python export_dag_gated.py --run-dir <run> [--names concept_names.csv | --resolve-names]
  make -C analysis/cloud export-dag-gated ID=173 RESOLVE_NAMES=1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent / "spark-vi"),
           str(_HERE.parent.parent / "charmpheno")):
    if _p not in sys.path:
        sys.path.append(_p)


def resolve_run(pattern) -> Path:
    """An exact run dir, or a glob (`.../0173-*`) matching exactly one dir that holds a
    saved model (`model/result/manifest.json`)."""
    p = Path(pattern)
    if (p / "model" / "result" / "manifest.json").exists():
        return p
    hits = [Path(m) for m in glob.glob(str(pattern))
            if (Path(m) / "model" / "result" / "manifest.json").exists()]
    if len(hits) != 1:
        raise SystemExit(f"[export-dag-gated] {pattern!r} matches {len(hits)} run dirs "
                         "with a saved model (<run>/model/result); pass an exact dir")
    return hits[0]


def given_dag_from_edges(edges) -> dict:
    """`[[child, parent | null], ...]` -> `{node: [parents]}`, every node a key."""
    out: dict = {}
    for c, p in edges:
        out.setdefault(c, [])
        if p is not None:
            out[c].append(p)
            out.setdefault(p, [])
    return {n: sorted(ps, key=str) for n, ps in out.items()}


def load_term_names(path) -> dict:
    """`{code: name}` from a `concept_id,concept_name` CSV (header optional); codes are
    ints where they parse as ints."""
    names = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0].strip():
                continue
            k = row[0].strip()
            try:
                k = int(k)
            except ValueError:
                if k == "concept_id":
                    continue
            names[k] = row[1]
    return names


def _mondo_nodes_file(cache_dir, version, *, fetch=True, run_dir=None):
    """The fit's `mondo_nodes.tsv`, looked up where a relative cache dir may resolve
    (cwd, the run dir, the repo root, analysis/cloud); fetched into the first
    candidate when absent and `fetch`. None when unavailable."""
    fname = f"{version}_mondo_nodes.tsv"
    c = Path(cache_dir)
    roots = [Path.cwd(), *( [Path(run_dir)] if run_dir else []), _HERE.parent.parent,
             _HERE]
    cands = [c] if c.is_absolute() else [r / c for r in roots]
    for d in cands:
        if (d / fname).exists() and (d / fname).stat().st_size > 0:
            return d / fname
    if not fetch:
        return None
    try:
        from charmpheno.mondo.ontology import download_cached
        return download_cached(version, "mondo_nodes.tsv", cands[0])
    except Exception as exc:          # offline, or no write access
        print(f"[export-dag-gated] could not fetch the Mondo {version} nodes file "
              f"({exc}); falling back to the bundle meta's names", flush=True)
        return None


def load_mondo_names(path, wanted) -> dict:
    """`{curie: label}` for the `wanted` ids from a Mondo `mondo_nodes.tsv`
    (columns include `id` and `name`)."""
    wanted = {str(w) for w in wanted}
    out = {}
    csv.field_size_limit(1 << 24)
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            i = row.get("id")
            if i in wanted and row.get("name"):
                out[i] = row["name"]
    return out


def _node_cid(n):
    if isinstance(n, int):
        return n
    s = str(n)
    try:
        return int(s.split(":", 1)[1]) if ":" in s else int(s)
    except ValueError:
        return None


def node_names_for(nodes, *, mondo_tsv=None, bundle_names=None) -> tuple:
    """`({node: label}, n from Mondo, n from the bundle meta)`: Mondo labels first,
    then the bundle meta's `name_by_id` (keyed by the bridge's integer cid, kept nodes
    only); an unnamed node keeps its id (the exporter's default)."""
    names = load_mondo_names(mondo_tsv, nodes) if mondo_tsv else {}
    n_mondo = sum(1 for n in nodes if str(n) in names)
    out = {n: names[str(n)] for n in nodes if str(n) in names}
    n_bundle = 0
    for n in nodes:
        if n in out:
            continue
        c = _node_cid(n)
        v = (bundle_names or {}).get(str(c)) if c is not None else None
        if v is not None and str(v) != str(n):
            out[n] = v
            n_bundle += 1
    return out, n_mondo, n_bundle


def unsupported_support(run, manifest) -> dict:
    """`{node id (str): support}` from the as-fitted node statistics TSV, or {}."""
    fname = ((manifest.get("node_statistics") or {}).get("file")
             or "node_statistics.tsv")
    p = Path(run) / fname
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("node") and row.get("support"):
                out[row["node"]] = int(float(row["support"]))
    return out


def format_sizes(sizes) -> str:
    from charmpheno.export.dag_gated import FILES
    lines = [f"  {'file':<14} {'raw':>12} {'gzip':>12}"]
    tr = tg = 0
    for f in FILES:
        r, g = sizes[f]
        tr += r
        tg += g
        lines.append(f"  {f:<14} {r:>12,} {g:>12,}")
    lines.append(f"  {'total':<14} {tr:>12,} {tg:>12,}"
                 f"   ({tr / 1e6:.2f} MB raw, {tg / 1e6:.2f} MB gzip)")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="the run dir (or a quoted glob such as '.../0173-*')")
    ap.add_argument("--out", default=None, help="output dir (default <run>/dag_export)")
    ap.add_argument("--single-file", default=None, metavar="HTML",
                    help="also write the browser page with the export embedded, one "
                         "self-contained file that opens without a server (default "
                         "<run>/dag_view.html; see dag-dashboard/README.md)")
    ap.add_argument("--no-single-file", action="store_true",
                    help="do not write the self-contained page")
    ap.add_argument("--npmi-file", default=None, metavar="JSON",
                    help="topic NPMI to ship per topic (topic_npmi.py output; default "
                         "<run>/npmi_top10.json when it exists)")
    ap.add_argument("--no-npmi", action="store_true", help="ship no NPMI")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--names", default=None,
                   help="term-name CSV (concept_id,concept_name), keyed by the vocab "
                        "feature id (resolve_vocab_names.py's output)")
    g.add_argument("--resolve-names", action="store_true",
                   help="run resolve_vocab_names.py (a targeted bq query, off-YARN) "
                        "into <run>/concept_names.csv first")
    ap.add_argument("--mondo-cache-dir", default=None,
                    help="override the manifest's corpus_manifest.mondo_cache_dir")
    ap.add_argument("--mondo-version", default=None,
                    help="override the manifest's corpus_manifest.mondo_version")
    ap.add_argument("--no-fetch-mondo", action="store_true",
                    help="never download the Mondo nodes file; fall back to the "
                         "bundle meta's names (kept nodes only)")
    ap.add_argument("--top-beta", type=int, default=25)
    ap.add_argument("--top-lift", type=int, default=25)
    ap.add_argument("--lift-pool", type=int, default=200)
    ap.add_argument("--created-utc", default=None,
                    help="pin manifest.created_utc (reproducible exports)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from charmpheno.egress import EGRESS_FLOOR
    from charmpheno.export.dag_gated import build_export, npmi_exportable, write_export
    from spark_vi.dag.nodeset import NodeSet
    from spark_vi.io.export import load_result

    run = resolve_run(args.run_dir)
    out = Path(args.out) if args.out else run / "dag_export"
    manifest = json.loads((run / "manifest.json").read_text())
    cm = manifest.get("corpus_manifest") or {}
    # Disclosure: the vocabulary must be floored at the egress floor (every shipped
    # term attested by >= EGRESS_FLOOR patients). Refuse when it is not, or unknown.
    mpc = cm.get("min_patient_count")
    if mpc is None or int(mpc) < EGRESS_FLOOR:
        raise SystemExit(
            f"[export-dag-gated] refusing: the run's corpus_manifest.min_patient_count "
            f"is {mpc!r}, but an export needs a vocabulary floored at >= {EGRESS_FLOOR} "
            "patients per term (the egress floor)")
    label_dag = json.loads((run / "label_dag.json").read_text())
    meta = json.loads((run / "bundle_meta.json").read_text())
    res = load_result(str(run / "model" / "result"))
    md = res.metadata["dag_node_set"]
    ns = NodeSet.from_dict(md)
    n_bg, tpn = int(md["n_bg"]), int(md["tpn"])
    lam = res.global_params["lambda"]
    if not isinstance(lam, dict):
        lam = {0: lam}
    vocab_maps = meta.get("vocab_maps") or [meta["vocab_map"]]
    domains = list(manifest.get("domain_names") or
                   [f"domain{m}" for m in range(len(vocab_maps))])
    given = given_dag_from_edges(label_dag.get("dag") or meta.get("label_dag"))
    print(f"[export-dag-gated] {run.name}: K={n_bg + len(ns.nodes) * tpn} "
          f"({n_bg} bg + {len(ns.nodes)} nodes x {tpn}); given DAG {len(given)} nodes "
          f"({len(ns.nodes)} kept / {len(ns.pruned)} pruned / {len(ns.spliced)} "
          f"spliced / {len(ns.unsupported)} unsupported); domains "
          + ", ".join(f"{d} {len(vm)}" for d, vm in zip(domains, vocab_maps)),
          flush=True)

    # term names
    names_path = args.names
    if args.resolve_names:
        names_path = str(run / "concept_names.csv")
        cmd = [sys.executable, str(_HERE / "resolve_vocab_names.py"), "--run-dir",
               str(run), "--bundle-meta", str(run / "bundle_meta.json"),
               "--out", names_path]
        print("[export-dag-gated] resolving term names: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    elif names_path is None and (run / "concept_names.csv").exists():
        names_path = str(run / "concept_names.csv")
    term_names = load_term_names(names_path) if names_path else {}
    if not term_names:
        print("[export-dag-gated] WARNING: no term names: the page will show term CODES "
              "only. Re-run with RESOLVE_NAMES=1 (make) / --resolve-names, or pass "
              "--names <csv>.", flush=True)
    print(f"[export-dag-gated] term names: {len(term_names)} from "
          f"{names_path or '(none: vocab.name stays empty)'}", flush=True)

    # node names
    version = args.mondo_version or cm.get("mondo_version")
    cache_dir = args.mondo_cache_dir or cm.get("mondo_cache_dir") or "data/mondo"
    tsv = (_mondo_nodes_file(cache_dir, version, fetch=not args.no_fetch_mondo,
                             run_dir=run) if version else None)
    node_names, n_mondo, n_bundle = node_names_for(
        list(given), mondo_tsv=tsv, bundle_names=meta.get("name_by_id"))
    print(f"[export-dag-gated] node names: {n_mondo} from {tsv or '(no Mondo file)'}, "
          f"{n_bundle} from the bundle meta, {len(given) - n_mondo - n_bundle} "
          "unnamed (id shown)", flush=True)

    stats = unsupported_support(run, manifest)
    extra = {n: stats.get(str(n)) for n in ns.unsupported}
    n_known = sum(v is not None for v in extra.values())
    print(f"[export-dag-gated] unsupported-node support from the as-fitted node "
          f"statistics: {n_known} of {len(extra)}", flush=True)

    npmi = None
    if not args.no_npmi:
        npath = Path(args.npmi_file) if args.npmi_file else run / "npmi_top10.json"
        if npath.exists():
            npmi = json.loads(npath.read_text())
            ok, why = npmi_exportable(npmi)
            if ok:
                print(f"[export-dag-gated] NPMI from {npath} (top {npmi.get('top_n')}, "
                      f"{npmi.get('split')} split, min_pair_count "
                      f"{npmi.get('min_pair_count')})", flush=True)
            else:
                print(f"[export-dag-gated] NPMI from {npath} NOT shipped: {why}",
                      flush=True)
                npmi = None
        else:
            print(f"[export-dag-gated] no NPMI ({npath} not found; run the readout "
                  "with --npmi first, or pass --npmi-file)", flush=True)
    files = build_export(
        lam, ns, n_bg=n_bg, tpn=tpn, given_dag=given, domains=domains,
        vocab_maps=vocab_maps, node_names=node_names, term_names=term_names,
        extra_support=extra, run=run.name, mondo_version=version or "",
        cdr=cm.get("cdr", ""), created_utc=args.created_utc,
        top_beta=args.top_beta, top_lift=args.top_lift, lift_pool=args.lift_pool,
        npmi=npmi)
    sizes = write_export(files, out)
    n_null = sum(s is None for s in files["nodes.json"]["support"])
    print(f"[export-dag-gated] wrote {out} ({len(files['nodes.json']['id'])} nodes, "
          f"{n_null} with support not recorded; {len(files['vocab.json']['code'])} "
          f"terms; {len(files['topics.json']['node'])} topics)", flush=True)
    print(format_sizes(sizes), flush=True)
    if not args.no_single_file:
        from charmpheno.export.dag_gated import embed_in_page
        page = _HERE.parent.parent / "dag-dashboard" / "index.html"
        html_out = Path(args.single_file) if args.single_file else run / "dag_view.html"
        html_out.write_text(embed_in_page(page.read_text(encoding="utf-8"), out),
                            encoding="utf-8")
        print(f"[export-dag-gated] wrote the self-contained page {html_out} "
              f"({html_out.stat().st_size / 1e6:.2f} MB): open it in a browser, no server "
              "needed", flush=True)
    return files, sizes


if __name__ == "__main__":
    main()
