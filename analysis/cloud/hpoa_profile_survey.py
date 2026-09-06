"""Stage-1 OFFLINE survey: HPOA disease->phenotype profiles for a Mondo branch.

Motivation (exps 0113-0115, insight 0082): spectral init fixed deep-node
starvation, but a residual remains — some deep node topics anchor on the most
SEPARABLE direction (demographic/etiologic strata) rather than the node's
clinical meaning. The candidate fix is a knowledge-aligned word-side prior:
boost each node's Dirichlet eta on the OMOP codes its HPO phenotype profile
maps to (and mildly downweight NOT-annotated ones). Whether that is even
possible depends on facts this survey measures:

  1. Which branch nodes have an HPOA profile at all (annotations key on
     OMIM/ORPHA/DECIPHER, reached via Mondo xrefs) — and at what depth.
  2. How much of each profile is REALIZABLE in OMOP: the HPO term (or an HPO
     descendant, per the ontology true-path rule) carries a SNOMED/ICD xref.
  3. How much frequency metadata exists to weight the prior.
  4. How promiscuous the realizable terms are across profiles (a term shared
     by every cardiac node adds no alignment).

Stage 1 reads PUBLIC ontology artifacts only (Mondo KGX nodes/edges,
`hp.obo`, `phenotype.hpoa`) — no CDR, no patient data, no Spark; it runs
off-cluster. Stage 2 (`hpoa_stage2_probe.py`, in-workspace) intersects the
realizable codes with the corpus vocabulary and counts persons under the
egress floor; `--emit-codes` writes its input — one row per (node, term,
SNOMED code) over each profile term's descendant-or-self closure, still pure
ontology data.

Driver-owned file: it IMPORTS the source-hashed mapping modules
(`mondo_to_omop_mapping`, `mondo_usage_core`) and never edits them, so no
bundle/corpus cache key moves.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from mondo_to_omop_mapping import _descendants, _disease_child_adjacency
from mondo_usage_core import parse_hpo_dag, parse_hpo_xrefs

_MONDO_RELEASE = "https://github.com/monarch-initiative/mondo/releases/download/v{v}/{f}"
_HPO_RELEASE = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download/{v}/{f}"
)
_HPO_LATEST = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/{f}"
)

# HPOA `database_id` prefixes reachable from Mondo's `xref` column. Mondo writes
# `Orphanet:377788` where HPOA keys `ORPHA:377788`; OMIM and DECIPHER match as-is.
_DB_PREFIXES = {"OMIM:": "OMIM:", "Orphanet:": "ORPHA:", "DECIPHER:": "DECIPHER:"}

# HPO frequency-subontology terms -> point estimate (midpoint of the term's
# defined range). Midpoints are a weighting convention, not a measurement — the
# survey reports how much of each profile carries ANY frequency, and the prior
# only needs a monotone weight, so the exact midpoint choice is low-stakes.
_FREQ_TERMS = {
    "HP:0040280": 1.0,     # Obligate (100%)
    "HP:0040281": 0.895,   # Very frequent (80-99%)
    "HP:0040282": 0.545,   # Frequent (30-79%)
    "HP:0040283": 0.17,    # Occasional (5-29%)
    "HP:0040284": 0.025,   # Very rare (1-4%)
    "HP:0040285": 0.0,     # Excluded (0%)
}


def normalize_frequency(raw) -> float | None:
    """HPOA `frequency` -> fraction in [0,1], or None when absent/unparseable.

    The column has three sanctioned shapes (HPOA format doc): an HP
    frequency-subontology term, a patient ratio ``n/m``, or a percentage
    ``17%`` (ranges ``a%-b%`` appear in the wild; take the midpoint)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _FREQ_TERMS:
        return _FREQ_TERMS[s]
    if s.endswith("%"):
        body = s[:-1].replace("%", "")
        try:
            if "-" in body:
                lo, hi = body.split("-", 1)
                return min(1.0, (float(lo) + float(hi)) / 200.0)
            return min(1.0, float(body) / 100.0)
        except ValueError:
            return None
    if "/" in s:
        try:
            n, m = s.split("/", 1)
            m = float(m)
            return min(1.0, float(n) / m) if m > 0 else None
        except ValueError:
            return None
    return None


def parse_hpoa(text: str) -> pd.DataFrame:
    """`phenotype.hpoa` -> DataFrame of its 12 tab-separated columns.

    Leading ``#`` metadata lines are dropped manually (not via a pandas comment
    char — free-text fields may legally contain ``#``)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str)
    need = {"database_id", "qualifier", "hpo_id", "frequency", "aspect"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"phenotype.hpoa is missing expected columns: {sorted(missing)}")
    return df


def mondo_branch_closure(nodes_df, edges_df, root: str) -> "tuple[set, dict, dict]":
    """(closure ids incl. root, id->min depth from root, id->name) over the
    Disease-only subclass_of adjacency (obsolete endpoints already dropped by
    the shared `_disease_child_adjacency`). Depth is BFS min-depth — the same
    notion the fit diagnostics bucket by."""
    child_adj = _disease_child_adjacency(edges_df, nodes_df)
    closure = {root} | _descendants(child_adj, root)
    depth = {root: 0}
    frontier = [root]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for c in child_adj.get(u, ()):
                if c in closure and c not in depth:
                    depth[c] = d
                    nxt.append(c)
        frontier = nxt
    names = dict(zip(nodes_df["id"], nodes_df["name"].astype(str)))
    return closure, depth, {i: names.get(i, i) for i in closure}


def mondo_hpoa_keys(nodes_df, ids: set) -> pd.DataFrame:
    """(mondo_id, db_id) rows: Mondo `xref` CURIEs rewritten to HPOA
    `database_id` form (Orphanet: -> ORPHA:), restricted to ``ids``."""
    sub = nodes_df[nodes_df["id"].isin(ids)][["id", "xref"]].copy()
    sub["xref"] = sub["xref"].fillna("").str.split("|")
    sub = sub.explode("xref")
    frames = []
    for src, dst in _DB_PREFIXES.items():
        hit = sub[sub["xref"].str.startswith(src)].copy()
        hit["db_id"] = dst + hit["xref"].str.slice(len(src))
        frames.append(hit[["id", "db_id"]])
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out.rename(columns={"id": "mondo_id"})


def build_profiles(hpoa_df: pd.DataFrame, keys_df: pd.DataFrame) -> pd.DataFrame:
    """Join HPOA onto Mondo nodes -> one row per (mondo_id, hpo_id, polarity).

    Aspect is restricted to ``P`` (phenotypic abnormality; inheritance and
    clinical-course rows do not describe observable codes). Polarity: a row is
    NEGATIVE when Qualifier=NOT **or** its frequency resolves to exactly 0
    (HPOA's "Excluded"/0% both mean tested-and-absent — boosting those would
    invert the annotation's meaning). Across sources (a disease often has both
    an OMIM and an ORPHA record) frequency aggregates by max and a row is
    negative only if EVERY source says so."""
    p = hpoa_df[hpoa_df["aspect"] == "P"].copy()
    p = p.merge(keys_df, left_on="database_id", right_on="db_id")
    p["freq"] = p["frequency"].map(normalize_frequency)
    p["neg"] = (p["qualifier"].fillna("").str.strip() == "NOT") | (p["freq"] == 0.0)
    g = p.groupby(["mondo_id", "hpo_id"]).agg(
        neg=("neg", "min"),           # negative only if unanimously negative
        freq=("freq", "max"),         # optimistic pool across sources
        has_freq=("freq", lambda s: s.notna().any()),
        n_sources=("database_id", "nunique"),
    ).reset_index()
    g["neg"] = g["neg"].astype(bool)
    g["has_freq"] = g["has_freq"].astype(bool)
    return g


def hpo_realizability(obo_text: str) -> "tuple[set, set, dict]":
    """(snomed_direct, snomed_closure, hp_labels) from ``hp.obo``.

    ``snomed_direct``: HP terms carrying a SNOMED xref themselves.
    ``snomed_closure``: HP terms with a SNOMED xref on themselves OR any HPO
    descendant — the true-path direction: a patient coded with the more
    specific phenotype has the profile's more general one. Equivalently (and
    computed as): a term is closure-realizable iff it is an ancestor-or-self
    of a direct-xref term, so one BFS up the parent map suffices."""
    labels, parents = parse_hpo_dag(obo_text)
    direct = {hp for hp, _n, vocab, _c in parse_hpo_xrefs(obo_text) if vocab == "SNOMED"}
    closure: set[str] = set()
    frontier = list(direct)
    while frontier:
        t = frontier.pop()
        if t in closure:
            continue
        closure.add(t)
        frontier.extend(parents.get(t, ()))
    return direct, closure, labels


def survey_rows(profiles, closure_ids, depth, names, snomed_direct, snomed_closure):
    """Per-node survey rows for EVERY node in the branch closure (a node with no
    HPOA profile still gets a row — absence is the finding)."""
    by_node = {m: g for m, g in profiles.groupby("mondo_id")}
    rows = []
    for mid in sorted(closure_ids):
        g = by_node.get(mid)
        pos = g[~g["neg"]] if g is not None else None
        n_pos = 0 if pos is None else len(pos)
        rows.append({
            "mondo_id": mid,
            "name": names.get(mid, mid),
            "depth": depth.get(mid, -1),
            "profile_n": n_pos,
            "not_n": 0 if g is None else int(g["neg"].sum()),
            "freq_known_n": 0 if pos is None else int(pos["has_freq"].sum()),
            "snomed_direct_n": 0 if pos is None else int(pos["hpo_id"].isin(snomed_direct).sum()),
            "snomed_closure_n": 0 if pos is None else int(pos["hpo_id"].isin(snomed_closure).sum()),
        })
    return pd.DataFrame(rows)


def profile_code_rows(profiles, hp_parents, xref_rows, vocabs=("SNOMED",)) -> pd.DataFrame:
    """One row per (mondo_id, hp_id, vocab, code): the codes that EVIDENCE a
    profile term, collected over the term's HPO descendant-OR-SELF closure.

    WHY descendant-or-self: the true-path rule, in the emit direction. A patient
    coded with a MORE specific phenotype instantiates the profile's more general
    term, so a term's evidence set is every SNOMED xref at or below it — exactly
    the codes `hpo_realizability`'s closure rule counted as realizable, now made
    explicit for stage 2 to intersect with the corpus vocabulary. Siblings must
    NOT leak: a code below a sibling evidences the sibling's ancestors only,
    which the downward walk guarantees by construction.

    Negative terms (NOT / frequency-0) are CARRIED with ``neg=True`` — the eta
    prior wants to downweight them, and dropping them here would force stage 2
    back into the HPOA parse. ``freq`` stays NaN (a blank TSV cell) when no
    source reported one. A term with no code in its closure yields no rows."""
    children: dict[str, list] = {}
    for child, ps in hp_parents.items():
        for parent in ps:
            children.setdefault(parent, []).append(child)
    codes_by_term: dict[str, set] = {}
    for hp, _name, vocab, code in xref_rows:
        if vocab in vocabs:
            codes_by_term.setdefault(hp, set()).add((vocab, code))

    memo: dict[str, tuple] = {}

    def _closure_codes(term):
        if term not in memo:
            seen = {term}
            out = set(codes_by_term.get(term, ()))
            stack = list(children.get(term, ()))
            while stack:
                t = stack.pop()
                if t in seen:
                    continue
                seen.add(t)
                out |= codes_by_term.get(t, set())
                stack.extend(children.get(t, ()))
            memo[term] = tuple(sorted(out))
        return memo[term]

    rows = []
    for r in profiles.itertuples(index=False):
        for vocab, code in _closure_codes(r.hpo_id):
            rows.append({"mondo_id": r.mondo_id, "hp_id": r.hpo_id,
                         "neg": bool(r.neg), "freq": r.freq,
                         "vocab": vocab, "code": code})
    return pd.DataFrame(
        rows, columns=["mondo_id", "hp_id", "neg", "freq", "vocab", "code"])


def promiscuity(profiles, snomed_closure, hp_labels, top_n: int = 15):
    """(per-term node counts Counter, top rows) over REALIZABLE positive profile
    terms. A term in hundreds of branch profiles cannot align any single node."""
    pos = profiles[~profiles["neg"] & profiles["hpo_id"].isin(snomed_closure)]
    counts = Counter(pos["hpo_id"])
    top = [(hp, n, hp_labels.get(hp, hp)) for hp, n in counts.most_common(top_n)]
    return counts, top


def build_report(rows: pd.DataFrame, prom_counts, prom_top, meta: dict) -> str:
    """Pooled markdown report. Public ontology facts only — no patient data was
    read anywhere in stage 1, so nothing here touches the egress floor."""
    L = [f"# HPOA profile survey — {meta['branch']} ({meta['branch_name']})", ""]
    L += [f"Stage 1 (offline, public ontology data only). Mondo v{meta['mondo_version']}; "
          f"hp.obo {meta['hpo_version']}; phenotype.hpoa {meta['hpoa_version']}.", ""]
    n = len(rows)
    has = rows[rows["profile_n"] > 0]
    L += ["## Pooled", "",
          f"- branch closure nodes: **{n}**",
          f"- nodes with >=1 positive HPOA profile term: **{len(has)}** "
          f"({100.0 * len(has) / max(n, 1):.1f}%)",
          f"- nodes with >=1 SNOMED-realizable term (closure rule): "
          f"**{int((rows['snomed_closure_n'] > 0).sum())}**",
          f"- nodes with >=5 SNOMED-realizable terms: "
          f"**{int((rows['snomed_closure_n'] >= 5).sum())}**",
          f"- nodes with >=1 NOT / excluded annotation: **{int((rows['not_n'] > 0).sum())}**", ""]
    if len(has):
        L += ["Among nodes WITH a profile (medians):", "",
              f"- profile size: {has['profile_n'].median():.0f} "
              f"(p25 {has['profile_n'].quantile(.25):.0f}, p75 {has['profile_n'].quantile(.75):.0f})",
              f"- SNOMED-realizable (direct xref): {has['snomed_direct_n'].median():.0f}",
              f"- SNOMED-realizable (closure): {has['snomed_closure_n'].median():.0f}",
              f"- frequency-annotated share: "
              f"{(has['freq_known_n'] / has['profile_n']).median():.2f}", ""]
    L += ["## By depth", "",
          "| depth | nodes | with profile | median profile n | median realizable (closure) |",
          "|---|---|---|---|---|"]
    for d, g in rows.groupby("depth"):
        gh = g[g["profile_n"] > 0]
        L.append(f"| {d} | {len(g)} | {len(gh)} | "
                 f"{gh['profile_n'].median():.0f} | {gh['snomed_closure_n'].median():.0f} |"
                 if len(gh) else f"| {d} | {len(g)} | 0 | - | - |")
    L += ["", "## Promiscuity of realizable terms", "",
          f"Distinct realizable profile terms: {len(prom_counts)}; "
          f"appearing in >=10 node profiles: {sum(1 for v in prom_counts.values() if v >= 10)}; "
          f">=50: {sum(1 for v in prom_counts.values() if v >= 50)}.", "",
          "| HP term | label | # node profiles |", "|---|---|---|"]
    L += [f"| {hp} | {lbl} | {cnt} |" for hp, cnt, lbl in prom_top]
    L += ["", "Stage 2 (in-workspace) intersects realizable codes with the corpus "
          "vocabulary and counts persons under the egress floor.", ""]
    return "\n".join(L)


def _download_cached(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        sys.stderr.write(f"[hpoa-survey] cache hit {dest}\n")
        return dest
    sys.stderr.write(f"[hpoa-survey] downloading {url}\n")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted release URLs)
    return dest


def _file_version(path: Path, tag: str) -> str:
    """Self-reported version line from an ontology artifact (provenance for the
    report, since --hpo-release may be 'latest')."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for _ in range(40):
            ln = f.readline()
            if tag in ln:
                return ln.split(tag, 1)[1].strip().strip('"')
    return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="MONDO:0004995")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--hpo-release", default="latest",
                   help="HPO release tag (e.g. v2026-09-01) or 'latest'; the report "
                        "records the files' self-reported versions either way")
    p.add_argument("--cache-dir", default="data/ontology")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-promiscuous", type=int, default=15)
    p.add_argument("--emit-codes", default=None, metavar="PATH",
                   help="also write a per-(node, term, code) TSV of the SNOMED "
                        "codes evidencing each profile term (descendant-or-self "
                        "closure; negatives carried) — stage 2's input")
    args = p.parse_args(argv)

    cache = Path(args.cache_dir)
    nodes_p = _download_cached(
        _MONDO_RELEASE.format(v=args.mondo_version, f="mondo_nodes.tsv"),
        cache / f"{args.mondo_version}_mondo_nodes.tsv")
    edges_p = _download_cached(
        _MONDO_RELEASE.format(v=args.mondo_version, f="mondo_edges.tsv"),
        cache / f"{args.mondo_version}_mondo_edges.tsv")
    hpo_url = (_HPO_LATEST if args.hpo_release == "latest" else _HPO_RELEASE)
    obo_p = _download_cached(hpo_url.format(v=args.hpo_release, f="hp.obo"),
                             cache / f"{args.hpo_release}_hp.obo")
    hpoa_p = _download_cached(hpo_url.format(v=args.hpo_release, f="phenotype.hpoa"),
                              cache / f"{args.hpo_release}_phenotype.hpoa")

    nodes_df = pd.read_csv(nodes_p, sep="\t", low_memory=False)
    edges_df = pd.read_csv(edges_p, sep="\t", low_memory=False)
    closure, depth, names = mondo_branch_closure(nodes_df, edges_df, args.branch)
    sys.stderr.write(f"[hpoa-survey] branch closure: {len(closure)} disease nodes\n")

    keys = mondo_hpoa_keys(nodes_df, closure)
    hpoa_df = parse_hpoa(hpoa_p.read_text(encoding="utf-8", errors="replace"))
    profiles = build_profiles(hpoa_df, keys)
    obo_text = obo_p.read_text(encoding="utf-8", errors="replace")
    snomed_direct, snomed_closure, hp_labels = hpo_realizability(obo_text)
    rows = survey_rows(profiles, closure, depth, names, snomed_direct, snomed_closure)
    prom_counts, prom_top = promiscuity(profiles, snomed_closure, hp_labels,
                                        args.top_promiscuous)

    meta = {
        "branch": args.branch, "branch_name": names.get(args.branch, args.branch),
        "mondo_version": args.mondo_version,
        "hpo_version": _file_version(obo_p, "data-version:"),
        "hpoa_version": _file_version(hpoa_p, "#version:"),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = args.branch.replace(":", "_")
    rows.to_csv(out / f"hpoa_profile_survey_{slug}.tsv", sep="\t", index=False)
    (out / f"hpoa_profile_survey_{slug}.md").write_text(
        build_report(rows, prom_counts, prom_top, meta), encoding="utf-8")
    sys.stderr.write(f"[hpoa-survey] wrote survey to {out}\n")

    if args.emit_codes:
        # Re-parse the DAG rather than threading it out of hpo_realizability:
        # the existing outputs (and their tests) stay byte-identical, at the
        # cost of one extra seconds-scale obo parse in an offline tool.
        _labels, hp_parents = parse_hpo_dag(obo_text)
        codes = profile_code_rows(profiles, hp_parents,
                                  parse_hpo_xrefs(obo_text))
        dest = Path(args.emit_codes)
        dest.parent.mkdir(parents=True, exist_ok=True)
        codes.to_csv(dest, sep="\t", index=False)
        sys.stderr.write(
            f"[hpoa-survey] emit-codes: {len(codes)} rows, "
            f"{codes['mondo_id'].nunique()} nodes, "
            f"{codes['code'].nunique()} distinct codes -> {dest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
