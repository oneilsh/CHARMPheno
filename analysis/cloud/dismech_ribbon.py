"""The DisMech ribbon: a flat disease label set read off `monarch-initiative/dismech`.

WHY THIS EXISTS (spec `docs/superpowers/specs/2026-10-07-dismech-ribbon-label-space-design.md`)
-----------------------------------------------------------------------------------------
Every topic-side problem of exps 0123–0131 came from NESTING the label set: a
parent and child compete for the same documents and the same words. DisMech's
disorder set is a deliberately flat cross-cut of Mondo (one `disease_term` per
disorder; subtypes hang below and are not labels here), so using it as the label
set takes the hierarchy out of the fit while the stacked read keeps it (§D1/§D4).

This module is PURE and driver-side: it reads a checkout of DisMech's
`kb/disorders/*.yaml` (public data, no CDR) into a small TSV, and reads that TSV
back into the `label_set` the native-Mondo build filters its kept terms by. The
TSV, not the 3,300-file checkout, is the committed, reproducible artifact — the
same convention `anchor_selection_data/priority_seed.tsv` set — and it carries the
DisMech commit it was cut from in its header so the pin is citable per experiment.

The classification columns (`category`, `harrisons_chapter`, and the specialist
nosologies) ride along ONLY as descriptors for §D4's candidate sets; nothing here
turns them into hierarchy. Spec decision 2026-10-07: candidate sets default to
Mondo ancestors, specialist nosologies are admitted case by case, and Harrison's /
`category` are too broad to share phenotype.

Identity. `label_set_identity` is what the bundle cache key folds
(`_case_finding_cache.compute_bundle_cache_key(label_set=...)`): the DisMech
commit, the member count and a digest of the sorted Mondo ids — so two ribbons
cut from different DisMech states, or one edited by hand, can never share a
cached bundle, while the TSV's descriptive columns are free to change.
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The nosology columns kept from DisMech's `classifications:` block, in TSV order.
# `harrisons_chapter` is the broad one (35% coverage, descriptive only); the rest
# are the specialist nosologies §D4 may admit as candidate sets.
CLASSIFICATION_KEYS = (
    "harrisons_chapter", "isds_skeletal_category", "icimd_category",
    "iuis_category", "icdo_morphology", "mechanistic_category",
    "channelopathy_category", "lysosomal_storage_category",
)
COLUMNS = ("mondo_id", "disorder_name", "category", *CLASSIFICATION_KEYS,
           "dismech_file")
_HEADER_PREFIX = "# dismech_commit: "
# Members a ribbon can never contain, whatever DisMech says. MONDO:0000001 is the
# ONTOLOGY ROOT ("disease"); DisMech's Dorsalgia.yaml carries it as its disease_term
# (a curation error, reported upstream), and as a label node it is an ancestor of
# every other member — exp 0135 launch 1 nested the whole ribbon under it.
EXCLUDED_MONDO_IDS = frozenset({"MONDO:0000001"})


@dataclass(frozen=True)
class RibbonRow:
    mondo_id: str
    disorder_name: str
    category: str
    classifications: dict = field(default_factory=dict)
    dismech_file: str = ""


@dataclass(frozen=True)
class Ribbon:
    """A loaded ribbon: the Mondo ids (the label set), the rows behind them, and
    the DisMech commit they were cut from."""
    commit: str
    rows: tuple
    path: str = ""

    @property
    def mondo_ids(self) -> set:
        return {r.mondo_id for r in self.rows}

    def classification_sets(self, key: str) -> dict:
        """`{classification value: set of mondo ids}` for one nosology column —
        §D4's candidate sets, when that nosology is judged phenotypically coherent.
        A disorder with several values of one nosology lands in each."""
        out: dict = {}
        for r in self.rows:
            for v in _split(r.classifications.get(key, "")):
                out.setdefault(v, set()).add(r.mondo_id)
        return out


def _split(v: str) -> list:
    return [x for x in str(v or "").split("|") if x]


def _classification_values(block) -> list:
    """DisMech writes a classification as a mapping or a list of mappings, each
    with `classification_value`; be liberal, keep every value, in order."""
    if block is None:
        return []
    items = block if isinstance(block, list) else [block]
    vals = []
    for it in items:
        if isinstance(it, dict):
            v = it.get("classification_value")
        else:
            v = it
        if v is not None and str(v) not in vals:
            vals.append(str(v))
    return vals


def row_from_disorder(doc: dict, *, dismech_file: str = "") -> RibbonRow | None:
    """One parsed `kb/disorders/*.yaml` -> a ribbon row, or None when the disorder
    carries no MONDO `disease_term` (DisMech has a few dozen such: poisonings,
    an infection or two, ageing — they have no Mondo node to label and are
    skipped, counted by the caller)."""
    term = ((doc.get("disease_term") or {}).get("term") or {})
    mondo_id = str(term.get("id") or "")
    if not mondo_id.startswith("MONDO:") or mondo_id in EXCLUDED_MONDO_IDS:
        return None
    cls = doc.get("classifications") or {}
    classifications = {
        k: "|".join(_classification_values(cls.get(k))) for k in CLASSIFICATION_KEYS
        if _classification_values(cls.get(k))}
    return RibbonRow(mondo_id=mondo_id,
                     disorder_name=str(doc.get("name") or term.get("label") or ""),
                     category=str(doc.get("category") or ""),
                     classifications=classifications,
                     dismech_file=dismech_file)


def read_disorders(kb_dir) -> tuple:
    """Parse every `*.yaml` under a DisMech `kb/disorders/` dir.

    Returns `(rows, skipped)`: rows sorted by Mondo id then file name, and the
    files skipped for having no MONDO disease_term (or an excluded one, see
    `EXCLUDED_MONDO_IDS`). Two disorders may share a Mondo
    term (DisMech keeps a few near-duplicates); both rows are kept here and the
    label set deduplicates on `mondo_ids`, so the receipt can report the
    collision."""
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    rows, skipped = [], []
    for p in sorted(Path(kb_dir).glob("*.yaml")):
        with open(p, "r", encoding="utf-8") as fh:
            doc = yaml.load(fh, Loader=loader) or {}
        r = row_from_disorder(doc, dismech_file=p.name)
        if r is None:
            skipped.append(p.name)
        else:
            rows.append(r)
    rows.sort(key=lambda r: (r.mondo_id, r.dismech_file))
    return rows, skipped


def write_ribbon_tsv(rows, path, *, commit: str) -> None:
    """The committed artifact: a `# dismech_commit:` header line, then the TSV."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"{_HEADER_PREFIX}{commit}\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r.mondo_id, r.disorder_name, r.category,
                        *(r.classifications.get(k, "") for k in CLASSIFICATION_KEYS),
                        r.dismech_file])


def load_ribbon(path) -> Ribbon:
    """Read a ribbon TSV back. Raises ValueError on a file without the commit
    header (an un-pinned label set is not a label set)."""
    text = Path(path).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    if not first.startswith(_HEADER_PREFIX):
        raise ValueError(
            f"{path}: missing '{_HEADER_PREFIX.strip()}' header; a ribbon TSV must "
            "name the DisMech commit it was cut from (dismech_ribbon.py --commit).")
    commit = first[len(_HEADER_PREFIX):].strip()
    rows = []
    for rec in csv.DictReader(io.StringIO(rest), delimiter="\t"):
        rows.append(RibbonRow(
            mondo_id=rec["mondo_id"], disorder_name=rec.get("disorder_name", ""),
            category=rec.get("category", ""),
            classifications={k: rec[k] for k in CLASSIFICATION_KEYS
                             if rec.get(k)},
            dismech_file=rec.get("dismech_file", "")))
    return Ribbon(commit=commit, rows=tuple(rows), path=str(path))


def label_set_identity(ribbon: Ribbon) -> str:
    """The cache-key token for a ribbon: `dismech:<commit12>:<n>:<digest12>` over
    the SORTED distinct Mondo ids. Descriptive columns do not enter — editing a
    name or a nosology tag must not orphan a bundle; adding or removing a disease
    must."""
    ids = sorted(ribbon.mondo_ids)
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:12]
    return f"dismech:{ribbon.commit[:12]}:{len(ids)}:{digest}"


def load_label_set(path) -> tuple:
    """`(mondo_ids, identity, ribbon)` — what the fit driver needs from a
    `--label-set <tsv>` argument."""
    rb = load_ribbon(path)
    return rb.mondo_ids, label_set_identity(rb), rb


def _main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Cut the DisMech ribbon TSV from a kb/disorders/ checkout.")
    p.add_argument("--kb", required=True, help="path to dismech/kb/disorders")
    p.add_argument("--commit", required=True,
                   help="the DisMech commit sha the checkout is at (git rev-parse HEAD)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows, skipped = read_disorders(a.kb)
    write_ribbon_tsv(rows, a.out, commit=a.commit)
    rb = load_ribbon(a.out)
    n_dup = len(rows) - len(rb.mondo_ids)
    sys.stderr.write(
        f"[ribbon] {len(rows)} disorder(s) with a MONDO term -> "
        f"{len(rb.mondo_ids)} distinct Mondo id(s) ({n_dup} shared term(s)); "
        f"{len(skipped)} file(s) skipped (no MONDO disease_term); "
        f"identity {label_set_identity(rb)}\n")
    for k in CLASSIFICATION_KEYS:
        n = sum(1 for r in rows if r.classifications.get(k))
        if n:
            sys.stderr.write(f"[ribbon]   {k}: {n} disorder(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
