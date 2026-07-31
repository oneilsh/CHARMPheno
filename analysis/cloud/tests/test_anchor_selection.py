"""Tests for the dismech #1079 priority-list seed parser."""

# A synthetic snippet exercising: intro prose under a header (no rows), both link
# forms (monarchinitiative.org and a dismech-issue link with a trailing MONDO
# paren), [x]/[ ] curation flags, a header count annotation to strip, and a
# disease appearing under two categories.
_SNIPPET = """\
# List

## Grouping methodology

Each disease was assigned to a category by keyword matching. No rows here.

### Neuroimmune (12 diseases, 10 curated in dismech)

- [x] [myasthenia gravis](https://monarchinitiative.org/MONDO:0009688)
- [ ] [neuromyelitis optica](https://monarchinitiative.org/MONDO:0019100)

### Neurodegenerative (150 diseases, 66 curated in dismech)

- [x] [episodic ataxia type 2](https://github.com/monarch-initiative/dismech/issues/1678) (MONDO:0007163)
- [x] [myasthenia gravis](https://monarchinitiative.org/MONDO:0009688)
"""


def test_parse_extracts_mondo_label_category_and_curation():
    from anchor_selection import parse_priority_seed

    rows = parse_priority_seed(_SNIPPET)
    # Four task lines carry a MONDO id; the prose section yields nothing.
    assert len(rows) == 4

    mg = rows[0]
    assert mg.mondo_id == "MONDO:0009688"
    assert mg.label == "myasthenia gravis"
    assert mg.category == "Neuroimmune"  # count annotation stripped
    assert mg.curated is True

    nmo = rows[1]
    assert nmo.mondo_id == "MONDO:0019100"
    assert nmo.curated is False

    # dismech-issue link form: MONDO id read from the trailing parenthesis.
    ea2 = rows[2]
    assert ea2.mondo_id == "MONDO:0007163"
    assert ea2.label == "episodic ataxia type 2"
    assert ea2.category == "Neurodegenerative"


def test_unique_diseases_exposes_multicategory_membership():
    from anchor_selection import parse_priority_seed, unique_diseases

    uniq = unique_diseases(parse_priority_seed(_SNIPPET))
    assert len(uniq) == 3  # MG counted once despite two category rows
    assert uniq["MONDO:0009688"] == ["Neurodegenerative", "Neuroimmune"]


def test_to_tsv_roundtrips_row_count():
    from anchor_selection import parse_priority_seed, to_tsv

    tsv = to_tsv(parse_priority_seed(_SNIPPET))
    lines = tsv.strip().split("\n")
    assert lines[0] == "mondo_id\tlabel\tcategory\tcurated"
    assert len(lines) == 1 + 4  # header + 4 rows
    assert "\tMONDO:0009688\t" not in tsv  # id is first column, no leading tab
    assert tsv.count("MONDO:0009688") == 2  # appears under both categories
