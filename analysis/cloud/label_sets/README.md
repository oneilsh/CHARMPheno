# DisMech ribbon (`dismech_ribbon.tsv`)

The flat disease label set of spec
`docs/superpowers/specs/2026-10-07-dismech-ribbon-label-space-design.md`: one row
per DisMech disorder (`kb/disorders/*.yaml`) that carries a MONDO `disease_term`.
Loaded by `analysis/cloud/dismech_ribbon.py` and handed to the native-Mondo build
as the `label_set` the powered set is intersected with (`gated_pc_cloud --label-set`,
`label_set:` in an experiment's front matter). Subtypes are not rows; the hierarchy
lives only in the stacked readout.

## Provenance

- Source: `monarch-initiative/dismech`, `kb/disorders/`, at the commit named in the
  file's first line (`# dismech_commit: <sha>`). That sha is the pin; the bundle
  cache key folds `dismech:<sha12>:<n>:<digest12>` over the sorted Mondo ids
  (`dismech_ribbon.label_set_identity`), so re-cutting at a new commit is a new
  corpus and editing a descriptive column is not.
- Disorders without a MONDO term (poisonings, a few infections, ageing) are
  skipped and counted on stderr by the cutter. Two DisMech curations of one Mondo
  term both appear as rows and collapse to one label node.

## Columns

`mondo_id`, `disorder_name`, `category` (DisMech's coarse tag — descriptive only),
the nosology columns `harrisons_chapter`, `isds_skeletal_category`,
`icimd_category`, `iuis_category`, `icdo_morphology`, `mechanistic_category`,
`channelopathy_category`, `lysosomal_storage_category` (pipe-joined when a
disorder carries several values; §D4's candidate sets where a nosology is judged
phenotypically coherent — Harrison's and `category` are not), and `dismech_file`.

## Regenerate

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/monarch-initiative/dismech /tmp/dismech
git -C /tmp/dismech sparse-checkout set kb/disorders
python analysis/cloud/dismech_ribbon.py --kb /tmp/dismech/kb/disorders \
  --commit "$(git -C /tmp/dismech rev-parse HEAD)" \
  --out analysis/cloud/label_sets/dismech_ribbon.tsv
```
