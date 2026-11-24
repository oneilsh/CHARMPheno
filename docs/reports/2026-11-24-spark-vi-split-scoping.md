# Scoping: splitting spark-vi into its own package (and a JSS paper)

**Date:** 2026-11-24. **Context:** the end of the spark-vi and DAG-gated walkthroughs
(REVIEW_LOG 2026-11-24), where the maintainer raised publishing spark-vi on its own,
possibly in the Journal of Statistical Software. This report sizes the work and lists
the decisions; it is not a plan of record (that would be a spec and a plan under
`docs/superpowers/`).

## What spark-vi is today

About 21,700 lines of package code and 23,500 of tests, pure Python, depending on
pyspark (>= 3.5, < 4), numpy and scipy (formulaic optional, for STM). It never imports
charmpheno (enforced by convention; the grep for OMOP / Mondo / charmpheno in the
package is empty). By subpackage (lines):

| subpackage | lines | what |
|---|---|---|
| `mllib/` | 7,600 | MLlib Estimator/Model shims: LDA, HDP, STM, DAG-gated LDA, the featurizer, DagHeads, persistence |
| `models/` | 5,000 | the VIModel implementations (LDA, online HDP, STM, DAG-gated LDA), spectral init |
| `readout/` | 3,100 | per-node logistic heads over a feature column, the closure product, Spark and local backends |
| `eval/` | 2,100 | exact ranking metrics (`eval.metrics`), topic evaluation (coherence, gated NPMI, sharpness, concentration recovery, seed panel) |
| `core/` | 1,600 | VIModel contract, VIRunner, LocalRunner, the shared SVI loop, retry, row-sparse statistics |
| `dag/` | 1,200 | node-set learning, the engine layout, graph operations |
| `io/` | 650 | any-URI file access, VIResult checkpoints |
| `inference/`, `diagnostics/` | 370 | concentration optimization, persist helpers |

## Proposed scope of the released package

**In:** the framework (`core`, `io`), the models and their MLlib shims (LDA, HDP, STM,
DAG-gated LDA, the featurizer), `dag` (the DAG-gated model needs node-set learning),
and the topic-evaluation tools that apply to any topic model (coherence / NPMI,
sharpness).

**Out** (maintainer's direction, this walkthrough): the readout. It is supervised
per-node logistic regression over a label DAG, used only by the clinical evaluation;
no other model has one, and it is not variational inference. With it go
`mllib/readout.py` (DagHeads) and its tests.

**To decide:**
1. `eval.metrics` (exact AUC / AP / calibration on mergeable tallies). Model-agnostic
   and NumPy-only, but its only consumer is the readout's evaluation. Lean: move it with
   the readout.
2. `eval/topic/{seed_panel,concentration_recovery}.py`: experiment helpers for our own
   simulator studies. Lean: move to charmpheno or a `benchmarks/` folder of the new
   repo, not the package.
3. The readout's new home: a `charmpheno` subpackage (simplest; charmpheno already
   depends on spark-vi) or a third small package (only worth it if someone else would
   use it).
4. Monorepo or a separate repository. A separate repository is the clean public
   artifact (its own issues, CI, releases); CHARMPheno then pins a released version
   or a git tag. The monorepo is cheaper to keep in sync while both move fast.

## Work packages

Ordered so the package is usable at every step.

1. **Move the readout out** (ADR superseding ADR 0051 §3's placement). `spark_vi/
   __init__.py` and `mllib/__init__.py` re-export it today; `closure_matrix` and the
   readout's uses of `spark_vi.core` helpers (broadcast, retry) become imports from the
   package. No cache keys are involved (the readout is not a bundle input).
2. **One persistence style.** OnlineLDA / OnlineHDP still use the older
   `_PersistableModel` (bare VIResult, no Params, not Pipeline-capable), and STM's
   model is not an MLlib `Model` at all (its own `save` / `load` over a local `Path`);
   move all three to `_StagePersistence` as the DAG-gated stages and the featurizer
   are.
3. **No private MLlib API.** Write the Params metadata (class, uid, timestamp, Spark
   version, `paramMap`, `defaultParamMap`) ourselves in MLlib's file layout, replacing
   `DefaultParamsWriter._get_metadata_to_save` and `DefaultParamsReader._parseMetaData`.
   Then declare a pyspark lower bound only and run CI on the bound and the latest
   release; no upper pin.
4. **Crash safety.** Local checkpoints: use the fresh-directory scheme the remote path
   already uses (closes the rename window, and leaves one strategy). HDFS: write the
   manifest without truncate-then-write (a temp file plus HDFS rename, which is atomic
   within one filesystem).
5. **Memory at scoring.** `DagGatedLDAModel.transform` broadcasts the dense K x V
   E[log β] (~400 MB at K ≈ 3,250 over three domains): float32 or per-domain shipping;
   the readout's dense `closure_matrix` leaves with the readout.
6. **Forked workers.** Cap BLAS threads in LocalRunner's workers (`threadpoolctl`, a
   small optional dependency) to avoid oversubscription; give the open-time seed pass
   an in-memory path so a leaves-first DAG-gated fit can run without Spark.
7. **Packaging hygiene.** Add a LICENSE file (pyproject says MIT; the repository has
   none); remove `spark-vi/probes/` (debug scripts) and `spark-vi/dist/` from the
   release; a CHANGELOG; version 0.x with semantic versioning.
8. **Documentation.** A user guide (install, the two engines, each model, persistence,
   resume), API reference generated from the docstrings (they are already thorough),
   and one tutorial per model.
9. **Replication materials on public data.** No All of Us data can appear. Candidates:
   a public text corpus (e.g. 20 Newsgroups, NeurIPS abstracts) for LDA / HDP / STM, and
   the synthetic DAG-gated simulator (`scripts/simulate_gated_omop.py`, made generic)
   with a public ontology for the gated model. Benchmarks against MLlib's online LDA and
   gensim for plain LDA (time, held-out perplexity), and engine agreement (Spark vs
   local).
10. **Extract the history.** `git filter-repo --subdirectory-filter spark-vi` keeps
    spark-vi's history (85 commits since the 2026-11-02 history scrub). Review commit
    messages before publishing: they name experiments, run directories and CDR
    identifiers. No patient-level data is in the tree, but the messages are worth a
    read; squashing to a clean initial commit is the alternative.

## What JSS will look at

A paper describing the software, the software under a free license, documentation,
and replication code whose results the reviewers can reproduce. Version pinning is
not a requirement; a stated, tested dependency range is the norm. The likely
storyline is one package: an SVI framework on Spark with an in-memory engine under the
same model contract, illustrated by LDA, HDP and STM, with the DAG-gated LDA (node-set
learning from a user's DAG and labelled documents) as the distinctive model. Check the
current author guidelines before writing.

## Suggested order

WP1 (the readout move) first: it decides the package boundary and is the only change
that touches CHARMPheno's own code paths. Then WP7 (license, cleanup) and WP2-3
(persistence, no private API), which are what a first public release needs. WP4-6 can
follow as 0.x releases; WP8-10 accompany the paper.
