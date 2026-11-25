# Scoping: splitting spark-vi into its own package (and a JOSS paper)

**Date:** 2026-11-24. **Context:** the end of the spark-vi and DAG-gated walkthroughs
(REVIEW_LOG 2026-11-24), where the maintainer raised publishing spark-vi on its own,
possibly in the Journal of Open Source Software (JOSS; first written here
as the Journal of Statistical Software, corrected by the maintainer 2026-11-25). This report sizes the work and lists
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

## Decisions (maintainer, 2026-11-24)

1. `eval.metrics` moves out with the readout.
2. The simulator-study helpers move to charmpheno. (The held-out predictive
   log-likelihood that STM and `predictive_gain` import from
   `eval/topic/concentration_recovery.py` is generic and stays in spark-vi; only the
   recovery-experiment parts move.)
3. The readout (with `DagHeads` and `eval.metrics`) becomes part of charmpheno.
4. A separate public repository, but not yet as the development home. spark-vi keeps
   being developed at `CHARMPheno/spark-vi/` (one commit for a change spanning both
   packages; the cluster, `make zip`, `--py-files` and the cache keys unchanged), and a
   publish script copies it into the public repository as clean, tagged release
   commits (no rewrite or review of the monorepo's commit history; an outside
   contribution comes back as a patch). When spark-vi stops moving with CHARMPheno, the
   public repository becomes canonical and CHARMPheno depends on a released version.
   Rejected for now: a git submodule (every cluster command needs a submodule update,
   cross-package changes become two ordered commits, a missed bump runs stale code
   silently) and `git subtree` push (publishes every commit message).

## Work packages

Ordered so the package is usable at every step.

1. **Move the readout out** — **done 2026-11-24** (ADR 0054; `668d43b`, plus the
   announced cache drop `b44064f` for a hashed docstring). Originally: an ADR
   superseding ADR 0051 §3's placement. `spark_vi/
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
7. **Packaging hygiene.** *Done 2026-11-25:* `spark-vi/LICENSE` (MIT; the
   dependencies are Apache-2.0 (pyspark), BSD-3 (numpy, scipy) and MIT (formulaic),
   all compatible, and no third-party code is vendored), `spark-vi/CHANGELOG.md`,
   versions documented as semver 0.x. `probes/` stays in the monorepo (the review log
   cites it); the wheel and sdist never carried it or `dist/`, and the publish script
   leaves both out. Original scope: Add a LICENSE file (pyproject says MIT; the repository has
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
   local). The MLlib parity test for plain LDA lives in charmpheno today
   (`charmpheno/tests/test_lda_compare.py`) and is spark-vi's only recovery check; it
   belongs in the released package's tests (two spark-vi test docstrings point at it).
10. **Extract the history.** `git filter-repo --subdirectory-filter spark-vi` keeps
    spark-vi's history (85 commits since the 2026-11-02 history scrub). Review commit
    messages before publishing: they name experiments, run directories and CDR
    identifiers. No patient-level data is in the tree, but the messages are worth a
    read; squashing to a clean initial commit is the alternative.

## What JOSS will look at

JOSS reviews the software more than the paper. The paper is short (roughly 250-1,000
words: a summary, a statement of need, the state of the field, references), and the
review is a checklist run against the public repository: an OSI license, a public
repository with an issue tracker, installation instructions that work, example usage,
API documentation, automated tests, community guidelines (how to contribute, report
issues, get support), and any performance or functionality claims the paper makes being
verifiable. The software must show substantial scholarly effort and a research use.
A stated, tested dependency range is the norm; pinning is not required. Check the
current author and review guidelines (and any policy on generative-AI assistance)
before submitting.

What this changes in the work packages: WP8 (documentation) and the community files
(CONTRIBUTING, issue templates) are review requirements; WP9 shrinks from full
replication materials to runnable examples on public data plus the evidence for any
claim the paper makes (e.g. Spark vs local agreement, parity with MLlib's LDA), and
benchmarks only if the paper claims performance. The storyline stays: an SVI framework
on Spark with an in-memory engine under the same model contract, illustrated by LDA,
HDP and STM, with the DAG-gated LDA (node-set learning from a user's DAG and labelled
documents) as the distinctive model.

## Suggested order

WP1 (the readout move) first: it decides the package boundary and is the only change
that touches CHARMPheno's own code paths. Then WP7 (license, cleanup) and WP2-3
(persistence, no private API), which are what a first public release needs. WP4-6 can
follow as 0.x releases; WP8-10 accompany the JOSS submission.
