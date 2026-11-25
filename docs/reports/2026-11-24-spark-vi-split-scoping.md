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
2. **One persistence style.** *Done 2026-11-25 for LDA / HDP (ADR 0055):* both models
   are `_ResultStagePersistence` stages (Params, `PipelineModel`); no backward
   compatibility for the old layout (maintainer); `_PersistableModel` is gone. *STM done
   2026-11-25 (ADR 0056):* `OnlineSTMEstimator` / `OnlineSTMModel` replace
   `StreamingSTM` / `STMModel` as MLlib stages on the same persistence (camelCase Params,
   `transform` = each document's gated mode θ, `PipelineModel`); covariates are
   DataFrame columns, a pre-built vector or a formula over the columns (the separate
   covariate frame is gone); the formula's spec persists as JSON (the formula plus the
   schema-frame it was fitted on, rebuilt on load), no pickle; no backward
   compatibility. The CHARMPheno drivers fit through the Estimator and keep writing
   their run dir as a VIResult checkpoint. Original scope: OnlineLDA / OnlineHDP still use the older
   `_PersistableModel` (bare VIResult, no Params, not Pipeline-capable), and STM's
   model is not an MLlib `Model` at all (its own `save` / `load` over a local `Path`);
   move all three to `_StagePersistence` as the DAG-gated stages and the featurizer
   are.
3. **No private MLlib API.** *Done 2026-11-25 (ADR 0055):* our own writer / reader of
   the metadata JSON, pinned to MLlib's document by a fixture from the private writer;
   pyspark `>=3.5`, no upper pin. The remaining underscore pyspark uses are listed in
   the ADR (the Params subclassing pattern, and the py4j Hadoop-filesystem access that
   PySpark has no public API for). Still to do: CI on the lower bound and the latest
   pyspark. Original scope: Write the Params metadata (class, uid, timestamp, Spark
   version, `paramMap`, `defaultParamMap`) ourselves in MLlib's file layout, replacing
   `DefaultParamsWriter._get_metadata_to_save` and `DefaultParamsReader._parseMetaData`.
   Then declare a pyspark lower bound only and run CI on the bound and the latest
   release; no upper pin.
4. **Crash safety.** Local checkpoints: use the fresh-directory scheme the remote path
   already uses (closes the rename window, and leaves one strategy). HDFS: write the
   manifest without truncate-then-write (a temp file plus HDFS rename, which is atomic
   within one filesystem).
5. **Memory at scoring.** *Done 2026-11-25 (`b0461f2`):* the dense K x V E[log β]
   (~400 MB at K ≈ 3,250 over three domains) is shipped once per executor as a
   memory-mapped file (`ship_array` / `load_shipped`), not broadcast once per Python
   worker; no pickled object near the 2 GB cap. Maintainer's choice among: float32
   (rejected: precision), per-domain broadcasts (only splits the object; total memory
   unchanged), shipping only each document's gated-open rows (exact and the most
   scalable, but a redesign; parked for DAGs that outgrow this), top-terms truncation
   (rejected: changes results). The readout's dense `closure_matrix` left with the
   readout.
6. **Forked workers.** Cap BLAS threads in LocalRunner's workers (`threadpoolctl`, a
   small optional dependency) to avoid oversubscription; give the open-time seed pass
   an in-memory path so a leaves-first DAG-gated fit can run without Spark.
   *Measured 2026-11-25:* oversubscription is not only a forked-worker problem. One
   in-process STM `local_update` over 1,500 synthetic documents (K = 40, V = 300) took
   ~80 s with OpenBLAS's default threads and ~3 s with one thread, on a 4-core machine
   with another test process running: a ~30x slowdown, enough that
   `test_stm_hardening.py` ran for hours. The E-steps are many small per-document
   matrix operations, where BLAS threads only contend. The test suite now sets one
   thread in `spark-vi/tests/conftest.py`; the library should cap threads inside its
   own E-steps (a `threadpool_limits` context around `local_update`, both engines)
   rather than rely on users setting `OMP_NUM_THREADS`.
   *Cluster probe, 2026-11-25* (`make blas-probe`, `analysis/cloud/blas_probe.py`: the
   DAG-gated per-document CAVI on synthetic documents, all 40 executor cores busy,
   0175/0176's geometry of 20 executors x 2 cores on 4-vCPU workers, OpenBLAS 0.3.25).
   Our Dataproc workers already run one thread: the Python workers see
   `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` (set by the cluster, not by us), so
   the cluster fits are not oversubscribed and setting it changes nothing (median ms
   per document, Spark's setting vs 1: 1.45 vs 1.43, 4.36 vs 4.41, 0.44 vs 0.43, 5.03
   vs 4.79 over K_open x n_unique = 200x40, 200x150, 400x40, 400x150). With the
   variables removed, OpenBLAS takes all 4 cores per process and the same work is
   1.1x, 4.3x, 4.8x and 4.1x slower (1.64, 18.75, 2.10, 20.41 ms). On the dev box
   (OpenBLAS 0.3.29, 4 cores, 2 tasks) the gated CAVI showed no such effect, so the
   cost depends on the BLAS build and the load; STM's E-step is exposed on both.
   What it means for the package: a user fitting outside a cluster that sets these
   variables (LocalRunner, a laptop, a plain Spark install) can lose a factor of 4 or
   more, so the cap belongs in the library. Open question: whether exp 0170's local
   engine on the master (240 s per iteration against 54 s on the executors) ran with
   the variables set; the driver's environment is not the YARN containers'.
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
benchmarks only if the paper claims performance. The storyline: an SVI framework on Spark
with an in-memory engine under the same model contract, illustrated by LDA, HDP and
STM, with the DAG-gated LDA as one of the shipped models.

Two papers (maintainer, 2026-11-25): the JOSS paper is about the framework; the
DAG-gated model's method (node-set learning from a user's DAG and labelled documents,
the gate schedule) is detailed in CHARMPheno's own paper, a methods-and-application
paper on rare-disease characterization, even though the code lives in spark-vi. So the
JOSS paper describes the DAG-gated model in a sentence or two and cites the methods
paper rather than deriving it, and spark-vi's documentation of that model points there
for the method. charmpheno's own dependency pins are not part of this release.

## Suggested order

WP1 (the readout move) first: it decides the package boundary and is the only change
that touches CHARMPheno's own code paths. Then WP7 (license, cleanup) and WP2-3
(persistence, no private API), which are what a first public release needs. WP4-6 can
follow as 0.x releases; WP8-10 accompany the JOSS submission.
