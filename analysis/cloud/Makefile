.PHONY: help zip cluster-overlay smoke bq-smoke lda-bq-smoke hdp-bq-smoke eval-bq-coherence build-dashboard-bundle lda-bq-fit-eval hdp-bq-fit-eval inspect smoke-dataproc setup next-exp exp summary summary-tail eval-exp build-dashboard-exp build-covariates

REPO_ROOT       := $(shell git rev-parse --show-toplevel)
SPARK_VI_ZIP    := $(REPO_ROOT)/spark-vi/dist/spark_vi.zip
CHARMPHENO_ZIP  := $(REPO_ROOT)/charmpheno/dist/charmpheno.zip
# Dependency-overlay zip: pure-Python deps the AoU Dataproc image lacks (see
# cluster-requirements.txt), shipped on --py-files alongside the source zips
# and run on the image's own Python (which provides numpy/pandas/scipy/pyarrow).
CLUSTER_REQS        := $(REPO_ROOT)/analysis/cloud/cluster-requirements.txt
CLUSTER_OVERLAY     := $(REPO_ROOT)/analysis/cloud/dist/formulaic_overlay.zip
CLUSTER_OVERLAY_DIR := $(REPO_ROOT)/analysis/cloud/dist/overlay-build
SCRIPT          := smoke_lda_cloud.py
BQ_SCRIPT       := bq_smoke_cloud.py
LDA_BQ_SCRIPT   := lda_bigquery_cloud.py
HDP_BQ_SCRIPT   := hdp_bigquery_cloud.py
EVAL_SCRIPT     := eval_coherence_cloud.py
BUNDLE_SCRIPT   := build_dashboard_cloud.py
INSPECT_SCRIPT  := inspect_app.py
SETUP_SCRIPT    := setup_workspace.py
WORKSPACE_ENV   := .workspace_env

# Args forwarded to lda-bq-smoke / hdp-bq-smoke / eval-bq-coherence / inspect
# (override on command line).
LDA_BQ_ARGS     ?=
HDP_BQ_ARGS     ?=
EVAL_ARGS       ?=
BUNDLE_ARGS     ?=
INSPECT_ARGS    ?=

# Required for eval-bq-coherence: path to the augmented VIResult dir written
# by lda_bigquery_cloud.py (local or GCS-mounted, e.g. /mnt/gcs/<bucket>/runs/<id>).
CHECKPOINT      ?=

# Checkpoint / resume knobs for long-running fits (override on command line).
# SAVE_DIR is the authoritative post-fit artifact when set; empty = no save.
# SAVE_INTERVAL = -1 means only the end-of-fit save fires (requires SAVE_DIR).
# RESUME_FROM points at a previously-written save dir; empty = fresh start.
# Empty strings are passed through as `--save-dir ""` etc. and treated by
# argparse identically to omitting the flag (matches the driver defaults).
# Emitted before LDA_BQ_ARGS / HDP_BQ_ARGS so a user-supplied --save-dir
# in those overrides wins on conflict (argparse takes the last occurrence).
SAVE_DIR        ?=
SAVE_INTERVAL   ?= -1
RESUME_FROM     ?=

# Optional HDFS/GCS cache for the (BQ load + BOW build) prep phase. Keyed by
# (source_table, person_mod, vocab_size, min_df, doc_spec); a matching prior
# run reloads in seconds instead of re-extracting from BQ + re-fitting
# CountVectorizer. HDFS default = session-only (lost when cluster shuts down);
# set CORPUS_CACHE_URI= (empty) to disable. Use gs://... for persistence.
CORPUS_CACHE_URI ?= hdfs:///user/dataproc/charm/corpus_cache

# Optional manual override for the CDR (passed through to setup_workspace.py).
# Use when `wb resource list` doesn't surface the CDR (external read-only
# data project, common case for AoU). Persists in .workspace_env afterwards.
CDR             ?=

# gcloud-submit fallback only — `smoke` doesn't need any of these.
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
CLUSTER ?=
REGION  ?=
BUCKET  ?=

SETUP_CMD := python3 $(SETUP_SCRIPT) $(if $(CDR),--cdr=$(CDR))

default: help

help:
	@echo "Cluster smoke targets (run from this dir on the Dataproc master):"
	@echo ""
	@echo "  make setup           - One-time per-cluster provisioning: discover workspace env"
	@echo "                         (project, CDR, buckets) via the 'wb' CLI -> $(WORKSPACE_ENV),"
	@echo "                         and build the cluster dependency overlay (deps the image"
	@echo "                         lacks, e.g. formulaic). Pass CDR=<proj>.<ds> when the CDR is"
	@echo "                         an external read-only project not surfaced by 'wb resource"
	@echo "                         list'."
	@echo "                         Workspace discovery auto-runs if $(WORKSPACE_ENV) is missing."
	@echo "  make smoke           - LDA shim smoke: spark-submit via YARN (client mode)"
	@echo "  make bq-smoke        - BigQuery read smoke (no spark_vi imports)"
	@echo "  make lda-bq-smoke    - End-to-end: BQ -> CountVectorizer -> LDA shim"
	@echo "                         Override args via LDA_BQ_ARGS='--K 10 --person-mod 5000'"
	@echo "                         Tune SVI step: LDA_BQ_ARGS='--tau0 64 --kappa 0.7'"
	@echo "                         (defaults τ₀=1024 κ=0.51 give ρ_0 ≈ 0.029 — slow on small corpora)"
	@echo "                         Checkpoint/resume: SAVE_DIR=/mnt/gcs/\$$BUCKET/runs/<id>/"
	@echo "                                            SAVE_INTERVAL=10  RESUME_FROM=<prev-save-dir>"
	@echo "  make hdp-bq-smoke    - End-to-end: BQ -> CountVectorizer -> OnlineHDP shim"
	@echo "                         Override args via HDP_BQ_ARGS='--T 100 --K 10 --person-mod 5000'"
	@echo "                         Tune SVI step: HDP_BQ_ARGS='--tau0 64 --kappa 0.7'"
	@echo "                         Disable γ opt: HDP_BQ_ARGS='--no-optimize-corpus-concentration'"
	@echo "                         Checkpoint/resume: SAVE_DIR=/mnt/gcs/\$$BUCKET/runs/<id>/"
	@echo "                                            SAVE_INTERVAL=10  RESUME_FROM=<prev-save-dir>"
	@echo "  make eval-bq-coherence - Full-corpus NPMI coherence on an augmented VIResult"
	@echo "                         CHECKPOINT=/mnt/gcs/\$$BUCKET/runs/<id>  (required)"
	@echo "                         Override args via EVAL_ARGS='--top-n 10'"
	@echo "                         HDP: EVAL_ARGS='--model-class hdp --hdp-k 50'"
	@echo "  make build-dashboard-bundle - 4-file dashboard bundle (+ .zip) from a checkpoint"
	@echo "                         CHECKPOINT=/mnt/gcs/\$$BUCKET/runs/<id>  (required)"
	@echo "                         Writes <checkpoint>/dashboard_bundle/ + dashboard_bundle.zip"
	@echo "                         Override via BUNDLE_ARGS='--vocab-top-n 3000 --top-n-codes-for-npmi 30'"
	@echo "                         HDP: BUNDLE_ARGS='--model-class hdp --hdp-top-k 50'"
	@echo "  make lda-bq-fit-eval - End-to-end fit + eval; auto-generates timestamped"
	@echo "                         SAVE_DIR under \$$(REPO_ROOT)/data/runs/<id>."
	@echo "                         LDA_BQ_ARGS='--K 50 --max-iter 200 ...'"
	@echo "                         EVAL_ARGS='--top-n 20'  SAVE_INTERVAL=10"
	@echo "                         To RESUME a prior run with more iterations:"
	@echo "                           make lda-bq-fit-eval RUN_ID=<existing-id> LDA_BQ_ARGS='--max-iter <N more>'"
	@echo "                         --max-iter on resume is ADDITIONAL iters (not a target total)."
	@echo "  make hdp-bq-fit-eval - End-to-end HDP fit + eval; same shape as lda-bq-fit-eval."
	@echo "                         HDP_BQ_ARGS='--T 150 --K 15 ...'"
	@echo "                         EVAL_ARGS='--model-class hdp --hdp-k 50'"
	@echo "                         RESUME: make hdp-bq-fit-eval RUN_ID=<existing-id> HDP_BQ_ARGS='--max-iter <N more>'"
	@echo "  next-exp [NO_EVAL=1]    Pick lowest-NNNN pending experiment; fit (+ eval unless NO_EVAL)."
	@echo "  exp ID=N [NO_EVAL=1]    Run/resume experiment N; fit (+ eval unless NO_EVAL)."
	@echo "  summary ID=N            Print summary.md for experiment N."
	@echo "  summary-tail ID=N       Print only the most-recent ## section of summary.md."
	@echo "  eval-exp [ID=N]         Re-run eval; if ID omitted, auto-selects most-recently-fit."
	@echo "  build-dashboard-exp     - build dashboard bundle (auto-picks freshest fit needing build)"
	@echo "  build-dashboard-exp ID=N - force rebuild for a specific experiment"
	@echo "  build-covariates EXP=N  - rebuild STM covariate cache for experiment N without re-running fit."
	@echo "                         Useful when iterating on covariate_formula. FORCE=1 invalidates cache."
	@echo "  make lda-bq-fit-eval-cancer - LDA fit + eval on the first-cancer-year cohort."
	@echo "                         Restricts to patients with a first malignant-cancer dx"
	@echo "                         (descendants of SNOMED 443392, minus NMSC and carcinoma"
	@echo "                         in situ) and windows their events to the 365 days starting"
	@echo "                         at that first dx. doc-unit=patient (one doc per patient's"
	@echo "                         post-dx year)."
	@echo "                         LDA_BQ_ARGS='--K 50 --max-iter 200 --person-mod 1 ...'"
	@echo "  make lda-bq-fit-eval-dementia - LDA fit + eval on the first-dementia-year cohort."
	@echo "                         Restricts to patients with a first all-cause dementia dx"
	@echo "                         (descendants of SNOMED 4182210 -- AD, vascular, Lewy body,"
	@echo "                         FTD, NOS) and windows their events to the 365 days starting"
	@echo "                         at that first dx. doc-unit=patient."
	@echo "                         LDA_BQ_ARGS='--K 40 --max-iter 200 --person-mod 1 ...'"
	@echo "                         RESUME: make lda-bq-fit-eval-cancer RUN_ID=<existing-id> LDA_BQ_ARGS='--max-iter <N more>'"
	@echo "  make inspect         - Live polling dashboard (run in a sibling terminal)"
	@echo "                         Override args via INSPECT_ARGS='--interval 3'"
	@echo "  make smoke-dataproc  - alternate: stage to GCS, gcloud dataproc jobs submit pyspark"
	@echo "  make zip             - (re)build spark_vi + charmpheno zips"
	@echo ""
	@echo "Vars (used only by smoke-dataproc; override on command line):"
	@echo "  PROJECT = $(PROJECT)"
	@echo "  CLUSTER = $(CLUSTER)"
	@echo "  REGION  = $(REGION)"
	@echo "  BUCKET  = $(BUCKET)        (no gs:// prefix)"
	@echo ""
	@echo "Checkpoint/resume vars (used by lda-bq-smoke / hdp-bq-smoke):"
	@echo "  SAVE_DIR      = $(SAVE_DIR)"
	@echo "                  authoritative post-fit artifact (loadable via OnlineXModel.load)"
	@echo "                  empty = no save"
	@echo "  SAVE_INTERVAL = $(SAVE_INTERVAL)"
	@echo "                  save every N iters during fit; -1 = end-of-fit only"
	@echo "                  (requires SAVE_DIR set)"
	@echo "  RESUME_FROM   = $(RESUME_FROM)"
	@echo "                  path to a previously-written save dir; empty = fresh start"
	@echo ""
	@echo "Example:"
	@echo "  make smoke-dataproc CLUSTER=my-cluster REGION=us-central1 BUCKET=my-bucket"
	@echo "  make lda-bq-smoke SAVE_DIR=/mnt/gcs/my-bucket/runs/lda-001/ SAVE_INTERVAL=10"
	@echo "  make lda-bq-smoke RESUME_FROM=/mnt/gcs/my-bucket/runs/lda-001/"

zip:
	$(MAKE) -C $(REPO_ROOT)/spark-vi zip
	$(MAKE) -C $(REPO_ROOT)/charmpheno zip

# Build the cluster dependency-overlay zip (pure-Python deps the image lacks).
# Built on the master so pip resolves Linux wheels for the cluster's own
# python3. build_spark_submit_cmd (run_experiment.py) detects
# $(CLUSTER_OVERLAY) and appends it to --py-files; it runs on the image's
# Python, so no interpreter override is needed. Only rebuilt when
# cluster-requirements.txt (or this Makefile) changes.
cluster-overlay: $(CLUSTER_OVERLAY)

# Compiled libs the AoU image already ships. pip resolves formulaic's full
# closure (pandas -> numpy, scipy, pyarrow ...), so we install it all to a
# build dir and then PRUNE these four -- leaving only the pure-Python leaves
# (formulaic, interface-meta, narwhals, wrapt, typing-extensions). This avoids
# shadowing the image's compiled libs: a stray numpy 2.x under the image's
# numpy-1.x-built pyarrow would crash with "AttributeError: _ARRAY_API not
# found". With the leaves on --py-files, the image's own numpy/pandas/scipy/
# pyarrow are used unchanged.
CLUSTER_IMAGE_LIBS := numpy pandas scipy pyarrow

# Rebuild when the requirements OR this Makefile's recipe changes.
$(CLUSTER_OVERLAY): $(CLUSTER_REQS) $(REPO_ROOT)/analysis/cloud/Makefile
	rm -rf $(CLUSTER_OVERLAY_DIR)
	mkdir -p $(CLUSTER_OVERLAY_DIR) $(dir $(CLUSTER_OVERLAY))
	python3 -m pip install --quiet --target $(CLUSTER_OVERLAY_DIR) -r $(CLUSTER_REQS)
	cd $(CLUSTER_OVERLAY_DIR) && for p in $(CLUSTER_IMAGE_LIBS); do \
		rm -rf "$$p" "$$p".libs "$$p"-*.dist-info "$$p".dist-info; \
	done
	rm -f $(CLUSTER_OVERLAY)
	cd $(CLUSTER_OVERLAY_DIR) && zip -qr $(CLUSTER_OVERLAY) . \
		-x "*.pyc" -x "*/__pycache__/*"
	@echo "Wrote $(CLUSTER_OVERLAY)"

# One-time per-cluster provisioning: re-discover and rewrite $(WORKSPACE_ENV),
# and build the cluster dependency overlay (pure-Python deps the image lacks).
# The overlay is a file-target keyed on cluster-requirements.txt, so it only
# rebuilds when that changes — `make setup` then `make exp ID=N` is enough on
# a fresh cluster.
setup: cluster-overlay
	$(SETUP_CMD)

# Auto-run setup the first time anything needs $(WORKSPACE_ENV) on a fresh
# cluster. Once the file exists, make won't re-run this — use `make setup`
# to regenerate.
$(WORKSPACE_ENV):
	$(SETUP_CMD)

smoke: zip
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --py-files $(SPARK_VI_ZIP) \
	    $(SCRIPT)

bq-smoke: $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    $(BQ_SCRIPT)

lda-bq-smoke: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) \
	    --save-dir "$(SAVE_DIR)" \
	    --save-interval $(SAVE_INTERVAL) \
	    --resume-from "$(RESUME_FROM)" \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    $(LDA_BQ_ARGS)

hdp-bq-smoke: zip $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(HDP_BQ_SCRIPT) \
	    --save-dir "$(SAVE_DIR)" \
	    --save-interval $(SAVE_INTERVAL) \
	    --resume-from "$(RESUME_FROM)" \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    $(HDP_BQ_ARGS)

eval-bq-coherence: zip $(WORKSPACE_ENV)
	@test -n "$(CHECKPOINT)" || { echo "ERROR: set CHECKPOINT=<dir>"; exit 1; }
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(EVAL_SCRIPT) \
	    --checkpoint "$(CHECKPOINT)" \
	    $(EVAL_ARGS)

build-dashboard-bundle: zip $(WORKSPACE_ENV)
	@test -n "$(CHECKPOINT)" || { echo "ERROR: set CHECKPOINT=<dir>"; exit 1; }
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(BUNDLE_SCRIPT) \
	    --checkpoint "$(CHECKPOINT)" \
	    $(BUNDLE_ARGS)

# Combined fit + eval. Auto-generates a fresh timestamped SAVE_DIR under
# $(REPO_ROOT)/data/runs/, runs the model-specific fit, then runs eval
# against that exact SAVE_DIR. Eval consumes the full BOW reproduced from
# the checkpoint's frozen vocab (no train/holdout split, no contract).
#
# Resume: pass RUN_ID=<existing-run-id> to continue a prior run with more
# iterations. If the dir exists, RESUME_FROM is auto-set and fit continues
# from the saved iteration count. --max-iter in *_BQ_ARGS is the number of
# ADDITIONAL iters to run, not a target total: resuming a 50-iter
# checkpoint with --max-iter 30 yields 80 total iters. The Robbins-Monro
# counter is preserved across the boundary so step sizes match a
# continuous run. Hyperparameters affecting the BOW (vocab-size, min-df,
# person-mod, source-table, doc-unit, doc-min-length) MUST match the
# original run — they're not yet auto-validated.
#
# data/runs/ is the project's repo-local artifact dir (gitignored). It's a
# Dataproc-master path, not a GCS-FUSE mount — artifacts don't survive a
# cluster teardown unless you rsync them off. For long-lived runs, override
# SAVE_DIR explicitly to a /mnt/gcs/<bucket>/... path.
#RUNS_DIR := $(REPO_ROOT)/data/runs
RUNS_DIR := /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs

# RUN_ID is the artifact directory name under $(RUNS_DIR). Empty (default)
# means auto-generate <model>_<timestamp>. Override to resume a prior run.
RUN_ID          ?=

# ─── Experiment-tracking targets (Increment 1) ──────────────────────────────
# See docs/superpowers/specs/2026-05-28-experiment-tracking-design.md.
#
# Author experiments locally in docs/experiments/NNNN-slug.md with YAML
# frontmatter; run them on the cluster via these targets.

# Pick lowest-NNNN with status: pending; fit + eval; write summary.md
next-exp: zip cluster-overlay $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --next --runs-dir $(RUNS_DIR) $(if $(NO_EVAL),--no-eval)

# Same pipeline pinned to specific id (auto-resumes if RUN_ID dir exists)
exp: zip cluster-overlay $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py --id $(ID) --runs-dir $(RUNS_DIR) $(if $(NO_EVAL),--no-eval)

# Print summary.md for an experiment (copy/paste back to chat)
summary:
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	@cat $(RUNS_DIR)/$(shell printf '%04d' $(ID))-*/summary.md

# Print just the most-recent ## section of summary.md (everything from the
# last H2 heading to EOF). Useful when older sessions are already pasted and
# you only want the latest fit / eval / build block.
summary-tail:
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	@awk '/^## /{buf=""} {buf=buf$$0"\n"} END{printf "%s",buf}' \
	    $(RUNS_DIR)/$(shell printf '%04d' $(ID))-*/summary.md

# Re-run eval only against an existing experiment's checkpoint; appends a
# timestamped ## Eval (NPMI) section to summary.md.
eval-exp: zip cluster-overlay $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py $(if $(ID),--id $(ID)) --eval-only --runs-dir $(RUNS_DIR)

# Build dashboard bundle for an experiment. With no ID, picks the most
# recent fit whose bundle is missing or older than the fit checkpoint.
# With ID=N, force-rebuilds the named experiment.
build-dashboard-exp: zip cluster-overlay $(WORKSPACE_ENV)
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py $(if $(ID),--id $(ID)) --build-only --runs-dir $(RUNS_DIR)

# Rebuild only the STM covariate cache for an experiment, without re-running
# the fit. Useful when iterating on covariate_formula: change the formula in
# the experiment frontmatter, then run this target to refresh the cache before
# the next `make exp EXP=N`.
#
# Usage:
#   make build-covariates EXP=NNNN
#   make build-covariates EXP=NNNN FORCE=1   # invalidate cache, force rebuild
build-covariates: zip cluster-overlay $(WORKSPACE_ENV)
ifndef EXP
	$(error EXP=NNNN required, e.g. make build-covariates EXP=0002)
endif
	. ./$(WORKSPACE_ENV) && \
	python $(REPO_ROOT)/scripts/run_experiment.py \
		--id $(EXP) --build-covariates-only --runs-dir $(RUNS_DIR) \
		$(if $(FORCE),--force-covariates)

lda-bq-fit-eval: zip $(WORKSPACE_ENV)
	@mkdir -p $(RUNS_DIR) && \
	if [ -n "$(RUN_ID)" ]; then \
	  RUN_ID_=$(RUN_ID); \
	else \
	  RUN_ID_=lda_$$(date +%Y%m%d_%H%M%S); \
	fi; \
	RUN_SAVE_DIR=$(RUNS_DIR)/$$RUN_ID_; \
	if [ -d "$$RUN_SAVE_DIR" ]; then \
	  echo "[fit-eval] RESUMING: RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=$$RUN_SAVE_DIR; \
	else \
	  echo "[fit-eval] FRESH:    RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=""; \
	fi; \
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) \
	    --save-dir "$$RUN_SAVE_DIR" \
	    --resume-from "$$RESUME_" \
	    --save-interval $(SAVE_INTERVAL) \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    --source-table condition_era \
	    --doc-unit patient_year \
	    --doc-min-length 20 \
	    --K 80 \
	    --subsampling-rate 0.2 \
	    --max-iter 20 \
	    --person-mod 10 \
	    --vocab-size 10000 \
	    --min-df 10 \
	    --tau0 64 \
	    --kappa 0.7 \
	    --print-topics-every 1 \
	    $(LDA_BQ_ARGS) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(EVAL_SCRIPT) \
	    --checkpoint "$$RUN_SAVE_DIR" \
	    $(EVAL_ARGS)

hdp-bq-fit-eval: zip $(WORKSPACE_ENV)
	@mkdir -p $(RUNS_DIR) && \
	if [ -n "$(RUN_ID)" ]; then \
	  RUN_ID_=$(RUN_ID); \
	else \
	  RUN_ID_=hdp_$$(date +%Y%m%d_%H%M%S); \
	fi; \
	RUN_SAVE_DIR=$(RUNS_DIR)/$$RUN_ID_; \
	if [ -d "$$RUN_SAVE_DIR" ]; then \
	  echo "[fit-eval] RESUMING: RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=$$RUN_SAVE_DIR; \
	else \
	  echo "[fit-eval] FRESH:    RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=""; \
	fi; \
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(HDP_BQ_SCRIPT) \
	    --save-dir "$$RUN_SAVE_DIR" \
	    --resume-from "$$RESUME_" \
	    --save-interval $(SAVE_INTERVAL) \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    --source-table condition_era \
	    --doc-unit patient_year \
	    --doc-min-length 20 \
	    --K 10 \
	    --T 80 \
	    --gamma 5 \
	    --subsampling-rate 0.2 \
	    --max-iter 20 \
	    --person-mod 10 \
	    --vocab-size 10000 \
	    --min-df 10 \
	    --tau0 64 \
	    --kappa 0.7 \
	    --print-topics-every 1 \
	    $(HDP_BQ_ARGS) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(EVAL_SCRIPT) \
	    --checkpoint "$$RUN_SAVE_DIR" \
	    --model-class hdp \
	    $(EVAL_ARGS)

#		--no-optimize-corpus-concentration \

# LDA fit + eval on the first-cancer-year cohort. Same scaffolding as
# lda-bq-fit-eval (auto-generates a timestamped RUN_ID, resumes if the
# dir exists, writes under $(RUNS_DIR)) but pins the cohort/doc-unit
# combination required for the post-dx-year semantics:
#   --cohort first_cancer_year   (descendants of SNOMED 443392 minus NMSC + in-situ)
#   --source-table condition_era (cohort window uses era_start dates)
#   --doc-unit patient           (one doc per patient's post-dx year;
#                                 NOT patient_year, which would split a
#                                 patient's year across calendar boundaries)
#   --doc-min-length 20          (per project convention)
# RUN_ID prefix is `cancer_lda_` so artifacts are easy to find vs the
# baseline LDA runs.
lda-bq-fit-eval-cancer: zip $(WORKSPACE_ENV)
	@mkdir -p $(RUNS_DIR) && \
	if [ -n "$(RUN_ID)" ]; then \
	  RUN_ID_=$(RUN_ID); \
	else \
	  RUN_ID_=cancer_lda_$$(date +%Y%m%d_%H%M%S); \
	fi; \
	RUN_SAVE_DIR=$(RUNS_DIR)/$$RUN_ID_; \
	if [ -d "$$RUN_SAVE_DIR" ]; then \
	  echo "[fit-eval] RESUMING: RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=$$RUN_SAVE_DIR; \
	else \
	  echo "[fit-eval] FRESH:    RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=""; \
	fi; \
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) \
	    --save-dir "$$RUN_SAVE_DIR" \
	    --resume-from "$$RESUME_" \
	    --save-interval $(SAVE_INTERVAL) \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    --cohort first_cancer_year \
	    --source-table condition_era \
	    --doc-unit patient \
	    --doc-min-length 20 \
	    --K 40 \
	    --subsampling-rate 0.2 \
	    --max-iter 150 \
	    --person-mod 1 \
	    --vocab-size 10000 \
	    --min-df 10 \
	    --tau0 64 \
	    --kappa 0.7 \
	    --print-topics-every 1 \
	    $(LDA_BQ_ARGS) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(EVAL_SCRIPT) \
	    --checkpoint "$$RUN_SAVE_DIR" \
	    $(EVAL_ARGS)


# LDA fit + eval on the first-dementia-year cohort. Mirrors the cancer
# target above (auto RUN_ID, resume support, K=40 default) but pins:
#   --cohort first_dementia_year   (descendants of SNOMED 4182210)
#   --source-table condition_era   (cohort window uses era_start dates)
#   --doc-unit patient             (one doc per patient's post-dx year)
#   --doc-min-length 20            (per project convention)
# RUN_ID prefix is `dementia_lda_` so artifacts are easy to identify.
#
# Dementia cohort note: all-cause (AD + vascular + DLB + FTD + NOS). On
# first run, sanity-check person count vs expectation (dementia is much
# rarer in incidence than cancer; expect thousands rather than tens of
# thousands of qualifying persons in AoU full-cohort).
lda-bq-fit-eval-dementia: zip $(WORKSPACE_ENV)
	@mkdir -p $(RUNS_DIR) && \
	if [ -n "$(RUN_ID)" ]; then \
	  RUN_ID_=$(RUN_ID); \
	else \
	  RUN_ID_=dementia_lda_$$(date +%Y%m%d_%H%M%S); \
	fi; \
	RUN_SAVE_DIR=$(RUNS_DIR)/$$RUN_ID_; \
	if [ -d "$$RUN_SAVE_DIR" ]; then \
	  echo "[fit-eval] RESUMING: RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=$$RUN_SAVE_DIR; \
	else \
	  echo "[fit-eval] FRESH:    RUN_ID=$$RUN_ID_  SAVE_DIR=$$RUN_SAVE_DIR"; \
	  RESUME_=""; \
	fi; \
	. ./$(WORKSPACE_ENV) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) \
	    --save-dir "$$RUN_SAVE_DIR" \
	    --resume-from "$$RESUME_" \
	    --save-interval $(SAVE_INTERVAL) \
	    --corpus-cache-uri "$(CORPUS_CACHE_URI)" \
	    --cohort first_dementia_year \
	    --source-table condition_era \
	    --doc-unit patient \
	    --doc-min-length 20 \
	    --K 40 \
	    --subsampling-rate 0.2 \
	    --max-iter 150 \
	    --person-mod 1 \
	    --vocab-size 10000 \
	    --min-df 10 \
	    --tau0 64 \
	    --kappa 0.7 \
	    --print-topics-every 1 \
	    $(LDA_BQ_ARGS) && \
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --driver-memory 4g \
	    --conf spark.executor.cores=4 \
	    --conf spark.executor.memory=6g \
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(EVAL_SCRIPT) \
	    --checkpoint "$$RUN_SAVE_DIR" \
	    $(EVAL_ARGS)


inspect:
	python $(INSPECT_SCRIPT) $(INSPECT_ARGS)

smoke-dataproc: zip
	@test -n "$(CLUSTER)" || { echo "ERROR: set CLUSTER=<name>"; exit 1; }
	@test -n "$(REGION)"  || { echo "ERROR: set REGION=<region>"; exit 1; }
	@test -n "$(BUCKET)"  || { echo "ERROR: set BUCKET=<bucket-name>"; exit 1; }
	gsutil cp $(SPARK_VI_ZIP) gs://$(BUCKET)/smoke/spark_vi.zip
	gsutil cp $(SCRIPT)       gs://$(BUCKET)/smoke/$(SCRIPT)
	gcloud dataproc jobs submit pyspark \
	    gs://$(BUCKET)/smoke/$(SCRIPT) \
	    --project=$(PROJECT) \
	    --cluster=$(CLUSTER) \
	    --region=$(REGION) \
	    --py-files=gs://$(BUCKET)/smoke/spark_vi.zip
