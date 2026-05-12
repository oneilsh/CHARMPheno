.PHONY: help zip smoke bq-smoke lda-bq-smoke hdp-bq-smoke eval-bq-coherence lda-bq-fit-eval hdp-bq-fit-eval inspect smoke-dataproc setup

REPO_ROOT       := $(shell git rev-parse --show-toplevel)
SPARK_VI_ZIP    := $(REPO_ROOT)/spark-vi/dist/spark_vi.zip
CHARMPHENO_ZIP  := $(REPO_ROOT)/charmpheno/dist/charmpheno.zip
SCRIPT          := smoke_lda_cloud.py
BQ_SCRIPT       := bq_smoke_cloud.py
LDA_BQ_SCRIPT   := lda_bigquery_cloud.py
HDP_BQ_SCRIPT   := hdp_bigquery_cloud.py
EVAL_SCRIPT     := eval_coherence_cloud.py
INSPECT_SCRIPT  := inspect_app.py
SETUP_SCRIPT    := setup_workspace.py
WORKSPACE_ENV   := .workspace_env

# Args forwarded to lda-bq-smoke / hdp-bq-smoke / eval-bq-coherence / inspect
# (override on command line).
LDA_BQ_ARGS     ?=
HDP_BQ_ARGS     ?=
EVAL_ARGS       ?=
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
	@echo "  make setup           - Discover workspace env (project, CDR, buckets) via the"
	@echo "                         'wb' CLI; write to $(WORKSPACE_ENV). Pass CDR=<proj>.<ds>"
	@echo "                         when the CDR is an external read-only project not"
	@echo "                         surfaced by 'wb resource list'. Auto-runs if missing."
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
	@echo "  make eval-bq-coherence - Held-out NPMI coherence on an augmented VIResult"
	@echo "                         CHECKPOINT=/mnt/gcs/\$$BUCKET/runs/<id>  (required)"
	@echo "                         Override args via EVAL_ARGS='--top-n 10'"
	@echo "                         HDP: EVAL_ARGS='--model-class hdp --hdp-k 50'"
	@echo "  make lda-bq-fit-eval - End-to-end fit + eval; auto-generates timestamped"
	@echo "                         SAVE_DIR under \$$(REPO_ROOT)/data/runs/<id>. Pass"
	@echo "                         --holdout-fraction > 0 in LDA_BQ_ARGS for the eval"
	@echo "                         to find a held-out partition."
	@echo "                         LDA_BQ_ARGS='--K 50 --max-iter 200 --holdout-fraction 0.1 --holdout-seed 42 ...'"
	@echo "                         EVAL_ARGS='--top-n 20'  SAVE_INTERVAL=10"
	@echo "                         To RESUME a prior run with more iterations:"
	@echo "                           make lda-bq-fit-eval RUN_ID=<existing-id> LDA_BQ_ARGS='--max-iter <new total>'"
	@echo "  make hdp-bq-fit-eval - End-to-end HDP fit + eval; same shape as lda-bq-fit-eval."
	@echo "                         HDP_BQ_ARGS='--T 150 --K 15 --holdout-fraction 0.1 --holdout-seed 42 ...'"
	@echo "                         EVAL_ARGS='--model-class hdp --hdp-k 50'"
	@echo "                         RESUME: make hdp-bq-fit-eval RUN_ID=<existing-id> HDP_BQ_ARGS='--max-iter <new total>'"
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

# Always re-discover and rewrite $(WORKSPACE_ENV).
setup:
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

# Combined fit + eval. Auto-generates a fresh timestamped SAVE_DIR under
# $(REPO_ROOT)/data/runs/, runs the model-specific fit, then runs eval
# against that exact SAVE_DIR. LDA_BQ_ARGS / HDP_BQ_ARGS must include
# --holdout-fraction > 0 for eval to find a held-out partition; otherwise
# the eval driver aborts with a missing-split error.
#
# Resume: pass RUN_ID=<existing-run-id> to continue a prior run with more
# iterations. If the dir exists, RESUME_FROM is auto-set and fit continues
# from the saved iteration count. Bump --max-iter in *_BQ_ARGS to the new
# total target. Hyperparameters affecting the BOW (vocab-size, min-df,
# person-mod, source-table, doc-unit, doc-min-length) MUST match the
# original run — they're not yet auto-validated.
#
# data/runs/ is the project's repo-local artifact dir (gitignored). It's a
# Dataproc-master path, not a GCS-FUSE mount — artifacts don't survive a
# cluster teardown unless you rsync them off. For long-lived runs, override
# SAVE_DIR explicitly to a /mnt/gcs/<bucket>/... path.
RUNS_DIR := $(REPO_ROOT)/data/runs

# RUN_ID is the artifact directory name under $(RUNS_DIR). Empty (default)
# means auto-generate <model>_<timestamp>. Override to resume a prior run.
RUN_ID          ?=

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
	$(MAKE) lda-bq-smoke SAVE_DIR=$$RUN_SAVE_DIR \
	  RESUME_FROM="$$RESUME_" \
	  SAVE_INTERVAL=$(SAVE_INTERVAL) \
	  LDA_BQ_ARGS="$(HDP_BQ_ARGS) \
		--source-table condition_era \
        --doc-unit patient_year \
		--doc-min-length 30 \
	    --K 25 \
		--subsampling-rate 0.1 \
		--max-iter 20 \
		--person-mod 10 \
		--vocab-size 10000 \
		--min-df 10 \
		--holdout-fraction 0.2 \
		--holdout-seed 42 \
		--tau0 64 \
		--kappa 0.7 \
		--print-topics-every 1" && \
	$(MAKE) eval-bq-coherence CHECKPOINT=$$RUN_SAVE_DIR EVAL_ARGS="$(EVAL_ARGS)"

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
	$(MAKE) hdp-bq-smoke \
	  SAVE_DIR=$$RUN_SAVE_DIR \
	  RESUME_FROM="$$RESUME_" \
	  SAVE_INTERVAL=$(SAVE_INTERVAL) \
	  HDP_BQ_ARGS="$(HDP_BQ_ARGS) \
		--source-table condition_era \
        --doc-unit patient_year \
		--doc-min-length 30 \
	    --K 10 \
		--T 80 \
		--gamma 50 \
		--subsampling-rate 0.1 \
		--max-iter 20 \
		--person-mod 10 \
		--vocab-size 10000 \
		--min-df 10 \
		--holdout-fraction 0.2 \
		--holdout-seed 42 \
		--tau0 64 \
		--kappa 0.7 \
		--print-topics-every 1" && \
	$(MAKE) eval-bq-coherence \
	  CHECKPOINT=$$RUN_SAVE_DIR EVAL_ARGS="--model-class hdp $(EVAL_ARGS)"

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
