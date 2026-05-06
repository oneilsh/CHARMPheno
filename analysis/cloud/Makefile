.PHONY: help zip smoke bq-smoke lda-bq-smoke inspect smoke-dataproc setup

REPO_ROOT       := $(shell git rev-parse --show-toplevel)
SPARK_VI_ZIP    := $(REPO_ROOT)/spark-vi/dist/spark_vi.zip
CHARMPHENO_ZIP  := $(REPO_ROOT)/charmpheno/dist/charmpheno.zip
SCRIPT          := smoke_lda_cloud.py
BQ_SCRIPT       := bq_smoke_cloud.py
LDA_BQ_SCRIPT   := lda_bigquery_cloud.py
INSPECT_SCRIPT  := inspect_app.py
SETUP_SCRIPT    := setup_workspace.py
WORKSPACE_ENV   := .workspace_env

# Args forwarded to lda-bq-smoke / inspect (override on command line).
LDA_BQ_ARGS     ?=
INSPECT_ARGS    ?=

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
	@echo "Example:"
	@echo "  make smoke-dataproc CLUSTER=my-cluster REGION=us-central1 BUCKET=my-bucket"

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
	    --conf spark.executor.memoryOverhead=2g \
	    --conf spark.python.worker.memory=2g \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) $(LDA_BQ_ARGS)

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
