.PHONY: help zip smoke bq-smoke lda-bq-smoke smoke-dataproc

REPO_ROOT      := $(shell git rev-parse --show-toplevel)
SPARK_VI_ZIP   := $(REPO_ROOT)/spark-vi/dist/spark_vi.zip
CHARMPHENO_ZIP := $(REPO_ROOT)/charmpheno/dist/charmpheno.zip
SCRIPT         := smoke_lda_cloud.py
BQ_SCRIPT      := bq_smoke_cloud.py
LDA_BQ_SCRIPT  := lda_bigquery_cloud.py

# Args forwarded to lda-bq-smoke (override on command line).
LDA_BQ_ARGS    ?=

# gcloud-submit fallback only — `smoke` doesn't need any of these.
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
CLUSTER ?=
REGION  ?=
BUCKET  ?=

default: help

help:
	@echo "Cluster smoke targets (run from this dir on the Dataproc master):"
	@echo ""
	@echo "  make smoke           - LDA shim smoke: spark-submit via YARN (client mode)"
	@echo "  make bq-smoke        - BigQuery read smoke (no spark_vi imports)"
	@echo "  make lda-bq-smoke    - End-to-end: BQ -> CountVectorizer -> LDA shim"
	@echo "                         Override args via LDA_BQ_ARGS='--K 10 --person-mod 5000'"
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

smoke: zip
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --py-files $(SPARK_VI_ZIP) \
	    $(SCRIPT)

bq-smoke:
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    $(BQ_SCRIPT)

lda-bq-smoke: zip
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --py-files $(SPARK_VI_ZIP),$(CHARMPHENO_ZIP) \
	    $(LDA_BQ_SCRIPT) $(LDA_BQ_ARGS)

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
