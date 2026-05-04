.PHONY: help zip smoke smoke-dataproc

REPO_ROOT    := $(shell git rev-parse --show-toplevel)
SPARK_VI_ZIP := $(REPO_ROOT)/spark-vi/dist/spark_vi.zip
SCRIPT       := smoke_lda_cloud.py

# gcloud-submit fallback only — `smoke` doesn't need any of these.
PROJECT ?= $(GOOGLE_CLOUD_PROJECT)
CLUSTER ?=
REGION  ?=
BUCKET  ?=

default: help

help:
	@echo "Cluster smoke targets (run from this dir on the Dataproc master):"
	@echo ""
	@echo "  make smoke           - spark-submit via YARN (client mode), reads local paths"
	@echo "  make smoke-dataproc  - alternate: stage to GCS, gcloud dataproc jobs submit pyspark"
	@echo "  make zip             - (re)build $(SPARK_VI_ZIP)"
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

smoke: zip
	spark-submit \
	    --master yarn \
	    --deploy-mode client \
	    --py-files $(SPARK_VI_ZIP) \
	    $(SCRIPT)

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
