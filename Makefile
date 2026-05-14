.PHONY: help install install-dev install-labeling data data-clean test test-all test-cluster \
        build zip clean precommit-install lint ingest-bundle label-phenotypes

default: help

help:
	@echo "Top-level orchestrator. Common targets:"
	@echo "  install          - Install root venv (scripts + dev tools)"
	@echo "  install-dev      - Install both package venvs (spark-vi, charmpheno) editable"
	@echo "  install-labeling - Install optional deps for the labeling script (anthropic SDK)"
	@echo "  data             - Fetch LDA beta and simulate synthetic OMOP data"
	@echo "  data-clean       - Delete the data/ cache and simulated outputs"
	@echo "  test             - Run unit tests in both packages (default loop, <10s)"
	@echo "  test-all         - Run unit + @slow integration tests"
	@echo "  test-cluster     - Run @cluster tests (manual, cluster-only)"
	@echo "  build            - Build wheels + sdists for both packages"
	@echo "  zip              - Build flat zips for both packages"
	@echo "  clean            - Remove dist/, build/, egg-info, pytest caches"
	@echo "  precommit-install - Install pre-commit hooks AND the nbstripout"
	@echo "                      git clean filter for .ipynb (one-time per clone)"
	@echo "  lint             - Run pre-commit against all tracked files"
	@echo ""
	@echo "Dashboard bundle pipeline (post-fit):"
	@echo "  ingest-bundle    - Unpack a dashboard bundle zip into dashboard/public/data/"
	@echo "                     ZIP=<path>  (e.g. /tmp/dashboard_bundle.zip)"
	@echo "  label-phenotypes - LLM-label the phenotypes in dashboard/public/data/."
	@echo "                     Requires CHARMPHENO_LABEL_KEY env var (or LABEL_KEY_FILE)"
	@echo "                     and \`make install-labeling\` to have run once."
	@echo "                     Override args via LABEL_ARGS='--model claude-haiku-4-5 --top-n 20'"

install:
	poetry install

install-dev: install
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi install; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno install; fi

# Optional: install the anthropic SDK for the post-fit labeling script.
# Kept in a separate poetry group so the cluster install path doesn't pull it.
install-labeling:
	poetry install --with labeling

# Unpack a bundle zip downloaded from the cloud into the dashboard's data dir.
# After this you typically run `make label-phenotypes`.
BUNDLE_OUT ?= dashboard/public/data
ingest-bundle:
	@test -n "$(ZIP)" || { echo "ERROR: set ZIP=<path/to/dashboard_bundle.zip>"; exit 1; }
	@test -f "$(ZIP)" || { echo "ERROR: not found: $(ZIP)"; exit 1; }
	mkdir -p $(BUNDLE_OUT)
	unzip -o "$(ZIP)" -d $(BUNDLE_OUT)
	@echo "[ingest] unpacked $(ZIP) -> $(BUNDLE_OUT)"

# Label phenotypes in an unpacked bundle. Read the API key from
# CHARMPHENO_LABEL_KEY (NOT ANTHROPIC_API_KEY — see scripts/label_phenotypes.py
# docstring for the rationale), or pass LABEL_KEY_FILE=<path> to read from disk.
# Override LABEL_ARGS to pass extra args (--top-n, --force, --limit, etc.).
LABEL_KEY_FILE ?=
LABEL_ARGS ?=
label-phenotypes:
	@if [ -z "$(LABEL_KEY_FILE)" ] && [ -z "$$CHARMPHENO_LABEL_KEY" ]; then \
		echo "ERROR: set CHARMPHENO_LABEL_KEY=sk-ant-... or LABEL_KEY_FILE=<path>"; \
		exit 1; \
	fi
	poetry run python scripts/label_phenotypes.py \
		--bundle-dir $(BUNDLE_OUT) \
		$(if $(LABEL_KEY_FILE),--api-key-file $(LABEL_KEY_FILE),) \
		$(LABEL_ARGS)

data:
	poetry run python scripts/fetch_lda_beta.py
	poetry run python scripts/simulate_lda_omop.py

data-clean:
	rm -rf data/cache data/simulated

test:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test; fi
	@if [ -d tests/scripts ]; then poetry run pytest tests/scripts -v; fi

# See spark-vi/Makefile and charmpheno/Makefile for the JAVA_HOME selection
# rationale (PySpark 3.5+ requires Java >=17). Same logic inlined here for
# the integration-suite branch; only export JAVA_HOME if a candidate exists.
JAVA_HOME_CANDIDATES := \
    /opt/homebrew/opt/openjdk@17 \
    /opt/homebrew/opt/openjdk \
    /usr/lib/jvm/temurin-17-jdk-amd64 \
    /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME := $(shell for d in $(JAVA_HOME_CANDIDATES); do if [ -d "$$d" ]; then echo "$$d"; break; fi; done)
JAVA_PREFIX := $(if $(JAVA_HOME),JAVA_HOME=$(JAVA_HOME) ,)

test-all:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test-all; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test-all; fi
	@if [ -d tests/integration ]; then \
		cd charmpheno && $(JAVA_PREFIX)poetry run pytest ../tests/integration -v -m "not cluster"; \
	fi

test-cluster:
	@if [ -d tests/integration ]; then poetry run pytest tests/integration -v -m cluster; fi

build:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi build; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno build; fi

zip:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi zip; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno zip; fi

clean:
	rm -rf .pytest_cache
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi clean; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno clean; fi

precommit-install:
	poetry run pre-commit install
	# Wire nbstripout as a git clean filter so .ipynb outputs are stripped
	# at `git add` time, not after the fact at commit time. Without this,
	# the pre-commit nbstripout hook still works but rejects the first
	# commit, requiring a second add+commit. The .gitattributes file
	# (committed) declares the filter; this step adds the local
	# .git/config entry that maps it to a runnable command.
	poetry run nbstripout --install

lint:
	poetry run pre-commit run --all-files
