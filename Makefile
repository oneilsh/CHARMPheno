.PHONY: help install install-dev data data-clean test test-all test-cluster \
        build zip clean precommit-install lint

default: help

help:
	@echo "Top-level orchestrator. Common targets:"
	@echo "  install          - Install root venv (scripts + dev tools)"
	@echo "  install-dev      - Install both package venvs (spark-vi, charmpheno) editable"
	@echo "  data             - Fetch LDA beta and simulate synthetic OMOP data"
	@echo "  data-clean       - Delete the data/ cache and simulated outputs"
	@echo "  test             - Run unit tests in both packages (default loop, <10s)"
	@echo "  test-all         - Run unit + @slow integration tests"
	@echo "  test-cluster     - Run @cluster tests (manual, cluster-only)"
	@echo "  build            - Build wheels + sdists for both packages"
	@echo "  zip              - Build flat zips for both packages"
	@echo "  clean            - Remove dist/, build/, egg-info, pytest caches"
	@echo "  precommit-install - Install the pre-commit hooks into .git/hooks/"
	@echo "  lint             - Run pre-commit against all tracked files"

install:
	poetry install

install-dev: install
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi install; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno install; fi

data:
	poetry run python scripts/fetch_lda_beta.py
	poetry run python scripts/simulate_lda_omop.py

data-clean:
	rm -rf data/cache data/simulated

test:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test; fi
	@if [ -d tests/scripts ]; then poetry run pytest tests/scripts -v; fi

test-all:
	@if [ -d spark-vi ]; then $(MAKE) -C spark-vi test-all; fi
	@if [ -d charmpheno ]; then $(MAKE) -C charmpheno test-all; fi
	@if [ -d tests/integration ]; then poetry run pytest tests/integration -v -m "not cluster"; fi

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

lint:
	poetry run pre-commit run --all-files
