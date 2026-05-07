.PHONY: help install dev clean build zip check test test-all test-cluster

default: help

# PySpark 3.5+ passes `-Djava.security.manager=allow` unconditionally, which
# is Java 12+ syntax — Java 11 trips on it with "Could not create
# SecurityManager". Pick the first installed Java >=17 candidate and only
# export JAVA_HOME if we found one; otherwise leave it unset so the PATH
# default (which on most machines is something modern) takes over rather
# than running with JAVA_HOME="" (which can pin a stale Java).
JAVA_HOME_CANDIDATES := \
    /opt/homebrew/opt/openjdk@17 \
    /opt/homebrew/opt/openjdk \
    /usr/lib/jvm/temurin-17-jdk-amd64 \
    /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME := $(shell for d in $(JAVA_HOME_CANDIDATES); do if [ -d "$$d" ]; then echo "$$d"; break; fi; done)
JAVA_PREFIX := $(if $(JAVA_HOME),JAVA_HOME=$(JAVA_HOME) ,)

help:
	@echo "spark-vi common targets:"
	@echo "  install      - poetry install (runtime + dev deps)"
	@echo "  dev          - alias for install"
	@echo "  test         - unit tests only (fast)"
	@echo "  test-all     - unit + @slow tests"
	@echo "  test-cluster - only @cluster tests (manual)"
	@echo "  build        - wheel + sdist into dist/ (auto-installs `build` if missing)"
	@echo "  zip          - flat dist/spark_vi.zip for --py-files"
	@echo "  clean        - remove dist/, build/, caches"

install:
	poetry install --with dev

dev: install

clean:
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache

test:
	$(JAVA_PREFIX)poetry run pytest tests/ -v

test-all:
	$(JAVA_PREFIX)poetry run pytest tests/ -v -m "not cluster"

test-cluster:
	$(JAVA_PREFIX)poetry run pytest tests/ -v -m cluster

build:
	@poetry run python -c "import build" 2>/dev/null || poetry run pip install build
	poetry run python -m build

zip: clean build
	mkdir -p dist
	zip -r dist/spark_vi.zip spark_vi -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/spark_vi.zip"
