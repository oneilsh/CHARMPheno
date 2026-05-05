.PHONY: help install dev clean build zip check test test-all test-cluster

default: help

JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX_CANDIDATES := /usr/lib/jvm/temurin-17-jdk-amd64 /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME := $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; else for d in $(JAVA_HOME_LINUX_CANDIDATES); do if [ -d "$$d" ]; then echo "$$d"; break; fi; done; fi)

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
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v

test-all:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m "not cluster"

test-cluster:
	JAVA_HOME=$(JAVA_HOME) poetry run pytest tests/ -v -m cluster

build:
	@poetry run python -c "import build" 2>/dev/null || poetry run pip install build
	poetry run python -m build

zip: clean build
	mkdir -p dist
	zip -r dist/spark_vi.zip spark_vi -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/spark_vi.zip"
