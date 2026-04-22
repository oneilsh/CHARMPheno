.PHONY: help install dev clean build zip check test test-all test-cluster

default: help

JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX := /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME ?= $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; elif [ -d "$(JAVA_HOME_LINUX)" ]; then echo "$(JAVA_HOME_LINUX)"; fi)

help:
	@echo "spark-vi common targets:"
	@echo "  install      - poetry install"
	@echo "  dev          - install + dev build tooling"
	@echo "  test         - unit tests only (fast)"
	@echo "  test-all     - unit + @slow tests"
	@echo "  test-cluster - only @cluster tests (manual)"
	@echo "  build        - wheel + sdist into dist/"
	@echo "  zip          - flat dist/spark_vi.zip for --py-files"
	@echo "  clean        - remove dist/, build/, caches"

install:
	poetry install

dev:
	poetry install
	poetry run pip install build

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
	poetry run python -m build

zip: clean build
	mkdir -p dist
	zip -r dist/spark_vi.zip spark_vi -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/spark_vi.zip"
