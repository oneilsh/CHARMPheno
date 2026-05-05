.PHONY: help install dev clean build zip test test-all test-cluster

default: help

JAVA_HOME_MACOS := /opt/homebrew/opt/openjdk@17
JAVA_HOME_LINUX := /usr/lib/jvm/java-17-openjdk-amd64
JAVA_HOME ?= $(shell if [ -d "$(JAVA_HOME_MACOS)" ]; then echo "$(JAVA_HOME_MACOS)"; elif [ -d "$(JAVA_HOME_LINUX)" ]; then echo "$(JAVA_HOME_LINUX)"; fi)

help:
	@echo "charmpheno common targets:"
	@echo "  install      - poetry install (runtime + dev deps) + editable install of ../spark-vi"
	@echo "  dev          - alias for install"
	@echo "  test         - unit tests only"
	@echo "  test-all     - unit + @slow tests"
	@echo "  test-cluster - @cluster tests only (manual)"
	@echo "  build        - wheel + sdist (auto-installs `build` if missing)"
	@echo "  zip          - flat dist/charmpheno.zip"
	@echo "  clean        - remove dist/, build/, caches"

install:
	poetry install --with dev
	poetry run pip install -e ../spark-vi

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
	zip -r dist/charmpheno.zip charmpheno -x "*.pyc" -x "*/__pycache__/*" -x "*.egg-info/*"
	@echo "Wrote dist/charmpheno.zip"
