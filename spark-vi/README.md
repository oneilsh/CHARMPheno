# spark-vi

Distributed variational inference for PySpark.

`spark-vi` is a domain-agnostic PySpark framework. Model authors subclass
`VIModel` and implement three methods; the framework handles Spark
orchestration, the training loop, convergence monitoring, and model export.

See the [framework design](../docs/architecture/SPARK_VI_FRAMEWORK.md) for the
architectural context and [CHARMPheno](../README.md) for the project this
framework was extracted from.

## Install

```bash
poetry install
```

## Test

```bash
make test          # unit tests only (fast)
make test-all      # unit + slow integration tests
```

## Build artifacts

```bash
make build         # dist/*.whl + dist/*.tar.gz
make zip           # dist/spark_vi.zip (flat, pure-Python; for --py-files)
```

Requires Java 17 for local Spark. The Makefile autodetects Homebrew
(`/opt/homebrew/opt/openjdk@17`) and common Linux paths; override with
`JAVA_HOME=... make test` if needed.

--------
