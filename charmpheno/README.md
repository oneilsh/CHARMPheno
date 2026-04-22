# charmpheno

Clinical phenotyping on PySpark. Built on [`spark-vi`](../spark-vi).

See the [research design](../docs/architecture/TOPIC_STATE_MODELING.md) for
the clinical context and [CHARMPheno](../README.md) for the project overall.

## Install (dev)

```bash
poetry install
poetry run pip install -e ../spark-vi
```

## Test

```bash
make test          # unit tests only (fast)
make test-all      # unit + @slow integration tests
```

## Build

```bash
make build         # dist/*.whl + dist/*.tar.gz
make zip           # dist/charmpheno.zip (flat, pure-Python)
```

Requires Java 17 for local Spark (same auto-detection as `spark-vi`).
