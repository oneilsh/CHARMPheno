# charmpheno

Clinical phenotyping on PySpark. Built on [`spark-vi`](../spark-vi).

See the [research design](../docs/architecture/TOPIC_STATE_MODELING.md) for
the clinical context and [CHARMPheno](../README.md) for the project overall.

Besides the OMOP / Mondo corpus code (`omop`, `mondo`) and the exporters (`export`), it
holds the case-finding readout (`readout`: per-node heads over a fitted model's θ, Spark
and local backends, the `DagHeads` MLlib stages in `readout.mllib`) and the exact
ranking metrics (`evaluate.metrics`), moved here from spark-vi by ADR 0054.

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
