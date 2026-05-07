"""charmpheno: clinical specialization on top of spark-vi.

Public surface:

    from charmpheno.omop import load_omop_parquet, validate

For topic modeling, use the spark_vi MLlib shims directly
(`spark_vi.mllib.OnlineHDPEstimator`, `spark_vi.mllib.VanillaLDAEstimator`)
or the underlying models with `VIRunner` — there is no clinical wrapper
class; clinical concerns live in the analysis driver scripts and in
`charmpheno.evaluate` / `charmpheno.omop`.
"""
__version__ = "0.1.0"
