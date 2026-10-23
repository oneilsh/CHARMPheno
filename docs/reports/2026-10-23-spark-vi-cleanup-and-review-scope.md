# spark-vi: cleanup scope and walkthrough timing (decisions, 2026-10-23)

**Status:** decided by the maintainer; work not started.

## Review state

The last maintainer walkthrough of spark-vi is **2026-08-12** (PC branch, 7 lessons).
The 2026-08-20 / 08-22 / 08-27 review-log entries are agent-recorded build notes, not
sign-offs, and are annotated as such. Since 2026-08-12, spark-vi's package code has had
53 commits and about 5,500 added lines (gross): multi-domain gated PC, the
supervised-head evolution, batched L-BFGS heads, runner broadcast/reduce changes, the
spectral init's scalable and guided forms, profile-η / count transform / alpha policy,
and the leaves-first gate schedule with its seeds. None of it has been walked through.

## Decisions

- **Cut the supervised-head machinery** (the `weight_y > 0` path: λ corrections, trust
  caps, the co-fit head and its provider seam, head-support variants used only by the
  co-fit). It never worked well, and the readout fits its heads post hoc. Removal gets
  an ADR; the post-hoc readout's batched L-BFGS is kept.
- **Keep the spectral init as an option**, even if the leaf seed (exp 0162) supersedes
  it as the default under the gate schedule. It has proved useful.
- **Keep the profile (HPO) seeds** as a comparator for the other seeding strategies.
- **No walkthrough yet.** It waits until the methods to keep are settled AND the user
  API is decided, including the split between the core (`spark_vi.models` /
  `spark_vi.core`) and the MLlib shim (`spark_vi.mllib`).

## Order of work

1. Settle the init question (exp 0162: leaf seed against spectral).
2. Cleanup: the supervised-head removal (ADR), plus anything else judged dead.
3. The user API: core against MLlib shim, and the two-engine design (Spark `VIRunner` and
   a local runner) with a home for the generic readout and evaluation pieces (heads over
   a label DAG, the closure product, gated NPMI, the identity census, the framings'
   generic cores); ADR + spec first.
4. The walkthrough of what remains, recorded in `docs/REVIEW_LOG.md` as sign-off.

## Where the code lives today, and spark-vi's domain-generality (audit, 2026-10-23)

| piece | location | lines (approx.) |
|---|---|---|
| inits: spectral (dense, scalable, guided), gate seeds (subtree, balanced, leaf, profile), heights | spark-vi `models/topic/{gated_init,spectral_init,spectral_init_scalable,dag_placement}.py` | ~3,200 |
| model-side heads (the co-fit supervised heads, to be cut) | spark-vi `models/topic/pc.py` | part of 1,746 |
| batched L-BFGS heads | spark-vi `models/topic/batched_lr.py` | 834 |
| distributed readout driver, readout DAG, closure-product stacking, depth/within-parent/de-novo reads | `analysis/cloud/{gated_pc_readout,distributed_readout,readout_dag}.py` | ~6,000 |
| framings 1–2 (differential, VOI, acquisition) | `analysis/cloud/{differential,voi_tables,voi_acquisition}.py` + `run_differential` | ~1,400+ |
| framing 3 / new-diagnosis evaluation | `analysis/cloud/{prevalence_check,compare_incident_runs}.py` + readout | ~240+ |
| gated NPMI, own-code census, digest | `analysis/cloud/{topic_npmi,inspect_topics}.py` | ~3,300 |
| charmpheno | OMOP assembly, BigQuery, cohorts, doc spec, dashboard export, LDA-era evaluation | 7,164; none of the readout / framings |

- **Hard rule held:** spark-vi imports nothing from charmpheno, analysis, BigQuery or
  OMOP code.
- **Soft leak, in behaviour:** a "domain 0 = conditions" convention is built into
  spark-vi features: the profile-η boost, `gateSeedProfile`, `gateSeedOwnCodes`, and
  their docs say "condition domain". Mechanically each is a fixed primary-domain index;
  for domain-generality they should take the domain as a parameter and say "primary
  domain".
- **Soft leak, in Params:** provenance Params for the HPO profile-η prior name the
  clinical tooling (`hpoa_stage2_probe`) in the shim's help text.
- **Soft leak, in prose:** comments and docstrings cite whole-Mondo scale, experiment
  numbers, HPO and patients (provenance of a design choice). Harmless to behaviour, but a
  standalone package would rephrase them generically.
- **The bigger structural point:** the readout, framings and evaluation (~11.6k lines)
  live in `analysis/cloud`, a set of driver scripts, not in either package, and they mix
  generic machinery with clinical rules (index dates, first-attestation sidecar, egress
  floor) in the same files. charmpheno's role as described in AGENTS.md ("downstream
  export, recovery-vs-ground-truth metrics") no longer matches where the active
  clinical logic is. The package-split ADR should place each of these pieces.
