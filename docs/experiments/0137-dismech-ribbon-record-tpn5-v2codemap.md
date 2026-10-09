---
id: 137
slug: dismech-ribbon-record-tpn5-v2codemap
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE RIBBON RECORD (spec 2026-10-07 §Experiments), on the REPAIRED code map.
# Changes vs 0135: tpn 3 -> 5 (0136: the ceiling binds); native-mondo-v2 (insight 0097:
# multi-map ancestor targets dropped — O90.3 no longer hands every pregnancy code to
# peripartum cardiomyopathy); readout by mass on the excess over the prior floor; the
# Mondo readout DAG (WP-B) and own-code census (WP-C') on by default.
#
# AUDIT FIRST (minutes into the run, before the fit is worth anything): the powering
# line's "multi-map guard: ... dropped: <named examples>" (Finding related to pregnancy
# should lead it) and the "attestation audit" line (codes shared across label nodes,
# exact-shared vs climb-tie, and the most-attested nodes by code count); the full
# per-node list is in <run>/code_map.tsv (0136's audit snippet).
# NOT in 0137, deliberately: a source-exact rung (condition_source_concept_id against
# Mondo's own source codes, main's usage-dashboard rung 1). It changes attestation for
# every node, so it is its own comparison after the record (0138).
# Peripartum CM should drop from 106 codes to a handful; tularemia (93), thrombophilia
# (162) and osteochondrosis (211) need a look. Kill and fix if another node's own codes
# are generic.
# READS: within-cohort and stacked de-novo macro (vs 0135; vs 0134 on shared nodes by
# Mondo id), root head, own-code census (n_profiles at tpn_max 5), DCM / peripartum /
# preeclampsia blocks with words.
# COST: K = 1200 + 5 x ~390 ~ 3,150 (0135: 2,373); bundle MISS (v2 key), sidecar MISS
# (code-map identity moved). Readout after the fit, as 0135 launch 3.
dag_source: mondo_native
mondo_branch: ""
label_set: analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
tpn: 5                      # tpn_max from 0136 (ceiling binding for ~1/3 of decidable nodes)
max_iter: 50
diag_only: true
init: spectral
spectral_method: scalable
spectral_d: 768             # 0134's: only 8 bg anchors + 3 per node are anchored
anchor_scope: closure       # on a flat forest, closure(node) = {node, root}: own block
spectral_topo_order: forward
count_transform: none
preindex_closure: false
readout_mode: distributed
readout_theta_mass: 0.99    # excess-over-prior-floor coverage (0135 launch 3 fix)
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 1200                  # 0134's (insight 0095: width is the root head's lever)
spectral_bg_anchors: 8      # 0134's: anchor eight bg rows as deflation seeds, random-init the rest
optimize_doc_concentration: false   # insight 0091; see 0134's LANDMINE note
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
spark_conf:
  # 0134's geometry. K is ~4x 0134's; if executors OOM in the E-step, raise
  # spark.executor.memory before touching min_positives.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0137 — DisMech ribbon record: tpn_max 5 on the repaired code map

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=137 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Audit, as soon as `code_map.tsv` appears in the run dir (right after the bundle is built):

```bash
grep -E "multi-map guard|hierarchy test|attestation audit|label set|powering:" "$RUN"/sweep_log.md
python3 - "$RUN" <<'PY'
import csv, json, sys, collections
run = sys.argv[1]
names = json.load(open(f"{run}/bundle_meta.json"))["name_by_id"]
n = collections.Counter(int(r["node_cid"]) for r in csv.DictReader(open(f"{run}/code_map.tsv"), delimiter="\t"))
for c, k in n.most_common(25):
    print(k, names.get(str(c), c))
PY
```

## Run log

**2026-10-09 — launch 1 (native-mondo-v2) stopped at the audit.** The multi-map guard
dropped 8 links over 2 concepts (Disorder of pregnancy ×5, Finding related to pregnancy
×3); peripartum CM and preeclampsia left the most-attested list. But the audit showed
single-target captures the guard cannot see: tularemia (GI-tract disorders), thrombophilia
(every DVT code), osteochondrosis (spine fractures, scoliosis). Relaunched on
native-mondo-v3 (anchor test; insight 0097 addendum); the audit line gains `anchor test:
N uncorroborated non-SNOMED target(s) dropped over M term(s): <names>`.

**2026-10-09 — launch 2 (native-mondo-v3) stopped at the audit.** The anchor test
dropped 151 targets over 147 terms, and the label set lost 15 members vs 0135 (dilated
cardiomyopathy, Down syndrome, Lyme disease, giardiasis, coccidioidomycosis, atrial
septal defect, IPAH, MALT lymphoma, …; 378 nodes), while tularemia (93) and thrombophilia
(165) were unchanged and HELLP rose to 97. Relaunched on native-mondo-v4 (hierarchy-
consistency test; insight 0097). Audit line: `hierarchy test: N over-broad non-SNOMED
target(s) dropped over M term(s): <names>`. Acceptance before the fit is read: DCM, Down
syndrome and Lyme back in the label set; tularemia and thrombophilia down to modest code
counts.

**2026-10-09 — launch 3 (native-mondo-v4, threshold 3) stopped at the audit.** Right on
the captures: 168 targets over 163 terms (pregnancy, vertebral column, …); DCM, Lyme,
giardiasis and peripartum CM back; tularemia leaves the label set (its real support is
under 100 once the GI capture is gone); osteochondrosis's spine capture gone. Wrong on a
few real diseases SNOMED subdivides more finely than Mondo: Down syndrome, acute kidney
injury, autoimmune thrombocytopenic purpura and acne left the label set. Thrombophilia
(162, every DVT code) unchanged: its Mondo subtree has no vein-thrombosis term (it holds
DIC, TTP, purpura fulminans), so the DVT codes arrive through one of its protected SNOMED
xrefs — traced with a vocabulary query. Relaunched on v5 (threshold 10; the receipt lists
dropped targets with their unrelated-term counts and the kept borderline band 3–9, so the
threshold is calibrated on data).

**Thrombophilia traced (vocabulary query, 2026-10-09):** the 106 DVT codes descend from
SCTID 111293003 *Venous thrombosis*, which Mondo lists as the SNOMED `same_as` of
*thrombophilia due to thrombin defect* (MONDO:0008559) — a Mondo curation error, protected
from the hierarchy test because it is a SNOMED xref. v6 adds a curated exclusion list
(`EXCLUDED_MONDO_XREFS`, this one entry) and REPORTS (never drops) any other Mondo SNOMED
xref that fails the hierarchy test, as `SUSPECT Mondo SNOMED xrefs ...`, for review into
the list. To report upstream to Mondo with the DisMech curation list.

## Results
