---
id: 139
slug: wpd-model-based-acquisition-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. WP-D part 3b on the ribbon record 0137, no refit: 0138 part 3 found the
# question ORDER right (greedy EIG >= static > random) and the presence-model ENGINE weak
# (calibrated only at a temper of 0.05-0.1; balanced top-1 on the full pool 0.12 for the
# 19-member connective-tissue set). Here the engine is the record's own: after each
# revealed code, θ is re-inferred from the revealed codes (the deployment fold-in over the
# saved λ and α — full-K CAVI, exactly OnlinePCLDAModel._transform) and the saved readout
# heads (readout_heads_gated_pc_stacked_mass99.npz) score it; the member posterior is the
# stacked product renormalized over the set (`model_acquisition.py`).
# SAME AS 0138 PART 3: candidate sets, query pool (top 150 condition + 150 drug codes by
# population EIG, attestation codes excluded), strategies, metrics, egress floor.
# DIFFERENT, by construction: a NO answer carries no information (a bag-of-codes model
# has no likelihood for an absent code), so the engine learns only from YES answers;
# no temper (the heads were found calibrated on full records, 0138 part 2); the step-0
# state is the empty record's posterior, the same for every patient.
# PRE-REGISTERED READS: (1) does the model engine beat the presence engine at equal
# questions (balanced top-1 and P(true) at 5/10/20, and on the full pool)? (2) does the
# order still hold under this engine (greedy >= static > random)? (3) calibration of the
# partial-record posteriors (the heads were fit on whole records; a few-code θ is off
# their training distribution — ECE says whether that matters); (4) the whole-record
# ceiling should land near 0138 part 2's conditional read (a check the engine
# reproduces the readout), and the gap between it and the full-pool ceiling is what the
# diagnosis codes themselves carry.
fit_of: 137
---

# 0139 — WP-D part 3b: model-based acquisition on 0137

## Run

Bundle load + one small collect per set (no transform, no solve); then driver-side
numpy, about 2.4 s per patient for all three strategies at K = 3,145 (so the 300-patient
cap is ~12 min per set):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/acquisition_model.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--acquisition '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --acq-engine model --acq-max-patients 300 --acq-tag groups_model"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--acquisition '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-prior uniform --acq-engine model --acq-tag marfan_eds_model"
EOF
nohup bash "$RUN"/acquisition_model.sh > "$RUN"/acquisition_model_log.md 2>&1 &
cat "$RUN"/acquisition_groups_model.md "$RUN"/acquisition_marfan_eds_model.md | cut -c1-170
```

(`--acq-prior uniform` on Marfan/EDS affects only the pool's ranking here; the model
engine's prior is the empty record's posterior.)

## Run log

## Results
