---
id: 139
slug: wpd-model-based-acquisition-on-0137
status: done
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

**2026-10-12 — run** (300 patients per Mondo group, 116 Marfan/EDS; ~12 min per group,
driver-only numpy as designed).

## Results

**The model engine is worse than the presence engine, and the reason is identifiable.**

| set | engine | empty / step 0 bal top-1 | greedy @20 | full pool | whole record (dx codes in) | ECE full pool |
|---|---|--:|--:|--:|--:|--:|
| connective tissue (19) | presence (0138, tau 0.1) | 0.053 | 0.070 | 0.122 | — | 0.046 |
| | model (θ + heads) | 0.059 | 0.071 | 0.103 | 0.318 | 0.410 |
| cardiomyopathy (8) | presence | 0.125 | 0.243 | 0.274 | — | 0.106 |
| | model | 0.125 | 0.135 | 0.145 | 0.216 | 0.171 |
| Marfan / EDS (3) | presence (uniform) | 0.333 | 0.611 | 0.701 | — | 0.253 |
| | model | 0.333 | 0.333 | 0.378 | 0.484 | 0.197 |

1. **The posterior barely moves.** Marfan/EDS stays at the empty-record call (top-1
   0.612 = the same member for every patient, balanced 0.333) through 20 questions in
   every strategy; cardiomyopathy gains 0.01 balanced top-1. Mechanism (arithmetic, not
   a fit artefact): the deployment fold-in uses the fitted α = 0.5 on each of 3,145
   topics, a prior of ~1,570 pseudo-counts, so a revealed code with a record count of a
   few moves θ by a fraction of a percent. Full records (hundreds to thousands of
   tokens) move it; a few-code record does not — θ stays near α / sum(α).
2. **The empty-record θ is off the heads' training distribution and they mis-call it**:
   connective tissue's empty-record posterior concentrates (1.5 bits of a possible 4.2)
   on a member that is right for 8% of patients while the population majority is 31%;
   ECE 0.41–0.47 there. The heads were fit on whole-record θ only.
3. **Even with the whole record — diagnosis codes included — the renormalized heads are
   weak which-member classifiers**: balanced top-1 0.32 / 0.22 / 0.48. They are
   one-vs-rest logistic heads, each calibrated on its own population (0138 part 2:
   cond AUC ~0.8 per edge); renormalizing independent marginals across members is not a
   multiclass model, and argmax drifts to members with large intercepts. Part 2's own
   bal_acc column said the same (0.17–0.76, mostly majority calls).
4. **Ordering**: under this engine the strategies barely separate (static ≥ greedy on
   P(true) in connective tissue and cardiomyopathy; the greedy rule's expected-KL score
   is tiny when nothing moves). Not informative about ordering; 0138's presence-engine
   result stands.

**Read.** For acquisition, the tempered presence engine (0138) is the better engine on
this record; the topic model's θ path is not usable from a few codes at α 0.5 × K 3,145,
and the readout heads are not a which-member classifier. Two engine fixes, both outside
this experiment: a deployment fold-in at a much smaller α (the heads would then need
refitting on that θ — this is a concrete use for the α ladder declined for now), and a
per-set multiclass (softmax) head fit within the parent's cohort, the natural "which
child" model. Insight 0098.
