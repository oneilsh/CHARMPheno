---
id: 140
slug: framing1-differential-engines-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Framing 1 of spec 2026-10-13 (clinician framings): "for this patient, which
# disease in group G, and what in their record points there?" — four engines on the
# same held-out patients of 0137 (`differential.py`, `gated_pc_readout --differential`):
#   E1 presence model with SUBTYPE profiles (D5), present AND absent codes, tempered;
#   E2 the stacked readout heads renormalized (0139's baseline);
#   E3 a multiclass softmax head on log θ, fit on train-split single-positive patients;
#   E4 gated per-disease likelihood (fold-in ELBO under {background ∪ block(m)}), tempered.
# Records lose attestation codes (members' + group's own) and [measured] tokens (D1, D2).
# Evaluation patients: test split, positive on exactly one member (multi-positive set
# aside and counted); tempers fit on one half, ALL engines scored on the other half.
# Sets: ^cardiomyopathy$ (8), ^connective tissue disorder$ (19), Marfan/EDS/hEDS (3).
# PRE-REGISTERED READS: (1) which engine has the best balanced top-1 at acceptable
# calibration (ECE ≲ 0.1) per set — that engine carries framings 2 and 3; (2) E2 should
# reproduce 0139's whole-record figures roughly (balanced top-1 0.32 / 0.22 / 0.48 —
# not exactly: attestation and [measured] are now removed, and only single-positive
# patients are scored); (3) whether E4's τ is tiny (generative overconfidence on long
# records) and whether, tempered, it is calibrated; (4) do the reasons read clinically
# (E1/E4 pooled supporting codes per member)?
fit_of: 137
---

# 0140 — framing 1: differential with reasons, four engines on 0137

## Run

The HDFS cache died with the last cluster, so the first command rebuilds the bundle
(~20 min, n2-standard-8 master); the other two reuse it. Each set is then minutes of
driver numpy (benchmarked: full-K fold-in ~0.03 s, E4 over 19 members ~0.3 s per
patient).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/differential.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --acq-tag groups"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-tag marfan_eds"
EOF
nohup bash "$RUN"/differential.sh > "$RUN"/differential_log.md 2>&1 &
```

Read:

```bash
cat "$RUN"/differential_groups.md "$RUN"/differential_marfan_eds.md | cut -c1-170
```

## Run log

## Results
