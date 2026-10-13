---
id: 142
slug: framing3-undiagnosed-lookalikes-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Framing 3 of spec 2026-10-13 (clinician framings): "who looks like disease
# X but carries no X code, and why?" (`lookalikes.py`, `gated_pc_readout --lookalikes`).
# For every member X of the candidate sets: held-out (test) patients WITHOUT an X label,
# ranked by the record's stacked readout P_stack(X) (the case-finding score; 0137 stacked
# de-novo macro AUC 0.822); the top 200 profiled against 3,000 random test patients and
# up to 1,000 of X's own — codes over-represented among the look-alikes AND more common
# in X's patients than in background ([measured] and X's attestation codes excluded).
# VALIDATION FOR FREE: concepts NAMED like X that the code map missed (0141: a second
# "Hypertrophic cardiomyopathy" concept) are real X diagnoses the label never saw; if
# the ranking finds undiagnosed-by-label cases they are over-represented in the top 200.
# PRE-REGISTERED READS: (1) per-X AUC (the ranking's quality, positives vs the rest);
# (2) the X-named-concept share, look-alikes vs random — the direct "found a real
# missed case" signal where such a concept exists; (3) do the profiles read as X-like
# features (and how many are treatments)?
# DEFERRED: the temporal check (later-coded X) needs the incident arm, off in 0137.
fit_of: 137
---

# 0142 — framing 3: undiagnosed look-alikes on 0137

## Run

Test-split transform + one collect (as 0138 part 2), then a broadcast-join collect of the
~30k records the profiles use. Bundle rebuild first if this cluster has no cache (~20 min).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/lookalikes.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--lookalikes '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --acq-tag groups"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--lookalikes '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-tag marfan_eds"
EOF
nohup bash "$RUN"/lookalikes.sh > "$RUN"/lookalikes_log.md 2>&1 &
cat "$RUN"/lookalikes_groups.md "$RUN"/lookalikes_marfan_eds.md | cut -c1-170
```

## Run log

## Results
