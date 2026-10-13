---
id: 142
slug: framing3-undiagnosed-lookalikes-on-0137
status: running
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

**2026-10-13 — first run** (21 targets with >= 20 test positives; 156 s for the
Marfan/EDS pass once scored).

- **Ranking quality**: AUC 0.70–0.95 (hEDS 0.951, SSc 0.931, SLE 0.936, EDS 0.909, RA
  0.895, PsA 0.895, dermatomyositis 0.888, HCM 0.886; lowest hypereosinophilic syndrome
  0.703, restrictive CM 0.742, ET 0.750, palmar fibromatosis 0.746).
- **X-named concept shares are large where measurable** — SLE look-alikes 63.0% vs random
  1.3%, RA 52.5% vs 2.3%, DCM 28.5%, ET 25.5%, sarcoidosis 19.0%, PMR 17.0%, EDS 14.5%,
  myocarditis 14.0% — **but the first run's "unmapped" label was wrong**: the name match
  did not test code-map membership (HCM's line counted 2 concepts, one of them the mapped
  4124693). And in lookback mode features are the PRE-index history while the label is
  "coded in the year after the index date", so a look-alike carrying an X-named concept
  may simply be a PREVIOUSLY diagnosed X patient not re-coded in the label year. The
  RA concept sitting in 2.3% of random test patients (≈ RA's own test prevalence) points
  that way. Fixed in the driver: look-alikes are split into X attestation code in the
  history / only an unmapped X-named concept / neither (undiagnosed by code), named
  concepts counted as mapped vs unmapped, and the undiagnosed group profiled on its own.
- **Profiles**: many top look-alikes carry a SIBLING diagnosis — AS and PsA look-alikes
  are RA patients (RA 78% / 70%), HCM's are DCM / systolic-HF patients, DCM's carry HCM /
  HOCM codes, ET's carry polycythemia vera and vice versa, Takotsubo's are DCM /
  cardiomyopathy. That is the differential problem showing up as look-alikes, not
  undiagnosed disease. The clinically compelling ones: hEDS / EDS (hypermobility syndrome,
  joint derangement, orthostatic hypotension, dysautonomia, migraine, fibromyalgia, IBS);
  SLE (hydroxychloroquine, C3/C4 tested normal, Sjögren's, connective-tissue disorder);
  dermatomyositis (ILD, myositis, ANA positive, Raynaud's); leiomyoma (menorrhagia, iron
  deficiency anemia from blood loss, endometriosis, uterine polyp); PV (thrombocytosis,
  hydroxyurea, high hematocrit); SSc (full NEGATIVE ENA panels — worked up and not
  confirmed); HES (raised eosinophils, severe persistent asthma).
- **Code-map gap candidate**: "Contracture of palmar fascia" (= Dupuytren's, i.e. palmar
  fibromatosis) in 15% of its look-alikes vs 0.2% random; the rest of that profile is
  cardiopulmonary noise (dyspnea, edema, orthopnea) — older men.

Re-run with the split:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup bash "$RUN"/lookalikes.sh > "$RUN"/lookalikes_log2.md 2>&1 &
cat "$RUN"/lookalikes_groups.md "$RUN"/lookalikes_marfan_eds.md | cut -c1-170
```

## Results
