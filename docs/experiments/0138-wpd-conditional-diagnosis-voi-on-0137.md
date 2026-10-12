---
id: 138
slug: wpd-conditional-diagnosis-voi-on-0137
status: running
model_class: gated_pc
# NOT A FIT. WP-D (spec 2026-10-07 §D4) on the ribbon record 0137, no refit:
#   part 1 — code-level VOI tables over Mondo candidate sets (off-cluster, numpy over the
#            saved λ + census signatures + readout_dag.json; `inspect_topics --voi`);
#   part 2 — conditional diagnosis on held-out patients: per readout-DAG edge P -> c, the
#            within-P-cohort cond AUC (the sober column), top-1 vs majority and ECE, for
#            the flat heads and the stacked product (`gated_pc_readout
#            --conditional-from-heads stacked_mass99`: test split only, saved heads, no
#            solve).
# CANDIDATE SETS (decided 2026-10-07): Mondo ancestors — first two: connective tissue
# disorders and cardiomyopathies; the full per-edge table comes free from part 2.
# PRE-REGISTERED READS:
#   part 1: are the top codes clinically discriminating (e.g. aortic dilatation/dissection
#           for Marfan vs EDS; LBBB / systolic HF for DCM vs HCM's outflow obstruction),
#           or care-context codes? Own codes are excluded from the ranking by default.
#   part 2: cond AUC by depth for flat vs stacked; top-1 above the majority baseline for
#           how many parents; ECE (calibration is the prerequisite for VOI to mean bits).
# part 3 (built 2026-10-12, voi_acquisition.py + gated_pc_readout --acquisition): simulated acquisition on held-out patients — hide codes,
# reveal in EIG order vs random vs information-content order, measure posterior
# concentration (the 2026-08-14 report's evaluation of VOI).
fit_of: 137
---

# 0138 — WP-D on 0137: code-level VOI over Mondo groups, and conditional diagnosis

## Run

Part 1 (minutes, master only; names resolved off-YARN):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--voi '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --own-codes --relevance-lambda 0.6 --own-top 5 --results-file results_readout_stacked_mass99.json --voi-top 12" 2>/dev/null | grep -vE "^\[inspect-topics\]|^make" | cut -c1-170
```

Census re-read with the usage dashboard's interpretation tools (relevance-ranked words,
KL from the corpus marginal, ANCHOR flag; own code must rank in the top 5 by RELEVANCE,
which is the strict rule for common diseases), before reading VOI:

```bash
make -C analysis/cloud inspect-topics ID=137 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --relevance-lambda 0.6 --own-top 5 --top-words 4 --digest-exemplars 4 --grep 'Ehlers-Danlos syndrome,|Marfan|Dilated cardio|thrombophilia'" 2>/dev/null | grep -vE "^\[inspect-topics\]|^make" | cut -c1-170 | sed -n '1,70p'
```

Part 2 (bundle rebuild on a fresh cluster ~20 min, test-split transform, one collect; no
solve):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--conditional-from-heads stacked_mass99" > "$RUN"/conditional_log.md 2>&1 &
grep -A 40 "conditional diagnosis" "$RUN"/conditional_log.md | cut -c1-170
```

Part 3 — simulated acquisition (`voi_acquisition.py`; bundle load + one small collect per
set, no transform, no solve). Three readouts: the two Mondo groups under the population
prior; Marfan vs EDS under the uniform prior; and the same with strata admitted to the
profiles (the census-rule sensitivity, hEDS's POTS topic):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
cat > "$RUN"/acquisition.sh <<'EOF'
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--acquisition '^cardiomyopathy\$\$|^connective tissue disorder\$\$' --acq-tag groups"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--acquisition '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-prior uniform --acq-tag marfan_eds"
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--acquisition '^Marfan syndrome|^Ehlers-Danlos syndrome' --acq-adhoc --acq-prior uniform --acq-include-strata --acq-tag marfan_eds_strata"
EOF
nohup bash "$RUN"/acquisition.sh > "$RUN"/acquisition_log.md 2>&1 &
cat "$RUN"/acquisition_groups.md "$RUN"/acquisition_marfan_eds.md "$RUN"/acquisition_marfan_eds_strata.md | cut -c1-170
```

PRE-REGISTERED READS (part 3): greedy reaches a given P(true) in fewer questions than
static, static fewer than random; balanced top-1 is the column that shows whether rare
members are ever recovered; ECE of the max posterior says how overconfident naive Bayes
over the asked codes is (expected: clearly, at 20 questions).

## Run log

**2026-10-12 — census re-read with relevance ranking and the strict own-code rule**
(`--relevance-lambda 0.6 --own-top 5`, thin < 0.1). Block topics 1,945: signature 1,044
(54%) · stratum 387 (20%) · thin 436 (22%) · duplicate 8 · starved 70 (4%). Decidable nodes
(280): n_profiles 0: 5 · 1: 60 · 2: 96 · 3: 68 · 4: 42 · 5: 9 — median 2, p90 4, 3% at the
ceiling (was 16% under the lenient rule). Rare diseases now read cleanly: Marfan = aortic
(ectasia, atenolol, warfarin) + dissection/pseudoaneurysm + skeletal (pectus, scoliosis),
the evidence-36 carotid/goiter topic thin; hEDS = EDS type 3 + mast cell + somatic
dysfunction, and a musculoskeletal profile, the hearing-loss topic thin and the
hypermobility-POTS topic a stratum (no hEDS own code in its top 5 — arguably a real hEDS
profile coded as hypermobility; watch); DCM = DCM/LBBB/chronic systolic HF // sacubitril,
carvedilol + a primary-CM/PVC profile, the decompensated-HF topic a stratum. Common
diseases still carry 3–4 (hypertension: diabetic-neuropathy, metabolic, lumbar-pain —
own code ranked by relevance still clears top 5 there); the ANCHOR flag marks the
own-code-dominated topics. **`tpn_max` 5 confirmed with headroom**; a per-disease tpn
from this census is the efficiency option for the next fit.

**2026-10-12 — parts 1 and 2 run.**

*Part 1 (VOI, population prior).* The tables read clinically. Connective tissue
disorder: methotrexate, hydroxychloroquine, etanercept, adalimumab, CCP and rheumatoid
factor → RA; complement C3/C4, mycophenolate, azathioprine → SLE; hydroxyurea and
thrombocytosis → essential thrombocythemia; psoriasis → psoriatic arthritis;
menorrhagia, ovarian cyst, pelvic pain → leiomyoma. Cardiomyopathy: carvedilol,
sacubitril, spironolactone, chronic systolic HF, LBBB → DCM; verapamil and
**disopyramide** → HCM; transplanted heart and amyloidosis → restrictive; the mRNA
COVID vaccine → myocarditis. Weak spots: (a) measurements favouring myocarditis or
restrictive CM are care-context labs (HDW, ALT, creatinine, LDL, bilirubin), i.e. "has
an inpatient lab panel"; diazepam/petrolatum → restrictive is the same thing in drugs;
(b) "Cardiomyopathy" (the group's own code) topped the table — the set node's own codes
are now excluded too; (c) eosinophilic asthma is an own code of the hypereosinophilic
node, so asthma/dander-IgE "favour" it — a code-map breadth question, not a VOI one;
(d) Mondo's *connective tissue disorder* is a broad grouping (leiomyoma, ET, pituitary
tumour are members); Marfan/EDS are in it but at ~0% prior, and under the population
prior EIG cannot favour a near-zero member, so the pre-registered Marfan-vs-EDS read
needs `--voi-prior uniform` (added). Posteriors of 95–100% from one code are the
presence model's, not validated — part 3 is what tests them.

*Part 2 (conditional diagnosis, test split).* The two printed rows are not rival
models: under the closure mask each depth >= 1 head is fit on its parent's positives,
so sigma(z_c) **is** P(c | parent); the P_stack(c) row ranks the marginal within the
parent cohort, where the ancestor product is which-child noise (relabelled in the
driver as the reference row). The answer row: cond AUC 0.81 / 0.80 / 0.81 / 0.79 /
0.79 / 0.78 / 0.77 at depths 0–6 (869 edges) — flat with depth; pooled ECE 0.020 but
per-node mean 0.086 (pooling flatters calibration). The marginal row falls from 0.79 to
0.66 with depth, ECE 0.093 — the expected mismatch, not a defect of the stacked heads.
Which-child top-1 beats the majority baseline on most parents, by a wide margin where
siblings are balanced (reproductive 0.87 vs 0.49, developmental disorders 0.63 vs 0.27,
digestive 0.83 vs 0.51, connective tissue 0.81 vs 0.64) and barely or not at all where
one child dominates (cardiovascular, nervous system, musculoskeletal, integumentary,
pregnancy, ENT); balanced accuracy 0.17–0.76, i.e. the heads mostly call the majority
child. That is the population prior speaking, the same thing part 1 shows.

**Read.** Conditional diagnosis works at about the level of the marginal readout (0.79
within cohort) and holds with depth; calibration is good pooled, uneven per node.
Part 3 (simulated acquisition) is justified. Re-read part 1 with `--voi-prior uniform`
on a tighter set for the Marfan/EDS contrast first.

**2026-10-12 — part 1 re-read, uniform prior.** The pattern matched only the EDS node
(members: generic EDS and hEDS; Marfan forms no set of its own, and no readout node
holds both — `--voi-adhoc` added for that contrast). Generic-EDS vs hEDS splits on a
clinical axis: arrhythmia, orthostatic hypotension, tachycardia, gastroparesis,
propranolol, fludrocortisone (dysautonomia/POTS) favour **generic EDS**; multiple joint
pain, myofascial pain, spine/sacrum disorders, naltrexone (low-dose, for pain) and
cromolyn (mast-cell) favour **hEDS**. The POTS side landing on generic EDS is the
census's doing: hEDS's hypermobility-POTS topic is classed a stratum (no hEDS own code
in its top 5) and so is not in hEDS's profile — the case flagged in the census entry,
now with a visible consequence. Measurements all favour hEDS and are work-up panels
(CBC differential, ferritin, B12, TSH, anion gap); contrast agents and mupirocin are care
context. So: drug and condition tables are informative, measurement tables mostly
measure care intensity — consistent across all three sets read.

**2026-10-12 — the pre-registered Marfan vs EDS read** (`--voi-adhoc --voi-prior
uniform`; members Marfan, generic EDS, hEDS). **Passes.** Marfan is favoured by aortic
valve disorder, dissection of aorta, ectasia of thoracic aorta, mitral valve disorder,
transplanted heart valve, atenolol and warfarin (valve replacement) — the textbook
aortic-root picture; generic EDS by orthostatic hypotension, tachycardia, arrhythmia,
gastroparesis, propranolol, tizanidine (dysautonomia); hEDS by multiple joint pain,
naltrexone, cromolyn. Odd ones: amoxapine and phenylephrine on the Marfan side (a
low-evidence drug and a peri-operative one). Measurements again read as work-up
intensity, not disease.

## Results
