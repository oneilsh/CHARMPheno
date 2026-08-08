# Hughes et al. (AISTATS 2018) vs. our All-of-Us replication

**Paper:** Hughes, Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez,
*"Semi-Supervised Prediction-Constrained Topic Models,"* AISTATS 2018
(PMLR v84, `proceedings.mlr.press/v84/hughes18a`). The **Antidepressant task**
(their §6, Fig. 3) is the one we replicate on AoU OMOP.

This doc is the setup crosswalk: what the paper did, what we do, and — where we
diverge — *why* (usually: AoU OMOP shape or "only to the extent we can with the
current known concept ids"). It covers **both** of our cohorts:

- `mdd_antidepressant` (exps **0070** in-mem / **0071** VI) — our first framing:
  per-index-drug outcome, one observed label cell per patient.
- `mdd_stable_treatment` (exp **0072**, VI) — the **Hughes-faithful rebuild**:
  fully-observed multi-label over a fixed drug set, all-history features.

The `mdd_stable_treatment` column is the one that matters for "did we replicate
Hughes." `mdd_antidepressant` is kept for continuity and as the semi-supervised
contrast (it exercises the per-cell `label_mask` the paper only mentions in
passing — "if documents are partially labeled … include prediction constraints
for observed labels").

---

## 1. The target number

Fig. 3 (center panel), **avg heldout AUC across the 11 meds**, K swept
{10, 25, 50, 100}:

- **PC-sLDA ≈ 0.60–0.65**, rising with K, **beating logistic regression
  slightly** and **improving reliably on its Gibbs-LDA init**.
- BP-sLDA (purely discriminative) overfits badly here; unsupervised Gibbs-LDA is
  the floor PC improves on.

**Target ~0.60–0.65, not 0.67–0.71.** The higher band is the later
JAMA-Psychiatry 2020 follow-up (different model, different cohort), and had crept
into our 0070/0071 docs — now corrected.

---

## 2. Side-by-side

| Dimension | Hughes et al. 2018 | `mdd_stable_treatment` (0072, faithful) | `mdd_antidepressant` (0070/0071) |
|---|---|---|---|
| **Cohort** | MDD patients, tertiary-care EHR | MDD (existence of dx) + stable-treatment interval | MDD new-user of an antidepressant |
| **N** | 29774 / 3721 / 3722 (train/val/test) | AoU-dependent (record at run) | AoU-dependent |
| **Unit of prediction** | patient | patient | patient |
| **Label** | subset of **11** antidepressants that "successfully treat" → fully-observed 11-vector | fully-observed **10**-vector: `y[i]=1` iff drug *i* is in the person's stable subset | one cell: did the **index** drug sustain ≥90d |
| **Label mask** | all-observed (all 11 heads train on all patients) | **all-ones** (all 10 heads, every patient) | **one observed cell/patient** (semi-supervised) |
| **"Success" def.** | med sustained / effective (stable treatment) | constant-drug-subset interval ≥90d, encounter-regular, in one observation period | index ingredient covered ≥`stability_days`, refill-gap-stitched |
| **Drug set** | 11 (incl. mirtazapine) | **10** (mirtazapine dropped — no validated AoU concept id) | 15 (`_DRUG_REGISTRY`) |
| **Features** | count vector of **full EHR code history** | **all prior history** (`date < index`), fused vocab | pre-index window `[index−lookback, index)`, fused vocab |
| **Vocab** | V=5126: ICD-9 dx + CPT procedures + meds | fused condition + drug + procedure OMOP concept_ids | same fused vocab |
| **Feature anchor** | code history up to prediction time | stable-interval **start** (`index_date`) | first-antidepressant `drug_era` date |
| **Model** | PC-sLDA, L multi-label logistic heads sharing π | VI-native PC (SVI), 10 heads sharing π | same machine, per-index-drug head |
| **π estimate** | NEF-MAP via exponentiated gradient (train=test) | CAVI local step (VI port) / NEF-MAP (in-mem ref) | same |
| **Init** | **Gibbs-LDA (unsupervised) warm-start** | `--warm-start-unsup-iters N` (weight_y=0 phase → fresh-RM supervised phase) | same knob available |
| **Objective** | `−Σ[log p(x|φ) + λ log p(y|π,η)] − log p(φ,η)`, π=MAP(x) | same, per-head sum over observed cells | same |
| **λ (weight_y)** | ~avg tokens/doc; selected on val | `weight_y` (sweep) | `weight_y` (sweep) |
| **Val split** | yes (29774/3721/3722) | **skipped for now** (user deferred) | skipped |
| **Fit** | batch gradient (differentiate through MAP) | distributed SVI, Robbins-Monro | in-mem L-BFGS-B (0070) / SVI (0071) |

---

## 3. Where we are faithful (0072)

- **Fully-observed multi-label.** `stable_treatment_label` emits a length-10
  `y` with an implicit all-ones mask — every head trains on every patient, exactly
  Hughes' "subset of the meds" framing (a multi-drug stable interval yields
  multiple positives; a single-drug interval yields one). This is the paper's
  distinctive setup: **L conditionally-independent logistic heads sharing one π**,
  which the third-party BP-sLDA/MED-sLDA baselines couldn't do.
- **All prior history features.** `all_history_feature_events` takes every event
  with `date < index_date` — matching "count vector of the patient's EHR code
  history" rather than a fixed pre-index window.
- **Fused dx + procedure + medication vocab** over OMOP concept_ids — the OMOP
  analogue of Hughes' ICD-9 + CPT + medication codewords.
- **Unsupervised warm-start.** Hughes initialize PC-sLDA from a Gibbs-LDA fit;
  our `--warm-start-unsup-iters N` runs a `weight_y=0` SVI phase, then warm-starts
  the supervised phase with a fresh Robbins-Monro schedule. The A/B (warm vs cold)
  isolates exactly the effect the paper leans on ("PC-sLDA improves on the
  baseline Gibbs predictions reliably").
- **The PC objective and π=MAP(x)** are the trusted in-mem reference's, oracle-
  validated on `toy_bars` (ADR 0038 / Phase A). The VI port's CAVI local step is a
  faithfulness-preserving substitute for NEF-MAP (train=test π; validated at
  recovery parity, weight_y=0).
- **Cohort filters:** age 18–80 at interval start; ≥1 MDD dx; ≥2 pre-treatment
  events; stable interval within one observation period — the paper's inclusion
  spirit, mapped onto OMOP tables.

## 4. Where we deliberately diverge (and why)

- **10 drugs, not 11.** Mirtazapine has no validated OMOP standard concept id on
  this CDR, and the user's constraint is explicit: align drugs/classes *"only to
  the extent we can with the current known concept ids."* The other 10 are all
  pinned, AoU-validated ids already in `_DRUG_REGISTRY` — no new concept-id risk.
- **"Success" is operationalized as a stable-treatment interval, not clinician-
  judged response.** AoU has no outcome flag; Hughes' label likewise came from a
  treatment-stability heuristic on EHR. Our definition: a constant-drug-subset
  interval ≥90d that is encounter-regular (visits ≤`max_gap_days` apart, endpoints
  bounded) and sits inside one observation period. The **first (earliest)** such
  interval defines the patient (user decision).
- **MDD indication = existence of an MDD dx (any date)**, a flagged reading of the
  user's criterion (b). This is looser than the sibling `mdd_antidepressant`
  index, which requires the dx on/before index. Deliberate; noted in the code.
- **No validation split yet.** User deferred it ("skip the val split for now").
  K / weight_y are swept on the heldout test split for now; add a 3-way split
  before quoting a tuned number as a replication.
- **AoU is OMOP, not the paper's ICD-9/CPT source system.** Concept-id vocab,
  drug_era coverage semantics, and cross-system data completeness differ. Per ADR
  0038, the *machine* is trusted independently (Phase A), so a null on AoU is a
  **data** finding (med-completeness / cross-system leakage), not a model bug.

## 5. `mdd_antidepressant` (0070/0071) — the semi-supervised contrast

Our original framing labels each patient only for the drug they actually
initiated → a D×C matrix with **one observed cell per row**. This is a legitimate
generalization the paper explicitly allows ("if documents are partially labeled")
but is **not** what Hughes ran — they had the full label vector per patient.
Keep 0070/0071 as the partial-label / per-cell-`label_mask` contrast, and read
0072 as the head-to-head with the paper. If 0072 replicates and 0070/0071 don't,
that is informative about label completeness, not about the model.

## 6. Reading order at run time

1. Convergence first: VI `vi_convergence` block (final ELBO, `n_iter`,
   **`|w_CK|max`** — a value ≈0 means the head never left init; every AUC is then
   0.5 by construction, as happened on the under-converged in-mem 0070 fit).
2. Then the per-drug AUC table (PC vs two-stage vs LR), macro-averaged over
   non-degenerate drugs.
3. Compare macro-AUC to **~0.60–0.65**. Above baselines and in-band ⇒ Hughes
   replicated on AoU. Converged head but null ⇒ AoU data finding (Phase C insight).
