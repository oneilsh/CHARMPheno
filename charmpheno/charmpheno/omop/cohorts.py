"""Cohort definitions for OMOP-shaped event data.

A "cohort" here is a function that takes an OMOP events DataFrame
(person_id, concept_id, date columns) and returns a subset filtered to
a specific clinical population over a specific observation window.
Cohorts are orthogonal to DocSpecs: a cohort selects WHICH patients
and WHICH dates make it through; a DocSpec then collapses surviving
events into documents.

Currently implemented:

- ``first_cancer_year``: patients with a first malignant-cancer diagnosis
  (excluding non-melanoma skin cancer and carcinoma in situ), windowed
  to the 365 days starting at that first dx. Requires >= 365 days of
  post-index observation_period coverage (so the doc window is fully
  observed) and, by default, >= 365 days of prior coverage (so "first" is
  meaningful); the prior lookback is configurable via ``prior_obs_days``
  (0 drops it, admitting prevalent cases). See ``_window_observed_cohort``.
- ``first_dementia_year``: patients with a first all-cause dementia
  diagnosis (Alzheimer's, vascular, Lewy body, FTD, dementia NOS — i.e.
  descendants of SNOMED "Dementia"), windowed to the 365 days starting
  at that first dx. Same observation-period bracketing as the cancer
  cohort.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def cohort_defs_version() -> str:
    """Content hash of this module's source, for cache-key invalidation.

    Folded into the corpus + covariate cache keys so that ANY change to a cohort
    definition or its helper logic auto-invalidates cached corpora — no manual
    version bump to remember. Coarse by design: any edit to this module changes
    the hash and invalidates ALL cohort caches (correctness over cache reuse).

    Falls back to a constant if the source is unavailable (e.g. a byte-compiled
    deploy with no .py in the zip), in which case invalidation relies on the
    manual ``v`` in the cache-key payloads — so bump that too on shape changes.
    """
    try:
        src = inspect.getsource(sys.modules[__name__])
    except (OSError, TypeError):
        return "src-unavailable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


# Top-level SNOMED concept whose descendants define the inclusion set for
# malignant cancers. concept_ancestor(443392) returns every malignant-
# cancer condition concept in OMOP.
_CANCER_ANCESTOR = 443392

# Ancestor concepts whose descendants are EXCLUDED from the "first cancer"
# definition.
#   - NMSC (BCC/SCC) is excluded because it's enormously common,
#     clinically minor, and would otherwise dominate the cohort.
#   - Carcinoma in situ is excluded because it's pre-invasive and follows
#     a different disease trajectory than invasive cancer.
_CANCER_EXCLUSION_ANCESTORS: tuple[int, ...] = (
    4115276,  # Squamous cell carcinoma of skin
    4112744,  # Basal cell carcinoma of skin
    4180978,  # Carcinoma in situ
)

# Window after the first cancer dx that defines a patient's "document".
# Matches the existing one-year doc convention used elsewhere in the
# project for patient_year DocSpec defaults.
_WINDOW_DAYS = 365


# Top-level SNOMED concept whose descendants define the inclusion set for
# all-cause dementia. concept_ancestor(4182210) should return AD,
# vascular dementia, DLB, FTD, dementia NOS, mixed dementia, etc.
#
# VERIFY ON FIRST RUN: a quick sanity check is to count descendants:
#   SELECT COUNT(*) FROM concept_ancestor
#   WHERE ancestor_concept_id = 4182210;
# Expect dozens-to-hundreds of descendants. If you see 0, swap for the
# correct OMOP concept_id for the SNOMED "Dementia" hierarchy in your
# vocab version.
#
# Choice rationale: we deliberately go broad (all-cause) rather than
# AD-only because EHR coding is notoriously mushy between AD vs
# "dementia NOS" vs vascular — a pure-AD cohort would silently exclude
# real-AD patients whose providers happened to code them differently.
# The post-onset phenotype cascade is also similar across dementia
# subtypes (delirium, falls, polypharmacy, behavioral disturbance,
# aspiration pneumonia, end-of-life care), so the breadth helps without
# diluting the signal.
_DEMENTIA_ANCESTOR = 4182210

# No exclusions for v1: capturing the full dementia-syndrome trajectory
# is the goal. Kept as a constant (not inlined) so adding exclusions
# later is a one-liner.
_DEMENTIA_EXCLUSION_ANCESTORS: tuple[int, ...] = ()

# Top-level OMOP concept whose descendants define Ehlers-Danlos syndrome.
# Provided by the domain owner (2026-07-03). No exclusions.
# VERIFY ON FIRST RUN (as with dementia): count the descendants —
#   SELECT COUNT(*) FROM concept_ancestor WHERE ancestor_concept_id = 79145;
# Expect a non-zero descendant set (the EDS subtypes). If you see 0, the id is
# wrong for this vocab version — re-confirm the OMOP concept for the SNOMED
# "Ehlers-Danlos syndrome" hierarchy before fitting.
_EDS_ANCESTOR = 79145

# Top-level SNOMED concept for diabetes mellitus. concept_ancestor(201820)
# returns the diabetes TYPE/etiology/status taxonomy (T1/T2/MODY/neonatal/
# gestational/secondary x remission/control/pregnancy) — 127 standard-condition
# descendants on the AoU vocab (offline-verified 2026-07-15). Classic
# complications (nephropathy/retinopathy/neuropathy/CKD) are NOT is-a
# descendants of 201820 (they live under 442793); by design they ride along as
# learned vocabulary in each type node's topic rather than as DAG nodes. VERIFY
# ON FIRST RUN: SELECT COUNT(*) FROM concept_ancestor WHERE
# ancestor_concept_id = 201820 (expect ~hundreds; 0 means wrong id for this
# vocab version).
_DIABETES_ANCESTOR = 201820

# Top-level OMOP concepts for five additional rare/uncommon autoimmune-ish
# diseases, chosen to be phenotypically DISTINCT from each other (unlike the
# diabetes TYPE taxonomy, whose subtypes are near-identical phenotypes) and
# common enough to yield fittable case counts on All of Us (counts confirmed by
# the domain owner, 2026-07-15; concept-ids vocab-verified offline). Together
# with EDS they form the "rare6" multi-disease case-finding forest. VERIFY ON
# FIRST RUN, as with the other anchors: SELECT COUNT(*) FROM concept_ancestor
# WHERE ancestor_concept_id = <id> (expect a non-zero descendant/subtype set).
_SARCOIDOSIS_ANCESTOR = 438688      # Sarcoidosis (AoU-over-represented, few 1000s)
_SLE_ANCESTOR = 257628              # Systemic lupus erythematosus (~6500, best-powered)
_SCLERODERMA_ANCESTOR = 40352976    # Scleroderma / systemic sclerosis (few 1000s)
_MYASTHENIA_GRAVIS_ANCESTOR = 76685  # Myasthenia gravis (~1100)
_AMYLOIDOSIS_ANCESTOR = 432595      # Amyloidosis (~1500)

# The multi-disease rare-disease forest foreground: a patient qualifies for the
# "rare6" arm if they have a first dx under ANY of these six anchors. The same
# tuple doubles as the label-DAG roots (each anchor becomes a subtree under a
# synthetic forest root; see case_finding_assembly). Distinct phenotypes + a
# whole-population clean background is the rare-disease case-finding thesis in
# practice (project_kg_rare_disease_casefinding).
_RARE6_ANCESTORS: tuple[int, ...] = (
    _EDS_ANCESTOR, _SARCOIDOSIS_ANCESTOR, _SLE_ANCESTOR,
    _SCLERODERMA_ANCESTOR, _MYASTHENIA_GRAVIS_ANCESTOR, _AMYLOIDOSIS_ANCESTOR,
)

# Major depressive disorder for the Hughes antidepressant-response replication
# (Phase C). Inclusion is the SNOMED "Major depressive disorder" hierarchy;
# bipolar disorder and schizoaffective disorder are EXCLUDED so the treated
# indication is unipolar depression — an antidepressant started in a bipolar or
# schizoaffective patient is a different clinical decision (mood stabiliser /
# antipsychotic co-therapy, distinct response criteria) and would confound a
# "the antidepressant worked" outcome. VERIFY ON FIRST RUN, per anchor:
#   SELECT COUNT(*) FROM concept_ancestor WHERE ancestor_concept_id = 440383;
# (expect a non-trivial MDD subtype set; 0 => wrong id for this vocab version).
# Likewise 439254 (bipolar) and 4224940 (schizoaffective) for the exclusions.
_MDD_ANCESTOR = 440383
_MDD_EXCLUSION_ANCESTORS: tuple[int, ...] = (
    439254,    # Bipolar disorder
    4224940,   # Schizoaffective disorder
)

# Anxiety-disorder hierarchy. Added as a SEPARATE disease entry: anxiety is a
# secondary antidepressant indication in Hughes, but the PRIMARY replication
# cohort's indication is major depression, so anxiety is kept registered (for a
# future secondary-indication arm) rather than folded into the MDD inclusion.
# VERIFY ON FIRST RUN:
#   SELECT COUNT(*) FROM concept_ancestor WHERE ancestor_concept_id = 441542;
_ANXIETY_ANCESTOR = 441542

# Disease registry for the generalized population+disease cohort. Each entry is
# fully described by concept ancestors; adding a rare disease is a new entry
# here + a SUPPORTED_COHORTS/COHORT_METADATA/apply_cohort line, no new function.
# A multi-ancestor entry (rare6) is a foreground union: apply_first_diagnosis_
# year_cohort already unions descendants of all inclusion_ancestors.
_DISEASE_REGISTRY: dict[str, dict] = {
    "cancer": {
        "inclusion_ancestors": (_CANCER_ANCESTOR,),
        "exclusion_ancestors": _CANCER_EXCLUSION_ANCESTORS,
    },
    "eds": {
        "inclusion_ancestors": (_EDS_ANCESTOR,),
        "exclusion_ancestors": (),
    },
    "diabetes": {
        "inclusion_ancestors": (_DIABETES_ANCESTOR,),
        "exclusion_ancestors": (),
    },
    "rare6": {
        "inclusion_ancestors": _RARE6_ANCESTORS,
        "exclusion_ancestors": (),
    },
    "mdd": {
        "inclusion_ancestors": (_MDD_ANCESTOR,),
        "exclusion_ancestors": _MDD_EXCLUSION_ANCESTORS,
    },
    "anxiety": {
        "inclusion_ancestors": (_ANXIETY_ANCESTOR,),
        "exclusion_ancestors": (),
    },
}


def disease_anchors(disease: str) -> tuple[int, ...]:
    """The DAG anchor concept-ids for a registered disease — its cohort inclusion
    ancestors reused as the label-DAG roots.

    A single-disease cohort (diabetes, eds, cancer) yields one anchor and a DAG
    rooted directly at it; the multi-disease forest (rare6) yields the full anchor
    set, which case_finding_assembly hangs under a synthetic forest root. Keeping
    the DAG anchors identical to the cohort inclusion ancestors makes ``disease``
    the single knob that determines both which patients are foreground and which
    subtree(s) their frontier is scored against.
    """
    try:
        return tuple(_DISEASE_REGISTRY[disease]["inclusion_ancestors"])
    except KeyError:
        raise ValueError(
            f"disease {disease!r} not in registry (known: {tuple(_DISEASE_REGISTRY)})"
        )


# Drug classes for the population_glp1 gated cohort, resolved by RxNorm
# Ingredient NAME (not hard-coded concept_ids, so it is portable across CDR
# vocab versions). VERIFY ON FIRST RUN that each name set resolves to a
# non-empty ingredient set on the target CDR (see apply_population_drug_cohort's
# build-time diagnostic).
_DRUG_REGISTRY: dict[str, dict] = {
    # Each class = descendants of its ATC-class concept (the authoritative
    # definition), with ingredient names kept as a belt-and-suspenders fallback.
    # ATC A10BJ = 1123618 "GLP-1 analogues"; A10BK = 1123627 "SGLT2 inhibitors".
    # tirzepatide is NOT under A10BJ (dual GIP/GLP-1), so it stays its own arm.
    # VERIFY the seed ids resolve to a non-trivial descendant set on the CDR.
    "glp1_ra": {
        "ingredient_names": (
            "semaglutide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide",
        ),
        "seed_concept_ids": (1123618,),
    },
    "sglt2i": {
        "ingredient_names": (
            "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
        ),
        "seed_concept_ids": (1123627,),
    },
    # tirzepatide (FDA 2022): pin the OMOP concept id as an explicit seed so the
    # class set is descendants-of-779705 regardless of how this CDR's vocab
    # names/classes the ingredient (name-only match under-counted it). VERIFY
    # 779705 on the CDR (SELECT ... FROM concept WHERE concept_id=779705).
    "tirzepatide": {
        "ingredient_names": ("tirzepatide",),
        "seed_concept_ids": (779705,),
    },
    # --- Antidepressants (Hughes Phase C replication) -----------------------
    # Fifteen RxNorm-ingredient antidepressants across four pharmacologic
    # classes, each a SEPARATE registry entry keyed by ingredient name and
    # carrying a "class" tag (SSRI/SNRI/TCA/Atyp) so the index step can record
    # WHICH drug (and class) a person initiated and the outcome labeler can tell
    # a same-ingredient refill from a switch to a different ingredient. Same
    # portable-by-name + pinned-id-fallback shape as the GLP-1 entries above:
    # ``ingredient_names`` resolves on the CDR vocab, ``seed_concept_ids`` pins
    # the AoU-validated OMOP standard concept ids so resolution is robust to
    # vocab naming drift. VERIFY ON FIRST RUN that each pinned id is a standard
    # RxNorm Ingredient that expands to a non-trivial drug_era-matchable set:
    #   SELECT COUNT(*) FROM concept_ancestor WHERE ancestor_concept_id = <id>;
    # (0 descendants => the pin is non-standard / misspec for this vocab
    # version — a non-standard pin has no concept_ancestor rollup and collapses
    # to just itself, silently under-counting exposure.)
    "fluoxetine":    {"ingredient_names": ("fluoxetine",),    "seed_concept_ids": (755695,),   "class": "SSRI"},
    "sertraline":    {"ingredient_names": ("sertraline",),    "seed_concept_ids": (739138,),   "class": "SSRI"},
    "paroxetine":    {"ingredient_names": ("paroxetine",),    "seed_concept_ids": (722031,),   "class": "SSRI"},
    "citalopram":    {"ingredient_names": ("citalopram",),    "seed_concept_ids": (797617,),   "class": "SSRI"},
    "escitalopram":  {"ingredient_names": ("escitalopram",),  "seed_concept_ids": (715939,),   "class": "SSRI"},
    "vilazodone":    {"ingredient_names": ("vilazodone",),    "seed_concept_ids": (40234834,), "class": "SSRI"},
    "venlafaxine":   {"ingredient_names": ("venlafaxine",),   "seed_concept_ids": (743670,),   "class": "SNRI"},
    "duloxetine":    {"ingredient_names": ("duloxetine",),    "seed_concept_ids": (715259,),   "class": "SNRI"},
    "desvenlafaxine":{"ingredient_names": ("desvenlafaxine",),"seed_concept_ids": (717607,),   "class": "SNRI"},
    "amitriptyline": {"ingredient_names": ("amitriptyline",), "seed_concept_ids": (710062,),   "class": "TCA"},
    "imipramine":    {"ingredient_names": ("imipramine",),    "seed_concept_ids": (778268,),   "class": "TCA"},
    "nortriptyline": {"ingredient_names": ("nortriptyline",), "seed_concept_ids": (721724,),   "class": "TCA"},
    "bupropion":     {"ingredient_names": ("bupropion",),     "seed_concept_ids": (750982,),   "class": "Atyp"},
    "trazodone":     {"ingredient_names": ("trazodone",),     "seed_concept_ids": (703547,),   "class": "Atyp"},
    "vortioxetine":  {"ingredient_names": ("vortioxetine",),  "seed_concept_ids": (44507700,), "class": "Atyp"},
}

# The 15-drug antidepressant set (registry keys), in class order. The
# antidepressant index + outcome steps iterate this to build the concept map;
# keeping it as an explicit tuple (rather than filtering _DRUG_REGISTRY on the
# "class" key) documents the exact Hughes drug list and its ordering.
_ANTIDEPRESSANT_INGREDIENTS: tuple[str, ...] = (
    "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram",
    "vilazodone",                                        # SSRI
    "venlafaxine", "duloxetine", "desvenlafaxine",       # SNRI
    "amitriptyline", "imipramine", "nortriptyline",      # TCA
    "bupropion", "trazodone", "vortioxetine",            # Atypical
)

# The 10 Hughes-aligned antidepressants (AISTATS 2018, supplement B.4) for the
# ``mdd_stable_treatment`` cohort — a STRICT SUBSET of the 15-drug set above.
# Hughes lists 11 drugs; mirtazapine is dropped because it has no validated
# OMOP standard concept id on this CDR (do NOT add it), and the 5 extras carried
# by ``_ANTIDEPRESSANT_INGREDIENTS`` (vilazodone, desvenlafaxine, imipramine,
# trazodone, vortioxetine) are intentionally excluded here so the stable-
# treatment label is a fully-observed indicator over EXACTLY this Hughes set.
# The fixed ordering below is the label column order (index i of the length-10
# outcome vector). All 10 are already in ``_DRUG_REGISTRY`` with AoU-validated
# pinned ids (no new concept-id reliance — nothing to VERIFY ON FIRST RUN beyond
# the existing per-ingredient pins).
_HUGHES_ANTIDEPRESSANTS: tuple[str, ...] = (
    "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram",
    "venlafaxine", "duloxetine",
    "amitriptyline", "nortriptyline",
    "bupropion",
)


# Names accepted by the CLI/loader. Add a new key here when adding a new
# cohort function so the registry stays the single source of truth.
SUPPORTED_COHORTS: tuple[str, ...] = (
    "first_cancer_year",
    "first_dementia_year",
    "cancer_or_dementia",
    "population_cancer",
    "population_cancer_sparse",
    "population_eds",
    "population_rare6",
    "population_sparse",
    "population_glp1",
    "mdd_antidepressant",
    "mdd_stable_treatment",
)

# Fixed salt for the general-population random-window assignment. Hashing
# person_id with a constant salt makes each person's sampled 1-year window
# deterministic and reproducible across runs (Spark's F.rand() is not
# resume-stable), while still spreading windows pseudo-uniformly across each
# person's observation history.
_RANDOM_WINDOW_SALT = 20260702


# User-facing metadata for each cohort. Consumed by the dashboard bundle
# builder (write into corpus_stats.json) so the UI's cohort selector has a
# label + description without having to duplicate this text in the
# frontend. Keep `label` short (fits in a dropdown); `description` is a
# one-paragraph blurb shown when the cohort is selected.
#
# The "full" entry is the unfiltered general-population corpus (i.e. the
# loader was called with cohort=None) and lets us treat the no-cohort
# case identically to the filtered ones for selector + metadata purposes.
COHORT_METADATA: dict[str, dict[str, str]] = {
    "full": {
        "id": "full",
        "label": "General Population (1 year windows)",
        "description": (
            "Unfiltered 1-year windows on 10% of AllOfUs condition data, "
            "no clinical inclusion or window constraint applied."
        ),
    },
    "first_cancer_year": {
        "id": "first_cancer_year",
        "label": "Cancer (1 year windows post-diagnosis)",
        "description": (
            "Patients with a first malignant-cancer diagnosis (SNOMED "
            "443392 and descendants), excluding non-melanoma skin cancer "
            "(BCC/SCC) and carcinoma in situ. The document window is the "
            "365 days starting at that first diagnosis. The follow-up window "
            "must be fully observed (365 days of post-index "
            "observation_period coverage); by default 'first' also requires "
            "365 days of prior coverage, relaxable via prior_obs_days."
        ),
    },
    "first_dementia_year": {
        "id": "first_dementia_year",
        "label": "Dementia (1 year windows post-diagnosis)",
        "description": (
            "Patients with a first all-cause dementia diagnosis (SNOMED "
            "4182210 and descendants — Alzheimer's, vascular dementia, "
            "Lewy body dementia, frontotemporal dementia, dementia NOS, "
            "mixed dementia). The document window is the 365 days "
            "starting at that first diagnosis, capturing the early-stage "
            "comorbidity cascade (delirium, falls, polypharmacy, "
            "behavioral disturbance, aspiration pneumonia). The follow-up "
            "window must be fully observed (365 days of post-index "
            "observation_period coverage); by default 'first' also requires "
            "365 days of prior coverage, relaxable via prior_obs_days."
        ),
    },
    "cancer_or_dementia": {
        "id": "cancer_or_dementia",
        "label": "Cancer or Dementia (combined, source-labeled)",
        "description": (
            "Union of the first-cancer-year and first-dementia-year cohorts, "
            "each document labeled by its source cohort. A patient qualifying "
            "for both contributes two documents (one per cohort). Used as an "
            "STM validation: a source_cohort covariate should produce strongly "
            "separable cancer vs dementia topic structure."
        ),
    },
    "population_cancer": {
        "id": "population_cancer",
        "label": "Population + Cancer (gated)",
        "description": (
            "The whole (sampled) population as a shared background, with a "
            "cancer subcohort carrying its own foreground topics. Disjoint, one "
            "document per person: patients with a first malignant-cancer "
            "diagnosis (SNOMED 443392 and descendants, excluding non-melanoma "
            "skin cancer and carcinoma in situ) get the 365-day post-diagnosis "
            "window and source_cohort='cancer'; every other person gets a "
            "deterministic random 365-day window anchored on one of their own "
            "condition-eras (so it captures real activity, not an empty calendar "
            "year) and source_cohort='general' (background-only, since 'general' "
            "has no foreground block). Trains general-population background "
            "topics against cancer-specific foreground under one gated STM."
        ),
    },
    "population_cancer_sparse": {
        "id": "population_cancer_sparse",
        "label": "Population + Cancer + Sparse (gated)",
        "description": (
            "As Population + Cancer, but the non-cancer arm is split by "
            "in-window coding density: heavily-coded years (>= 20 events) stay "
            "source_cohort='general' (background-only), while light-coder years "
            "(5-19 events) become source_cohort='sparse' — their own foreground "
            "block. Lets a gated STM surface whether light-coder general years "
            "are generic-checkup content or carry real structure (exp 0029). "
            "Three disjoint source_cohort tags: cancer, general, sparse."
        ),
    },
    "population_eds": {
        "id": "population_eds",
        "label": "Population + Ehlers-Danlos (gated)",
        "description": (
            "The whole population as a shared background, with an "
            "Ehlers-Danlos syndrome (EDS) subcohort carrying its own foreground "
            "topics — a rare-disease-on-a-background-population example. Disjoint, "
            "one document per person: persons with a first EDS diagnosis (OMOP "
            "79145 and descendants) get the 365-day post-diagnosis window and "
            "source_cohort='eds'; every other person gets a deterministic random "
            "365-day window anchored on one of their own condition-eras and "
            "source_cohort='general' (background-only). K=60 (40 shared "
            "background + 20 EDS foreground), fit on the full population as a "
            "gated block-wise correlated STM with a sex-at-birth (M/F) and age "
            "prevalence covariate."
        ),
    },
    "population_rare6": {
        "id": "population_rare6",
        "label": "Population + Rare-disease forest (gated)",
        "description": (
            "The whole population as a shared clean background, with a six-disease "
            "rare-disease foreground: Ehlers-Danlos syndrome (OMOP 79145), "
            "sarcoidosis (438688), systemic lupus erythematosus (257628), "
            "scleroderma/systemic sclerosis (40352976), myasthenia gravis (76685), "
            "and amyloidosis (432595). Disjoint, one document per person: a person "
            "with a first dx under ANY of the six anchors gets the 365-day "
            "post-diagnosis window and source_cohort='rare6'; every other person "
            "gets a deterministic random event-anchored 365-day window and "
            "source_cohort='general' (background-only). Unlike a single-disease "
            "type taxonomy, the six diseases are phenotypically distinct, so a "
            "gated placement model can both find cases against the background and "
            "route each to the right disease subtree — the rare-disease "
            "case-finding thesis in practice."
        ),
    },
    "population_sparse": {
        "id": "population_sparse",
        "label": "Population + Sparse (gated)",
        "description": (
            "The whole population split by in-window coding density — no disease "
            "anchor. Each person is windowed to one deterministic random "
            "event-anchored 365-day span, then split by event count: heavily-coded "
            "years (>= 20 events) are source_cohort='general' (background-only), "
            "light-coder years (5-19 events) become source_cohort='sparse' — their "
            "own foreground block. K=50 (40 background + 10 sparse). A gated "
            "block-wise correlated STM reads the sparse foreground topics to show "
            "what light-coder general years are made of, against a clean "
            "whole-population background."
        ),
    },
    "population_glp1": {
        "id": "population_glp1",
        "label": "Population + GLP-1 vs SGLT2i (gated)",
        "description": (
            "The whole population as a shared background, with two drug "
            "foreground arms anchored on the first year after starting a "
            "medication (incident new-user: a year of prior coverage, a "
            "fully-observed follow-up year): glp1_ra (GLP-1 receptor agonists) "
            "and sglt2i (SGLT2 inhibitors, the active comparator). Users of both "
            "are assigned to the earlier drug's arm only when that index year is "
            "monotherapy (the other drug started > 1 year away); in-window "
            "both-users are excluded. Tirzepatide (dual GIP/GLP-1) users are "
            "excluded entirely (kept out of the GLP-1 arm and the background). "
            "Documents are the conditions in that year; "
            "drugs are the anchor only. The general background carries the same "
            "1-year-prior + 1-year-follow-up observability bracket. A gated "
            "block-wise correlated STM then shows what is distinctive to each "
            "arm and its (anti-)correlations with the background comorbidity "
            "topics."
        ),
    },
    "mdd_antidepressant": {
        "id": "mdd_antidepressant",
        "label": "MDD antidepressant initiators (Hughes replication)",
        "description": (
            "Major-depression patients (SNOMED 440383 and descendants, "
            "excluding bipolar 439254 and schizoaffective 4224940) at their "
            "first antidepressant drug_era across a 15-drug set (SSRI/SNRI/TCA/"
            "atypical). Incident new-user bracket: a year of prior coverage and "
            "a fully-observed follow-up window (>= the stability horizon), plus "
            "a qualifying major-depression diagnosis on or before the index "
            "date. Unlike the topic-model cohorts this cohort returns a "
            "PER-PERSON index table (person_id, index_date, index drug + class, "
            "source_cohort) — the input to the >=90-day antidepressant-stability "
            "outcome labeler for the Hughes 'the drug worked' replication."
        ),
    },
    "mdd_stable_treatment": {
        "id": "mdd_stable_treatment",
        "label": "MDD stable antidepressant treatment (Hughes-faithful)",
        "description": (
            "The Hughes et al. (AISTATS 2018, supplement B.4) 'stable-treatment' "
            "antidepressant cohort over the 10 Hughes-aligned antidepressants "
            "(fluoxetine, sertraline, paroxetine, citalopram, escitalopram, "
            "venlafaxine, duloxetine, amitriptyline, nortriptyline, bupropion). "
            "A patient qualifies with age 18-80 at the stable-interval start, "
            ">= 1 major-depression diagnosis (SNOMED 440383 and descendants, "
            "excluding bipolar 439254 / schizoaffective 4224940), >= 2 events "
            "before their first antidepressant era, and a qualifying STABLE "
            "INTERVAL: a maximal interval whose active antidepressant SUBSET is "
            "constant, lasting >= 90 days, with encounters at least every 13 "
            "months (max visit gap <= 395 days, bounding both endpoints). The "
            "FIRST such interval defines the label — a fully-observed length-10 "
            "indicator of which drugs are in that interval's stable subset "
            "(usually one, occasionally a held combination). An add-on or switch "
            "is a boundary that splits the interval. Returns a PER-PERSON index "
            "table (person_id, index_date=stable_start, stable_end, drug_subset, "
            "source_cohort)."
        ),
    },
}


def cohort_metadata(cohort: str | None) -> dict[str, str]:
    """Return the user-facing metadata dict for a cohort name.

    ``cohort=None`` is treated as the ``"full"`` (unfiltered) cohort. An
    unknown name raises KeyError — callers should validate against
    ``SUPPORTED_COHORTS`` before calling this.
    """
    key = cohort if cohort is not None else "full"
    return COHORT_METADATA[key]


def apply_cohort(
    cond_df: DataFrame,
    cohort: str,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Dispatch on cohort name. Raises ValueError on unknown names.

    Kept as a thin registry rather than inlined in the loader so adding
    a new cohort means adding a function below + a SUPPORTED_COHORTS
    entry, without touching the loader call site.

    prior_obs_days is the per-cohort prior-observation lookback (default
    ``_WINDOW_DAYS`` = 365); see :func:`_window_observed_cohort`.
    """
    if cohort == "first_cancer_year":
        return apply_first_cancer_year_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "first_dementia_year":
        return apply_first_dementia_year_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "cancer_or_dementia":
        return apply_cancer_or_dementia_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_cancer":
        return apply_population_cancer_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_cancer_sparse":
        return apply_population_cancer_sparse_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_eds":
        return apply_population_disease_cohort(
            cond_df, disease="eds", spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_rare6":
        return apply_population_disease_cohort(
            cond_df, disease="rare6", spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_sparse":
        return apply_population_sparse_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_glp1":
        return apply_population_drug_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "mdd_antidepressant":
        return apply_mdd_antidepressant_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "mdd_stable_treatment":
        # NB: prior_obs_days is not a knob for this cohort — its observability
        # gate is "the stable interval falls within one observation period"
        # (not a fixed prior/forward bracket), so the loader's prior_obs_days is
        # intentionally not threaded through here.
        return apply_mdd_stable_treatment_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
        )
    raise ValueError(
        f"cohort {cohort!r} not supported (supported: {SUPPORTED_COHORTS})"
    )


def _window_observed_cohort(
    first_dx: DataFrame,
    observation_period: DataFrame,
    *,
    prior_obs_days: int,
    window_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Keep the (person_id, index_date) rows that are adequately observed.

    Two observation-period gates, joined against ``observation_period``:

    - **Prior lookback**: ``index_date >= observation_period_start_date +
      prior_obs_days``. At the default 365 this makes "first dx" mean "first
      with a year of prior coverage", excluding prevalent cases whose true
      first dx predates the record. ``prior_obs_days=0`` drops the lookback
      (admitting those prevalent cases); the gate then only requires the
      index to fall within an observation period at all.
    - **Follow-up**: ``index_date + window_days <=
      observation_period_end_date``, so the document window is fully observed
      (absence of a code in the window is informative, not merely unobserved).
      Independent of ``prior_obs_days``.

    Returns ``(person_id, index_date)`` for the surviving rows.
    """
    return (
        first_dx.join(observation_period, on="person_id", how="inner")
        .where(F.col("index_date") >= F.date_add(
            F.col("observation_period_start_date"), prior_obs_days))
        .where(F.date_add(F.col("index_date"), window_days)
               <= F.col("observation_period_end_date"))
        # A person may have several observation_period rows that each satisfy
        # the gates; distinct() collapses them so a survivor is one row, not one
        # per qualifying period (which would fan out duplicate documents in the
        # downstream cond_df join and over-weight multi-period patients).
        .select("person_id", "index_date")
        .distinct()
    )


def _concept_set_from_ancestors(
    ca_df: DataFrame,
    *,
    inclusion_ancestors: Sequence[int],
    exclusion_ancestors: Sequence[int] = (),
) -> DataFrame:
    """Build a concept-id set from a concept_ancestor DataFrame.

    Includes-then-excludes: (⋃ descendants of ``inclusion_ancestors``) −
    (⋃ descendants of ``exclusion_ancestors``). A concept reachable from both
    an inclusion and an exclusion ancestor is excluded. ``ca_df`` must have
    ``ancestor_concept_id`` and ``descendant_concept_id`` columns; returns a
    distinct single-column ``concept_id`` DataFrame. Predicates on
    ``ancestor_concept_id`` push down to BQ, so only the ~thousands of relevant
    concept ids materialize, not the full concept_ancestor table.
    """
    included = (
        ca_df.where(F.col("ancestor_concept_id").isin(list(inclusion_ancestors)))
        .select(F.col("descendant_concept_id").alias("concept_id"))
        .distinct()
    )
    if exclusion_ancestors:
        excluded = (
            ca_df.where(F.col("ancestor_concept_id").isin(list(exclusion_ancestors)))
            .select(F.col("descendant_concept_id").alias("concept_id"))
        )
        included = included.subtract(excluded).distinct()
    return included


def apply_first_diagnosis_year_cohort(
    cond_df: DataFrame,
    *,
    inclusion_ancestors: Sequence[int],
    exclusion_ancestors: Sequence[int] = (),
    window_days: int = _WINDOW_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Filter to persons with a first qualifying dx + a ``window_days`` window.

    Generalizes the per-disease first-year cohorts: the concept set is
    (⋃ descendants of ``inclusion_ancestors``) − (⋃ descendants of
    ``exclusion_ancestors``) via :func:`_concept_set_from_ancestors`; the
    document window is ``[index_date, index_date + window_days)``. Same
    observation-period bracketing as the cancer/dementia cohorts
    (:func:`_window_observed_cohort`, ``prior_obs_days`` lookback). Returns
    ``cond_df``'s schema, filtered.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    concepts = _concept_set_from_ancestors(
        ca,
        inclusion_ancestors=inclusion_ancestors,
        exclusion_ancestors=exclusion_ancestors,
    )

    first_dx = (
        cond_df.join(F.broadcast(concepts), on="concept_id", how="inner")
        .groupBy("person_id")
        .agg(F.min(date_col).alias("index_date"))
    )

    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_dx, op, prior_obs_days=prior_obs_days,
    )

    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )


def apply_first_cancer_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Filter to patients with a first cancer dx + 1-year follow-up window.

    Args:
        cond_df: events DataFrame from load_omop_bigquery (must have
            ``person_id``, ``concept_id``, and ``date_col``).
        spark, cdr_dataset, billing_project: same shape as
            load_omop_bigquery — needed to read concept_ancestor +
            observation_period from the same CDR.
        date_col: name of the calendar-date column on ``cond_df`` used
            both to find the first cancer dx and to bound the doc window.
            ``condition_start_date`` for condition_occurrence,
            ``condition_era_start_date`` for condition_era.

    Returns:
        A DataFrame with the same schema as ``cond_df``, filtered to
        rows where the person had a qualifying first cancer dx and the
        row's date lies in [index_date, index_date + 365d).
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    # Build the cancer concept set as (descendants of 443392) - (descendants
    # of exclusion ancestors). Predicates on ancestor_concept_id push down
    # to BQ, so this only materializes ~thousands of concept ids, not the
    # full concept_ancestor table.
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    included = (
        ca.where(F.col("ancestor_concept_id") == _CANCER_ANCESTOR)
          .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    excluded = (
        ca.where(F.col("ancestor_concept_id").isin(
            list(_CANCER_EXCLUSION_ANCESTORS),
        ))
        .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    cancer_concepts = included.subtract(excluded).distinct()

    # First cancer dx date per person. Broadcasting cancer_concepts is
    # safe — it's a few thousand integers.
    first_dx = (
        cond_df.join(F.broadcast(cancer_concepts), on="concept_id", how="inner")
               .groupBy("person_id")
               .agg(F.min(date_col).alias("index_date"))
    )

    # Observation-period gating (prior lookback + fully-observed follow-up);
    # see _window_observed_cohort. prior_obs_days controls the lookback.
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_dx, op, prior_obs_days=prior_obs_days,
    )

    # Filter the events: cohort members only, in the doc window. Not
    # broadcasting cohort_df: at AoU scale a cancer cohort can run into
    # the hundreds of thousands of persons and the planner is in a
    # better position than we are to pick the join strategy.
    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
               .where(F.col(date_col) >= F.col("index_date"))
               .where(F.col(date_col) < F.date_add(
                   F.col("index_date"), _WINDOW_DAYS,
               ))
               .drop("index_date")
    )


def apply_first_dementia_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Filter to patients with a first dementia dx + 1-year follow-up.

    Mirrors :func:`apply_first_cancer_year_cohort` but anchored on the
    SNOMED "Dementia" hierarchy with no ancestor exclusions — all-cause
    dementia is intentional (see module-level comment on _DEMENTIA_ANCESTOR).

    Args:
        cond_df: events DataFrame from load_omop_bigquery (must have
            ``person_id``, ``concept_id``, and ``date_col``).
        spark, cdr_dataset, billing_project: same shape as
            load_omop_bigquery — needed to read concept_ancestor +
            observation_period from the same CDR.
        date_col: name of the calendar-date column on ``cond_df`` used
            both to find the first dementia event and to bound the doc
            window. ``condition_start_date`` for condition_occurrence,
            ``condition_era_start_date`` for condition_era.

    Returns:
        A DataFrame with the same schema as ``cond_df``, filtered to
        rows where the person had a qualifying first dementia dx and
        the row's date lies in [index_date, index_date + 365d).
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    dementia_concepts = (
        ca.where(F.col("ancestor_concept_id") == _DEMENTIA_ANCESTOR)
          .select(F.col("descendant_concept_id").alias("concept_id"))
          .distinct()
    )
    # Exclusion subtract is a no-op for v1 (empty tuple) but kept here
    # symmetric with the cancer cohort so adding exclusions later is a
    # one-line change.
    if _DEMENTIA_EXCLUSION_ANCESTORS:
        excluded = (
            ca.where(F.col("ancestor_concept_id").isin(
                list(_DEMENTIA_EXCLUSION_ANCESTORS),
            ))
            .select(F.col("descendant_concept_id").alias("concept_id"))
        )
        dementia_concepts = dementia_concepts.subtract(excluded).distinct()

    first_event = (
        cond_df.join(F.broadcast(dementia_concepts), on="concept_id", how="inner")
               .groupBy("person_id")
               .agg(F.min(date_col).alias("index_date"))
    )

    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_event, op, prior_obs_days=prior_obs_days,
    )

    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
               .where(F.col(date_col) >= F.col("index_date"))
               .where(F.col(date_col) < F.date_add(
                   F.col("index_date"), _WINDOW_DAYS,
               ))
               .drop("index_date")
    )


def _combine_cohorts(
    cancer_events: DataFrame, dementia_events: DataFrame,
) -> DataFrame:
    """Tag each cohort's events with source_cohort and union (no dedup).

    A comorbid patient's cancer-window events (tagged "cancer") and
    dementia-window events (tagged "dementia") both survive, so they become
    two distinct documents downstream via PatientCohortDocSpec.
    """
    c = cancer_events.withColumn("source_cohort", F.lit("cancer"))
    d = dementia_events.withColumn("source_cohort", F.lit("dementia"))
    return c.unionByName(d)


def apply_cancer_or_dementia_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Combined cancer-or-dementia cohort with a source_cohort label column.

    Composes the two single-disease cohorts and unions their tagged events.
    Returns cond_df's schema plus a `source_cohort` string column. Both arms
    share the same ``prior_obs_days`` lookback.
    """
    cancer = apply_first_cancer_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    dementia = apply_first_dementia_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    return _combine_cohorts(cancer, dementia)


def _random_observed_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = 0,
) -> DataFrame:
    """Window each person to ONE deterministic random event-anchored year.

    Unlike the disease cohorts, the general population has no index event to
    anchor on. Rather than a random CALENDAR window (usually empty — bursty
    coding over long observation periods), we anchor on the person's own coding:
    :func:`_random_event_windows` picks a random fully-observed event date and
    windows the following ``window_days``. The choice is a deterministic,
    resume-stable pseudo-random pick (hash-based).

    Returns ``cond_df``'s schema, filtered to each person's sampled window.
    Persons with no event that has a fully-observed forward window are dropped.
    """
    op = (
        spark.read.format("bigquery")
        .option("table", f"{cdr_dataset}.observation_period")
        .option("parentProject", billing_project)
        .load()
        .select(
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
        )
    )
    windows = _random_event_windows(
        cond_df, op, date_col=date_col, window_days=window_days,
        prior_obs_days=prior_obs_days,
    )

    return (
        cond_df.join(windows, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )


def _random_event_windows(
    cond_df: DataFrame,
    observation_period: DataFrame,
    *,
    date_col: str,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = 0,
) -> DataFrame:
    """Anchor one deterministic random fully-observed window per person ON an event.

    A random CALENDAR window over the observation period is usually empty:
    EHR coding is bursty (clustered around real health events) while
    observation periods span years, so a random year misses the activity and the
    document is dropped by doc_min_length. Instead we anchor on the coding
    itself: for each person, consider their event dates whose forward
    ``window_days`` is fully observed (``event_date + window_days <=
    observation_period_end`` for some period covering the event), and pick ONE
    deterministically (min ``hash(person_id, event_date, salt)``, so it is a
    reproducible pseudo-random choice — Spark's ``F.rand()`` is not
    resume-stable). The window ``[index_date, index_date + window_days)`` then
    contains at least the anchoring event and its surrounding activity.

    Args:
        cond_df: events (``person_id`` + ``date_col``).
        observation_period: ``person_id`` + observation_period start/end dates.

    Returns ``(person_id, index_date)``; persons with no event that has a
    fully-observed forward window are dropped.
    """
    events = cond_df.select(
        "person_id", F.col(date_col).alias("event_date"),
    ).distinct()

    # Eligible anchors: an event whose forward window is inside SOME observation
    # period of that person (event within the period, and window end <= period
    # end). A person may have several periods; any one covering the window works.
    eligible = (
        events.join(observation_period, on="person_id", how="inner")
        .where(F.col("event_date") >= F.date_add(
            F.col("observation_period_start_date"), prior_obs_days))
        .where(
            F.date_add(F.col("event_date"), window_days)
            <= F.col("observation_period_end_date")
        )
        .select("person_id", "event_date")
        .distinct()
    )

    # Deterministic pseudo-random pick: the eligible event with the smallest
    # hash per person (ties broken by earliest date for stability).
    ranked = eligible.withColumn(
        "h", F.hash(F.col("person_id"), F.col("event_date"), F.lit(_RANDOM_WINDOW_SALT)),
    )
    chosen = ranked.withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("person_id").orderBy(
                F.col("h").asc(), F.col("event_date").asc(),
            )
        ),
    ).where(F.col("rn") == 1)

    return chosen.select("person_id", F.col("event_date").alias("index_date"))


def apply_population_disease_cohort(
    cond_df: DataFrame,
    *,
    disease: str,
    window_days: int = _WINDOW_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole-population background + a single disease foreground, disjoint.

    Generalizes :func:`apply_population_cancer_cohort` to any registered
    ``disease`` (see ``_DISEASE_REGISTRY``). One document per person:

    - disease arm — persons with a first qualifying dx (via
      :func:`apply_first_diagnosis_year_cohort` on the registry's ancestors),
      windowed to ``window_days`` post-dx, tagged ``source_cohort=<disease>``.
    - ``general`` arm — every OTHER person, on a deterministic random
      event-anchored ``window_days`` window (:func:`_random_observed_year_cohort`),
      tagged ``source_cohort='general'`` (background-only).

    ``window_days`` threads to BOTH arms so foreground and background windows
    match. Arms are disjoint by person (general = ``left_anti`` of the disease
    arm's persons). Returns ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    try:
        spec = _DISEASE_REGISTRY[disease]
    except KeyError:
        raise ValueError(
            f"disease {disease!r} not in registry "
            f"(known: {tuple(_DISEASE_REGISTRY)})"
        )

    diseased = apply_first_diagnosis_year_cohort(
        cond_df,
        inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"],
        window_days=window_days,
        spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    diseased_persons = diseased.select("person_id").distinct()

    non_diseased = cond_df.join(diseased_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_diseased, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days,
    )

    return (
        diseased.withColumn("source_cohort", F.lit(disease))
        .unionByName(general.withColumn("source_cohort", F.lit("general")))
    )


def apply_population_cancer_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole-population background + a cancer foreground subcohort, disjoint.

    One document per person, tagged with a ``source_cohort`` column:

    - **cancer** — patients with a qualifying first cancer dx (the existing
      :func:`apply_first_cancer_year_cohort`), windowed to the 365 days after
      that diagnosis. These carry the cancer foreground topics.
    - **general** — every OTHER person (no qualifying cancer dx), windowed to a
      deterministic random 365-day span anchored on one of their own
      condition-eras (:func:`_random_observed_year_cohort` →
      :func:`_random_event_windows`). ``source_cohort='general'`` is not a
      foreground group, so these documents resolve to background-only via
      :meth:`TopicBlockPartition.allowed_indices`.

    The arms are disjoint by person (the general arm is the ``left_anti`` of the
    cancer arm's persons), so no patient contributes two documents. Returns
    ``cond_df``'s schema plus a ``source_cohort`` string column.
    """
    return apply_population_disease_cohort(
        cond_df, disease="cancer", window_days=_WINDOW_DAYS,
        spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )


def _bucket_general_by_density(
    general_events: DataFrame,
    *,
    sparse_min: int = 5,
    dense_min: int = 20,
) -> DataFrame:
    """Split the windowed general arm into 'sparse' and 'general' by density.

    Tags each person's in-window events by their per-person event count:

    - ``count >= dense_min``            -> ``source_cohort='general'`` (dense;
      background-only, the usual heavily-coded general year).
    - ``sparse_min <= count < dense_min`` -> ``source_cohort='sparse'`` (a
      light-coder year, given its OWN foreground block so exp 0029 can read
      whether such years are generic-checkup content or real signal).
    - ``count < sparse_min``            -> dropped (they fall under
      ``doc_min_length`` anyway).

    The count is the raw in-window event-row count, a proxy for eventual
    document length (which is measured on post-vocab tokens in
    ``to_bow_dataframe``, so the bucketing is approximate at the vocab-filter
    boundary). Row-preserving: every kept person's event rows survive, tagged.
    """
    counts = general_events.groupBy("person_id").agg(
        F.count(F.lit(1)).alias("_n_events")
    )
    return (
        general_events.join(counts, on="person_id", how="inner")
        .withColumn(
            "source_cohort",
            F.when(F.col("_n_events") >= dense_min, F.lit("general"))
            .when(F.col("_n_events") >= sparse_min, F.lit("sparse")),
        )
        .where(F.col("source_cohort").isNotNull())
        .drop("_n_events")
    )


def apply_population_cancer_sparse_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """population_cancer + a third 'sparse' foreground for light-coder years.

    Identical to :func:`apply_population_cancer_cohort` except the general
    (non-cancer) arm is split by in-window coding density
    (:func:`_bucket_general_by_density`): heavily-coded years stay
    ``source_cohort='general'`` (background-only), while light-coder years
    (``sparse_min..dense_min-1`` events) become ``source_cohort='sparse'`` — a
    dedicated foreground block. exp 0029 reads the sparse foreground topics to
    test whether light-coder general years are generic-checkup content or carry
    real structure. Three disjoint ``source_cohort`` tags: cancer, general,
    sparse. Returns ``cond_df``'s schema plus ``source_cohort``.
    """
    cancer = apply_first_cancer_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    cancer_persons = cancer.select("person_id").distinct()

    non_cancer = cond_df.join(cancer_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_cancer, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
    )
    general_tagged = _bucket_general_by_density(general)

    return cancer.withColumn("source_cohort", F.lit("cancer")).unionByName(
        general_tagged
    )


def apply_population_sparse_cohort(
    cond_df: DataFrame,
    *,
    window_days: int = _WINDOW_DAYS,
    sparse_min: int = 5,
    dense_min: int = 20,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole population split by in-window coding density — no disease arm.

    Every person is windowed to one deterministic random event-anchored
    ``window_days`` span (:func:`_random_observed_year_cohort`), then split by
    per-person in-window event count (:func:`_bucket_general_by_density`):
    ``>= dense_min`` -> ``source_cohort='general'`` (dense background),
    ``sparse_min..dense_min-1`` -> ``source_cohort='sparse'`` (a light-coder
    foreground block), ``< sparse_min`` dropped. Unlike
    :func:`apply_population_cancer_sparse_cohort`, there is NO disease foreground
    — cancer/EDS patients simply fold into the population, giving a clean
    whole-population reference for reading what light-coder years contain.

    ``prior_obs_days`` is accepted for a uniform :func:`apply_cohort` signature
    but unused (there is no disease index event to be "first" of). Returns
    ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    general = _random_observed_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days,
    )
    return _bucket_general_by_density(
        general, sparse_min=sparse_min, dense_min=dense_min,
    )


def _ingredient_concept_ids(
    concept_df: DataFrame, ingredient_names: Sequence[str],
    *, extra_concept_ids: Sequence[int] = (),
) -> DataFrame:
    """Resolve seed drug concept_ids by RxNorm Ingredient name + explicit pins.

    ``concept_df`` must have ``concept_id``, ``concept_name``, ``vocabulary_id``,
    ``concept_class_id``. Returns the distinct ``concept_id`` of standard RxNorm
    ingredients whose name matches ``ingredient_names``, UNION any
    ``extra_concept_ids`` pinned explicitly. The pins are included even when they
    do not match the name/class filter — a robustness hatch for newer drugs
    whose vocab naming/classing may not match the expected Ingredient name (e.g.
    tirzepatide). These are SEED ids; expand them to a full class set with
    :func:`_expand_descendants` before matching ``drug_era``.
    """
    names_lower = [n.lower() for n in ingredient_names]
    by_name = (
        concept_df
        .where(F.col("vocabulary_id") == "RxNorm")
        .where(F.col("concept_class_id") == "Ingredient")
        .where(F.lower(F.col("concept_name")).isin(names_lower))
        .select("concept_id")
    )
    if extra_concept_ids:
        extra = concept_df.sparkSession.createDataFrame(
            [(int(c),) for c in extra_concept_ids], ["concept_id"],
        )
        by_name = by_name.unionByName(extra)
    return by_name.distinct()


def _expand_descendants(ca_df: DataFrame, seed_ids: DataFrame) -> DataFrame:
    """All descendants (including the seeds themselves) of ``seed_ids`` via
    ``concept_ancestor``.

    ``ca_df`` has ``ancestor_concept_id``, ``descendant_concept_id``;
    ``seed_ids`` is a single-column ``concept_id`` DataFrame. Returns the
    distinct ``concept_id`` set = seeds ∪ their descendants. ``drug_era`` is
    ingredient-level so the ingredient (a self-descendant) matches directly;
    expanding to descendants also captures any clinical-drug-level rows and
    guards against incomplete ingredient rollup for newer drugs. Self is
    unioned back in case ``concept_ancestor`` lacks a self-row for a seed.
    """
    descendants = (
        ca_df.join(
            seed_ids,
            ca_df.ancestor_concept_id == seed_ids.concept_id,
            how="inner",
        )
        .select(ca_df.descendant_concept_id.alias("concept_id"))
    )
    return descendants.unionByName(seed_ids.select("concept_id")).distinct()


def _first_drug_era_dates(
    drug_era_df: DataFrame, ingredient_concept_ids: DataFrame,
) -> DataFrame:
    """Earliest drug_era start per person among the given ingredient concept_ids.

    ``drug_era_df`` has ``person_id``, ``drug_concept_id``,
    ``drug_era_start_date``. Returns ``(person_id, index_date)`` — the person's
    first exposure to the class. Persons with no matching era are absent.
    """
    return (
        drug_era_df.join(
            F.broadcast(ingredient_concept_ids),
            drug_era_df.drug_concept_id == ingredient_concept_ids.concept_id,
            how="inner",
        )
        .groupBy("person_id")
        .agg(F.min("drug_era_start_date").alias("index_date"))
    )


def _assign_drug_groups(
    g: DataFrame, s: DataFrame, t: DataFrame,
    *, window_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Assign each drug-exposed person to exactly one foreground group.

    Inputs are per-class first-era dates ``(person_id, index_date)`` for
    glp1_ra (``g``), sglt2i (``s``), tirzepatide (``t``). Two drug arms
    (glp1_ra, sglt2i); tirzepatide is resolved only to EXCLUDE its users:

    - has tirzepatide                    -> EXCLUDED (dropped; kept out of the
      single arms AND, via the caller's left_anti on all tracked drug persons,
      out of the general background: a dual GIP/GLP-1 patient is neither a pure
      GLP-1-RA new-user nor an untreated background person)
    - has glp1_ra AND sglt2i, no tirzepatide:
        - ``|g - s| > window_days`` -> the EARLIER drug's single arm (the second
          drug starts OUTSIDE the index year, so that year is genuine
          monotherapy; index = earlier of g, s)
        - ``|g - s| <= window_days`` -> EXCLUDED (the second drug starts INSIDE
          the index year, contaminating it; no combination-therapy arm)
    - has glp1_ra only                   -> ``glp1_ra``      (index = g)
    - has sglt2i only                    -> ``sglt2i``       (index = s)

    Returns ``(person_id, source_cohort, index_date)`` for the two arms only.
    """
    g2 = g.select("person_id", F.col("index_date").alias("g_date"))
    s2 = s.select("person_id", F.col("index_date").alias("s_date"))
    t2 = t.select("person_id", F.col("index_date").alias("t_date"))
    joined = (
        g2.join(s2, on="person_id", how="full_outer")
          .join(t2, on="person_id", how="full_outer")
    )
    has_g = F.col("g_date").isNotNull()
    has_s = F.col("s_date").isNotNull()
    has_t = F.col("t_date").isNotNull()
    gap = F.abs(F.datediff(F.col("g_date"), F.col("s_date")))
    earlier_index = F.least(F.col("g_date"), F.col("s_date"))
    # For a long-gap both-user, the arm is whichever drug was initiated first.
    earlier_arm = F.when(F.col("g_date") <= F.col("s_date"), F.lit("glp1_ra")) \
                   .otherwise(F.lit("sglt2i"))

    source = (
        F.when(has_t, F.lit(None))                 # tirzepatide user -> excluded
        .when(has_g & has_s & (gap > window_days), earlier_arm)  # clean monotherapy year
        .when(has_g & has_s, F.lit(None))          # in-window both-user -> excluded
        .when(has_g, F.lit("glp1_ra"))
        .when(has_s, F.lit("sglt2i"))
        .otherwise(F.lit(None))
    )
    index_date = (
        F.when(has_t, F.lit(None))
        .when(has_g & has_s & (gap > window_days), earlier_index)
        .when(has_g & has_s, F.lit(None))
        .when(has_g, F.col("g_date"))
        .when(has_s, F.col("s_date"))
        .otherwise(F.lit(None))
    )
    return (
        joined.withColumn("source_cohort", source)
        .withColumn("index_date", index_date)
        .where(F.col("source_cohort").isNotNull())
        .select("person_id", "source_cohort", "index_date")
    )


def _coinitiation_gap_histogram(g: DataFrame, s: DataFrame) -> DataFrame:
    """Bucketed |g - s| gap counts for persons who are new-users of BOTH
    glp1_ra and sglt2i. A no-fit diagnostic on the both-user population: how many
    fall > 1 year apart (recovered to a single arm) vs in-window (excluded).
    Returns ``(bucket, n)``.
    """
    both = (
        g.select("person_id", F.col("index_date").alias("g_date"))
        .join(s.select("person_id", F.col("index_date").alias("s_date")),
              on="person_id", how="inner")
    )
    gap = F.abs(F.datediff(F.col("g_date"), F.col("s_date")))
    bucket = (
        F.when(gap <= 7, "0-7")
        .when(gap <= 30, "8-30")
        .when(gap <= 90, "31-90")
        .when(gap <= 180, "91-180")
        .when(gap <= 365, "181-365")
        .otherwise("366+")
    )
    return (
        both.withColumn("bucket", bucket)
        .groupBy("bucket")
        .agg(F.count(F.lit(1)).alias("n"))
    )


def apply_population_drug_cohort(
    cond_df: DataFrame,
    *,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = _WINDOW_DAYS,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
) -> DataFrame:
    """Whole-population background + two drug foreground arms, disjoint.

    Anchors on ``drug_era``: each per-class set is the descendants
    (:func:`_expand_descendants`) of its seed ingredients (resolved by name +
    optional pinned ids via :func:`_ingredient_concept_ids`), and the first
    matching era per person gives that class's index date. These are partitioned
    by :func:`_assign_drug_groups` into glp1_ra and sglt2i (a both-GLP1+SGLT2i
    user goes to the earlier drug's arm only when the two starts are >
    ``window_days`` apart, else is excluded). Tirzepatide is resolved only to
    exclude its users (from the arms and, via the general left_anti, the
    background); there is no tirzepatide or combination arm.
    Chosen index dates are new-user-bracketed
    (:func:`_window_observed_cohort`: ``prior_obs_days`` prior coverage + observed
    ``window_days`` follow-up). The ``general`` arm is every person with NO
    tracked drug exposure, windowed to a random observed year with the SAME
    prior+forward bracket (:func:`_random_observed_year_cohort`). Documents are
    the person's condition rows in ``[index_date, index_date + window_days)``.
    Returns ``cond_df``'s schema plus a ``source_cohort`` column.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    concept = _read("concept").select(
        "concept_id", "concept_name", "vocabulary_id", "concept_class_id",
    )
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    drug_era = _read("drug_era").select(
        "person_id", "drug_concept_id", "drug_era_start_date",
    )
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )

    def _first_dates(class_key: str) -> DataFrame:
        spec = _DRUG_REGISTRY[class_key]
        seeds = _ingredient_concept_ids(
            concept, spec["ingredient_names"],
            extra_concept_ids=spec.get("seed_concept_ids", ()),
        )
        # cache: concept_set is scanned twice (count + the era join), dates is
        # reused downstream (partition + gap histogram + general left_anti) —
        # avoids re-scanning concept_ancestor / drug_era each time.
        concept_set = _expand_descendants(ca, seeds).cache()
        dates = _first_drug_era_dates(drug_era, concept_set).cache()
        # Loud resolution: log BOTH the concept-set size AND the resulting person
        # count. A thin/empty concept set is a mis-spec (wrong or non-standard
        # seed id — a non-standard pin has no concept_ancestor hierarchy, so the
        # set collapses to just itself); zero persons despite a plausible set
        # means the seed doesn't match how drug_era tags the drug. Either is a
        # definition bug, not a rare drug.
        print(f"[cohort population_glp1] {class_key}: resolved "
              f"{concept_set.count()} drug concept(s), "
              f"{dates.count()} persons with a first era", flush=True)
        return dates

    g = _first_dates("glp1_ra")
    s = _first_dates("sglt2i")
    t = _first_dates("tirzepatide")

    # Build-time diagnostic: both-user |g-s| gap distribution. Buckets past
    # window_days (> 365d, the "366+" bucket) are recovered into a single arm;
    # the rest are excluded. Ascending gap order (not lexicographic).
    _gap_bucket_order = ("0-7", "8-30", "31-90", "91-180", "181-365", "366+")
    print("[cohort population_glp1] GLP-1/SGLT2i both-user |g-s| gap histogram "
          f"(> {window_days}d -> single arm, else excluded):", flush=True)
    _gap_counts = {
        r["bucket"]: r["n"] for r in _coinitiation_gap_histogram(g, s).collect()
    }
    for bucket in _gap_bucket_order:
        print(f"[cohort population_glp1]   {bucket}: {_gap_counts.get(bucket, 0)}",
              flush=True)

    assigned = _assign_drug_groups(g, s, t, window_days=window_days)

    # New-user observation bracket on the chosen index; rejoin to recover the tag.
    bracketed = _window_observed_cohort(
        assigned.select("person_id", "index_date"), op,
        prior_obs_days=prior_obs_days, window_days=window_days,
    )
    drug_docs = assigned.join(bracketed, on=["person_id", "index_date"], how="inner")
    drug_windows = (
        cond_df.join(drug_docs, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )

    # general = persons with NO tracked drug exposure at all (excluded both-users
    # and inadequately-observed initiators are NOT background).
    drug_persons = (
        g.select("person_id")
        .unionByName(s.select("person_id"))
        .unionByName(t.select("person_id"))
        .distinct()
    )
    non_drug = cond_df.join(drug_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_drug, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        window_days=window_days, prior_obs_days=prior_obs_days,
    ).withColumn("source_cohort", F.lit("general"))

    return drug_windows.unionByName(general)


def lookback_feature_label_events(events_df, index_df, *, date_col,
                                  lookback_days, label_window_days):
    """Split raw events into a pre-index feature frame and a forward label frame.

    `index_df` = (person_id, index_date, source_cohort). Feature frame = events in
    [index_date - lookback_days, index_date); label frame = events in
    [index_date, index_date + label_window_days). Each frame carries source_cohort
    (index_date dropped). Events only occur within observation periods, so the
    lookback naturally yields the available history (up to lookback_days); the
    >=1yr-prior observation requirement is enforced upstream in the index table.
    """
    joined = events_df.join(F.broadcast(index_df), on="person_id", how="inner")
    feature = (joined
               .where(F.col(date_col) < F.col("index_date"))
               .where(F.col(date_col) >= F.date_sub(F.col("index_date"), lookback_days))
               .drop("index_date"))
    label = (joined
             .where(F.col(date_col) >= F.col("index_date"))
             .where(F.col(date_col) < F.date_add(F.col("index_date"), label_window_days))
             .drop("index_date"))
    return feature, label


def case_finding_index_table(cond_df, *, disease, spark, cdr_dataset,
                             billing_project, date_col, prior_obs_days=365,
                             label_window_days=_WINDOW_DAYS):
    """(person_id, index_date, source_cohort) for the disease + general arms.

    Foreground: first qualifying dx (min over the disease's inclusion-minus-
    exclusion descendants), gated by _window_observed_cohort so the symmetric
    bracket [index - prior_obs_days, index + label_window_days) is observed.
    Background: _random_event_windows over everyone else, same gate. No windowing
    of events here — just the gated index per person; lookback_feature_label_events
    does the windowing. Reuses the same helpers as the forward cohorts."""
    spec = _DISEASE_REGISTRY[disease]

    def _read(table):
        return (spark.read.format("bigquery")
                .option("table", f"{cdr_dataset}.{table}")
                .option("parentProject", billing_project).load())

    ca = _read("concept_ancestor").select("ancestor_concept_id", "descendant_concept_id")
    concepts = _concept_set_from_ancestors(
        ca, inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"])
    first_dx = (cond_df.join(F.broadcast(concepts), on="concept_id", how="inner")
                .groupBy("person_id").agg(F.min(date_col).alias("index_date")))
    op = _read("observation_period").select(
        "person_id", "observation_period_start_date", "observation_period_end_date")

    fg = (_window_observed_cohort(first_dx, op, prior_obs_days=prior_obs_days,
                                  window_days=label_window_days)
          .withColumn("source_cohort", F.lit(disease)))
    non = cond_df.join(fg.select("person_id").distinct(), on="person_id", how="left_anti")
    bg = (_random_event_windows(non, op, date_col=date_col,
                                window_days=label_window_days, prior_obs_days=prior_obs_days)
          .withColumn("source_cohort", F.lit("general")))
    return fg.unionByName(bg)


# --- Antidepressant initiation index + >=90-day stability outcome (Phase C) ---
# The Hughes "antidepressant PC replication" core, decision-independent: a
# per-drug incident-new-user index over a 15-drug antidepressant set restricted
# to a major-depression sub-population, and a pure per-drug >=90-day continuation
# ("the drug worked") outcome labeler. Both are built from the same
# ingredient/class concept map so the index DRUG and a downstream SWITCH are
# defined against one authoritative set. No driver / BQ->memory bridge here —
# these are the composable pieces a future driver wires together.


def _antidepressant_concept_map(
    concept_df: DataFrame,
    ca_df: DataFrame,
    *,
    ingredients: Sequence[str] = _ANTIDEPRESSANT_INGREDIENTS,
) -> DataFrame:
    """Map every antidepressant drug concept_id -> its ingredient name + class.

    For each registered antidepressant ``ingredient`` (a ``_DRUG_REGISTRY`` key),
    resolve its seed concept id by RxNorm Ingredient name with a pinned-id
    fallback (:func:`_ingredient_concept_ids`) and expand to the full descendant
    set (:func:`_expand_descendants`), tagging every resulting concept_id with
    the ingredient's registry name and ``class`` tag. The union over all
    ``ingredients`` is the single lookup the index + outcome steps join
    ``drug_era`` against: it both DEFINES the 15-drug antidepressant set and
    carries the ingredient/class identity needed to record which drug a person
    started and to tell a same-ingredient refill from a switch to a DIFFERENT
    ingredient.

    ``concept_df`` needs concept_id/concept_name/vocabulary_id/concept_class_id;
    ``ca_df`` needs ancestor_concept_id/descendant_concept_id. Returns a distinct
    ``(concept_id, drug_name, drug_class)``. Distinct ingredients have disjoint
    descendant sets, so a concept_id maps to exactly one ingredient in practice;
    ``distinct()`` keeps the frame a set but does not arbitrate genuine overlap
    (none is expected across these 15 mono-ingredient sets).
    """
    mapped: DataFrame | None = None
    for name in ingredients:
        spec = _DRUG_REGISTRY[name]
        seeds = _ingredient_concept_ids(
            concept_df, spec["ingredient_names"],
            extra_concept_ids=spec.get("seed_concept_ids", ()),
        )
        tagged = (
            _expand_descendants(ca_df, seeds)
            .withColumn("drug_name", F.lit(name))
            .withColumn("drug_class", F.lit(spec["class"]))
        )
        mapped = tagged if mapped is None else mapped.unionByName(tagged)
    return mapped.distinct()


def _first_antidepressant_index(
    drug_era_df: DataFrame,
    concept_map: DataFrame,
) -> DataFrame:
    """Per person: the first antidepressant era + which ingredient it was.

    ``drug_era_df`` has person_id/drug_concept_id/drug_era_start_date;
    ``concept_map`` is ``(concept_id, drug_name, drug_class)`` from
    :func:`_antidepressant_concept_map`. The index DATE is the earliest
    antidepressant ``drug_era_start_date`` across the whole 15-drug set (reusing
    :func:`_first_drug_era_dates`); the index DRUG is the ingredient whose era
    starts on that date. Same-day co-initiation of two ingredients (a genuine
    tie) is broken deterministically by lowest ``drug_concept_id`` so the index
    is single-valued and resume-stable. Returns ``(person_id, index_date,
    index_drug_concept_id, index_drug_name, index_drug_class)``; persons with no
    antidepressant era are absent.
    """
    ad_concepts = concept_map.select("concept_id").distinct()
    index_dates = _first_drug_era_dates(drug_era_df, ad_concepts)

    # Recover the ingredient whose era starts ON the index date. Joining on
    # person_id then filtering start == index_date, then an inner join to
    # concept_map, keeps only the antidepressant era(s) that opened the index.
    starts = (
        drug_era_df.join(index_dates, on="person_id", how="inner")
        .where(F.col("drug_era_start_date") == F.col("index_date"))
        .join(
            F.broadcast(concept_map),
            drug_era_df["drug_concept_id"] == concept_map["concept_id"],
            how="inner",
        )
    )
    ranked = starts.withColumn(
        "_rn",
        F.row_number().over(
            Window.partitionBy("person_id").orderBy(F.col("drug_concept_id").asc())
        ),
    )
    return (
        ranked.where(F.col("_rn") == 1).select(
            "person_id",
            "index_date",
            F.col("drug_concept_id").alias("index_drug_concept_id"),
            F.col("drug_name").alias("index_drug_name"),
            F.col("drug_class").alias("index_drug_class"),
        )
    )


def _mdd_antidepressant_index(
    cond_df: DataFrame,
    drug_era_df: DataFrame,
    observation_period: DataFrame,
    concept_map: DataFrame,
    mdd_concepts: DataFrame,
    *,
    date_col: str,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Pure core of the MDD antidepressant-initiator index (no BQ reads).

    Composes the reusable primitives so the whole index is unit-testable on
    synthetic frames:

    1. First antidepressant era + its ingredient/class per person
       (:func:`_first_antidepressant_index`).
    2. Incident-new-user bracket (:func:`_window_observed_cohort`):
       ``prior_obs_days`` of prior coverage AND a fully-observed ``window_days``
       follow-up. ``window_days`` must be >= the outcome ``stability_days`` so the
       stability window is guaranteed observed (the labeler then treats every
       member as uncensored). Survivors are rejoined to recover the drug fields.
    3. **Major-depression indication rule** (the tunable temporal choice): keep a
       person only if they have a qualifying MDD condition (``mdd_concepts``, the
       inclusion-minus-exclusion descendant set) dated **on or before the index
       date** — i.e. the antidepressant is being started against an already-coded
       depression indication. This is a semi-join (no fan-out). Debatable knob:
       one could instead require the MDD dx within a bounded pre-index window
       (e.g. 365d) or allow a short post-index grace (dx coded at the prescribing
       visit); "any MDD dx up to index" is the most permissive faithful rule and
       maximises the replication N.

    ``cond_df`` has person_id/concept_id/``date_col``. Returns ``(person_id,
    index_date, index_drug_concept_id, index_drug_name, index_drug_class,
    source_cohort='mdd_antidepressant')``, one row per surviving person.
    """
    index = _first_antidepressant_index(drug_era_df, concept_map)

    bracketed = _window_observed_cohort(
        index.select("person_id", "index_date"), observation_period,
        prior_obs_days=prior_obs_days, window_days=window_days,
    )
    index = index.join(bracketed, on=["person_id", "index_date"], how="inner")

    # MDD indication on/before index: semi-join over persons whose earliest
    # qualifying MDD dx precedes (or equals) their index date.
    mdd_events = (
        cond_df.join(F.broadcast(mdd_concepts), on="concept_id", how="inner")
        .select("person_id", F.col(date_col).alias("_mdd_date"))
    )
    qualifying = (
        mdd_events.join(index.select("person_id", "index_date"),
                        on="person_id", how="inner")
        .where(F.col("_mdd_date") <= F.col("index_date"))
        .select("person_id")
        .distinct()
    )

    return (
        index.join(qualifying, on="person_id", how="inner")
        .withColumn("source_cohort", F.lit("mdd_antidepressant"))
        .select(
            "person_id", "index_date", "index_drug_concept_id",
            "index_drug_name", "index_drug_class", "source_cohort",
        )
    )


def apply_mdd_antidepressant_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    window_days: int = _WINDOW_DAYS,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """MDD antidepressant-initiator index table (Hughes Phase C replication).

    Reads the CDR (concept, concept_ancestor, drug_era, observation_period),
    builds the 15-drug antidepressant concept map + the MDD inclusion-minus-
    exclusion condition set, and delegates the index logic to the pure
    :func:`_mdd_antidepressant_index`. ``drug_era`` is projected with
    ``drug_era_end_date`` and ``gap_days`` (beyond the start used for the index)
    so the SAME read feeds :func:`antidepressant_stability_label` downstream.

    Unlike the topic-model cohorts (which return a windowed events frame), this
    returns a PER-PERSON index table ``(person_id, index_date,
    index_drug_concept_id, index_drug_name, index_drug_class, source_cohort)`` —
    the direct input to the >=90-day stability outcome labeler. ``window_days``
    (the observed follow-up requirement) must be >= the labeler's
    ``stability_days``; it defaults to a full observed year.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    concept = _read("concept").select(
        "concept_id", "concept_name", "vocabulary_id", "concept_class_id",
    )
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    # Project end_date + gap_days too (not just start): the outcome labeler needs
    # drug_era_end_date to measure continuous coverage.
    drug_era = _read("drug_era").select(
        "person_id", "drug_concept_id", "drug_era_start_date",
        "drug_era_end_date", "gap_days",
    )
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )

    concept_map = _antidepressant_concept_map(concept, ca)
    spec = _DISEASE_REGISTRY["mdd"]
    mdd_concepts = _concept_set_from_ancestors(
        ca,
        inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"],
    )
    return _mdd_antidepressant_index(
        cond_df, drug_era, op, concept_map, mdd_concepts,
        date_col=date_col, window_days=window_days, prior_obs_days=prior_obs_days,
    )


def antidepressant_stability_label(
    drug_era_df: DataFrame,
    index_df: DataFrame,
    *,
    drug_concept_sets: DataFrame,
    stability_days: int = 90,
    grace_gap_days: int = 30,
) -> DataFrame:
    """Per-drug >=90-day antidepressant-stability outcome ("the drug worked").

    A PURE transform (no BQ reads) implementing a Hughes-style "sustained
    continuation" definition. ``index_df`` is the MDD initiator index
    (``person_id``, ``index_date``, ``index_drug_name`` at least);
    ``drug_era_df`` has ``person_id``, ``drug_concept_id``,
    ``drug_era_start_date``, ``drug_era_end_date``; ``drug_concept_sets`` is the
    ``(concept_id, drug_name, ...)`` antidepressant map from
    :func:`_antidepressant_concept_map` — it both restricts eras to the 15-drug
    set and names each era's ingredient (so a switch to a DIFFERENT ingredient is
    detectable). Returns one row per index person: ``(person_id,
    index_drug_name, worked)`` with ``worked`` a boolean.

    Definitional choices (all tunable; the debatable ones are flagged):

    - **Positive (worked=True)**: the INDEX ingredient has continuous coverage
      spanning >= ``stability_days`` from ``index_date``, where consecutive eras
      of the SAME ingredient are stitched when the gap from one era's end to the
      next era's start is <= ``grace_gap_days`` (a gap-and-islands stitch on the
      island anchored at ``index_date``). Coverage span is
      ``datediff(coverage_end, index_date)`` — a calendar-day span; the +/-1-day
      inclusive/exclusive convention is a (minor, debatable) choice.
    - **Negative (worked=False)**: index-ingredient coverage ends before
      ``stability_days`` (discontinuation / a gap > ``grace_gap_days`` before
      reaching it), OR the person initiates a DIFFERENT antidepressant ingredient
      within ``[index_date, index_date + stability_days)`` (**switch = failure**,
      overriding an otherwise-sufficient index coverage — debatable: some
      definitions score a switch-then-stable as success). Switching is keyed on
      INGREDIENT, so a dose/formulation change of the same ingredient is a refill,
      not a switch.
    - **Uncensored assumption**: the cohort's :func:`_window_observed_cohort`
      follow-up gate (``window_days`` >= ``stability_days``) guarantees the
      stability window is observed, so every ``index_df`` member is treated as
      uncensored — absence of a continuing era is a real discontinuation, not an
      unobserved one. This function depends on that upstream bracket.

    Defaults: ``stability_days=90`` (Hughes ~3-month sustained continuation),
    ``grace_gap_days=30`` (permissible refill gap; 30/45/60 are all defensible).
    """
    cm = drug_concept_sets.select(
        F.col("concept_id").alias("_cm_concept_id"),
        F.col("drug_name").alias("_era_drug_name"),
    )
    eras = (
        drug_era_df.join(
            F.broadcast(cm),
            drug_era_df["drug_concept_id"] == F.col("_cm_concept_id"),
            how="inner",
        )
        .join(
            index_df.select("person_id", "index_date", "index_drug_name"),
            on="person_id", how="inner",
        )
        .select(
            "person_id", "index_date", "index_drug_name", "_era_drug_name",
            "drug_era_start_date", "drug_era_end_date",
        )
    )

    # Continuous coverage of the INDEX ingredient from index_date, via a
    # gap-and-islands stitch. Island 1 is the run anchored at index_date (the
    # earliest index-ingredient era starts exactly at index_date, since
    # index_date is the person's first antidepressant era start).
    idx_eras = eras.where(
        (F.col("_era_drug_name") == F.col("index_drug_name"))
        & (F.col("drug_era_end_date") >= F.col("index_date"))
    )
    w = Window.partitionBy("person_id").orderBy("drug_era_start_date")
    prev_max_end = F.max("drug_era_end_date").over(
        w.rowsBetween(Window.unboundedPreceding, -1)
    )
    is_break = prev_max_end.isNull() | (
        F.col("drug_era_start_date") > F.date_add(prev_max_end, grace_gap_days)
    )
    islands = idx_eras.withColumn("_break", is_break.cast("int"))
    islands = islands.withColumn(
        "_island",
        F.sum("_break").over(w.rowsBetween(Window.unboundedPreceding, 0)),
    )
    coverage = (
        islands.where(F.col("_island") == 1)
        .groupBy("person_id", "index_date")
        .agg(F.max("drug_era_end_date").alias("_coverage_end"))
        .withColumn(
            "_covered",
            F.datediff(F.col("_coverage_end"), F.col("index_date"))
            >= F.lit(stability_days),
        )
        .select("person_id", "_covered")
    )

    # Switch = a DIFFERENT antidepressant ingredient initiated within the window.
    switched = (
        eras.where(
            (F.col("_era_drug_name") != F.col("index_drug_name"))
            & (F.col("drug_era_start_date") >= F.col("index_date"))
            & (F.col("drug_era_start_date")
               < F.date_add(F.col("index_date"), stability_days))
        )
        .select("person_id")
        .distinct()
        .withColumn("_switched", F.lit(True))
    )

    # Every cohort member is uncensored: a member with no qualifying coverage
    # row (never reached the island / discontinued immediately) is worked=False.
    return (
        index_df.select("person_id", "index_drug_name")
        .join(coverage, on="person_id", how="left")
        .join(switched, on="person_id", how="left")
        .withColumn("_covered", F.coalesce(F.col("_covered"), F.lit(False)))
        .withColumn("_switched", F.coalesce(F.col("_switched"), F.lit(False)))
        .withColumn("worked", F.col("_covered") & (~F.col("_switched")))
        .select("person_id", "index_drug_name", "worked")
    )


# --- Hughes-faithful "stable-treatment" antidepressant cohort/outcome ---------
# A DISTINCT cohort from the per-index-drug ``mdd_antidepressant`` one above: it
# replaces "which drug did a person INITIATE + did that drug work" with the
# Hughes et al. (AISTATS 2018, supplement B.4) "stable treatment" construction —
# WHICH antidepressant SUBSET a person was stably held on over their first
# qualifying stable interval. Defined over the 10 Hughes-aligned antidepressants
# (``_HUGHES_ANTIDEPRESSANTS``, a strict subset of the 15-drug set), it turns a
# person's antidepressant drug_eras into a sequence of maximal constant-subset
# intervals, keeps those that are both long enough (>= 90 days) and regularly
# encountered (a visit at least every 13 months), and labels the FIRST such
# interval with a fully-observed length-10 indicator over the fixed drug order.
# All cores are pure (no BQ reads) and unit-testable on synthetic Spark frames;
# a single BQ wrapper reads the CDR and delegates. The existing
# ``mdd_antidepressant`` index / stability outcome above are left INTACT.


def _antidepressant_era_subsets(
    drug_era_df: DataFrame,
    concept_map: DataFrame,
) -> DataFrame:
    """Restrict ``drug_era`` to the antidepressant set and name each era's drug.

    ``drug_era_df`` has person_id/drug_concept_id/drug_era_start_date/
    drug_era_end_date; ``concept_map`` is ``(concept_id, drug_name, drug_class)``
    from :func:`_antidepressant_concept_map` (built with
    ``_HUGHES_ANTIDEPRESSANTS`` for this cohort, so the set is the 10 Hughes
    drugs). Inner-joins the eras to the map, restricting to those 10 ingredients
    and tagging every era with its ingredient name. Returns ``(person_id,
    drug_name, era_start, era_end)`` — the raw exposure spans the sweep-line
    interval builder consumes. One row per antidepressant era; a person's non-
    antidepressant eras (and any non-Hughes antidepressant) drop out here.
    """
    cm = concept_map.select(
        F.col("concept_id").alias("_cm_id"),
        F.col("drug_name").alias("drug_name"),
    )
    return (
        drug_era_df.join(
            F.broadcast(cm),
            drug_era_df["drug_concept_id"] == F.col("_cm_id"),
            how="inner",
        )
        .select(
            "person_id",
            "drug_name",
            F.col("drug_era_start_date").alias("era_start"),
            F.col("drug_era_end_date").alias("era_end"),
        )
    )


def _stable_drug_intervals(
    era_subsets: DataFrame,
    *,
    min_days: int = 90,
) -> DataFrame:
    """Maximal constant-subset antidepressant intervals of >= ``min_days``.

    A sweep-line over the antidepressant era spans (``era_subsets`` from
    :func:`_antidepressant_era_subsets`). The active drug SUBSET changes only at
    an era boundary, so the elementary segments are the gaps between consecutive
    boundary points, where a boundary is either an ``era_start`` (a drug turns
    on) or ``era_end + 1`` (a drug turns off — ``era_end`` is inclusive). For
    each segment ``[seg_start, seg_end)`` the active subset is the drugs whose
    era covers ``seg_start`` (``era_start <= seg_start <= era_end``), collected
    via a range-join into ``sort_array(collect_set(drug_name))``; a segment with
    NO active antidepressant (an off-all-ADs gap) yields the empty subset.

    Consecutive same-subset segments are then merged with a gap-and-islands
    window (break when the sorted-subset key changes from the previous segment —
    so an add-on ``{A} -> {A,B}`` splits, a switch ``{A} -> {B}`` splits, and an
    empty off-all-ADs segment both breaks the run and is dropped). Each island
    collapses to ``(min seg_start, max seg_end)``; the reported ``interval_end``
    is ``max(seg_end) - 1`` (back to the last inclusive covered day). Kept iff
    the subset is non-empty and ``datediff(interval_end, interval_start) >=
    min_days``.

    Returns ``(person_id, interval_start, interval_end, drug_subset)`` with
    ``drug_subset`` a sorted ``array<string>`` of ingredient names.
    """
    es = era_subsets
    # Boundary points per person: era starts (drug on) and era_end+1 (drug off).
    starts = es.select("person_id", F.col("era_start").alias("pt"))
    ends = es.select("person_id", F.date_add(F.col("era_end"), 1).alias("pt"))
    points = starts.unionByName(ends).distinct()

    w_pt = Window.partitionBy("person_id").orderBy("pt")
    segs = (
        points.withColumn("seg_end", F.lead("pt").over(w_pt))
        .withColumnRenamed("pt", "seg_start")
        # The last boundary opens no segment (nothing is active past it).
        .where(F.col("seg_end").isNotNull())
    )

    # Active subset per segment: LEFT range-join so off-all-ADs segments survive
    # with an empty subset (collect_set drops the null drug_name -> []).
    s = segs.alias("s")
    e = es.alias("e")
    seg_subsets = (
        s.join(
            e,
            (F.col("s.person_id") == F.col("e.person_id"))
            & (F.col("s.seg_start") >= F.col("e.era_start"))
            & (F.col("s.seg_start") <= F.col("e.era_end")),
            how="left",
        )
        .select(
            F.col("s.person_id").alias("person_id"),
            F.col("s.seg_start").alias("seg_start"),
            F.col("s.seg_end").alias("seg_end"),
            F.col("e.drug_name").alias("drug_name"),
        )
        .groupBy("person_id", "seg_start", "seg_end")
        .agg(F.sort_array(F.collect_set("drug_name")).alias("drug_subset"))
    )

    # Gap-and-islands: a new island starts whenever the sorted-subset key differs
    # from the previous segment's (subset change OR an empty-subset segment,
    # whose "" key differs from any non-empty neighbour). Same window idiom as
    # antidepressant_stability_label's coverage stitch.
    w_seg = Window.partitionBy("person_id").orderBy("seg_start")
    keyed = seg_subsets.withColumn("_key", F.concat_ws(",", F.col("drug_subset")))
    keyed = keyed.withColumn("_prev", F.lag("_key").over(w_seg))
    keyed = keyed.withColumn(
        "_break",
        (F.col("_prev").isNull() | (F.col("_key") != F.col("_prev"))).cast("int"),
    )
    keyed = keyed.withColumn(
        "_island",
        F.sum("_break").over(w_seg.rowsBetween(Window.unboundedPreceding, 0)),
    )

    intervals = (
        keyed.groupBy("person_id", "_island")
        .agg(
            F.min("seg_start").alias("interval_start"),
            F.max("seg_end").alias("_seg_end_max"),
            # All segments in an island share the same subset, so any is correct.
            F.first("drug_subset").alias("drug_subset"),
        )
        # seg_end is exclusive (era_end+1 based); step back to the inclusive last
        # covered day so interval_end is a real coverage date.
        .withColumn("interval_end", F.date_sub(F.col("_seg_end_max"), 1))
    )
    return (
        intervals.where(
            (F.size("drug_subset") > 0)
            & (F.datediff(F.col("interval_end"), F.col("interval_start"))
               >= F.lit(min_days))
        )
        .select("person_id", "interval_start", "interval_end", "drug_subset")
    )


def _encounter_regular_intervals(
    intervals: DataFrame,
    visit_df: DataFrame,
    *,
    max_gap_days: int = 395,
) -> DataFrame:
    """Keep intervals whose encounters recur at least every ``max_gap_days``.

    ``intervals`` is ``(person_id, interval_start, interval_end, drug_subset)``
    from :func:`_stable_drug_intervals`; ``visit_df`` has ``person_id`` and
    ``visit_start_date``. For each interval, take the ``visit_start_date`` values
    in ``[interval_start, interval_end]``, order them, and require ALL THREE gap
    kinds to be ``<= max_gap_days`` (395d ~= 13 months, the Hughes regularity
    rule):

    - each consecutive visit-to-visit gap,
    - the leading gap ``interval_start -> first visit``,
    - the trailing gap ``last visit -> interval_end``.

    An interval with NO visit in range is dropped (the inner join removes it —
    an un-encountered interval cannot be "regularly encountered"). A single-visit
    interval has no consecutive gap (``coalesce(..., 0)``), so only the two
    endpoint gaps gate it. Returns ``intervals``'s schema, filtered.
    """
    v = visit_df.select("person_id", "visit_start_date")
    j = (
        intervals.join(v, on="person_id", how="inner")
        .where(
            (F.col("visit_start_date") >= F.col("interval_start"))
            & (F.col("visit_start_date") <= F.col("interval_end"))
        )
    )
    w = Window.partitionBy(
        "person_id", "interval_start", "interval_end"
    ).orderBy("visit_start_date")
    j = j.withColumn("_prev_visit", F.lag("visit_start_date").over(w))
    j = j.withColumn(
        "_cons_gap", F.datediff(F.col("visit_start_date"), F.col("_prev_visit"))
    )
    agg = j.groupBy("person_id", "interval_start", "interval_end").agg(
        F.first("drug_subset").alias("drug_subset"),
        F.max("_cons_gap").alias("_max_gap"),
        F.min("visit_start_date").alias("_first_visit"),
        F.max("visit_start_date").alias("_last_visit"),
    )
    agg = agg.withColumn(
        "_start_gap", F.datediff(F.col("_first_visit"), F.col("interval_start"))
    )
    agg = agg.withColumn(
        "_end_gap", F.datediff(F.col("interval_end"), F.col("_last_visit"))
    )
    return (
        agg.where(
            (F.coalesce(F.col("_max_gap"), F.lit(0)) <= F.lit(max_gap_days))
            & (F.col("_start_gap") <= F.lit(max_gap_days))
            & (F.col("_end_gap") <= F.lit(max_gap_days))
        )
        .select("person_id", "interval_start", "interval_end", "drug_subset")
    )


def _first_stable_interval(intervals: DataFrame) -> DataFrame:
    """One row per person: the earliest (by ``interval_start``) stable interval.

    The user's decision is that the FIRST qualifying stable interval defines the
    patient's label + feature anchor. Ties on ``interval_start`` (not expected —
    a person's intervals are disjoint) are broken by ``interval_end`` for
    determinism. Returns ``(person_id, interval_start, interval_end,
    drug_subset)``.
    """
    w = Window.partitionBy("person_id").orderBy(
        F.col("interval_start").asc(), F.col("interval_end").asc()
    )
    return (
        intervals.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .select("person_id", "interval_start", "interval_end", "drug_subset")
    )


def _mdd_stable_treatment_index(
    cond_df: DataFrame,
    drug_era_df: DataFrame,
    visit_df: DataFrame,
    person_df: DataFrame,
    observation_period: DataFrame,
    concept_map: DataFrame,
    mdd_concepts: DataFrame,
    ad_concept_ids: DataFrame,
    *,
    date_col: str,
    min_days: int = 90,
    max_gap_days: int = 395,
    min_history_events: int = 2,
    age_min: int = 18,
    age_max: int = 80,
) -> DataFrame:
    """Pure core of the Hughes stable-treatment cohort (no BQ reads).

    Composes the reusable primitives so the whole cohort is unit-testable on
    synthetic frames:

    1. Antidepressant eras -> constant-subset intervals >= ``min_days``
       (:func:`_antidepressant_era_subsets` -> :func:`_stable_drug_intervals`).
    2. Encounter-regularity filter, max gap <= ``max_gap_days`` on all three gap
       kinds (:func:`_encounter_regular_intervals`).
    3. The FIRST surviving interval per person (:func:`_first_stable_interval`);
       its ``interval_start`` is the index date / feature anchor.

    Then four person-level qualifiers (each a semi-join, no fan-out):

    - **(a) age 18-80 at the stable-interval start** — ``age =
      year(index_date) - year_of_birth`` (the repo's year-difference convention;
      ``person_df`` supplies ``year_of_birth``), kept if ``age_min <= age <=
      age_max`` inclusive.
    - **(b) major-depression indication** — >= 1 qualifying MDD condition
      (``mdd_concepts``, the inclusion-minus-exclusion descendant set). NOTE: the
      user's criterion (b) is stated as mere existence of an MDD dx, so this is
      an EXISTENCE semi-join over ANY date (unlike the sibling
      :func:`_mdd_antidepressant_index`, which requires the MDD dx on/before the
      index) — a deliberate, flagged reading of the written criterion.
    - **(c) sufficient pre-treatment history** — >= ``min_history_events`` events
      in ``cond_df`` dated STRICTLY BEFORE the person's first antidepressant
      drug_era across the 10-drug set (``ad_concept_ids`` via
      :func:`_first_drug_era_dates`). "Events (any domain)" is approximated by
      ``cond_df`` (the events frame handed in).
    - **(d) interval within one observation period** — the whole stable interval
      ``[index_date, stable_end]`` must fall inside a single
      ``observation_period`` row (a direct join + ``distinct`` collapse of the
      possibly-several qualifying periods).

    ``cond_df`` has person_id/concept_id/``date_col``. Returns ``(person_id,
    index_date=stable_start, stable_end, drug_subset, source_cohort=
    'mdd_stable_treatment')``, one row per surviving person.
    """
    era_subsets = _antidepressant_era_subsets(drug_era_df, concept_map)
    intervals = _stable_drug_intervals(era_subsets, min_days=min_days)
    regular = _encounter_regular_intervals(
        intervals, visit_df, max_gap_days=max_gap_days,
    )
    first = _first_stable_interval(regular).select(
        "person_id",
        F.col("interval_start").alias("index_date"),
        F.col("interval_end").alias("stable_end"),
        "drug_subset",
    )

    # (a) age 18-80 at stable start (year-difference convention).
    aged = (
        first.join(
            person_df.select("person_id", "year_of_birth"),
            on="person_id", how="inner",
        )
        .withColumn("_age", F.year(F.col("index_date")) - F.col("year_of_birth"))
        .where((F.col("_age") >= age_min) & (F.col("_age") <= age_max))
        .select("person_id", "index_date", "stable_end", "drug_subset")
    )

    # (d) the whole stable interval falls within one observation period.
    op_ok = (
        aged.join(observation_period, on="person_id", how="inner")
        .where(
            (F.col("index_date") >= F.col("observation_period_start_date"))
            & (F.col("stable_end") <= F.col("observation_period_end_date"))
        )
        .select("person_id", "index_date", "stable_end", "drug_subset")
        .distinct()
    )

    # (b) >= 1 qualifying MDD condition (existence).
    mdd_persons = (
        cond_df.join(F.broadcast(mdd_concepts), on="concept_id", how="inner")
        .select("person_id")
        .distinct()
    )

    # (c) >= min_history_events events strictly before the first AD drug_era.
    first_ad = _first_drug_era_dates(drug_era_df, ad_concept_ids).select(
        "person_id", F.col("index_date").alias("_first_ad_date"),
    )
    history_ok = (
        cond_df.join(first_ad, on="person_id", how="inner")
        .where(F.col(date_col) < F.col("_first_ad_date"))
        .groupBy("person_id")
        .agg(F.count(F.lit(1)).alias("_n_history"))
        .where(F.col("_n_history") >= F.lit(min_history_events))
        .select("person_id")
    )

    return (
        op_ok.join(mdd_persons, on="person_id", how="inner")
        .join(history_ok, on="person_id", how="inner")
        .withColumn("source_cohort", F.lit("mdd_stable_treatment"))
        .select(
            "person_id", "index_date", "stable_end", "drug_subset",
            "source_cohort",
        )
    )


def stable_treatment_label(
    index_df: DataFrame,
    *,
    drug_order: Sequence[str] = _HUGHES_ANTIDEPRESSANTS,
) -> DataFrame:
    """Explode ``drug_subset`` into a fixed-order length-N indicator vector.

    ``index_df`` carries ``drug_subset`` (a ``array<string>`` of ingredient
    names, e.g. from :func:`_mdd_stable_treatment_index`); ``drug_order`` is the
    fixed drug column order (default ``_HUGHES_ANTIDEPRESSANTS``, length 10). The
    label ``y`` is a fully-observed indicator: ``y[i] = 1.0`` iff
    ``drug_order[i]`` is in the person's stable subset, else ``0.0`` — so a
    single-drug interval yields exactly one positive and a held combination
    (e.g. ``{fluoxetine, sertraline}``) yields multiple. Every head trains on
    every patient (the mask is all-ones by construction, since the subset is a
    fully-observed set over the fixed drug order), so no mask column is emitted.
    Returns ``(person_id, y)`` with ``y`` an ``array<double>`` of length
    ``len(drug_order)``.
    """
    y = F.array(*[
        F.when(
            F.array_contains(F.col("drug_subset"), F.lit(name)), F.lit(1.0)
        ).otherwise(F.lit(0.0))
        for name in drug_order
    ])
    return index_df.select("person_id", y.alias("y"))


def apply_mdd_stable_treatment_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    min_days: int = 90,
    max_gap_days: int = 395,
    min_history_events: int = 2,
    age_min: int = 18,
    age_max: int = 80,
) -> DataFrame:
    """Hughes-faithful stable-treatment antidepressant cohort index table.

    Reads the CDR (``concept``, ``concept_ancestor``, ``drug_era``,
    ``visit_occurrence``, ``person``, ``observation_period``), builds the
    10-drug Hughes antidepressant concept map (``_HUGHES_ANTIDEPRESSANTS``) + the
    MDD inclusion-minus-exclusion condition set, and delegates the whole cohort
    to the pure :func:`_mdd_stable_treatment_index`.

    ``visit_occurrence`` is a NEW read for this cohort (the encounter-regularity
    guard); ``person.year_of_birth`` drives the age gate. Unlike the topic-model
    cohorts (which return a windowed events frame), this returns a PER-PERSON
    index table ``(person_id, index_date=stable_start, stable_end, drug_subset,
    source_cohort)`` — feed ``drug_subset`` to :func:`stable_treatment_label` for
    the length-10 outcome vector.
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    concept = _read("concept").select(
        "concept_id", "concept_name", "vocabulary_id", "concept_class_id",
    )
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    drug_era = _read("drug_era").select(
        "person_id", "drug_concept_id", "drug_era_start_date",
        "drug_era_end_date",
    )
    visit = _read("visit_occurrence").select("person_id", "visit_start_date")
    person = _read("person").select("person_id", "year_of_birth")
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )

    concept_map = _antidepressant_concept_map(
        concept, ca, ingredients=_HUGHES_ANTIDEPRESSANTS,
    )
    ad_concept_ids = concept_map.select("concept_id").distinct()
    spec = _DISEASE_REGISTRY["mdd"]
    mdd_concepts = _concept_set_from_ancestors(
        ca,
        inclusion_ancestors=spec["inclusion_ancestors"],
        exclusion_ancestors=spec["exclusion_ancestors"],
    )
    return _mdd_stable_treatment_index(
        cond_df, drug_era, visit, person, op, concept_map, mdd_concepts,
        ad_concept_ids, date_col=date_col, min_days=min_days,
        max_gap_days=max_gap_days, min_history_events=min_history_events,
        age_min=age_min, age_max=age_max,
    )
