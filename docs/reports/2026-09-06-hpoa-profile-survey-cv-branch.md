# Scouting: HPOA phenotype profiles as node-topic priors — stage-1 feasibility survey (CV branch)

**Date:** 2026-09-06
**Feeds:** the alignment-residual question of insight 0082 (exps 0114/0115: spectral
init fixes starvation; the residual is objective misalignment — separability is not
phenotype). Candidate fix under evaluation: a knowledge-aligned WORD-side prior —
boost each node's Dirichlet eta on the OMOP codes its HPO profile maps to, mildly
downweight NOT-annotated ones. The gate already supplies the patient-side guidance
(MixEHR-Guided's mechanism, in hard form); what 0114 showed missing is word-side.
**Tooling:** `analysis/cloud/hpoa_profile_survey.py` (`make -C analysis/cloud
hpoa-profile-survey`), tests in `analysis/cloud/tests/test_hpoa_profile_survey.py`.
**Data:** PUBLIC ontology artifacts only — Mondo KGX v2026-06-02 (the fit's pin),
hp.obo + phenotype.hpoa v2026-06-23. No CDR reads, no patient data, no egress
exposure anywhere in stage 1.

## Verdict

**Feasible, with a known shape.** 41% of the CV branch carries an HPOA profile;
~30% of the branch (495/1628 nodes) has >=5 SNOMED-realizable profile terms — and
coverage HOLDS at the depths where starvation lived. Frequency metadata is
near-universal (95% of profile terms on profiled nodes), so a frequency-weighted
prior is on the table, not hypothetical. Two structural caveats below (promiscuity;
the leakage-strip interaction) shape the design but do not kill it.

## Provenance finding first (time-sensitive)

**HPO removed all ontology xrefs in release v2026-09-01** (release notes: "XREFs
are no longer maintained in the ontology"; 17,412 mappings dropped, SNOMED CT and
UMLS included). The maintained successors are SSSOM sets (HP-MeSH, HP-UMLS, HP-MP,
HP-uPheno at data.monarchinitiative.org/mappings/) — none is HP->SNOMED.
Consequences:

- **v2026-06-23 is the pin of record** for the HP->SNOMED xref map this project
  uses (`mondo_usage_core.parse_hpo_xrefs`, the usage-dashboard HPO axis, and this
  survey). 4,594 SNOMEDCT + 12,816 UMLS xrefs. Anything downloading "latest" hp.obo
  now silently loses the map (this survey run against latest returned realizability
  ZERO across the board — that is how the removal was noticed).
- Future-proofing, if ever needed: HP-UMLS SSSOM (maintained) -> SNOMED via UMLS,
  a heavier licensed path. Not needed while the frozen xrefs serve.

## Method (one paragraph)

Branch closure from the Mondo KGX edges via the shared disease-only subclass
adjacency (`mondo_to_omop_mapping`), root MONDO:0004995 -> 1,628 disease nodes with
BFS min-depth. Node -> HPOA keys via the Mondo `xref` CURIEs (OMIM: as-is,
`Orphanet:` rewritten to `ORPHA:`, DECIPHER: as-is). HPOA rows: aspect P only;
polarity NEGATIVE when Qualifier=NOT or frequency resolves to exactly 0 ("Excluded"
— boosting those would invert the annotation); frequency normalized from all three
sanctioned shapes (HP frequency term -> range midpoint, n/m ratio, percent);
sources pooled per (node, term) by max frequency. Realizability: an HP term counts
as SNOMED-realizable when it or any HPO DESCENDANT carries a SNOMED xref (the
true-path direction: a patient coded with the more specific phenotype has the
general one) — computed as ancestor-or-self of the direct-xref set.

## Results (pooled — counts of ontology nodes, no patient data)

| | |
|---|---|
| branch closure nodes | 1,628 |
| nodes with >=1 positive profile term | 665 (40.8%) |
| nodes with >=1 SNOMED-realizable term | 654 |
| nodes with >=5 SNOMED-realizable terms | **495 (30.4%)** |
| nodes with any NOT/excluded annotation | 100 (246 negative rows total) |
| median profile size (profiled nodes) | 14 (p25 6, p75 30) |
| median SNOMED-realizable (closure) | 11 |
| frequency-annotated share of profile terms | 0.95 median per node; 0.85 pooled |

By depth (nodes / with profile / >=5 realizable):

| depth | nodes | with profile | >=5 realizable | >=10 realizable |
|---|---|---|---|---|
| 2 | 107 | 28 | 27 | 23 |
| 3 | 432 | 170 | 139 | 112 |
| 4 | 461 | 162 | 123 | 89 |
| 5 | 368 | 166 | 113 | 68 |
| 6 | 167 | 96 | 60 | 40 |
| 7 | 71 | 37 | 31 | 23 |
| 8 | 15 | 5 | 1 | 1 |

**The depth profile is the point:** at depths 4-6 — where 0113's starvation lived
and 0114's residual misalignment lives — a quarter to a third of nodes carry a
usable (>=5-term) realizable profile. The knowledge prior reaches the right
stratum. Nodes without a profile simply keep the current behavior (spectral init,
flat eta): the prior is per-node additive, so partial coverage costs nothing.

## Structural caveats (design-shaping, not disqualifying)

1. **Promiscuity concentrates in disease-name terms.** 302 of 2,059 realizable
   terms appear in >=10 profiles; the head is cardiac disease-phenotypes
   (Congestive heart failure 120 profiles, ASD 90, VSD 83, Dilated cardiomyopathy
   82, HCM 73, AF 64) plus syndromic giveaways from genetic diseases (Seizure 78,
   Hypertelorism 72, Global developmental delay 70, Short stature 66, Hypotonia
   63). A raw boost on these adds no per-node alignment. An IDF-style downweight
   by profile-frequency within the branch is required, not optional.
2. **The leakage strip eats the promiscuous head — conveniently.** The fit strips
   every DAG-node code from the bag-of-words (`strip_mode both`, insight 0079).
   Profile terms that ARE disease codes (the CHF/DCM/ASD head above) map to
   stripped tokens, so their boost lands on nothing. What survives the strip is
   exactly the SNOMED Finding-branch symptom vocabulary (Dyspnea, Syncope,
   Palpitations, edema...) that insight 0070 showed stays feature-side because
   Mondo correctly excludes phenotypes. The prior's effective mass therefore
   self-selects toward the specific, symptom-shaped tail — the aligned part.
   Stage 2 must split realizable codes into strip-side vs surviving to report the
   EFFECTIVE realizable count per node.
3. **NOT is a garnish, as suspected:** 100 nodes, 246 rows branch-wide. Carried
   (it is nearly free and semantically forced — see polarity rule), but it will
   not move fits.

## Stage 2 (in-workspace, needs the cluster/BQ)

Per node: (a) map realizable HP terms -> SNOMED source codes -> OMOP standard
concepts ('Maps to'), (b) split by strip-survival against the fit's label-code
set, (c) intersect with the corpus vocabulary of the 0113/0114 bundle, (d) count
distinct persons with >=1 surviving profile code — pooled figures and
counts-of-nodes only, egress floor respected. That yields the final per-node
"effective prior support" number and the go/no-go for the eta-boost experiment.

## Repro

```bash
make -C analysis/cloud hpoa-profile-survey            # defaults: CV branch, pinned versions
# or directly:
PYTHONPATH=analysis/cloud python3 analysis/cloud/hpoa_profile_survey.py \
    --branch MONDO:0004995 --mondo-version 2026-06-02 --hpo-release v2026-06-23 \
    --cache-dir data/ontology --out-dir /tmp/hpoa_survey
```

Per-node table (public ontology facts): `2026-09-06-hpoa-profile-survey-MONDO_0004995.tsv`
alongside this report.

## Stage-2 results (2026-09-06, `hpoa-stage2-probe ID=114`; egress-safe figures)

4,330 SNOMED codes → 3,798 standard Condition concepts. **The binding fact is the
label-space intersection: only 36 of the 299 powered label-DAG nodes carry an HPOA
profile** — the other 618 profiled branch nodes sit BELOW the min_positives=100 label
space (HPOA annotates specific OMIM/ORPHA diseases; the powered label nodes are mostly
mid-level categories HPOA does not key to). Where the prior CAN act, it acts well:

- in-vocab profile tokens per probed node: median 45 (p25 20, p75 92); 27/36 nodes ≥20
  tokens; zero nodes with none — the strip does NOT hollow profiles out.
- coverage of observed positive TRAIN docs (15 probed nodes with n_pos ≥ 100): **median
  0.76**; 14/15 ≥ 0.50, 9 ≥ 0.75, 5 ≥ 0.90. Pooled: 20,702 of 44,469 positive cells
  carry ≥1 profile token.
- Exemplars: PAH 0.95, temporal arteritis 0.94, GPA 0.93 (autoimmune/vascular diseases
  with rich symptom profiles); varicose disease 0.09 (a profile of rare syndromic
  contexts, not the common condition — the annotation-population mismatch case).

**Implication.** As scoped (own-node profiles only) the eta prior reaches 12% of the
label space. The natural extension is a **Mondo-descendant profile roll-up**: credit each
powered label node with the union of its (unpowered, profiled) Mondo-descendants'
profiles, frequency-max-pooled, flagged self-vs-inherited. This is roll-up along the
LABEL DAG's own subsumption axis — the same direction the gate pools patients — and is
NOT the messy HPO-side "infer higher-level terms" inference deliberately deferred in this
report; there is no lower-level annotation to remove because the descendants are not
label nodes. Promiscuity/IDF weighting then handles the blur near the root. Decision
pending.

## Stage-2b results (2026-09-06, `--rollup` profiles, rebuilt bundle; egress-safe figures)

(Cluster restart wiped the HDFS bundle cache; `gated-pc-readout ID=114` rebuilt it from
the manifest and reproduced the recorded readout EXACTLY — macro 0.7567/0.5031,
detection 0.5723 — a clean byte-stability check on the rebuild path.)

Rolled-up profiles (838 credited branch nodes, 374,136 code rows): **nodes probed
36 → 132 of 299** (44% of the label space; the remaining 167 have no realizable
profiled descendant). And inheritance did NOT dilute:

- in-vocab tokens per node: median 82 (p25 38, p75 274); 113/132 nodes ≥20.
- coverage (101 nodes with n_pos ≥ 100): **median 0.87**; 96 ≥ 0.50, 76 ≥ 0.75,
  35 ≥ 0.90. Pooled: 877,174 of 1,087,885 positive cells carry ≥1 profile token.
- The low tail is INFORMATIVE, not noise — the annotation-population mismatch class:
  varicose disease 0.09, vascular occlusion disorder 0.18, **mitral valve disorder
  0.23** — labels whose HPOA evidence comes from rare congenital/syndromic contexts
  while the corpus population is the common acquired disease. For these the prior
  would push toward the wrong (rare-syndromic) presentation; being soft it cannot
  dominate, and the per-node coverage table (workspace-internal) supports gating the
  boost on coverage if wanted.

**GO.** Reach 44% of the label space at median 0.87 coverage clears the bar. Next
build: the eta-prior mechanism in the fit path (node-specific eta from the rolled-up
profile codes: freq x IDF weights, IDF over the 132 credited label nodes; NOT rows
mildly downweighted; nodes without profiles keep flat eta), then the record arm
**random init x profile-eta vs 0113** (macro 0.7813 to beat).
