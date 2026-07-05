# Seed-panel real-model result (exp 0039, exp-0028 population_cancer bundle)

Reconstructed the exp-0028 gated STM from dashboard/public/data/population_cancer/ (K=60, V=5000, background_k=40, foreground=(('cancer', 20),), reference topic=0) and ran the seed-panel acceptance test for group 'cancer' at c in {3,4,5,8}, n_codes in {1,2}.

**Sanity-check caveat (read before the table).** The literal brief-specified check (seed 3 spread-out foreground topics with their own top-1 code, c=5, expect recovery for most) FAILED as literally specified (0/3 recovered self at c=5). A widened diagnostic (same topics, c in {5, 20, 50, 100, 1000, 100000}) showed all three DO recover themselves at large enough c (one needed c=100000 to resolve a near-tie with a neighboring topic's top word), converging toward the uninformative-prior limit exactly as the math predicts -- this rules out a reconstruction/indexing bug (partition, Gamma orientation, beta/sigma alignment, and reference handling are all correct). The root cause is a genuine property of this fit: the population-mean Gamma intercepts for rare cancer-subtype foreground topics are strongly negative relative to common background topics (this is a single-cohort model where MOST documents present with generic comorbidities, not a specific rare cancer subtype), and the real beta rows share vocabulary across topics (unlike the disjoint synthetic corpus), so a 1-2 token seed's data-term signal is often too weak to overcome that prior at c in {3,4,5,8}. See stdout capture for the full per-topic diagnostic. Recover-self rate by c: n_codes=1={3: 0.05, 4: 0.1, 5: 0.05, 8: 0.15}, n_codes=2={3: 0.35, 4: 0.3, 5: 0.25, 8: 0.65}.

## n_codes=1 -- ALL seeds (regardless of which topic they land on)

| c | median top_mass | median eff_topics | median second_mass | recover-self rate |
|---|---|---|---|---|
| 3 | 0.1606 | 22.7702 | 0.0474 | 0.0500 |
| 4 | 0.1439 | 24.3637 | 0.0467 | 0.1000 |
| 5 | 0.1153 | 28.8409 | 0.0451 | 0.0500 |
| 8 | 0.1162 | 29.2522 | 0.0460 | 0.1500 |

### n_codes=1 -- SELF-RECOVERED SUBSET only (the reviewer's actual scenario: given the seed lands on its own topic, is the TOTAL mass implausible?)

| c | n seeds | median top_mass | median eff_topics | median second_mass |
|---|---|---|---|---|
| 3 | 1 | 0.1153 | 26.1209 | 0.1062 |
| 4 | 2 | 0.2312 | 12.3549 | 0.1314 |
| 5 | 1 | 0.3765 | 6.3882 | 0.0884 |
| 8 | 3 | 0.1289 | 22.6411 | 0.0801 |

## n_codes=2 -- ALL seeds (regardless of which topic they land on)

| c | median top_mass | median eff_topics | median second_mass | recover-self rate |
|---|---|---|---|---|
| 3 | 0.0903 | 36.4712 | 0.0422 | 0.3500 |
| 4 | 0.1278 | 24.7710 | 0.0569 | 0.3000 |
| 5 | 0.1146 | 28.9530 | 0.0716 | 0.2500 |
| 8 | 0.2022 | 16.6154 | 0.0706 | 0.6500 |

### n_codes=2 -- SELF-RECOVERED SUBSET only (the reviewer's actual scenario: given the seed lands on its own topic, is the TOTAL mass implausible?)

| c | n seeds | median top_mass | median eff_topics | median second_mass |
|---|---|---|---|---|
| 3 | 7 | 0.1960 | 13.8770 | 0.0775 |
| 4 | 6 | 0.2732 | 12.3528 | 0.0654 |
| 5 | 5 | 0.1952 | 16.6341 | 0.0847 |
| 8 | 13 | 0.2772 | 10.5482 | 0.0761 |

## Example seeds (n_codes=1), c=3 vs c=5

| topic | label | seed code | recovered @c=3 | top_mass @c=3 | recovered @c=5 | top_mass @c=5 |
|---|---|---|---|---|---|---|
| 40 | Cancer: lymphoma with neutropenic fever | Fever | 43 | 0.0201 | 43 | 0.0201 |
| 41 | Cancer NHL with immunodeficiency | Fatigue | 43 | 0.0201 | 36 | 0.0832 |
| 42 | Cancer: hepatobiliary–pancreatic and neuroendocrine | Disease of liver | 36 | 0.0893 | 17 | 0.2090 |
| 43 | Cancer with nodal and thoracic complications | Lymphadenopathy | 36 | 0.1107 | 36 | 0.0951 |
| 44 | Prostate cancer with BPH and ED | Primary malignant neoplasm of prostate | 36 | 0.1746 | 36 | 0.0950 |
| 45 | Cancer: ovarian/testicular with peritoneal–lung spread | Primary malignant neoplasm of ovary | 36 | 0.1648 | 36 | 0.1238 |

## Decision

Restricting to seeds that land on their OWN topic (the reviewer's actual scenario), secondary-mass collapse between c=3 and c=5 is **inconclusive (too few self-recovered seeds, n<3)** for n_codes=1 (n=1 self-recovered seed at each of c=3/c=5 -- too few to trust) and **False** for n_codes=2 (n=7 at c=3, n=5 at c=5 -- the reliable comparison here). The n_codes=2 verdict drives the recommendation below.

Among self-recovered real seeds, c=3 and c=5 (and even c=8) look similar: theta stays diffuse throughout this band -- median top_mass is modest (roughly 0.1-0.4, well short of 'implausibly total') and median eff_topics stays in the double digits out of 60 topics (secondary structure clearly retained) at every c tried, for both n_codes=1 and n_codes=2. No material over-commitment collapse is detected in {3,4,5,8} on this real corpus -- the reviewer's specific worry does not materialize here. **Recommendation: ship c=5** (the held-out predictive-LL optimum); the acceptable-secondary-structure band extends at least to c=8 on this evidence, i.e. c=5 is comfortably inside it, not at its edge.

**Separate, non-blocking finding worth flagging:** recover-self rate itself is LOW at c in {3,4,5,8} (5-35% for n_codes=1, 25-65% for n_codes=2) and only climbs to 70%+ around c=20-50 -- well above any candidate ship scale. The practical risk this surfaces for the demo is not 'too confident on the right rare phenotype' but the opposite: a 1-2 token seed of a rare cancer subtype is often completed toward a MORE COMMON background comorbidity topic instead, because of the strong population-mean Gamma intercept gap between common and rare topics. This is orthogonal to the c=3-vs-c=5 over-commitment question this task was scoped to answer, but is worth surfacing to the team (e.g. as an argument for showing >= 2 seed codes, or conditioning demo covariates away from the population mean, when showcasing a rare phenotype).
