# References

Central bibliography for CHARMPheno. Entries are grouped by theme. Most were
consolidated from inline citations across `docs/` (architecture, decisions,
insights) and Python docstrings under `spark-vi/`; a `— used in:` pointer marks
papers tied to a specific implementation file. A few marked **(landscape)** are
positioning references from literature review that are not (yet) cited in code.

New here? Add the paper under the closest theme (or start a new one), with
authors, year, title, venue, a link if available, and a short note on its role.

---

## Topic models — core

- **Blei, Ng & Jordan (2003).** Latent Dirichlet Allocation. *JMLR* 3:993–1022.
  — LDA; also the asymmetric-α Newton–Raphson estimator (App. A.4.2). *used in:* `spark-vi/spark_vi/inference/concentration_optimization.py`
- **Hoffman, Blei & Bach (2010).** Online Learning for Latent Dirichlet Allocation. *NeurIPS* 23.
  — online/streaming LDA; scalar-η Newton and ρ_t schedule. *used in:* `spark-vi/spark_vi/models/topic/lda.py`
- **Hoffman, Blei, Wang & Paisley (2013).** Stochastic Variational Inference. *JMLR* 14:1303–1347.
  — SVI theory; κ, τ₀ learning-rate schedule. *used in:* `spark-vi/spark_vi/core/config.py`
- **Teh, Jordan, Beal & Blei (2006).** Hierarchical Dirichlet Processes. *JASA* 101(476):1566–1581.
  — HDP.
- **Wang, Paisley & Blei (2011).** Online Variational Inference for the HDP. *AISTATS*, PMLR 15:752–760.
  — OnlineHDP reference. *used in:* `spark-vi/spark_vi/models/topic/online_hdp.py`
- **Wallach, Mimno & McCallum (2009).** Rethinking LDA: Why Priors Matter. *NeurIPS* 22.
  — asymmetric-α / symmetric-β argument.
- **Blei & Lafferty (2007).** A Correlated Topic Model of Science. *Annals of Applied Statistics* 1(1):17–35.
  — logistic-normal / CTM; full-Σ + inverse-Wishart basis underpinning STM. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Roberts, Stewart & Airoldi (2016).** A Model of Text for Experimentation in the Social Sciences. *JASA* 111(515):988–1003.
  — Structural Topic Model (STM). *used in:* `spark-vi/spark_vi/models/topic/stm.py`
- **Roberts, Stewart & Tingley (2019).** stm: An R Package for Structural Topic Models. *JSS* 91(2).
  — reference implementation (`sigma.prior`, spectral init).
- **Blei & Lafferty (2006).** Dynamic Topic Models. *ICML*, 113–120.
  — corpus-level topic drift (DTM).
- **Lee & Seung (2001).** Algorithms for Non-negative Matrix Factorization. *NIPS*.
  — implicit-φ ("Lee/Seung trick") in the LDA/HDP E-step. *used in:* `spark-vi/spark_vi/models/topic/lda.py`
- **Eisenstein, Ahmed & Xing (2011).** Sparse Additive Generative Models of Text (SAGE). *ICML*.
  — word-level sparse log-deviation background/foreground; alternative to gating.
- **Chang, Boyd-Graber, Gerrish, Wang & Blei (2009).** Reading Tea Leaves: How Humans Interpret Topic Models. *NeurIPS* 22.
  — perplexity vs. interpretability.

## Topic hierarchies, supervision & gating

- **Ramage, Manning & Dumais (2011).** Partially Labeled Topic Models for Interpretable Text Mining (PLDA). *KDD*.
  — generative-restriction basis for "gated LDA"; the model our `TopicBlockPartition` gating reimplements. *used in:* `docs/insights/0028-...-plda.md`
- **Ramage, Hall, Nallapati & Manning (2009).** Labeled LDA: A Supervised Topic Model for Credit Attribution in Multi-labeled Corpora. *EMNLP*.
  — the label-restricted special case (0 background) of PLDA. **(landscape)**
- **Blei, Griffiths & Jordan (2010).** The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of Topic Hierarchies. *JACM* 57(2):7. (arXiv:0710.0845)
  — hLDA / nCRP; the canonical "coarse-near-root, specific-near-leaves, doc-uses-a-path" hierarchy. Closest generative analogue to the ontology background-cascade idea (but learns the tree). **(landscape)**
- **Paisley, Wang, Blei & Jordan (2015).** Nested Hierarchical Dirichlet Processes. *IEEE TPAMI* 37(2):256–270. (arXiv:1210.6738)
  — nHDP; per-document distribution over paths on a shared tree (soft/multi-path generalization of hLDA). **(landscape)**
- **Li & McCallum (2006).** Pachinko Allocation: DAG-Structured Mixture Models of Topic Correlations. *ICML*.
  — DAG (not tree) topic hierarchy; the multi-parent generalization relevant to MONDO. **(landscape)**
- **Perotte, Wood, Elhadad & Bartlett (2011).** Hierarchically Supervised Latent Dirichlet Allocation (HSLDA). *NeurIPS* 24.
  — flat LDA topics + ICD-9-tree label supervision (child label requires parent). Hierarchy on the *label-prediction* side, not topic access. **(landscape)**

## Phenotyping / EHR topic models

- **Li, Nair, Lu, Wen, Wang et al. (2020).** Inferring Multimodal Latent Topics from EHRs (MixEHR). *Nature Communications* 11:2536.
  — multi-view Dirichlet phenotype topic model; the lineage this project positions against.
- **MixEHR-Guided (MixEHR-G).** Modeling EHRs with a guided multi-modal topic model for large-scale automatic phenotyping. *Journal of Biomedical Informatics*, 2022. https://www.sciencedirect.com/science/article/pii/S1532046422001976
  — PheCode/surrogate-feature priors make topics identifiable with known phenotypes. (Li lab, McGill; first author to confirm.) **(landscape)**
- **Song, Hu, Verma, Buckeridge & Li (2022).** Automatic Phenotyping by a Seed-guided Topic Model (MixEHR-Seed). *KDD '22* (ACM SIGKDD). DOI:10.1145/3534678.3542675
  — dual seed-topic / regular-topic distributions per phenotype; PheWAS-catalog seeds. **(landscape)**
- **Yang, Song, Zabad, … & Li (2025/2026).** PheCode-guided Multi-modal Topic Modeling of EHRs Improves Disease Incidence Prediction and GWAS Discovery from UK Biobank (MixEHR-SAGE). *Briefings in Bioinformatics* 27(1):bbag030; medRxiv 2025.05.28.25328511. https://academic.oup.com/bib/article/27/1/bbag030/8454868
  — unifies MixEHR-G prior init + MixEHR-Seed inference; explicitly excludes rare disease (future work). **(landscape)**
- **Li, Yang & Li (2024).** MixEHR-SurG: A Joint Proportional Hazard and Guided Topic Model for Inferring Mortality-associated Topics from EHRs. *Journal of Biomedical Informatics*, 2024. (arXiv:2312.13454; PMID 38631461)
  — survival integration in the MixEHR lineage. **(landscape)**
- **Wang, Wang, Song, Buckeridge & Li (2024).** MixEHR-Nest: Identifying Subphenotypes within EHRs through Hierarchical Guided-Topic Modeling. arXiv:2410.13217.
  — 2-level phenotype→subtopic hierarchy, PheCode/CCS-guided; nearest work to the ontology-cascade idea. **(landscape)**
- **Zou, Pesaranghader, Song, Verma, Buckeridge & Li (2022).** Modeling EHR Data Using an End-to-End Knowledge-Graph-Informed Topic Model (GAT-ETM). *Scientific Reports* 12. https://www.nature.com/articles/s41598-022-22956-w
  — graph-attention embeddings of EHR codes from a *taxonomy* KG, fed into an Embedded Topic Model (logistic-normal). The main "KG-informed EHR topic model" prior art; graph is over codes (no code↔ontology mapping problem). **(landscape)**
- **Pivovarov, Perotte, Grave, Angiolillo, Wiggins & Elhadad (2015).** Learning Probabilistic Phenotypes from Heterogeneous EHR Data (UPhenome). *JBI* 58:156–165.
- **Ghassemi et al. (2014).** Unfolding Physiological State: Mortality Modelling in Intensive Care Units. *KDD*.
- **Li, D.C., Therneau, Chute & Liu (2014).** Discovering Associations Among Diagnosis Groups Using Topic Modeling. *AMIA Joint Summits*.
- **Halpern, Horng, Choi & Sontag (2016).** Electronic Medical Record Phenotyping Using the Anchor and Learn Framework. *JAMIA* 23(4):731–740.
- **Chen et al. (2016).** Word-distance-dependent CRF for diagnosis codes. — cited in `docs/architecture/TOPIC_STATE_MODELING.md`.
- **Hubbard et al. (2021).** Estimating Patient Phenotypes and Outcome–Exposure Associations (Bayesian latent class; PCORI). — reviewer-comparison prior art.
- **Meaney et al. (2022).** Comparison of Methods for Estimating Temporal Topic Models From Primary Care Clinical Text. *JMIR Medical Informatics*.
- **Mehmood, Zahra, Iqbal, Qahmash & Hussain (2026).** A Systematic Review of Topic Modeling Techniques for EHRs. *Healthcare* 14(2):282.
- **Kumari et al. (2026).** Recent Advancements in Topic Modeling Techniques for Healthcare and Bioinformatics. *Advanced Intelligent Systems*.
- **O'Neil et al. (2024).** Large-scale HDP phenotyping application (CHARM prior work). *npj Digital Medicine*. https://www.nature.com/articles/s41746-024-01286-3
  — the team's own prior work; motivates CHARMPheno.
- **Pfaff et al. (2023).** N3C macrovisit construction. — encounter/document-unit standardization.

## Anchor-word / spectral initialization

- **Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu & Zhu (2013).** A Practical Algorithm for Topic Modeling with Provable Guarantees. *ICML*. (arXiv:1212.4777)
  — anchor-word / spectral init. (A "2014" variant is also referenced in `docs/insights/0029`.) *used in:* `spark-vi/spark_vi/models/topic/spectral_init.py`
- **Lee & Mimno (2014).** Low-dimensional Embeddings for Interpretable Anchor-based Topic Inference. *EMNLP*.
  — random-projection dimension default. *used in:* `docs/decisions/0032`
- **Damle & Sun (2017).** A Geometric Approach to Archetypal Analysis and NMF. *Technometrics*. (arXiv:1405.4275)
  — d = K+1 extreme-point preservation.

## Inference / concentration optimization / online EM

- **Blei & Jordan (2006).** Variational Inference for Dirichlet Process Mixtures. *Bayesian Analysis* 1(1):121–143.
  — stick-breaking concentration priors. *used in:* `spark-vi/spark_vi/inference/concentration_optimization.py`
- **Minka (2000/2003).** Estimating a Dirichlet Distribution. — fixed-point alternative to Newton for α.
- **Cappé & Moulines (2009).** On-line Expectation–Maximization Algorithm for Latent Data Models. *JRSS-B* 71(3):593–613.
  — stochastic-EM framing of STM's Γ/Σ updates. *used in:* `docs/decisions/0023`
- **Teh, Newman & Welling (2007).** A Collapsed Variational Bayesian Inference Algorithm for LDA. *NeurIPS* 20.
  — mean-field under-dispersion.
- **Asuncion, Welling, Smyth & Teh (2009).** On Smoothing and Inference for Topic Models. *UAI*.

## Coherence & diversity metrics

- **Röder, Both & Hinneburg (2015).** Exploring the Space of Topic Coherence Measures. *WSDM*.
  — NPMI coherence. *used in:* `spark-vi/spark_vi/eval/topic/coherence.py`
- **Aletras & Stevenson (2013).** Evaluating Topic Coherence Using Distributional Semantics. *IWCS*.
  — rare-pair handling. *used in:* `spark-vi/spark_vi/eval/topic/coherence.py`
- **Hill (1973).** Diversity and Evenness: A Unifying Notation and Its Consequences. *Ecology* 54(2):427–432.
  — inverse-Simpson / Hill-number "effective topics." *used in:* `spark-vi/spark_vi/eval/topic/concentration.py`
- **Jost (2006).** Entropy and Diversity. *Oikos* 113(2):363–375.
  — Hill numbers. *used in:* `spark-vi/spark_vi/eval/topic/concentration.py`

## Language-model smoothing (predictive-gain eval)

- **Zhai & Lafferty (2004).** A Study of Smoothing Methods for Language Models Applied to Ad Hoc Information Retrieval. *SIGIR* / *ACM TOIS*.
  *used in:* `spark-vi/spark_vi/mllib/topic/predictive_gain.py`
- **MacKay & Peto (1995).** A Hierarchical Dirichlet Language Model. *Natural Language Engineering* 1(3):289–308.
  *used in:* `spark-vi/spark_vi/mllib/topic/predictive_gain.py`

## Linear algebra: covariance selection / matrix completion

- **Dempster (1972).** Covariance Selection. *Biometrics* 28(1):157–175.
  — zero-precision = conditional independence; max-det completion. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Grone, Johnson, Sá & Wolkowicz (1984).** Positive Definite Completions of Partial Hermitian Matrices. *Linear Algebra Appl.* 58:109–124. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Speed & Kiiveri (1986).** Gaussian Markov Distributions over Finite Graphs. *Annals of Statistics* 14(1):138–150.
  — iterative proportional scaling. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Lauritzen (1996).** Graphical Models. Oxford University Press.
  — chordal-pattern closed form. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Higham (2002).** Computing the Nearest Correlation Matrix — a Problem from Finance. *IMA J. Numerical Analysis* 22(3):329–343.
  — PSD-projection fallback. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Dykstra (1983).** An Algorithm for Restricted Least-Squares Regression. *JASA* 78:837–842.
  — alternating projection. *used in:* `spark-vi/spark_vi/models/topic/_linalg.py`
- **Lewandowski, Kurowicka & Joe (2009).** Generating Random Correlation Matrices Based on Vines and Extended Onion Method. *J. Multivariate Analysis* 100(9):1989–2001.
  — LKJ unit-diagonal correlation parameterization. *used in:* `docs/decisions/0034`

## Streaming statistics

- **Welford (1962).** Note on a Method for Calculating Corrected Sums of Squares and Products. *Technometrics* 4(3):419–420.
  — streaming mean/variance. *used in:* `spark-vi/spark_vi/mllib/topic/stm.py`
- **Chan, Golub & LeVeque (1979).** Updating Formulae and a Pairwise Algorithm for Computing Sample Variances. STAN-CS-79-773, Stanford.
  — parallel/tree-reduce Welford. *used in:* `spark-vi/spark_vi/mllib/topic/stm.py`
- **Weiss (2005).** A Course in Probability (Thm. 4.4.7, law of total variance). Addison-Wesley.
  *used in:* `spark-vi/spark_vi/mllib/topic/stm.py`

## Continuous-time dynamics / temporal / causality

- **Fasen (2013).** Statistical Estimation of Multivariate Ornstein–Uhlenbeck Processes and Applications to Co-integration. *J. Econometrics* 172(2):325–337.
- **Singh, Ghosh & Adhikari (2018).** Fast Bayesian Inference of the Multivariate Ornstein–Uhlenbeck Process. *Phys. Rev. E* 98:012136.
- **Tran, Lesaffre, Verbeke & Duyck (2021).** Latent Ornstein–Uhlenbeck Models for Bayesian Analysis of Multivariate Longitudinal Categorical Responses. *Biometrics* 77(3):689–701.
- **Gaiffas & Matulewicz (2019).** Sparse Inference of the Drift of a High-Dimensional Ornstein–Uhlenbeck Process. *J. Multivariate Analysis* 169:1–20.
  — distributed sparse L₁ drift estimation for the planned Stage-2 dynamics.
- **Wahl, Sidorenko & Kurths (2016).** Granger-Causality Maps of Diffusion Processes. *Phys. Rev. E* 93:022213.
- **Blei & Frazier (2011).** Distance Dependent Chinese Restaurant Processes. *JMLR* 12:2461–2488.
  — ddCRP with temporal kernel.
- **Schulam & Saria (2015).** A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-resolution Structure. *NeurIPS* 28.
- **Ranganath, Perotte, Elhadad & Blei (2016).** Deep Survival Analysis. *MLHC*. (arXiv:1608.02158)

## Sparse high-dimensional time series

- **Basu & Michailidis (2015).** Regularized Estimation in Sparse High-Dimensional Time Series Models. *Annals of Statistics* 43(4):1535–1567.
- **Basu, Shojaie & Michailidis (2015).** Network Granger Causality with Inherent Grouping Structure. *JMLR* 16(13):417–453.

## Bayesian sparsity priors

- **Carvalho, Polson & Scott (2010).** The Horseshoe Estimator for Sparse Signals. *Biometrika* 97(2):465–480.
- **Piironen & Vehtari (2017).** Sparsity Information and Regularization in the Horseshoe and Other Shrinkage Priors. *Electronic J. Statistics* 11(2):5018–5051.

## Compositional data analysis

- **Egozcue, Pawlowsky-Glahn, Mateu-Figueras & Barceló-Vidal (2003).** Isometric Logratio Transformations for Compositional Data Analysis. *Mathematical Geology* 35(3):279–300.

## Distributed VI & probabilistic-programming frameworks

- **Masegosa et al. (2019).** AMIDST: A Java Toolbox for Scalable Probabilistic Machine Learning. *Knowledge-Based Systems* 163:595–597.
- **Masegosa et al. (2017).** Scaling up Bayesian Variational Inference Using Distributed Computing Clusters. *IJAR* 88:91–108.
- **Akbayrak, Şenöz, Sarı & de Vries (2022).** Probabilistic Programming with Stochastic Variational Message Passing (ForneyLab). *IJAR* 148:235–252.
- **Masegosa & Gómez-Olmedo (2025).** Toward Variational Structural Learning of Bayesian Networks. *IEEE Access* 13:26130–26141.
- **Tran, Kucukelbir, Dieng, Rudolph, Liang & Blei (2016).** Edward: A Library for Probabilistic Modeling, Inference, and Criticism. arXiv:1610.09787.
- **Tran, Hoffman, Moore, Suter, Vasudevan & Radul (2018).** Simple, Distributed, and Accelerated Probabilistic Programming (Edward2). *NeurIPS* 31.
- **Carpenter et al. (2017).** Stan: A Probabilistic Programming Language. *JSS* 76(1). — named as prior art in `docs/architecture/SPARK_VI_FRAMEWORK.md`.
- **Salvatier, Wiecki & Fonnesbeck (2016).** Probabilistic Programming in Python Using PyMC3. *PeerJ CS* 2:e55. — named as prior art.
- **Bingham et al. (2019).** Pyro: Deep Universal Probabilistic Programming. *JMLR* 20. — named as prior art.
- **Phan, Pradhan & Jankowiak (2019).** Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro. arXiv:1912.11554. — named as prior art.
- **Ge, Xu & Ghahramani (2018).** Turing: A Language for Flexible Probabilistic Inference. *AISTATS*. — named as prior art.

## Privacy & synthetic data

- **Jälkö, Dikmen & Honkela (2017).** Differentially Private Variational Inference for Non-conjugate Models. *UAI 2017*. https://www.auai.org/uai2017/proceedings/papers/152.pdf
  — DPVI: DP-SGD-style gradient clipping + Gaussian noise on doubly-stochastic VI. Applies to our non-conjugate models (STM, gated LDA); §3.1 argues for DP-protecting only global params (β, Γ) while never releasing per-document latents θ_d — the backbone for the on-device / patient-owned split and the GenFed line.
- **Su, Wang, Schiavazzi & Liu (2023).** Privacy-Preserving Data Synthesis via Differentially Private Normalizing Flows with Application to Electronic Health Records Data. *AAAI Summer Symposium Series (SuSS-23)*.
  — DP synthetic-EHR generation (masked autoregressive flows under a Gaussian-DP budget) on a small heterogeneous cohort; the synthetic-data-release counterpart to DPVI, and a reference point for the DP-vs-rare-sample tension.
- **Dong, Roth & Su (2022).** Gaussian Differential Privacy. *JRSS-B* 84(1):3–37.
  — the DP accounting used by Su et al. (2023).
