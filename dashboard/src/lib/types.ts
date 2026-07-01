export interface Model { K: number; V: number; alpha: number[]; beta: number[][] }
// `quality` and `description` come from the post-fit labeling step
// (scripts/label_phenotypes.py). Missing/null on freshly-exported
// bundles . the dashboard treats null as "unlabeled / show everything".
export type PhenotypeQuality =
  'phenotype' | 'background' | 'anchor' | 'mixed' | 'dead'
export interface ThetaPercentiles {
  p5: number
  p25: number
  p50: number
  p75: number
  p95: number
}
export interface Phenotype {
  id: number; label: string; description: string;
  quality: PhenotypeQuality | null;
  npmi: number | null; pair_coverage: number | null; corpus_prevalence: number;
  theta_histogram?: (number | null)[]
  theta_percentiles?: ThetaPercentiles
  original_topic_id: number
}
export interface PhenotypesBundle {
  phenotypes: Phenotype[]
  theta_histogram_bin_edges?: number[]
  theta_histogram_min_count?: number
}
export interface VocabCode {
  id: number; code: string; description: string; domain: string; corpus_freq: number
}
export interface VocabBundle { codes: VocabCode[] }
// Inline cohort metadata embedded in corpus_stats.json by the bundle
// builder (see charmpheno.omop.cohorts.cohort_metadata). Optional —
// older bundles exported before multi-cohort support shipped won't
// carry it, and the UI falls back to the top-level manifest entry.
export interface CohortMeta {
  id: string; label: string; description: string
}
export interface CorpusStats {
  corpus_size_docs: number; mean_codes_per_doc: number; k: number; v: number; v_full: number
  cohort?: CohortMeta
}
export type CovariateRecipe =
  | { kind: 'intercept' }
  | { kind: 'main'; var: string }
  | { kind: 'dummy'; var: string; level: string }
  | { kind: 'interaction'; factors: CovariateRecipe[] }

export interface CovariateControl {
  name: string
  type: 'continuous' | 'categorical'
  range?: [number, number]
  default?: number
  reference?: string
  levels?: string[]
  proportions?: Record<string, number>
}
export interface CovariateSchema {
  k: number
  controls: CovariateControl[]
  design_columns: { name: string; recipe: CovariateRecipe }[]
  unsupported: string[]
}
export type CovariateEffects = { covariate: string; per_topic: number[] }[]

export interface GatingSpec {
  group_var: string
  groups: string[]
  topic_blocks: string[]
}

export interface Correlation {
  topic_order: number[]
  block_labels: string[]
  R: (number | null)[][]
  identified: boolean[][]
  support: number[][]
}

export interface DashboardBundle {
  model: Model; phenotypes: PhenotypesBundle; vocab: VocabBundle; corpusStats: CorpusStats
  covariateSchema?: CovariateSchema
  covariateEffects?: CovariateEffects
  gating?: GatingSpec
  correlation?: Correlation
}

// Top-level data/manifest.json: lists which per-cohort bundles are
// available and which one to load by default. The selector in the
// masthead pulls its options from `cohorts`; `default` is used the
// first time a user lands on the dashboard (subsequent visits restore
// the last-selected cohort from localStorage).
export interface CohortManifestEntry {
  id: string; label: string; description: string
}
export interface CohortManifest {
  default: string
  cohorts: CohortManifestEntry[]
}

// In-memory only; not part of the bundle:
export interface SyntheticPatient {
  id: string; theta: number[]; code_bag: number[]; neighbors: string[]
  // True when the patient's dominant phenotype is NOT classified as dead
  // or mixed. Basic mode hides patients with isClean=false; advanced
  // mode shows them. Always true when no quality labels are available.
  isClean: boolean
}
export interface SyntheticCohort {
  patients: SyntheticPatient[]; seed: number
}
