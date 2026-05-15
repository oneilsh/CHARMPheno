export interface Model { K: number; V: number; alpha: number[]; beta: number[][] }
// `quality` and `description` come from the post-fit labeling step
// (scripts/label_phenotypes.py). Missing/null on freshly-exported
// bundles . the dashboard treats null as "unlabeled / show everything".
export type PhenotypeQuality =
  'phenotype' | 'background' | 'anchor' | 'mixed' | 'dead'
export interface Phenotype {
  id: number; label: string; description: string;
  quality: PhenotypeQuality | null;
  npmi: number; pair_coverage: number; corpus_prevalence: number;
  original_topic_id: number
}
export interface PhenotypesBundle { phenotypes: Phenotype[] }
export interface VocabCode {
  id: number; code: string; description: string; domain: string; corpus_freq: number
}
export interface VocabBundle { codes: VocabCode[] }
export interface CorpusStats {
  corpus_size_docs: number; mean_codes_per_doc: number; k: number; v: number; v_full: number
}
export interface DashboardBundle {
  model: Model; phenotypes: PhenotypesBundle; vocab: VocabBundle; corpusStats: CorpusStats
}

// In-memory only; not part of the bundle:
export interface SyntheticPatient {
  id: string; theta: number[]; code_bag: number[]; neighbors: string[]
}
export interface SyntheticCohort {
  patients: SyntheticPatient[]; seed: number
}
