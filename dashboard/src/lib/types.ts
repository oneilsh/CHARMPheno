export interface Model { K: number; V: number; alpha: number[]; beta: number[][] }
export interface Phenotype {
  id: number; label: string; npmi: number; corpus_prevalence: number;
  junk_flag: boolean; original_topic_id: number
}
export interface PhenotypesBundle { phenotypes: Phenotype[]; npmi_threshold: number }
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
