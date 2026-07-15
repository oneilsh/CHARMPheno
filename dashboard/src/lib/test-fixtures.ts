// Shared minimal-but-valid DashboardBundle fixtures for STM-era dashboard
// tests. Keep this file's exports additive as later tasks extend it — don't
// narrow or repurpose an existing export's shape once other tests depend on
// it.
import type { DashboardBundle } from './types'

// A small STM bundle: K=3 (reference topic 0 + two free topics 1, 2).
//
// The `age` covariate effects are deliberately chosen so the covariate-driven
// prevalence order of topics 1 and 2 INVERTS between the off state
// (covariateActive: false, which reads raw corpus_prevalence) and
// covariateActive: true with age=80:
//
//   - corpus_prevalence: topic 1 (0.30) > topic 2 (0.20)  -> off-state order.
//   - covariateEffects: at age=80, softmax(Gamma^T x) gives topic 2 a much
//     larger share than topic 1 (topic 2 carries a strong positive age
//     coefficient; topic 1's is slightly negative), so the covariate-on
//     order flips to topic 2 > topic 1.
//
// See src/lib/atlas/PhenotypeBrowser.test.ts for the re-sort assertion that
// depends on this inversion.
export function makeStmBundleFixture(): DashboardBundle {
  return {
    model: {
      K: 3,
      V: 4,
      alpha: [0.1, 0.1, 0.1],
      beta: [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
      ],
    },
    phenotypes: {
      phenotypes: [
        { id: 0, label: 'Reference', description: '', quality: 'background', npmi: null, pair_coverage: null, corpus_prevalence: 0.5, original_topic_id: 0 },
        { id: 1, label: 'Topic A', description: '', quality: 'phenotype', npmi: 0.2, pair_coverage: 0.9, corpus_prevalence: 0.3, original_topic_id: 1 },
        { id: 2, label: 'Topic B', description: '', quality: 'phenotype', npmi: 0.3, pair_coverage: 0.9, corpus_prevalence: 0.2, original_topic_id: 2 },
      ],
    },
    vocab: {
      codes: [
        { id: 0, code: 'C0', description: 'code 0', domain: 'condition', corpus_freq: 0.4 },
        { id: 1, code: 'C1', description: 'code 1', domain: 'condition', corpus_freq: 0.3 },
        { id: 2, code: 'C2', description: 'code 2', domain: 'condition', corpus_freq: 0.2 },
        { id: 3, code: 'C3', description: 'code 3', domain: 'condition', corpus_freq: 0.1 },
      ],
    },
    corpusStats: { corpus_size_docs: 1000, mean_codes_per_doc: 8, k: 3, v: 4, v_full: 4 },
    covariateSchema: {
      k: 3,
      controls: [{ name: 'age', type: 'continuous', range: [0, 100], default: 60 }],
      design_columns: [
        { name: 'Intercept', recipe: { kind: 'intercept' } },
        { name: 'age', recipe: { kind: 'main', var: 'age' } },
      ],
      unsupported: [],
    },
    covariateEffects: [
      { covariate: 'Intercept', per_topic: [0, 1.0, 0.5] },
      { covariate: 'age', per_topic: [0, -0.02, 0.03] },
    ],
    correlation: {
      topic_order: [0, 1, 2],
      block_labels: ['background', 'background', 'background'],
      R: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      identified: [
        [true, true, true],
        [true, true, true],
        [true, true, true],
      ],
      support: [
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
      ],
      reference_topic: 0,
    },
  }
}
