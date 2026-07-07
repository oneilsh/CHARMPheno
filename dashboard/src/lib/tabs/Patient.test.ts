import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import Patient from './Patient.svelte'
import { bundle, cohort, resetConditioningForCohort } from '../store'
import { generateCohort } from '../cohort'
import { makeStmBundleFixture } from '../test-fixtures'
import type { DashboardBundle } from '../types'

afterEach(() => cleanup())

beforeEach(() => {
  bundle.set(null)
  cohort.set(null)
  resetConditioningForCohort()
})

// Explore Cohort is now a pure viewer: source-cohort / covariate / Regenerate
// controls moved to the Simulate Cohort tab. These guard that the panel renders
// the current cohort and no longer carries those moved controls. The underlying
// set/sample group-masking behavior is tested at the generateCohort level in
// cohort.test.ts.

// STM fixture extended with a gated group/topic-block structure (as in
// cohort.test.ts) so the gated-only affordances (color-by-group) render.
function makeGatedBundle(): DashboardBundle {
  const base = makeStmBundleFixture()
  return {
    ...base,
    gating: {
      group_var: 'g',
      groups: ['cancer', 'dementia'],
      topic_blocks: ['background', 'cancer', 'dementia'],
      group_proportions: { cancer: 0.8, dementia: 0.2 },
    },
  }
}

function seedCohort(b: DashboardBundle) {
  cohort.set(generateCohort({
    model: b.model,
    meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
    n: 20,
    seed: 1,
    nNeighbors: 3,
    qualityByPhenotype: b.phenotypes.phenotypes.map((p) => p.quality),
  }))
}

it('renders a gated STM cohort and offers color-by-group, without the moved controls', () => {
  const b = makeGatedBundle()
  bundle.set(b)
  seedCohort(b)
  const { getByText, queryByText } = render(Patient)
  // Gated-only affordance stays on this tab.
  expect(getByText('color by group')).toBeTruthy()
  // Controls that moved to Simulate Cohort are gone.
  expect(queryByText('Regenerate cohort')).toBeNull()
  expect(queryByText(/use set/i)).toBeNull()
  expect(queryByText('Source cohort')).toBeNull()
})

it('does not crash for a non-STM bundle and shows no Regenerate control', () => {
  const b: DashboardBundle = {
    model: { K: 2, V: 2, alpha: [1, 1], beta: [[.9, .1], [.1, .9]] },
    phenotypes: { phenotypes: [
      { id: 0, label: 'A', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.5, original_topic_id: 0 },
      { id: 1, label: 'B', description: '', quality: null, npmi: null, pair_coverage: null, corpus_prevalence: 0.5, original_topic_id: 1 },
    ] },
    vocab: { codes: [
      { id: 0, code: 'C0', description: 'c0', domain: 'condition', corpus_freq: 0.5 },
      { id: 1, code: 'C1', description: 'c1', domain: 'condition', corpus_freq: 0.5 },
    ] },
    corpusStats: { corpus_size_docs: 100, mean_codes_per_doc: 6, k: 2, v: 2, v_full: 2 },
  }
  bundle.set(b)
  seedCohort(b)
  const { queryByText } = render(Patient)
  expect(queryByText('Regenerate cohort')).toBeNull()
})
