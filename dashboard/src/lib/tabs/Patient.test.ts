import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup, fireEvent } from '@testing-library/svelte'
import { get } from 'svelte/store'
import Patient from './Patient.svelte'
import {
  bundle, cohort, patientConditioning, resetConditioningForCohort,
} from '../store'
import { generateCohort } from '../cohort'
import { makeStmBundleFixture } from '../test-fixtures'
import type { DashboardBundle } from '../types'

afterEach(() => cleanup())

beforeEach(() => {
  bundle.set(null)
  cohort.set(null)
  resetConditioningForCohort()
})

// STM fixture extended with a gated group/topic-block structure so
// set-vs-sample conditioning has an observable effect (masked topic 0
// for out-of-group patients), mirroring the fixtures in cohort.test.ts.
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

it('renders the conditioning bar (group selector) for a gated STM bundle', () => {
  const b = makeGatedBundle()
  bundle.set(b)
  seedCohort(b)
  const { getByText } = render(Patient)
  expect(getByText('Regenerate cohort')).toBeTruthy()
  // ConditioningBar's group section label comes from gating.group_var
  // (no group_var_label set on this fixture).
  expect(getByText('g')).toBeTruthy()
})

it('does not crash and still offers Regenerate for a non-STM bundle', async () => {
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
  const { getByText } = render(Patient)
  const btn = getByText('Regenerate cohort')
  expect(btn).toBeTruthy()
  const before = get(cohort)
  await fireEvent.click(btn)
  const after = get(cohort)
  expect(after).not.toBeNull()
  expect(after!.seed).not.toBe(before!.seed)
})

it('Regenerate in "set" mode applies patientConditioning\'s group to every patient (masked out-of-group topic ~0)', async () => {
  const b = makeGatedBundle()
  bundle.set(b)
  seedCohort(b)
  patientConditioning.set({ covariateActive: false, values: {}, group: 'cancer' })

  const { getByText } = render(Patient)
  // Switch the sample-vs-set toggle to "set" mode.
  await fireEvent.click(getByText(/use set/i))
  await fireEvent.click(getByText('Regenerate cohort'))

  const c = get(cohort)!
  expect(c.patients.length).toBeGreaterThan(0)
  for (const p of c.patients) {
    expect(p.group).toBe('cancer')
    // topic 2 is dementia-only (topic_blocks[2] === 'dementia'); masked to 0
    // for every patient since the whole cohort is fixed to 'cancer'.
    expect(p.theta[2]).toBe(0)
  }
})

it('Regenerate in "sample" mode (default) draws each patient its own group', async () => {
  const b = makeGatedBundle()
  bundle.set(b)
  seedCohort(b)
  patientConditioning.set({ covariateActive: false, values: {}, group: null })

  const { getByText } = render(Patient)
  await fireEvent.click(getByText('Regenerate cohort'))

  const c = get(cohort)!
  const groupsSeen = new Set(c.patients.map((p) => p.group))
  expect(groupsSeen.has('cancer') || groupsSeen.has('dementia')).toBe(true)
  for (const p of c.patients) {
    if (p.group === 'cancer') expect(p.theta[2]).toBe(0)
  }
})
