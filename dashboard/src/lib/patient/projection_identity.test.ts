import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { cohort, patientProjection } from '../store'
import type { SyntheticCohort } from '../types'
import type { UMAP } from 'umap-js'

// The cached UMAP layout is keyed on the cohort OBJECT, not its seed: the
// load-time cohort and the first Simulate click both use seed 42, so a
// seed-equality check left the old layout in place after a regeneration
// (new patients drawn onto stale coordinates). These guard that identity.

beforeEach(() => { cohort.set(null); patientProjection.set(null) })

const stubUmap = {} as unknown as UMAP
function mkCohort(seed: number): SyntheticCohort {
  return { seed, patients: [] }
}

it('drops the cached projection when a NEW cohort reuses the same seed', () => {
  const a = mkCohort(42)
  cohort.set(a)
  patientProjection.set({ patientCoords: [[0, 0]], cohort: a, umap: stubUmap })
  expect(get(patientProjection)).not.toBeNull()

  // A distinct cohort object with the SAME seed (the load-vs-first-Simulate
  // collision) must still invalidate the layout.
  const b = mkCohort(42)
  cohort.set(b)
  expect(get(patientProjection)).toBeNull()
})

it('keeps the projection while the same cohort object is re-set', () => {
  const a = mkCohort(7)
  cohort.set(a)
  patientProjection.set({ patientCoords: [[1, 1]], cohort: a, umap: stubUmap })
  cohort.set(a) // same object reference
  expect(get(patientProjection)).not.toBeNull()
})
