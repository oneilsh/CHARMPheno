import { writable, derived } from 'svelte/store'
import type { DashboardBundle, SyntheticCohort } from './types'

export const bundle = writable<DashboardBundle | null>(null)
export const cohort = writable<SyntheticCohort | null>(null)

export const selectedPhenotypeId = writable<number | null>(null)
export const selectedPatientId = writable<string | null>(null)
export const simulatorPrefix = writable<number[]>([])     // vocab indices (trimmed)
export const advancedView = writable<boolean>(false)
export const colorMode = writable<'npmi' | 'prevalence'>('npmi')
export const hoveredCodeIdx = writable<number | null>(null)

// Condition search: vocab index of a condition the user has pinned via the
// search box. Triggers persistent phenotype-highlight on the atlas. Distinct
// from hoveredCodeIdx (transient, set by CodePanel mouseover).
export const searchedConditionIdx = writable<number | null>(null)

// Phenotype browser filter+sort state.
export const phenotypeFilter = writable<string>('')
export const phenotypeSortBy = writable<'label' | 'prevalence' | 'coherence'>(
  'prevalence',
)

export const phenotypesById = derived(bundle, ($b) =>
  $b ? new Map($b.phenotypes.phenotypes.map((p) => [p.id, p])) : new Map()
)

export const patientsById = derived(cohort, ($c) =>
  $c ? new Map($c.patients.map((p) => [p.id, p])) : new Map()
)
