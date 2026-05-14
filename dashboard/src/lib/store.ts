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

export const phenotypesById = derived(bundle, ($b) =>
  $b ? new Map($b.phenotypes.phenotypes.map((p) => [p.id, p])) : new Map()
)

export const patientsById = derived(cohort, ($c) =>
  $c ? new Map($c.patients.map((p) => [p.id, p])) : new Map()
)
