import { writable, derived } from 'svelte/store'
import type { DashboardBundle, SyntheticCohort } from './types'
import { computeJsdMds } from './mds'
import { jsd, phenotypesContainingCode } from './inference'

export const bundle = writable<DashboardBundle | null>(null)
export const cohort = writable<SyntheticCohort | null>(null)

// Cached 2D UMAP projection of the current cohort. Held in a store (not
// PatientMap-local state) so navigating away from the Patient tab and
// back does not retrigger UMAP fitting on a cohort that hasn't changed.
// PatientMap invalidates this whenever `seed` differs from $cohort.seed.
export const patientProjection = writable<{
  patientCoords: number[][]
  seed: number
} | null>(null)

// Reset the cached projection whenever the cohort itself is regenerated.
// A new cohort has a new seed by construction, so the seed-equality check
// in PatientMap would catch it too - this just avoids briefly rendering
// stale coords against fresh patients.
cohort.subscribe(($c) => {
  if (!$c) { patientProjection.set(null); return }
  patientProjection.update((p) => (p && p.seed === $c.seed ? p : null))
})

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

// Phenotype-to-patients pin: a phenotype id the user wants to find patients
// for. Set by the "Find patients with this phenotype" action in the
// phenotype-atlas CodePanel; the patient atlas adds an amber ring to any
// patient whose theta on this phenotype is at or above OTHER_THRESHOLD
// (i.e., the phenotype appears as a labeled band in that patient's
// profile). Independent of searchedConditionIdx so a patient can carry
// both rings simultaneously.
export const searchedPhenotypeForPatients = writable<number | null>(null)

// Set of phenotype ids whose top relevance-ranked conditions include the
// searched condition. Computed once when searchedConditionIdx changes, so
// per-patient consumers (the profile-bar match dots, the patient-table
// row highlight) can read it without recomputing per patient.
export const searchedPhenotypeSet = derived(
  [bundle, searchedConditionIdx],
  ([$b, $idx]) => {
    if (!$b || $idx === null) return null
    return phenotypesContainingCode({
      beta: $b.model.beta,
      corpusFreq: $b.vocab.codes.map((c) => c.corpus_freq),
      codeIdx: $idx,
    })
  }
)

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

// JSD-MDS coords for phenotypes. Computed once when the bundle loads so the
// phenotype-atlas (TopicMap) and the patient-atlas (PatientMap) share the
// same 2D space. PatientMap projects each patient as a theta-weighted
// barycenter of these coords.
export const phenotypeCoords = derived(bundle, ($b) =>
  $b ? computeJsdMds($b.model.beta) : ([] as number[][])
)

// 1D ordering of phenotypes by similarity. Built via a greedy nearest-
// neighbor walk through the K x K JSD distance matrix: pick a starting
// phenotype, then repeatedly hop to the closest unvisited one. The
// resulting sequence has adjacents that are actually similar (low JSD),
// which is what the palette needs - sorting by MDS-x alone collapses the
// 2D atlas onto a single axis and mis-orders phenotypes that share x but
// differ in y. Starts from the phenotype with the smallest mean distance
// to all others (a rough "center" of the phenotype space) for stability.
export const phenotypeOrder = derived(bundle, ($b) => {
  if (!$b) return [] as number[]
  const beta = $b.model.beta
  const K = beta.length
  const D: number[][] = Array.from({ length: K }, () => new Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const d = Math.sqrt(Math.max(0, jsd(beta[i], beta[j])))
      D[i][j] = d
      D[j][i] = d
    }
  }
  let start = 0
  let bestMean = Infinity
  for (let i = 0; i < K; i++) {
    let s = 0
    for (let j = 0; j < K; j++) s += D[i][j]
    if (s < bestMean) { bestMean = s; start = i }
  }
  const visited = new Set<number>([start])
  const order: number[] = [start]
  while (order.length < K) {
    const last = order[order.length - 1]
    let pick = -1
    let pickD = Infinity
    for (let k = 0; k < K; k++) {
      if (visited.has(k)) continue
      if (D[last][k] < pickD) { pickD = D[last][k]; pick = k }
    }
    order.push(pick)
    visited.add(pick)
  }
  return order
})
