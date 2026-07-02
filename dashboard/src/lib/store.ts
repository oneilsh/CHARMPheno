import { writable, derived } from 'svelte/store'
import type { UMAP } from 'umap-js'
import type { CohortManifest, DashboardBundle, Phenotype, PhenotypeQuality, SyntheticCohort } from './types'
import { computeJsdMds } from './mds'
import { jsd, phenotypesContainingCode } from './inference'
import { buildDesignVector, covariatePrevalence, allowedMaskForGroup, covariatePrevalenceGated, maskGroupPrevalence } from './covariate'

export const bundle = writable<DashboardBundle | null>(null)
export const cohort = writable<SyntheticCohort | null>(null)

// The top-level cohort manifest — populated once on app boot from
// data/manifest.json. Null while loading; thereafter immutable for the
// session. Drives the masthead selector's options.
export const manifest = writable<CohortManifest | null>(null)

// Which cohort's bundle is currently loaded (matches the `id` of one of
// the entries in `manifest.cohorts`). Persisted across sessions in
// localStorage so reloading the page restores the user's last choice.
// Set to null while the initial manifest is still being fetched.
const COHORT_STORAGE_KEY = 'charmpheno.selectedCohort'
const initialSelectedCohort: string | null = (() => {
  try { return localStorage.getItem(COHORT_STORAGE_KEY) } catch { return null }
})()
export const selectedCohort = writable<string | null>(initialSelectedCohort)
selectedCohort.subscribe((id) => {
  try {
    if (id) localStorage.setItem(COHORT_STORAGE_KEY, id)
  } catch { /* private mode / disabled storage: best-effort persistence */ }
})

// Cached 2D UMAP projection of the current cohort. Held in a store (not
// PatientMap-local state) so navigating away from the Patient tab and
// back does not retrigger UMAP fitting on a cohort that hasn't changed.
// PatientMap invalidates this whenever `seed` differs from $cohort.seed.
// We also keep the fitted UMAP instance so the Simulator can call
// `.transform()` on new theta samples and plot them on the same atlas.
export const patientProjection = writable<{
  patientCoords: number[][]
  seed: number
  umap: UMAP
} | null>(null)

// True while a UMAP fit is in flight. Promoted to a store so both
// PatientMap and the Simulator's mini-atlas can read it without
// racing each other into a duplicate fit.
export const patientProjectionFitting = writable<boolean>(false)

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

// Patient atlas: color points by each patient's recorded gating group
// instead of dominant phenotype. Only meaningful for gated STM bundles
// (SyntheticPatient.group is null otherwise); PatientMap falls back to the
// default dominant-phenotype coloring when off or when patients carry no
// group.
export const colorByGroup = writable<boolean>(false)

// Patient-prevalence threshold τ for the histogram-derived "fraction
// above τ" prevalence reader. Fixed at 0.02 — a patient is counted as
// "having" the phenotype when at least 2% of their coded activity is
// attributed to the topic. Exposed as a store (rather than a bare
// constant) so the several components that read $tauThreshold continue
// to work unchanged; there is no longer a user-facing slider.
export const tauThreshold = writable<number>(0.02)

export interface Conditioning {
  covariateActive: boolean
  values: Record<string, number | string>
  group: string | null
}

function createConditioning() {
  return writable<Conditioning>({ covariateActive: false, values: {}, group: null })
}

// Per-panel, independent conditioning state. Each survives its own panel's
// unmount/remount (fixing the Phase-1 tab-switch-resets bug); state is shared
// by NO other panel. Reset only on cohort/bundle change (see below).
export const atlasConditioning = createConditioning()
export const simulatorConditioning = createConditioning()
export const patientConditioning = createConditioning()

export function resetConditioningForCohort(): void {
  for (const c of [atlasConditioning, simulatorConditioning, patientConditioning])
    c.set({ covariateActive: false, values: {}, group: null })
}

// Back-compat alias: the shipped four-quadrant prevalenceReader reads the
// Phenotype Atlas's conditioning.
export const conditioning = atlasConditioning

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
export const phenotypeSortBy = writable<'label' | 'prevalence' | 'coherence' | 'topic_mass'>(
  'prevalence',
)


// Fraction of patients with theta_k > tau, derived from the histogram.
// Sum bin fractions where the bin's lower edge >= tau. Suppressed bins
// (null) contribute 0 (round-to-zero rule, matches the privacy model).
// HDP / legacy bundles without a histogram fall back to corpus_prevalence
// so existing components continue to work without conditionals.
export function fractionAboveTau(
  p: Phenotype,
  edges: number[] | undefined,
  tau: number,
): number {
  if (!p.theta_histogram || !edges) return p.corpus_prevalence
  let s = 0
  for (let i = 0; i < p.theta_histogram.length; i++) {
    if (edges[i] >= tau) {
      const v = p.theta_histogram[i]
      if (v != null) s += v
    }
  }
  return s
}

// Convenience derived store: a (Phenotype) -> number reader bound to the
// current bundle's bin_edges + the current tau slider. Components that
// display prevalence-as-fraction-above-tau subscribe to this rather than
// calling fractionAboveTau directly, so the value updates reactively as
// the slider moves.
//
// Four-quadrant logic:
//   1. plain (no covariate, no gating)          -> fractionAboveTau (unchanged default)
//   2. covariate active, no gating              -> softmax(Gamma^T x) via covariatePrevalence
//   3. covariate active + gating                -> mask-before-softmax via covariatePrevalenceGated
//   4. gating only (no covariate)               -> fractionAboveTau base then maskGroupPrevalence
export const prevalenceReader = derived(
  [bundle, tauThreshold, conditioning],
  ([$b, $tau, $cond]) => {
    const schema = $b?.covariateSchema
    const effects = $b?.covariateEffects
    const gating = $b?.gating
    const edges = $b?.phenotypes.theta_histogram_bin_edges

    // Covariate axis: when active and renderable, base = softmax(Gamma^T x).
    const covariateOn =
      $cond.covariateActive && !!schema && !!effects && schema.unsupported.length === 0
    if (covariateOn) {
      const x = buildDesignVector(schema!.design_columns, $cond.values)
      if (gating) {
        const mask = allowedMaskForGroup(gating.topic_blocks, $cond.group)
        const prev = covariatePrevalenceGated(effects!, x, mask)
        return (p: Phenotype) => prev[p.id] ?? 0
      }
      const prev = covariatePrevalence(effects!, x)
      return (p: Phenotype) => prev[p.id] ?? 0
    }

    // Non-covariate base = fractionAboveTau; mask by group when gated.
    if (gating) {
      // Build the per-topic base indexed by topic id (= displayed index, the
      // same key topic_blocks uses), then mask hidden foreground to 0.
      const base: number[] = []
      for (const p of $b!.phenotypes.phenotypes) base[p.id] = fractionAboveTau(p, edges, $tau)
      const masked = maskGroupPrevalence(base, gating.topic_blocks, $cond.group)
      return (p: Phenotype) => masked[p.id] ?? 0
    }
    return (p: Phenotype) => fractionAboveTau(p, edges, $tau)
  }
)

// Returns a predicate (p) -> boolean for whether a phenotype should be shown
// in the current view mode. Simple mode hides `dead` and `mixed` topics;
// advanced mode shows everything. Follows the prevalenceReader pattern —
// consumers use `.filter($isVisibleInCurrentMode)` directly.
export const isVisibleInCurrentMode = derived(advancedView, ($adv) =>
  (p: { quality: PhenotypeQuality | null }) =>
    $adv || (p.quality !== 'dead' && p.quality !== 'mixed')
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
