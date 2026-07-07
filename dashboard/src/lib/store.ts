import { writable, derived } from 'svelte/store'
import type { UMAP } from 'umap-js'
import type { CohortManifest, DashboardBundle, Phenotype, PhenotypeQuality, SyntheticCohort } from './types'
import { computeJsdMds } from './mds'
import { jsd, phenotypesContainingCode } from './inference'
import { cohortCoverage, sampleThetaCohort } from './conditioning/coverage'

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

// Compare subtab: the two phenotypes being contrasted in the Difference pane.
// Set by clicking a correlation cell (a = row phenotype, b = column phenotype);
// a === b (a diagonal click) clears it.
export const comparePair = writable<{ a: number; b: number } | null>(null)
export const simulatorPrefix = writable<number[]>([])     // vocab indices (trimmed)
export const advancedView = writable<boolean>(false)

// Patient atlas: color points by each patient's recorded gating group
// instead of dominant phenotype. Only meaningful for gated STM bundles
// (SyntheticPatient.group is null otherwise); PatientMap falls back to the
// default dominant-phenotype coloring when off or when patients carry no
// group.
export const colorByGroup = writable<boolean>(false)

// Patient-coverage threshold τ. A patient is counted as "having" the phenotype
// when at least 1% of their coded activity is attributed to the topic. Exposed
// as a store so the components that read $tauThreshold work unchanged; there is
// no user-facing slider.
export const tauThreshold = writable<number>(0.01)

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

// Back-compat alias: the shipped four-quadrant coverageReader reads the
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
export type PhenotypeSortKey =
  | 'id' | 'label' | 'cohort' | 'prevalence' | 'coherence'
export const phenotypeSortBy = writable<PhenotypeSortKey>('prevalence')
export const phenotypeSortDir = writable<'asc' | 'desc'>('desc')


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

const ATLAS_COHORT_N = 1500
const ATLAS_COHORT_SEED = 20260706

function isStmBundle(b: DashboardBundle | null): b is DashboardBundle {
  return !!b && !!b.covariateEffects && !!b.correlation
}

// Marginal (baseline) atlas cohort: the corpus's natural covariate/group mix.
// Recomputed ONLY when the bundle changes (not on covariate edits), so it is a
// stable per-bundle reference for absolute bubble scaling. Null for non-STM.
export const atlasBaselineThetaCohort = derived(bundle, ($b) =>
  isStmBundle($b)
    ? sampleThetaCohort({ bundle: $b, active: false, values: {}, n: ATLAS_COHORT_N, seed: ATLAS_COHORT_SEED })
    : null
)

// Display atlas cohort: the marginal baseline when covariates are off (reused —
// no resample), or a covariate-fixed cohort (group still marginal) when on.
export const atlasThetaCohort = derived(
  [bundle, atlasConditioning, atlasBaselineThetaCohort],
  ([$b, $cond, $baseline]) => {
    if (!isStmBundle($b)) return null
    if (!$cond.covariateActive) return $baseline
    return sampleThetaCohort({ bundle: $b, active: true, values: $cond.values, n: ATLAS_COHORT_N, seed: ATLAS_COHORT_SEED })
  }
)

// (Phenotype) -> coverage. STM bundles read the sampled cohort (fraction of
// patients with θ > τ); non-STM bundles fall back to the empirical θ-histogram
// fractionAboveTau, unchanged. The atlas encodes cohort as COLOR (not a filter),
// so coverage is never masked by group.
function coverageFrom(
  cohort: number[][] | null, b: DashboardBundle | null, tau: number,
): (p: Phenotype) => number {
  const edges = b?.phenotypes.theta_histogram_bin_edges
  if (cohort && b) {
    const cov = cohortCoverage(cohort, tau, b.model.K)
    return (p: Phenotype) => cov[p.id] ?? 0
  }
  return (p: Phenotype) => fractionAboveTau(p, edges, tau)
}

// Display coverage reader (the atlas's current state).
export const coverageReader = derived(
  [bundle, atlasThetaCohort, tauThreshold],
  ([$b, $cohort, $tau]) => coverageFrom($cohort, $b, $tau)
)

// Marginal (baseline) coverage reader — the stable absolute-scale anchor for
// bubble size (see TopicMap). Equals coverageReader when covariates are off.
export const baselineCoverageReader = derived(
  [bundle, atlasBaselineThetaCohort, tauThreshold],
  ([$b, $cohort, $tau]) => coverageFrom($cohort, $b, $tau)
)

// ── Predictive-gain readers (additive; Task 6a plumbing) ───────────────────
// These consume the hydrated per-phenotype fields set by bundle.ts's
// hydratePredictiveGain (bundle-level `predictive_gain` distributed onto each
// Phenotype by index). Task 6a only makes the data available: the headline
// readout and TopicMap encoding still read `coverageReader` unchanged —
// see Task 6b for the value-sensitive swap once real numbers are in.

// (Phenotype) -> number reader for "presence": the fraction of a topic's
// docs clearing its own permutation null (held-out predictive signal).
// Falls back to the existing prevalence reader's fractionAboveTau base when
// `presence` hasn't been hydrated (older/non-gated bundles), so callers can
// adopt this reader without a conditional at each call site.
export const presenceReader = derived(
  [bundle, tauThreshold],
  ([$b, $tau]) => {
    const edges = $b?.phenotypes.theta_histogram_bin_edges
    return (p: Phenotype) =>
      typeof p.presence === 'number' && Number.isFinite(p.presence)
        ? p.presence
        : fractionAboveTau(p, edges, $tau)
  }
)

// (Phenotype) -> number reader for "depth": unique-contribution share.
// null/undefined (undefined depth, or a bundle with no predictive_gain at
// all) reads as 0 rather than throwing/NaN-ing through downstream math.
export const depthReader = derived([bundle], () =>
  (p: Phenotype) => p.depth ?? 0
)

// (Phenotype) -> number reader for "mean_gain": the topic's mean unique
// held-out predictive contribution (nats). Mirrors presenceReader/depthReader
// — null/undefined (older/non-gated bundles with no predictive_gain block)
// reads as 0 rather than throwing/NaN-ing through downstream math. This is
// the primary size-source for the Task 6b TopicMap bubble-size swap (see
// meanGainReader usage in atlas/TopicMap.svelte).
export const meanGainReader = derived([bundle], () =>
  (p: Phenotype) => p.mean_gain ?? 0
)

// Bundle-level predictive_gain diagnostics accessor (bin edges, null_band,
// scale, etc.) for components that need more than the per-phenotype
// hydrated fields. Null when the bundle has no predictive_gain block.
export const predictiveGain = derived(bundle, ($b) => $b?.phenotypes.predictive_gain ?? null)

// Returns a predicate (p) -> boolean for whether a phenotype should be shown
// in the current view mode. Simple mode hides `dead` and `mixed` topics;
// advanced mode shows everything. Follows the coverageReader pattern —
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
