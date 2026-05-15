import { UMAP, cosine } from 'umap-js'
import { get } from 'svelte/store'
import { createRng } from '../sampling'
import { cohort, patientProjection, patientProjectionFitting } from '../store'
import type { SyntheticPatient } from '../types'

// Shared UMAP parameters - both PatientMap and the simulator's mini-atlas
// rely on these to produce the SAME layout, so the simulator's transformed
// sim points land where the cohort dots do.
export const UMAP_N_NEIGHBORS = 50

function fitCohortUmap(patients: SyntheticPatient[], seed: number) {
  const thetas = patients.map((p) => p.theta)
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.min(UMAP_N_NEIGHBORS, Math.max(2, patients.length - 1)),
    minDist: 0.15,
    distanceFn: cosine,
    random: createRng(seed),
  })
  const patientCoords = umap.fit(thetas)
  return { patientCoords, seed, umap }
}

// Triggers a UMAP fit on the current cohort if one isn't cached for this
// seed. Idempotent and concurrency-safe via the patientProjectionFitting
// store flag. Returns nothing - callers subscribe to `patientProjection`
// to observe the result.
export function ensurePatientProjection(): void {
  const c = get(cohort)
  if (!c) return
  const existing = get(patientProjection)
  if (existing && existing.seed === c.seed) return
  if (get(patientProjectionFitting)) return
  patientProjectionFitting.set(true)
  setTimeout(() => {
    try {
      patientProjection.set(fitCohortUmap(c.patients, c.seed))
    } finally {
      patientProjectionFitting.set(false)
    }
  }, 0)
}
