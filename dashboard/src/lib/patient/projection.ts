import { UMAP, cosine } from 'umap-js'
import { get } from 'svelte/store'
import { createRng } from '../sampling'
import { cohort, patientProjection, patientProjectionFitting } from '../store'
import type { SyntheticCohort } from '../types'

// Shared UMAP parameters - both PatientMap and the simulator's mini-atlas
// rely on these to produce the SAME layout, so the simulator's transformed
// sim points land where the cohort dots do.
export const UMAP_N_NEIGHBORS = 50

// The cohort whose fit is currently in flight (null when idle). Lets a newer
// cohort supersede an older fit: ensurePatientProjection starts a fresh fit for
// the new cohort object, and the stale fit both stops early (its epoch callback
// returns false) and discards its result on completion. The previous
// synchronous implementation couldn't race - it ran to completion inside one
// setTimeout - but fitAsync yields between epochs, so a regenerate mid-fit is
// now possible. Keyed on the cohort object (not its seed) so two same-seed
// cohorts never alias each other.
let fittingCohort: SyntheticCohort | null = null

async function fitCohortUmapAsync(c: SyntheticCohort): Promise<void> {
  try {
    const patients = c.patients
    const thetas = patients.map((p) => p.theta)
    const umap = new UMAP({
      nComponents: 2,
      nNeighbors: Math.min(UMAP_N_NEIGHBORS, Math.max(2, patients.length - 1)),
      minDist: 0.15,
      distanceFn: cosine,
      random: createRng(c.seed),
    })
    // fitAsync yields to the event loop after every optimization epoch, so the
    // UI stays responsive while the layout settles. The only synchronous cost
    // is the one-time kNN graph build inside initializeFit; the deferred start
    // (below) keeps that off the critical first-paint path. The callback aborts
    // the run if a newer cohort has superseded this one.
    const patientCoords = await umap.fitAsync(thetas, () => fittingCohort === c)
    if (fittingCohort !== c) return // superseded; a newer fit owns the store
    patientProjection.set({ patientCoords, cohort: c, umap })
  } catch {
    // Leave the projection null so a later ensurePatientProjection can retry.
  } finally {
    if (fittingCohort === c) {
      fittingCohort = null
      patientProjectionFitting.set(false)
    }
  }
}

// Triggers a UMAP fit on the current cohort if one isn't cached for this seed.
// Idempotent: a no-op if the projection is already current or a fit for this
// seed is already running. Returns immediately - callers subscribe to
// `patientProjection` to observe the result. Safe to call eagerly on load
// (App.svelte) so the layout is ready before the user reaches the Patient tab.
export function ensurePatientProjection(): void {
  const c = get(cohort)
  if (!c) return
  const existing = get(patientProjection)
  if (existing && existing.cohort === c) return
  if (fittingCohort === c) return
  fittingCohort = c
  patientProjectionFitting.set(true)
  // Defer the start so the synchronous kNN build doesn't block the paint that
  // triggered this call (first paint on load, or the tab swap that mounts the map).
  setTimeout(() => { void fitCohortUmapAsync(c) }, 0)
}
