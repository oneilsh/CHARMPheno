import type { Model, SyntheticCohort, SyntheticPatient } from './types'
import {
  createRng, sampleDirichlet, sampleCategorical, samplePoisson,
} from './sampling'

export interface CohortInput {
  model: Model
  meanCodesPerDoc: number
  n: number
  seed: number
  nNeighbors: number
}

function cosineNeighbors(thetas: number[][], k: number): number[][] {
  const n = thetas.length
  const norms = thetas.map((t) => Math.hypot(...t))
  const neighbors: number[][] = []
  const sims = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { sims[j] = -Infinity; continue }
      let dot = 0
      for (let d = 0; d < thetas[i].length; d++) dot += thetas[i][d] * thetas[j][d]
      const denom = (norms[i] || 1) * (norms[j] || 1)
      sims[j] = denom > 0 ? dot / denom : -Infinity
    }
    const idx = sims.map((_, i2) => i2)
    idx.sort((a, b) => sims[b] - sims[a])
    neighbors.push(idx.slice(0, k))
  }
  return neighbors
}

export function generateCohort(input: CohortInput): SyntheticCohort {
  const { model, meanCodesPerDoc, n, seed, nNeighbors } = input
  const rng = createRng(seed)
  const thetas: number[][] = []
  const bags: number[][] = []
  for (let i = 0; i < n; i++) {
    const theta = sampleDirichlet(model.alpha, rng)
    const nCodes = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const bag: number[] = []
    for (let c = 0; c < nCodes; c++) {
      const z = sampleCategorical(theta, rng)
      const w = sampleCategorical(model.beta[z], rng)
      bag.push(w)
    }
    thetas.push(theta)
    bags.push(bag)
  }
  const nbrIdx = cosineNeighbors(thetas, Math.min(nNeighbors, n - 1))
  const pad = (i: number) => `synth_${i.toString().padStart(4, '0')}`
  const patients: SyntheticPatient[] = thetas.map((theta, i) => ({
    id: pad(i),
    theta,
    code_bag: bags[i],
    neighbors: nbrIdx[i].map(pad),
  }))
  return { patients, seed }
}
