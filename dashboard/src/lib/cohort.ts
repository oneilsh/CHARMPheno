import type {
  Model, SyntheticCohort, SyntheticPatient, PhenotypeQuality,
} from './types'
import {
  createRng, sampleDirichlet, sampleCategorical, samplePoisson,
} from './sampling'

export interface CohortInput {
  model: Model
  meanCodesPerDoc: number
  n: number
  seed: number
  nNeighbors: number
  // Optional. Phenotype-quality labels keyed by phenotype id. When
  // provided, sampling becomes adaptive: we keep drawing until at least
  // one of {clean, messy} counts is ≥ MIN_DOMINANT, then truncate each
  // bucket down to a multiple of ROUND (so both are "round" numbers).
  // `n` becomes the initial batch size. Without these, all patients are
  // tagged isClean=true and exactly `n` are sampled.
  qualityByPhenotype?: (PhenotypeQuality | null)[]
}

// Adaptive-sizing knobs (used only when qualityByPhenotype is provided).
// We sample until max(cleanFloor, messyFloor) >= MIN_DOMINANT, then keep
// floor(count/ROUND)*ROUND from each bucket. That guarantees both bucket
// sizes are round numbers and the larger bucket is at least MIN_DOMINANT.
const ROUND = 100
const MIN_DOMINANT = 1000

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

function dominantIdx(theta: number[]): number {
  let best = 0
  for (let k = 1; k < theta.length; k++) if (theta[k] > theta[best]) best = k
  return best
}

export function generateCohort(input: CohortInput): SyntheticCohort {
  const {
    model, meanCodesPerDoc, n, seed, nNeighbors,
    qualityByPhenotype,
  } = input
  const rng = createRng(seed)
  let thetas: number[][] = []
  let bags: number[][] = []
  let isCleanFlags: boolean[] = []

  // Sample one (theta, bag, isClean) without storing it yet, so the
  // rejection-style path can choose whether to accept.
  const drawOne = () => {
    const theta = sampleDirichlet(model.alpha, rng)
    const nCodes = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const bag: number[] = []
    for (let c = 0; c < nCodes; c++) {
      const z = sampleCategorical(theta, rng)
      const w = sampleCategorical(model.beta[z], rng)
      bag.push(w)
    }
    let isClean = true
    if (qualityByPhenotype) {
      const q = qualityByPhenotype[dominantIdx(theta)]
      isClean = !(q === 'dead' || q === 'mixed')
    }
    return { theta, bag, isClean }
  }

  if (qualityByPhenotype) {
    // Adaptive sampling: draw an initial batch of `n`, then keep going
    // until max(cleanFloor, messyFloor) >= MIN_DOMINANT. Each draw lands
    // in either the clean or messy bucket - we never reject. At the end
    // we truncate from the tail of each bucket down to a ROUND multiple
    // so both displayed counts are round numbers regardless of the
    // model's clean/messy mix.
    let cleanCount = 0
    let messyCount = 0
    const drawAndPush = () => {
      const d = drawOne()
      thetas.push(d.theta)
      bags.push(d.bag)
      isCleanFlags.push(d.isClean)
      if (d.isClean) cleanCount++
      else messyCount++
    }
    for (let i = 0; i < n; i++) drawAndPush()
    const meets = () =>
      Math.max(Math.floor(cleanCount / ROUND), Math.floor(messyCount / ROUND))
        * ROUND >= MIN_DOMINANT
    // Hard cap protects against degenerate models with ~0 of one bucket.
    const hardCap = Math.max(n, MIN_DOMINANT) * 5
    while (!meets() && thetas.length < hardCap) drawAndPush()

    // Truncate each bucket down to its ROUND multiple, preserving
    // sample order. Doing this with two separate kept-counters is
    // simpler than a global truncation that respects both quotas.
    const cleanKeep = Math.floor(cleanCount / ROUND) * ROUND
    const messyKeep = Math.floor(messyCount / ROUND) * ROUND
    let keptClean = 0
    let keptMessy = 0
    const t2: number[][] = []
    const b2: number[][] = []
    const f2: boolean[] = []
    for (let i = 0; i < thetas.length; i++) {
      if (isCleanFlags[i] && keptClean < cleanKeep) {
        t2.push(thetas[i]); b2.push(bags[i]); f2.push(true); keptClean++
      } else if (!isCleanFlags[i] && keptMessy < messyKeep) {
        t2.push(thetas[i]); b2.push(bags[i]); f2.push(false); keptMessy++
      }
    }
    thetas = t2; bags = b2; isCleanFlags = f2
  } else {
    for (let i = 0; i < n; i++) {
      const d = drawOne()
      thetas.push(d.theta)
      bags.push(d.bag)
      isCleanFlags.push(d.isClean)
    }
  }

  const total = thetas.length
  const nbrIdx = cosineNeighbors(thetas, Math.min(nNeighbors, total - 1))
  const pad = (i: number) => `S${i.toString().padStart(4, '0')}`
  const patients: SyntheticPatient[] = thetas.map((theta, i) => ({
    id: pad(i),
    theta,
    code_bag: bags[i],
    neighbors: nbrIdx[i].map(pad),
    isClean: isCleanFlags[i],
  }))
  return { patients, seed }
}
