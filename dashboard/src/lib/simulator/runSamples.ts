import { variationalEStep } from '../inference'
import { createRng, sampleCategorical, samplePoisson } from '../sampling'

export interface SimulatorRunInput {
  alpha: number[]
  beta: number[][]
  meanCodesPerDoc: number
  prefix: number[]
  nSamples: number
  seed: number
}
export interface SimulatorRunResult {
  thetaSamples: number[][]
  codeCountsSamples: Map<number, number>[]
}

export function runSimulator(input: SimulatorRunInput): SimulatorRunResult {
  const { alpha, beta, meanCodesPerDoc, prefix, nSamples, seed } = input
  const prefixCounts = new Map<number, number>()
  for (const w of prefix) prefixCounts.set(w, (prefixCounts.get(w) ?? 0) + 1)
  const rng = createRng(seed)
  const thetas: number[][] = []
  const bags: Map<number, number>[] = []
  for (let s = 0; s < nSamples; s++) {
    const nNew = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const sampleCounts = new Map(prefixCounts)
    let est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    for (let n = 0; n < nNew; n++) {
      const z = sampleCategorical(est.theta, rng)
      const w = sampleCategorical(beta[z], rng)
      sampleCounts.set(w, (sampleCounts.get(w) ?? 0) + 1)
    }
    est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    thetas.push(est.theta)
    const completion = new Map<number, number>()
    for (const [w, c] of sampleCounts) {
      const pre = prefixCounts.get(w) ?? 0
      if (c - pre > 0) completion.set(w, c - pre)
    }
    bags.push(completion)
  }
  return { thetaSamples: thetas, codeCountsSamples: bags }
}

export function quantiles(values: number[], qs: number[]): number[] {
  const sorted = values.slice().sort((a, b) => a - b)
  return qs.map((q) => {
    if (sorted.length === 0) return 0
    const pos = q * (sorted.length - 1)
    const lo = Math.floor(pos), hi = Math.ceil(pos)
    if (lo === hi) return sorted[lo]
    return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo)
  })
}
