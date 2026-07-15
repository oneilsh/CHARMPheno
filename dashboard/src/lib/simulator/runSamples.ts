import { variationalEStep } from '../inference'
import { createRng, sampleCategorical, samplePoisson } from '../sampling'

export interface SimulatorRunInput {
  alpha: number[]
  beta: number[][]
  meanCodesPerDoc: number
  prefix: number[]
  nSamples: number
  seed: number
  // When true, refit theta via a variational E-step after every drawn
  // code so each generated token shifts the next token's distribution.
  // When false (default), theta is fit once on the prefix, used to draw
  // all `nNew` codes, then refit once at the end for reporting. The
  // autoregressive path is more faithful to the generative story but
  // scales as O(nSamples * nNew * E-step) rather than O(nSamples * 2).
  autoregressive?: boolean
  // STM only: a factory returning one conditioned theta draw. When present,
  // each sample's generative AND reported theta is this logistic-normal draw
  // instead of a Dirichlet prior/E-step estimate. The draw already
  // incorporates the prefix (see conditioning/recordPosterior.ts), so it is
  // reported as-is rather than re-inferred via the Dirichlet E-step, which
  // would re-diffuse it.
  conditionedTheta?: () => number[]
}
export interface SimulatorRunResult {
  thetaSamples: number[][]
  codeCountsSamples: Map<number, number>[]
  // code id -> (generating topic id -> total tokens of that code emitted by that
  // topic, summed over all samples). Each generated token draws z ~ theta then
  // w ~ beta[z], so this records which phenotype generated each predicted code —
  // used to group the posterior-predictive codes by their generating phenotype.
  codeTopicCounts: Map<number, Map<number, number>>
}

export function runSimulator(input: SimulatorRunInput): SimulatorRunResult {
  const {
    alpha, beta, meanCodesPerDoc, prefix, nSamples, seed,
    autoregressive = false, conditionedTheta,
  } = input
  const prefixCounts = new Map<number, number>()
  for (const w of prefix) prefixCounts.set(w, (prefixCounts.get(w) ?? 0) + 1)
  const rng = createRng(seed)
  const thetas: number[][] = []
  const bags: Map<number, number>[] = []
  const codeTopicCounts = new Map<number, Map<number, number>>()
  for (let s = 0; s < nSamples; s++) {
    const nNew = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const sampleCounts = new Map(prefixCounts)
    // Generative theta for THIS sample: conditioned logistic-normal draw when
    // provided (STM), else the prefix E-step's Dirichlet-based estimate.
    let genTheta: number[]
    let est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    if (conditionedTheta) {
      genTheta = conditionedTheta()
    } else {
      genTheta = est.theta
    }
    for (let n = 0; n < nNew; n++) {
      const z = sampleCategorical(genTheta, rng)
      const w = sampleCategorical(beta[z], rng)
      sampleCounts.set(w, (sampleCounts.get(w) ?? 0) + 1)
      // Attribute this generated token to the phenotype (topic z) that emitted it.
      let tm = codeTopicCounts.get(w)
      if (!tm) { tm = new Map<number, number>(); codeTopicCounts.set(w, tm) }
      tm.set(z, (tm.get(z) ?? 0) + 1)
      if (autoregressive && !conditionedTheta) {
        est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
        genTheta = est.theta
      }
    }
    // For a conditioned (STM) draw, report the conditioned/posterior theta
    // directly — it already incorporates the prefix, and re-inferring it
    // through the Dirichlet E-step would re-diffuse it (the rainbow bug).
    // Only the non-conditioned (Dirichlet) path refines via the E-step.
    if (conditionedTheta) {
      thetas.push(genTheta)
    } else {
      est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
      thetas.push(est.theta)
    }
    const completion = new Map<number, number>()
    for (const [w, c] of sampleCounts) {
      const pre = prefixCounts.get(w) ?? 0
      if (c - pre > 0) completion.set(w, c - pre)
    }
    bags.push(completion)
  }
  return { thetaSamples: thetas, codeCountsSamples: bags, codeTopicCounts }
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
