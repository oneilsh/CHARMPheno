import type { Phenotype } from './types'

// argmax over a theta vector. The "true" dominant phenotype: which
// phenotype has the largest weight in this patient's mix.
export function dominant(theta: number[]): number {
  let best = 0
  for (let k = 1; k < theta.length; k++) if (theta[k] > theta[best]) best = k
  return best
}

// The dominant we DISPLAY. In advanced mode this is just `dominant`. In
// basic mode we skip phenotypes flagged dead/mixed - those are hidden
// from the basic-view UI. If every phenotype in theta is dead/mixed
// (extreme edge case) we fall back to the true dominant so the caller
// always gets a valid index back.
export function displayedDominant(
  theta: number[],
  phenotypes: Phenotype[],
  advancedView: boolean,
): number {
  if (advancedView) return dominant(theta)
  let best = -1
  let bestV = -1
  for (let k = 0; k < theta.length; k++) {
    const q = phenotypes[k]?.quality
    if (q === 'dead' || q === 'mixed') continue
    if (theta[k] > bestV) { bestV = theta[k]; best = k }
  }
  return best >= 0 ? best : dominant(theta)
}

// Fraction of draws whose DISPLAYED dominant phenotype is each k — a distribution
// over phenotypes that sums to 1. Unlike the mean of theta (which flattens toward
// an even mix as the draws disagree, since averaging a distribution over draws
// smooths it), this "most often leads" vote concentrates when the draws agree on a
// leading phenotype and only spreads when they genuinely disagree — the same
// agree/ambiguous signal the per-sample structure strip shows, summarised.
export function dominantVote(
  thetaSamples: number[][],
  phenotypes: Phenotype[],
  advancedView: boolean,
): number[] {
  const K = thetaSamples[0]?.length ?? phenotypes.length
  const vote = new Array<number>(K).fill(0)
  if (thetaSamples.length === 0) return vote
  for (const t of thetaSamples) vote[displayedDominant(t, phenotypes, advancedView)] += 1
  for (let k = 0; k < K; k++) vote[k] /= thetaSamples.length
  return vote
}
