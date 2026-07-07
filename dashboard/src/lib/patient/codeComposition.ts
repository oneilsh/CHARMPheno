import type { Model } from '../types'

// Sentinel phenotype id for the aggregated long-tail band, matching
// ProfileBar / ContributingCodes ($selectedPhenotypeId === -1).
export const OTHER_ID = -1

export interface PhenotypeSegment {
  k: number      // phenotype id, or OTHER_ID for the aggregated tail
  weight: number // share of this code's attribution; segments sum to 1 (or 0 if unexplained)
}

export interface CodeRow {
  w: number
  count: number
  segments: PhenotypeSegment[]
}

// Per-code posterior split phi(w,k) = theta[k]*beta[k][w] / sum_j theta[j]*beta[j][w],
// reduced to the phenotypes the profile bar shows (theta >= threshold) plus an
// aggregated Other bucket for the tail. Patient-conditioned: depends on theta.
export function codeComposition(
  theta: number[],
  codeBag: number[],
  beta: Model['beta'],
  K: number,
  otherThreshold = 0.05,
): CodeRow[] {
  const counts = new Map<number, number>()
  for (const w of codeBag) counts.set(w, (counts.get(w) ?? 0) + 1)

  const rows: CodeRow[] = []
  for (const [w, count] of counts) {
    let z = 0
    for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
    const segments: PhenotypeSegment[] = []
    let other = 0
    if (z > 0) {
      for (let j = 0; j < K; j++) {
        const weight = (beta[j][w] * theta[j]) / z
        if (weight === 0) continue
        if (theta[j] >= otherThreshold) segments.push({ k: j, weight })
        else other += weight
      }
    }
    if (other > 0) segments.push({ k: OTHER_ID, weight: other })
    rows.push({ w, count, segments })
  }
  return rows
}

// Aggregate reconciliation for the profile-bar residual overlay. theta_data(k) is the
// occurrence-weighted, normalized code evidence; the prior-supported remainder is the
// part of theta(k) the codes do not account for. explained + prior === theta per k.
//
// EXACTNESS: additive "prior + evidence" is exact only for the Dirichlet/LDA engine
// (gamma = alpha + sum_w c*phi). Under STM (softmax(eta_hat)) this is a principled
// heuristic; present it as an approximate cue, not an identity. See insight 0028.
export function explainedVsPrior(
  theta: number[],
  codeBag: number[],
  beta: Model['beta'],
  K: number,
): { explained: number[]; prior: number[] } {
  const counts = new Map<number, number>()
  for (const w of codeBag) counts.set(w, (counts.get(w) ?? 0) + 1)

  const raw = new Array(K).fill(0)
  for (const [w, c] of counts) {
    let z = 0
    for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
    if (z <= 0) continue
    for (let j = 0; j < K; j++) raw[j] += (c * beta[j][w] * theta[j]) / z
  }
  const total = raw.reduce((a, b) => a + b, 0)
  const thetaData = total > 0 ? raw.map((x) => x / total) : new Array(K).fill(0)

  const explained = new Array(K).fill(0)
  const prior = new Array(K).fill(0)
  for (let k = 0; k < K; k++) {
    explained[k] = Math.min(theta[k], thetaData[k])
    prior[k] = Math.max(0, theta[k] - thetaData[k])
  }
  return { explained, prior }
}
