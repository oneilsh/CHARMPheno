import { sampleStandardNormal } from '../sampling'
import type { CovariateEffects, Correlation } from '../types'

// Lower-triangular Cholesky factor L with L Lᵀ = A. Throws if A is not
// positive-definite (a non-positive pivot). Textbook Cholesky-Banachiewicz;
// the covariance sub-blocks it factors here are small (~40x40).
export function cholesky(A: number[][]): number[][] {
  const n = A.length
  const L: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j]
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k]
      if (i === j) {
        if (sum <= 0) throw new Error('cholesky: matrix is not positive-definite')
        L[i][j] = Math.sqrt(sum)
      } else {
        L[i][j] = sum / L[j][j]
      }
    }
  }
  return L
}

// One draw from Normal(mean, L Lᵀ): mean + L z, z standard-normal.
export function mvnDraw(mean: number[], L: number[][], rng: () => number): number[] {
  const n = mean.length
  const z = new Array<number>(n)
  for (let i = 0; i < n; i++) z[i] = sampleStandardNormal(rng)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = mean[i]
    for (let k = 0; k <= i; k++) s += L[i][k] * z[k]
    out[i] = s
  }
  return out
}

// Faithful STM forward draw: theta = softmax(eta), eta ~ Normal(Gamma^T x, Sigma)
// (logistic-normal prior; Blei & Lafferty 2007). The reference topic is pinned
// eta = 0 and excluded from Gamma's non-zero rows and from Sigma (correlation.R
// over the K-1 free topics). For a gated draw we restrict to the allowed set
// (background union the selected group) so the Sigma sub-block never includes a
// cross-group (unidentified/null) cell and is positive-definite by construction.
export function sampleConditionedTheta(args: {
  effects: CovariateEffects
  x: number[]
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
  rng: () => number
}): number[] {
  const { effects, x, correlation, topicBlocks, group, rng } = args
  const K = effects[0]?.per_topic.length ?? 0
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order        // display id per R row (free topics)

  // Allowed display-topic ids: all topics if not gated, else background plus the
  // selected group's foreground (null group = background only).
  const allowed = (k: number): boolean => {
    if (!topicBlocks) return true
    const b = topicBlocks[k]
    return b === 'background' || b === group
  }

  // Free R rows to sample: in topic_order, allowed, and not the reference.
  const freeIdx: number[] = []          // indices into correlation.R / order
  for (let r = 0; r < order.length; r++) {
    const k = order[r]
    if (k !== ref && allowed(k)) freeIdx.push(r)
  }

  // Mean eta over the free rows: mu_k = Gamma^T x (sum over covariate effects).
  const mean = freeIdx.map((r) => {
    const k = order[r]
    let m = 0
    for (const e of effects) m += e.per_topic[k] * x[effects.indexOf(e)]
    return m
  })

  // Sigma sub-block over the free rows (guaranteed non-null / PD).
  const Sigma = freeIdx.map((ri) =>
    freeIdx.map((rj) => correlation.R[ri][rj] as number))

  const etaFree = freeIdx.length
    ? mvnDraw(mean, cholesky(Sigma), rng)
    : []

  // Assemble eta over all K display topics: reference -> 0, free -> drawn,
  // masked -> -Infinity (exactly zero after softmax).
  const eta = new Array<number>(K).fill(-Infinity)
  if (ref >= 0 && allowed(ref)) eta[ref] = 0
  freeIdx.forEach((r, i) => { eta[order[r]] = etaFree[i] })

  const finite = eta.filter((e) => e !== -Infinity)
  const mx = finite.length ? Math.max(...finite) : 0
  const exp = eta.map((e) => (e === -Infinity ? 0 : Math.exp(e - mx)))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}
