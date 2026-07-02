import { sampleStandardNormal } from '../sampling'

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
