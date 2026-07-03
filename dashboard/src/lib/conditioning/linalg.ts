import { cholesky } from './logisticNormal'

// Cholesky with adaptive diagonal loading (ridge / Tikhonov regularization).
// The exported topic-correlation sub-block is only guaranteed positive-definite
// within a single block; a background-union-group sub-block can be indefinite,
// on which the bare `cholesky` throws. Adding a small multiple of the identity
// (grown geometrically until the factor exists) yields a usable, minimally
// perturbed factor. The load starts negligibly small so a genuinely-PD input is
// essentially unperturbed.
export function choleskyPD(A: number[][]): number[][] {
  const n = A.length
  if (n === 0) return []
  let meanDiag = 0
  for (let i = 0; i < n; i++) meanDiag += A[i][i]
  meanDiag = Math.abs(meanDiag) / n || 1
  const base = meanDiag * 1e-12
  for (let t = 0; t < 60; t++) {
    const load = t === 0 ? 0 : base * Math.pow(2, t)
    try {
      const M = load === 0 ? A : A.map((row, i) => row.map((v, j) => (i === j ? v + load : v)))
      return cholesky(M)
    } catch {
      // not PD at this load; increase and retry
    }
  }
  throw new Error('choleskyPD: not factorable even with regularization')
}

function forwardSub(L: number[][], b: number[]): number[] {
  const n = L.length
  const y = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k]
    y[i] = s / L[i][i]
  }
  return y
}

function backSub(L: number[][], y: number[]): number[] {
  const n = L.length
  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]
    x[i] = s / L[i][i]
  }
  return x
}

// Solve A x = b for symmetric (regularized-)PD A via its Cholesky factor.
export function solveSPD(A: number[][], b: number[]): number[] {
  const L = choleskyPD(A)
  return backSub(L, forwardSub(L, b))
}

// Inverse of symmetric (regularized-)PD A, column-by-column against I.
export function invSPD(A: number[][]): number[][] {
  const n = A.length
  const L = choleskyPD(A)
  const inv: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let j = 0; j < n; j++) {
    const e = new Array<number>(n).fill(0)
    e[j] = 1
    const col = backSub(L, forwardSub(L, e))
    for (let i = 0; i < n; i++) inv[i][j] = col[i]
  }
  return inv
}
