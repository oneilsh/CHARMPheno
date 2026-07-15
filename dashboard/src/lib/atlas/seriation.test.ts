import { describe, it, expect } from 'vitest'
import {
  jacobiEigenSymmetric,
  fiedlerOrder,
  seriateWithinBlocks,
  seriateRect,
  seriateTSPCorr,
} from './seriation'

// --- eigensolver ----------------------------------------------------------

function reconstruct(values: number[], vectors: number[][]): number[][] {
  // A = Σ_k λ_k v_k v_kᵀ
  const n = values.length
  const A = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        A[i][j] += values[k] * vectors[k][i] * vectors[k][j]
      }
    }
  }
  return A
}

describe('jacobiEigenSymmetric', () => {
  it('diagonalizes a 2×2 with known eigenvalues', () => {
    const { values } = jacobiEigenSymmetric([[2, 1], [1, 2]])
    // eigenvalues of [[2,1],[1,2]] are 1 and 3
    expect(values.slice().sort((a, b) => a - b)).toEqual([expect.closeTo(1, 10), expect.closeTo(3, 10)])
  })

  it('reconstructs a symmetric matrix from its eigendecomposition', () => {
    const A = [
      [4, 1, 0.5, 0.2],
      [1, 3, 0.3, 0.1],
      [0.5, 0.3, 2, 0.4],
      [0.2, 0.1, 0.4, 1],
    ]
    const { values, vectors } = jacobiEigenSymmetric(A)
    const R = reconstruct(values, vectors)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) expect(R[i][j]).toBeCloseTo(A[i][j], 8)
  })

  it('eigenvectors are orthonormal', () => {
    const A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    const { vectors } = jacobiEigenSymmetric(A)
    const dot = (u: number[], v: number[]) => u.reduce((s, x, i) => s + x * v[i], 0)
    for (let a = 0; a < 3; a++) {
      expect(dot(vectors[a], vectors[a])).toBeCloseTo(1, 8)
      for (let b = a + 1; b < 3; b++) expect(dot(vectors[a], vectors[b])).toBeCloseTo(0, 8)
    }
  })
})

// --- Fiedler ordering -----------------------------------------------------

// Robinson similarity: W[i][j] decreases with |i-j| (neighbors most similar).
function robinson(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0 : n - Math.abs(i - j))),
  )
}

describe('fiedlerOrder', () => {
  it('leaves an already-seriated Robinson matrix in order (or reversed)', () => {
    const n = 6
    const ord = fiedlerOrder(robinson(n))
    const identity = ord.every((v, i) => v === i)
    const reversed = ord.every((v, i) => v === n - 1 - i)
    expect(identity || reversed).toBe(true)
  })

  it('recovers the latent linear order from a shuffled Robinson matrix', () => {
    const n = 7
    const base = robinson(n)
    // pi[newRow] = trueRow — a fixed non-trivial shuffle
    const pi = [3, 0, 5, 1, 6, 2, 4]
    const Wshuf = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, b) => base[pi[a]][pi[b]]),
    )
    const ord = fiedlerOrder(Wshuf)
    // Mapping the recovered order back to true positions must be monotonic.
    const truePos = ord.map((k) => pi[k])
    const inc = truePos.every((v, i) => i === 0 || truePos[i - 1] < v)
    const dec = truePos.every((v, i) => i === 0 || truePos[i - 1] > v)
    expect(inc || dec).toBe(true)
  })
})

// --- Block-preserving seriation -------------------------------------------

describe('seriateWithinBlocks', () => {
  it('reorders within a block but never moves items across block boundaries', () => {
    // Two blocks: A (indices 0..4) carries a shuffled Robinson structure; B
    // (indices 5..6) is a trivial 2-item block.
    const nA = 5
    const baseA = robinson(nA)
    const piA = [2, 4, 0, 3, 1]
    const n = nA + 2
    const R: (number | null)[][] = Array.from({ length: n }, () => new Array(n).fill(0))
    for (let a = 0; a < nA; a++)
      for (let b = 0; b < nA; b++) {
        // scale into [-1,1]; similarity() will map back, ordering is scale-free
        R[a][b] = a === b ? 1 : baseA[piA[a]][piA[b]] / nA
      }
    const blockLabels = ['A', 'A', 'A', 'A', 'A', 'B', 'B']

    const perm = seriateWithinBlocks(R, blockLabels)

    // Block A stays within [0,5), block B within [5,7): partition preserved.
    for (let d = 0; d < nA; d++) expect(perm[d]).toBeGreaterThanOrEqual(0)
    for (let d = 0; d < nA; d++) expect(perm[d]).toBeLessThan(nA)
    expect(new Set(perm.slice(0, nA))).toEqual(new Set([0, 1, 2, 3, 4]))
    expect(perm.slice(nA)).toEqual([5, 6])

    // Within A, the recovered order is monotonic in the latent positions.
    const truePos = perm.slice(0, nA).map((k) => piA[k])
    const inc = truePos.every((v, i) => i === 0 || truePos[i - 1] < v)
    const dec = truePos.every((v, i) => i === 0 || truePos[i - 1] > v)
    expect(inc || dec).toBe(true)
  })

  it('is a valid permutation of 0..n-1', () => {
    const R: (number | null)[][] = [
      [1, 0.2, null],
      [0.2, 1, 0.1],
      [null, 0.1, 1],
    ]
    const perm = seriateWithinBlocks(R, ['A', 'A', 'A'])
    expect(perm.slice().sort((a, b) => a - b)).toEqual([0, 1, 2])
  })
})

// --- TSP (Hamiltonian-path) seriation -------------------------------------

describe('seriateTSPCorr', () => {
  // A Robinson CORRELATION matrix: R decreases with |i-j| (so dissimilarity
  // grows with distance). Values kept in (0, 0.5] so a uniform +offset test
  // below never clamps.
  function robinsonCorr(n: number): number[][] {
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : (1 - Math.abs(i - j) / n) * 0.5)),
    )
  }

  it('recovers the latent linear order from a shuffled Robinson correlation', () => {
    const n = 7
    const base = robinsonCorr(n)
    const pi = [3, 0, 5, 1, 6, 2, 4]
    const shuf = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, b) => base[pi[a]][pi[b]]),
    )
    const ord = seriateTSPCorr(shuf)
    const truePos = ord.map((k) => pi[k])
    const inc = truePos.every((v, i) => i === 0 || truePos[i - 1] < v)
    const dec = truePos.every((v, i) => i === 0 || truePos[i - 1] > v)
    expect(inc || dec).toBe(true)
  })

  it('is invariant to a uniform correlation offset (the common-factor pedestal)', () => {
    const n = 7
    const base = robinsonCorr(n)
    const pi = [3, 0, 5, 1, 6, 2, 4]
    const shuf = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, b) => base[pi[a]][pi[b]]),
    )
    // Inflate every off-diagonal correlation by the same amount — a stand-in for
    // a shared latent factor. Path-length seriation must not change.
    const offset = shuf.map((row, a) => row.map((r, b) => (a === b ? 1 : r + 0.3)))
    expect(seriateTSPCorr(offset)).toEqual(seriateTSPCorr(shuf))
  })

  it('returns a valid permutation and tolerates nulls', () => {
    const R: (number | null)[][] = [
      [1, 0.4, null, 0.1],
      [0.4, 1, 0.2, null],
      [null, 0.2, 1, 0.5],
      [0.1, null, 0.5, 1],
    ]
    const ord = seriateTSPCorr(R)
    expect(ord.slice().sort((a, b) => a - b)).toEqual([0, 1, 2, 3])
  })
})

describe('seriateRect', () => {
  it('recovers the row and column gradients of a shuffled rank-1 matrix', () => {
    // Planted bicluster: M[a][b] = f[a]·g[b], both monotone -> the leading
    // singular vectors are f and g, so ordering by them recovers the gradients.
    const f = [1, 2, 3, 4, 5]
    const g = [1, 2, 3, 4]
    const base = f.map((fa) => g.map((gb) => fa * gb))
    // shuffle rows by piR, cols by piC
    const piR = [3, 0, 4, 1, 2]
    const piC = [2, 0, 3, 1]
    const M = piR.map((r) => piC.map((c) => base[r][c]))

    const { rowOrder, colOrder } = seriateRect(M)

    const rowTrue = rowOrder.map((k) => piR[k])
    const colTrue = colOrder.map((k) => piC[k])
    const mono = (xs: number[]) =>
      xs.every((v, i) => i === 0 || xs[i - 1] < v) || xs.every((v, i) => i === 0 || xs[i - 1] > v)
    expect(mono(rowTrue)).toBe(true)
    expect(mono(colTrue)).toBe(true)
  })

  it('returns valid permutations and tolerates nulls', () => {
    const M: (number | null)[][] = [
      [0.2, null, 0.5],
      [0.1, 0.3, null],
    ]
    const { rowOrder, colOrder } = seriateRect(M)
    expect(rowOrder.slice().sort((a, b) => a - b)).toEqual([0, 1])
    expect(colOrder.slice().sort((a, b) => a - b)).toEqual([0, 1, 2])
  })
})
