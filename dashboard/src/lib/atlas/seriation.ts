// Spectral seriation for the correlation heatmap.
//
// Goal: reorder topics so that similar (highly correlated) ones sit adjacent,
// revealing the block-diagonal / gradient ("Robinson") structure a raw index
// order hides — without hierarchical clustering, and preserving the gated
// block partition (we seriate WITHIN each block, never across).
//
// Method: spectral seriation (Atkins, Boman & Hendrickson 1998, "A spectral
// algorithm for seriation and the consecutive-ones problem", SIAM J. Comput.).
// Given a symmetric nonnegative similarity matrix W, form the Laplacian
// L = D − W (D = diag of row sums), and order the items by the value of the
// FIEDLER vector — the eigenvector of the second-smallest eigenvalue of L. The
// smallest eigenvalue is 0 with the constant eigenvector; the Fiedler vector is
// the smoothest non-trivial mode, and sorting by it is the provably optimal
// linear arrangement for a Robinsonian similarity.
//
// Eigenpairs come from a hand-rolled cyclic Jacobi solver (no matrix library):
// symmetric, small (per-block, ~tens of topics), and numerically robust.

// --- Symmetric eigensolver (cyclic Jacobi) --------------------------------

export interface Eigen {
  values: number[] // eigenvalue per index
  vectors: number[][] // vectors[k] = k-th eigenvector (length n), for values[k]
}

// Classic cyclic Jacobi diagonalization of a real symmetric matrix. Returns
// eigenvalues and eigenvectors (as rows of `vectors`). Rotations are applied as
// A <- Jᵀ A J, accumulating J into V so its columns are the eigenvectors.
export function jacobiEigenSymmetric(Ain: number[][], maxSweeps = 100, tol = 1e-14): Eigen {
  const n = Ain.length
  const A = Ain.map((row) => row.slice())
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  )

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Sum of squared off-diagonals: our convergence measure.
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q]
    if (off <= tol) break

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q]
        if (Math.abs(apq) < 1e-300) continue
        // Rotation angle that annihilates A[p][q]: tan(2θ) = 2·apq/(app − aqq).
        const theta = (A[q][q] - A[p][p]) / (2 * apq)
        let t = 1 / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        if (theta < 0) t = -t
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        // A <- A J (rotate columns p,q)
        for (let k = 0; k < n; k++) {
          const akp = A[k][p]
          const akq = A[k][q]
          A[k][p] = c * akp - s * akq
          A[k][q] = s * akp + c * akq
        }
        // A <- Jᵀ A (rotate rows p,q)
        for (let k = 0; k < n; k++) {
          const apk = A[p][k]
          const aqk = A[q][k]
          A[p][k] = c * apk - s * aqk
          A[q][k] = s * apk + c * aqk
        }
        // V <- V J (accumulate eigenvectors)
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p]
          const vkq = V[k][q]
          V[k][p] = c * vkp - s * vkq
          V[k][q] = s * vkp + c * vkq
        }
      }
    }
  }

  const values = A.map((row, i) => row[i])
  const vectors = Array.from({ length: n }, (_, k) =>
    Array.from({ length: n }, (_, i) => V[i][k]),
  )
  return { values, vectors }
}

// --- Fiedler ordering ------------------------------------------------------

// Index of the eigenvector for the LARGEST eigenvalue (leading mode).
export function leadingEigenvector(A: number[][]): number[] {
  const { values, vectors } = jacobiEigenSymmetric(A)
  let best = 0
  for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k
  return vectors[best]
}

// Order the rows of a symmetric nonnegative similarity matrix W by their Fiedler
// component. Returns a permutation `ord` where ord[displayPos] = originalIndex.
export function fiedlerOrder(W: number[][]): number[] {
  const n = W.length
  if (n <= 2) return W.map((_, i) => i)

  // Laplacian L = D − W (off-diagonal weights only; diagonal = degree).
  const L: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    let deg = 0
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      deg += W[i][j]
      L[i][j] = -W[i][j]
    }
    L[i][i] = deg
  }

  const { values, vectors } = jacobiEigenSymmetric(L)
  // Eigen-indices sorted by eigenvalue; [0] is ~0 (constant mode), [1] = Fiedler.
  const byVal = values.map((_, k) => k).sort((a, b) => values[a] - values[b])
  const fiedler = vectors[byVal[1]]
  // Stable sort of item indices by Fiedler component (ties keep input order).
  return W.map((_, i) => i).sort((a, b) => fiedler[a] - fiedler[b] || a - b)
}

// --- Block-preserving seriation -------------------------------------------

// Map a correlation R (in [-1, 1], possibly null) to a nonnegative similarity in
// [0, 1]: perfectly anti-correlated -> 0, uncorrelated -> 0.5, correlated -> 1.
function similarity(r: number | null): number {
  if (r == null || Number.isNaN(r)) return 0.5 // neutral for unmeasured pairs
  return (Math.max(-1, Math.min(1, r)) + 1) / 2
}

// Seriate a SYMMETRIC correlation sub-matrix (a within-block panel): map R to a
// nonnegative similarity and Fiedler-order it. Returns an order of 0..m-1.
export function seriateSymmetricCorr(subR: (number | null)[][]): number[] {
  const m = subR.length
  if (m <= 2) return Array.from({ length: m }, (_, i) => i)
  const W: number[][] = Array.from({ length: m }, (_, a) =>
    Array.from({ length: m }, (_, b) => (a === b ? 0 : similarity(subR[a][b]))),
  )
  return fiedlerOrder(W)
}

// --- TSP (Hamiltonian-path) seriation -------------------------------------
//
// Order items so the total dissimilarity between ADJACENT items is minimal —
// the shortest open path visiting each item once (the "path length" seriation
// criterion; Hahsler, Hornik & Buchta 2008, "Getting things in order: an
// introduction to the R package seriation", J. Stat. Soft. 25(3)).
//
// Why this and not spectral (Fiedler) ordering: a path over n items always has
// exactly n-1 edges, so adding a constant to EVERY dissimilarity shifts every
// ordering's cost by the same (n-1)·c and cannot change the optimum. A common
// latent factor (e.g. shared care-burden across a group of topics) inflates
// every correlation by roughly the same amount — a uniform dissimilarity offset
// — so path-length seriation structurally ignores it and follows the residual
// local cluster structure, exactly the structure a mean-dominated spectral
// order washes out. Solved heuristically (nearest-neighbour construction +
// 2-opt); deterministic (fixed start, no randomness), and blocks are small
// (tens of topics), so the quadratic 2-opt passes are negligible.

// Correlation -> dissimilarity: perfectly correlated -> 0, uncorrelated -> 1,
// anti-correlated -> 2. Unmeasured pairs are treated as uncorrelated (1).
function dissimilarity(r: number | null): number {
  if (r == null || Number.isNaN(r)) return 1
  return 1 - Math.max(-1, Math.min(1, r))
}

// Nearest-neighbour path from item 0, then 2-opt until no improving reversal
// remains. Returns an order of 0..n-1. Operates on an (n×n) dissimilarity D.
function pathTSP(D: number[][]): number[] {
  const n = D.length
  if (n <= 2) return Array.from({ length: n }, (_, i) => i)

  // Nearest-neighbour construction from a fixed start (determinism).
  const visited = new Array<boolean>(n).fill(false)
  const tour = [0]
  visited[0] = true
  for (let k = 1; k < n; k++) {
    const last = tour[tour.length - 1]
    let best = -1
    let bestD = Infinity
    for (let j = 0; j < n; j++) {
      if (!visited[j] && D[last][j] < bestD) {
        bestD = D[last][j]
        best = j
      }
    }
    tour.push(best)
    visited[best] = true
  }

  // 2-opt on the OPEN path: reversing tour[i..j] swaps the boundary edges
  // (i-1,i) and (j,j+1). Endpoints contribute no edge (cost 0), which keeps the
  // move count equal on both sides — the source of the uniform-offset invariance.
  let improved = true
  while (improved) {
    improved = false
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const before = (i > 0 ? D[tour[i - 1]][tour[i]] : 0) + (j < n - 1 ? D[tour[j]][tour[j + 1]] : 0)
        const after = (i > 0 ? D[tour[i - 1]][tour[j]] : 0) + (j < n - 1 ? D[tour[i]][tour[j + 1]] : 0)
        if (after + 1e-12 < before) {
          for (let lo = i, hi = j; lo < hi; lo++, hi--) {
            const t = tour[lo]
            tour[lo] = tour[hi]
            tour[hi] = t
          }
          improved = true
        }
      }
    }
  }
  return tour
}

// Seriate a SYMMETRIC correlation sub-matrix by path-length (TSP) ordering:
// map R to a dissimilarity and order items so adjacent ones are most similar.
export function seriateTSPCorr(subR: (number | null)[][]): number[] {
  const m = subR.length
  if (m <= 2) return Array.from({ length: m }, (_, i) => i)
  const D = subR.map((row, a) => row.map((r, b) => (a === b ? 0 : dissimilarity(r))))
  return pathTSP(D)
}

// Seriate a RECTANGULAR cross-block panel: rows and columns are different topic
// sets, so there is no symmetric similarity. Order rows by the leading left
// singular vector of M and columns by the CO-ORIENTED right singular vector
// (v ∝ Mᵀu), so the dominant co-cluster lands on the panel's main "diagonal".
// nulls (unmeasured pairs) are treated as 0. Returns independent row/col orders.
export function seriateRect(M: (number | null)[][]): {
  rowOrder: number[]
  colOrder: number[]
} {
  const nr = M.length
  const nc = nr ? M[0].length : 0
  const rowId = Array.from({ length: nr }, (_, i) => i)
  const colId = Array.from({ length: nc }, (_, i) => i)
  if (nr <= 1 || nc <= 1) return { rowOrder: rowId, colOrder: colId }

  const A = M.map((row) => row.map((v) => (v == null || Number.isNaN(v) ? 0 : v)))
  // Row Gram G = A Aᵀ (nr×nr); its leading eigenvector ∝ u1 (left singular vec).
  const G: number[][] = Array.from({ length: nr }, (_, a) =>
    Array.from({ length: nr }, (_, a2) => {
      let s = 0
      for (let b = 0; b < nc; b++) s += A[a][b] * A[a2][b]
      return s
    }),
  )
  const u = leadingEigenvector(G)
  // Co-oriented column coordinate v = Aᵀ u (∝ v1); keeps the bicluster aligned.
  const v = new Array<number>(nc).fill(0)
  for (let b = 0; b < nc; b++) {
    let s = 0
    for (let a = 0; a < nr; a++) s += A[a][b] * u[a]
    v[b] = s
  }
  const rowOrder = rowId.slice().sort((a, b) => u[a] - u[b] || a - b)
  const colOrder = colId.slice().sort((a, b) => v[a] - v[b] || a - b)
  return { rowOrder, colOrder }
}

// Seriate each contiguous block of `blockLabels` independently by spectral
// (Fiedler) ordering of its within-block similarity sub-matrix, leaving the
// block partition (positions and membership) intact. Returns a permutation
// `perm` where perm[displayPos] = original matrix index.
export function seriateWithinBlocks(
  R: (number | null)[][],
  blockLabels: string[],
): number[] {
  const n = blockLabels.length
  const perm = new Array<number>(n)
  let start = 0
  while (start < n) {
    let end = start + 1
    while (end < n && blockLabels[end] === blockLabels[start]) end++
    const m = end - start
    const sub = Array.from({ length: m }, (_, a) =>
      Array.from({ length: m }, (_, b) => R[start + a][start + b]),
    )
    const local = seriateSymmetricCorr(sub) // local indices in seriated order
    for (let k = 0; k < m; k++) perm[start + k] = start + local[k]
    start = end
  }
  return perm
}
