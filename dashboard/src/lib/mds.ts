import { jsd } from './inference'

export function jacobiEig(A: number[][], maxSweeps = 60, tol = 1e-10): { values: number[]; vectors: number[][] } {
  const n = A.length
  const a = A.map((r) => r.slice())
  const v: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  )
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) off += Math.abs(a[p][q])
    if (off < tol) break
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p][q]
      if (Math.abs(apq) < 1e-14) continue
      const theta = (a[q][q] - a[p][p]) / (2 * apq)
      // When diagonals are equal theta=0; Math.sign(0)===0 would skip the
      // rotation and leave degenerate eigenvalues unresolved (e.g. unit square).
      // Set t=1 (rotation by π/4) in that case.
      const t = theta === 0
        ? 1
        : Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
      const c = 1 / Math.sqrt(t * t + 1)
      const s = t * c
      const app = a[p][p], aqq = a[q][q]
      a[p][p] = app - t * apq
      a[q][q] = aqq + t * apq
      a[p][q] = 0; a[q][p] = 0
      for (let r = 0; r < n; r++) {
        if (r !== p && r !== q) {
          const arp = a[r][p], arq = a[r][q]
          a[r][p] = c * arp - s * arq
          a[p][r] = a[r][p]
          a[r][q] = s * arp + c * arq
          a[q][r] = a[r][q]
        }
      }
      for (let r = 0; r < n; r++) {
        const vrp = v[r][p], vrq = v[r][q]
        v[r][p] = c * vrp - s * vrq
        v[r][q] = s * vrp + c * vrq
      }
    }
  }
  const values = a.map((row, i) => row[i])
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i])
  return {
    values: order.map((i) => values[i]),
    vectors: v.map((row) => order.map((i) => row[i])),
  }
}

export function classicalMds(distance: number[][], d = 2): number[][] {
  const n = distance.length
  const Dsq = distance.map((row) => row.map((v) => v * v))
  const rowMeans = Dsq.map((r) => r.reduce((a, b) => a + b, 0) / n)
  const grandMean = rowMeans.reduce((a, b) => a + b, 0) / n
  const B: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => -0.5 * (Dsq[i][j] - rowMeans[i] - rowMeans[j] + grandMean))
  )
  const { values, vectors } = jacobiEig(B)
  const coords: number[][] = Array.from({ length: n }, () => new Array(d).fill(0))
  for (let k = 0; k < d; k++) {
    const lam = Math.max(values[k], 0)
    const s = Math.sqrt(lam)
    for (let i = 0; i < n; i++) coords[i][k] = vectors[i][k] * s
  }
  return coords
}

// 2D PCA of a (n x d) data matrix. Returns the top-2 PCs and projected
// coords (n x 2). Used to project synthetic patient theta vectors into a
// de-novo 2D space that captures the directions of greatest variance in
// patient phenotype mixes . richer cluster structure than constraining
// patients to the phenotype atlas convex hull.
//
// Implementation: compute K x K covariance, Jacobi-eigendecompose, take
// the top-2 eigenvectors as the PC basis, project each row onto them.
// O(n*K^2) for covariance + O(K^3) for the eig. Cheap at our scale
// (n=500, K=80) compared to O(n^3) classical MDS on patient pairs.
export function pca2d(data: number[][]): { coords: number[][]; basis: number[][]; mean: number[] } {
  const n = data.length
  if (n === 0) return { coords: [], basis: [[], []], mean: [] }
  const d = data[0].length
  const mean = new Array<number>(d).fill(0)
  for (const row of data) for (let j = 0; j < d; j++) mean[j] += row[j]
  for (let j = 0; j < d; j++) mean[j] /= n
  // Covariance (1 / (n - 1)) * sum (x - mean)(x - mean)^T
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  const denom = Math.max(1, n - 1)
  for (const row of data) {
    for (let i = 0; i < d; i++) {
      const di = row[i] - mean[i]
      for (let j = i; j < d; j++) {
        const dj = row[j] - mean[j]
        cov[i][j] += di * dj
      }
    }
  }
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) {
    cov[i][j] /= denom
    if (i !== j) cov[j][i] = cov[i][j]
  }
  const { vectors } = jacobiEig(cov)
  const pc1 = vectors.map((row) => row[0])
  const pc2 = vectors.map((row) => row[1])
  const basis = [pc1, pc2]
  const coords: number[][] = data.map((row) => {
    let a = 0, b = 0
    for (let j = 0; j < d; j++) {
      const c = row[j] - mean[j]
      a += c * pc1[j]
      b += c * pc2[j]
    }
    return [a, b]
  })
  return { coords, basis, mean }
}

// Project external points (rows of `data`) into a PCA basis that was fit on
// a different dataset. Useful for placing reference points (e.g. the one-hot
// phenotype basis vectors) into the patient-space PCA so they appear at the
// positions a pure-on-that-phenotype patient would land.
export function projectPca(data: number[][], basis: number[][], mean: number[]): number[][] {
  const [pc1, pc2] = basis
  const d = mean.length
  return data.map((row) => {
    let a = 0, b = 0
    for (let j = 0; j < d; j++) {
      const c = row[j] - mean[j]
      a += c * pc1[j]
      b += c * pc2[j]
    }
    return [a, b]
  })
}

export function computeJsdMds(beta: number[][]): number[][] {
  const K = beta.length
  const D: number[][] = Array.from({ length: K }, () => new Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const v = Math.sqrt(Math.max(0, jsd(beta[i], beta[j])))
      D[i][j] = v
      D[j][i] = v
    }
  }
  return classicalMds(D, 2)
}
