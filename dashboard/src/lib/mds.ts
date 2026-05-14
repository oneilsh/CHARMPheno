import { jsd } from './inference'

function jacobiEig(A: number[][], maxSweeps = 60, tol = 1e-10): { values: number[]; vectors: number[][] } {
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
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
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
