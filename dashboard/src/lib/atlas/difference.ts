import { relevance } from '../inference'

// Rank conditions by the difference in term relevance between two phenotypes A
// and B. Relevance is the LDAvis measure (Sievert & Shirley 2014):
// lambda*log(beta) + (1-lambda)*log(beta/p). delta(w) = rel_A(w) - rel_B(w);
// large positive delta marks a condition distinctive of A, large negative of B.
// The metric is a pure beta (term-weight) contrast, so it is defined even for
// phenotype pairs whose topic correlation is unidentified (a cross-group cell).
export interface RankedDelta {
  index: number
  delta: number
  relA: number
  relB: number
}

export function topDifferentialCodes(input: {
  betaA: number[]
  betaB: number[]
  pw: number[]
  lambda: number
  n: number
}): { aSide: RankedDelta[]; bSide: RankedDelta[] } {
  const { betaA, betaB, pw, lambda, n } = input
  const rows: RankedDelta[] = betaA.map((_, i) => {
    const relA = relevance(betaA[i], pw[i] ?? 0, lambda)
    const relB = relevance(betaB[i], pw[i] ?? 0, lambda)
    return { index: i, delta: relA - relB, relA, relB }
  })
  // Ascending-delta candidates need finite delta; a -Infinity delta means A has
  // beta 0 there (belongs to B side, not A) and vice versa. NaN (both 0) drops.
  const finite = rows.filter((r) => Number.isFinite(r.delta))
  const desc = finite.slice().sort((a, b) => b.delta - a.delta)
  const asc = finite.slice().sort((a, b) => a.delta - b.delta)
  return { aSide: desc.slice(0, n), bSide: asc.slice(0, n) }
}
