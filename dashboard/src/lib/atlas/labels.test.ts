import { describe, it, expect } from 'vitest'
import {
  labeledPhenotypeIds, labelRanks, labelHash, selectLabels,
  type LabelablePhenotype, type LabelCandidate,
} from './labels'

// Minimal phenotype stand-ins: the selector only reads id + corpus_prevalence
// (+ an optional visibility predicate). The point of this helper is that the
// labeled SET is a pure function of corpus_prevalence + block membership, so it
// does NOT reshuffle when the live covariate prevalence reader changes.
const p = (id: number, prev: number): LabelablePhenotype => ({ id, corpus_prevalence: prev })

describe('labeledPhenotypeIds', () => {
  it('ungated: returns the global top-N by corpus_prevalence', () => {
    const phenos = [p(0, 0.10), p(1, 0.50), p(2, 0.30), p(3, 0.05), p(4, 0.40)]
    const got = labeledPhenotypeIds(phenos, { perGroup: 2 })
    // top 2 by prevalence: id1 (0.50), id4 (0.40)
    expect(got).toEqual(new Set([1, 4]))
  })

  it('gated: takes top-perGroup WITHIN each block, so a minority block is still anchored', () => {
    // background block dominates prevalence; the foreground block is smaller but
    // must still get its own label(s) rather than being crowded out globally.
    const phenos = [
      p(0, 0.90), p(1, 0.80), p(2, 0.70), // background
      p(3, 0.05), p(4, 0.03),             // foreground (all smaller than any background)
    ]
    const blocks = ['background', 'background', 'background', 'cancer', 'cancer']
    const got = labeledPhenotypeIds(phenos, { blocks, perGroup: 1 })
    // top-1 of background (id0) AND top-1 of cancer (id3)
    expect(got).toEqual(new Set([0, 3]))
  })

  it('respects the visibility predicate (dead/mixed hidden in simple mode)', () => {
    const phenos = [p(0, 0.50), p(1, 0.40), p(2, 0.30)]
    const hidden = new Set([0]) // pretend id0 is a hidden (dead) topic
    const got = labeledPhenotypeIds(phenos, {
      perGroup: 1,
      isVisible: (q) => !hidden.has(q.id),
    })
    // id0 excluded, so the top visible is id1
    expect(got).toEqual(new Set([1]))
  })

  it('a group with fewer than perGroup members returns all of them (no crash)', () => {
    const phenos = [p(0, 0.9), p(3, 0.05)]
    const blocks = ['background', 'cancer'] // one member per block, perGroup=3
    const got = labeledPhenotypeIds(phenos, { blocks, perGroup: 3 })
    expect(got).toEqual(new Set([0, 3]))
  })

  it('breaks prevalence ties deterministically by lower id', () => {
    const phenos = [p(5, 0.2), p(2, 0.2), p(9, 0.2)]
    const got = labeledPhenotypeIds(phenos, { perGroup: 2 })
    // all tied at 0.2 -> pick the two lowest ids: 2 and 5
    expect(got).toEqual(new Set([2, 5]))
  })
})

describe('labelRanks', () => {
  it('ungated: ranks all phenotypes by prevalence, most prevalent = 0', () => {
    const phenos = [p(0, 0.10), p(1, 0.50), p(2, 0.30)]
    const ranks = labelRanks(phenos, {})
    expect(ranks.get(1)).toBe(0) // 0.50
    expect(ranks.get(2)).toBe(1) // 0.30
    expect(ranks.get(0)).toBe(2) // 0.10
  })

  it('gated: ranks are INDEPENDENT per block, so each block starts at 0', () => {
    const phenos = [p(0, 0.9), p(1, 0.8), p(2, 0.05), p(3, 0.03)]
    const blocks = ['background', 'background', 'cancer', 'cancer']
    const ranks = labelRanks(phenos, { blocks })
    expect(ranks.get(0)).toBe(0) // top of background
    expect(ranks.get(1)).toBe(1)
    expect(ranks.get(2)).toBe(0) // top of cancer, even though globally tiny
    expect(ranks.get(3)).toBe(1)
  })

  it('is a stable ordering: a progressive cutoff only adds ids as it grows', () => {
    const phenos = [p(0, 0.10), p(1, 0.50), p(2, 0.30), p(3, 0.40)]
    const ranks = labelRanks(phenos, {})
    const shownAt = (cut: number) =>
      new Set([...ranks].filter(([, r]) => r < cut).map(([id]) => id))
    // cut=1 subset of cut=2 subset of cut=3 (monotonic reveal, no reshuffle)
    const s1 = shownAt(1), s2 = shownAt(2), s3 = shownAt(3)
    expect([...s1].every((id) => s2.has(id))).toBe(true)
    expect([...s2].every((id) => s3.has(id))).toBe(true)
    expect(s1).toEqual(new Set([1]))       // only the most prevalent
    expect(s2).toEqual(new Set([1, 3]))    // + next
  })

  it('excludes hidden phenotypes (no rank entry)', () => {
    const phenos = [p(0, 0.5), p(1, 0.4)]
    const ranks = labelRanks(phenos, { isVisible: (q) => q.id !== 0 })
    expect(ranks.has(0)).toBe(false)
    expect(ranks.get(1)).toBe(0)
  })
})

describe('labelHash', () => {
  it('is deterministic for a given id + seed', () => {
    expect(labelHash(7)).toBe(labelHash(7))
    expect(labelHash(7, 42)).toBe(labelHash(7, 42))
  })

  it('returns a value in [0, 1)', () => {
    for (const id of [0, 1, 2, 17, 59, 1000]) {
      const h = labelHash(id)
      expect(h).toBeGreaterThanOrEqual(0)
      expect(h).toBeLessThan(1)
    }
  })

  it('decorrelates from id order (does not monotonically track id)', () => {
    // The whole point is spatial de-biasing: hash order must NOT equal id order,
    // otherwise labels would still clump by whatever id happens to correlate with.
    const ids = Array.from({ length: 30 }, (_, i) => i)
    const byHash = ids.slice().sort((a, b) => labelHash(a) - labelHash(b))
    expect(byHash).not.toEqual(ids)
  })

  it('reseeds to a different order', () => {
    const ids = Array.from({ length: 30 }, (_, i) => i)
    const a = ids.slice().sort((x, y) => labelHash(x, 1) - labelHash(y, 1))
    const b = ids.slice().sort((x, y) => labelHash(x, 2) - labelHash(y, 2))
    expect(a).not.toEqual(b)
  })
})

describe('selectLabels', () => {
  // Candidates on a grid; cx/cy are pre-zoom g-space coords. The identity
  // transform {k:1,x:0,y:0} maps them straight onto the [0,W]x[0,H] viewport.
  const grid = (n: number): LabelCandidate[] =>
    Array.from({ length: n }, (_, i) => ({ id: i, cx: (i % 10) * 10 + 5, cy: 5 }))

  const ident = { k: 1, x: 0, y: 0 }
  const box = { width: 100, height: 100 }

  it('labels ceil(baseFraction * inViewCount) candidates at k=1', () => {
    const cands = grid(10)
    const got = selectLabels(cands, { transform: ident, ...box, baseFraction: 0.5 })
    expect(got.size).toBe(5) // ceil(0.5 * 10)
  })

  it('rounds the target UP so a sparse view still shows at least one label', () => {
    const cands = grid(3)
    const got = selectLabels(cands, { transform: ident, ...box, baseFraction: 0.1 })
    // 0.1 * 3 = 0.3 -> ceil -> 1
    expect(got.size).toBe(1)
  })

  it('picks exactly the lowest-hash ids (stable, spatially unbiased set)', () => {
    const cands = grid(10)
    const expected = new Set(
      cands.slice().sort((a, b) => labelHash(a.id) - labelHash(b.id)).slice(0, 4).map((c) => c.id),
    )
    const got = selectLabels(cands, { transform: ident, ...box, baseFraction: 0.4 })
    expect(got).toEqual(expected)
  })

  it('is independent of input array order', () => {
    const cands = grid(10)
    const shuffled = [...cands].reverse()
    const a = selectLabels(cands, { transform: ident, ...box, baseFraction: 0.4 })
    const b = selectLabels(shuffled, { transform: ident, ...box, baseFraction: 0.4 })
    expect(b).toEqual(a)
  })

  it('excludes candidates outside the viewport under the current transform', () => {
    // Two points: one inside [0,100]^2, one far to the right. Identity transform.
    const cands: LabelCandidate[] = [
      { id: 0, cx: 50, cy: 50 },
      { id: 1, cx: 500, cy: 50 },
    ]
    const got = selectLabels(cands, { transform: ident, ...box, baseFraction: 1 })
    expect(got.has(0)).toBe(true)
    expect(got.has(1)).toBe(false)
  })

  it('accounts for the zoom/pan transform when testing visibility', () => {
    // A point at cx=500 is off-screen at k=1, but a pan of x=-450 brings it to
    // screen x=50 (in view), while the point at cx=50 pans to x=-400 (out).
    const cands: LabelCandidate[] = [
      { id: 0, cx: 50, cy: 50 },
      { id: 1, cx: 500, cy: 50 },
    ]
    const panned = { k: 1, x: -450, y: 0 }
    const got = selectLabels(cands, { transform: panned, ...box, baseFraction: 1 })
    expect(got.has(1)).toBe(true)
    expect(got.has(0)).toBe(false)
  })

  it('raises the labeled fraction as zoom (k) grows, capped at all-in-view', () => {
    // 10 candidates stacked near the origin so they stay in view at every k
    // (isolating the fraction-vs-k behavior from the viewport filter).
    const cands: LabelCandidate[] = Array.from({ length: 10 }, (_, i) => ({ id: i, cx: 1, cy: 1 }))
    const at = (k: number) =>
      selectLabels(cands, { transform: { k, x: 0, y: 0 }, ...box, baseFraction: 0.2 }).size
    expect(at(1)).toBe(2)  // ceil(0.2 * 1 * 10)
    expect(at(3)).toBe(6)  // ceil(0.6 * 10)
    expect(at(10)).toBe(10) // min(1, 2.0) -> all in view
  })

  it('returns an empty set when nothing is in view', () => {
    const cands: LabelCandidate[] = [{ id: 0, cx: 999, cy: 999 }]
    const got = selectLabels(cands, { transform: ident, ...box, baseFraction: 1 })
    expect(got.size).toBe(0)
  })
})
