import { describe, it, expect } from 'vitest'
import { labeledPhenotypeIds, labelRanks, type LabelablePhenotype } from './labels'

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
