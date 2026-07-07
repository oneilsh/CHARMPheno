import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { bundle, selectedPhenotypeId, tauThreshold, coverageReader, selectedPhenotypeLiveDist } from './store'
import { makeStmBundleFixture } from './test-fixtures'

const EDGES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

beforeEach(() => {
  bundle.set(null)
  selectedPhenotypeId.set(null)
  tauThreshold.set(0.1) // on a bin edge, so tail bins and the >τ count agree
})

it('selectedPhenotypeLiveDist is null without a selection and populated with one', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.theta_histogram_bin_edges = EDGES
  bundle.set(b)
  expect(get(selectedPhenotypeLiveDist)).toBeNull()
  selectedPhenotypeId.set(1)
  const d = get(selectedPhenotypeLiveDist)!
  expect(d.n).toBeGreaterThan(0)
  expect(d.histogram.reduce((a, x) => a + x, 0)).toBeCloseTo(1, 6)
})

it("the histogram's tail above τ equals the phenotype's coverage bubble (same live cohort)", () => {
  // The whole point of tying them: the bubble IS the area above τ of this
  // histogram. Both read the same atlas cohort, so for a non-gated topic (no
  // masking) the tail mass must equal coverageReader exactly.
  const b = makeStmBundleFixture()
  b.phenotypes.theta_histogram_bin_edges = EDGES
  bundle.set(b)
  selectedPhenotypeId.set(1)
  const d = get(selectedPhenotypeLiveDist)!
  const tau = get(tauThreshold)
  const tail = EDGES.slice(0, -1).reduce((s, lo, i) => s + (lo >= tau ? d.histogram[i] : 0), 0)
  const cov = get(coverageReader)
  expect(tail).toBeCloseTo(cov(b.phenotypes.phenotypes[1]), 6)
})
