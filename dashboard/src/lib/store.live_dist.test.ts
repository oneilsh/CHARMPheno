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
  // Use the store's own (τ-aligned) edges, not the exported grid.
  const edges = d.binEdges
  const tail = edges.slice(0, -1).reduce((s, lo, i) => s + (lo >= tau ? d.histogram[i] : 0), 0)
  const cov = get(coverageReader)
  expect(tail).toBeCloseTo(cov(b.phenotypes.phenotypes[1]), 6)
})

it('a rare gated foreground topic gets positive coverage from its in-group cohort (not a vanished 0)', () => {
  // Gated K=3 fixture, cancer foreground at a 2% marginal share. The marginal
  // 1500-patient atlas cohort holds ~30 cancer patients — enough here, but the
  // coverage MUST come from the in-group cohort and match its own histogram tail
  // (the vanished-bubble fix: foreground coverage read from a full in-group
  // sample, and the selected-phenotype histogram reads the SAME cohort).
  const b = makeStmBundleFixture()
  b.phenotypes.theta_histogram_bin_edges = EDGES
  ;(b as any).gating = {
    group_var: 'g', groups: ['cancer'],
    topic_blocks: ['background', 'cancer', 'background'],
    group_proportions: { cancer: 0.02 },
  }
  bundle.set(b)
  const cov = get(coverageReader)
  // Cancer foreground (topic 1) is expressed by its in-group cohort -> positive.
  expect(cov(b.phenotypes.phenotypes[1])).toBeGreaterThan(0)

  // Bubble = histogram tail invariant, now over the in-group cohort.
  selectedPhenotypeId.set(1)
  const d = get(selectedPhenotypeLiveDist)!
  const tau = get(tauThreshold)
  const tail = d.binEdges.slice(0, -1).reduce((s, lo, i) => s + (lo >= tau ? d.histogram[i] : 0), 0)
  expect(tail).toBeCloseTo(cov(b.phenotypes.phenotypes[1]), 6)
})
