import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ContributingCodes from './ContributingCodes.svelte'
import { bundle, selectedPhenotypeId } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())
beforeEach(() => {
  bundle.set(makeStmBundleFixture())
  selectedPhenotypeId.set(null)
})

it('renders one row per unique code with no selection', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  const { container } = render(ContributingCodes, { props: { theta, codeBag: [0, 0, 1] } })
  expect(container.querySelectorAll('li.code').length).toBe(2) // codes 0 and 1
  expect(container.querySelectorAll('li.code .seg').length).toBeGreaterThan(0)
})

it('still renders all codes when a phenotype is selected (focus, not filter)', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  selectedPhenotypeId.set(0)
  const { container } = render(ContributingCodes, { props: { theta, codeBag: [0, 0, 1] } })
  expect(container.querySelectorAll('li.code').length).toBe(2)
})
