import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ContributingCodes from './ContributingCodes.svelte'
import { bundle, selectedPhenotypeId } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'
import { copy } from '../copy'

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

it('does not claim a selection in the header when there are no rows (empty record)', () => {
  // A phenotype is selected but the patient has no codes at all, so
  // codeComposition() returns zero rows and the body falls back to
  // emptyRecord. The header must not contradict that by still naming the
  // selected phenotype (no h3, no open-in-atlas, no subMatch/subOther).
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  selectedPhenotypeId.set(0)
  const { container, getByText } = render(ContributingCodes, { props: { theta, codeBag: [] } })
  expect(container.querySelector('h3')).toBeNull()
  expect(container.querySelector('.open-in-atlas')).toBeNull()
  getByText(copy.contributingCodes.emptyRecord)
  getByText(copy.contributingCodes.composition)
})
