import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import PhenotypeBrowser from './PhenotypeBrowser.svelte'
import { bundle, phenotypeSortBy, atlasConditioning } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())

it('renders no Topic mass column', () => {
  bundle.set(makeStmBundleFixture())
  const { queryByText } = render(PhenotypeBrowser)
  expect(queryByText('Topic mass')).toBeNull()
})

it('re-sorts when conditioning changes the prevalence order', async () => {
  bundle.set(makeStmBundleFixture())
  phenotypeSortBy.set('prevalence')
  atlasConditioning.set({ covariateActive: false, values: {}, group: null })
  const { container } = render(PhenotypeBrowser)
  const firstRowId = () => container.querySelector('tbody tr')?.getAttribute('data-pid')
  const before = firstRowId()
  // Marginal coverage favors topic 2 (age-loaded); a young profile flips it to topic 1.
  atlasConditioning.set({ covariateActive: true, values: { age: 0 }, group: null })
  await Promise.resolve()
  const after = firstRowId()
  expect(after).not.toBe(before)
})
