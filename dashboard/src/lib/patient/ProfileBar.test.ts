import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ProfileBar from './ProfileBar.svelte'
import { bundle } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())
beforeEach(() => bundle.set(makeStmBundleFixture()))

it('renders a prior sub-segment on bands the codes do not fully explain', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  // codeBag speaks only to code 0 → other bands lean on the prior. showResidual
  // gates the overlay (it is off by default, e.g. on the neighbor strip).
  const { container } = render(ProfileBar, { props: { theta, codeBag: [0], showResidual: true } })
  expect(container.querySelectorAll('.band .prior-fill').length).toBeGreaterThan(0)
})

it('renders NO prior sub-segment when showResidual is false (neighbor-strip default)', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  const { container } = render(ProfileBar, { props: { theta, codeBag: [0] } })
  expect(container.querySelectorAll('.band .prior-fill').length).toBe(0)
})
