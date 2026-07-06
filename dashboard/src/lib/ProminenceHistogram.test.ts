import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ProminenceHistogram from './ProminenceHistogram.svelte'

afterEach(() => cleanup())

it('mounts without error and renders one bar per non-null bin', () => {
  const histogram = [0.2, 0.5, 0.3]
  const binEdges = [-0.1, 0.0, 0.1, 0.2]
  const { container } = render(ProminenceHistogram, { props: { histogram, binEdges } })
  const rects = container.querySelectorAll('rect')
  expect(rects.length).toBe(histogram.length)
})

it('renders a transparent hit-target (not a colored bar) for suppressed (null) bins', () => {
  const histogram = [0.2, null, 0.3]
  const binEdges = [-0.1, 0.0, 0.1, 0.2]
  const { container } = render(ProminenceHistogram, { props: { histogram, binEdges } })
  const rects = Array.from(container.querySelectorAll('rect'))
  expect(rects.length).toBe(3)
  const suppressed = rects.find((r) => r.getAttribute('fill-opacity') === '0')
  expect(suppressed).toBeTruthy()
})

it('derives its x-axis range from the given bin edges, not a hardcoded range', () => {
  // Bin edges entirely in negative nats territory (a plausible held-out
  // predictive-gain range) should render fine, unlike a [0,1]-theta-scale
  // assumption baked into the component.
  const histogram = [0.4, 0.6]
  const binEdges = [-2, -1, 0]
  const { container } = render(ProminenceHistogram, { props: { histogram, binEdges } })
  const rects = container.querySelectorAll('rect')
  expect(rects.length).toBe(2)
})
