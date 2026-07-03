import { it, expect, afterEach } from 'vitest'
import { render, cleanup, fireEvent } from '@testing-library/svelte'
import { get } from 'svelte/store'
import CorrelationHeatmap from './CorrelationHeatmap.svelte'
import { selectedPhenotypeId } from '../store'
import type { Correlation } from '../types'

afterEach(() => cleanup())

const correlation: Correlation = {
  topic_order: [0, 1, 2],
  block_labels: ['background', 'A', 'B'],
  R: [[1, 0.4, 0.5], [0.4, 1, 0.3], [0.5, 0.3, 1]],
  identified: [[true, true, true], [true, true, false], [true, false, true]],
  support: [[300, 200, 100], [200, 200, 100], [100, 100, 100]],
}

it('greys unidentified cells and colors identified ones', () => {
  const { container } = render(CorrelationHeatmap, { props: { correlation } })

  // Unidentified cell (1,2): identified=false with non-null R -> must carry the "no joint support" NA styling/title.
  // This proves that identified:false alone triggers NA regardless of R being present.
  const naCell = container.querySelector('[data-row="1"][data-col="2"]')
  expect(naCell).toBeTruthy()
  expect(naCell?.classList.contains('na')).toBe(true)
  expect(naCell?.getAttribute('data-tip') ?? naCell?.querySelector('title')?.textContent ?? naCell?.getAttribute('title'))
    .toMatch(/no joint support/i)

  // Its mirror (2,1) must also be greyed (symmetric NA).
  const naCellMirror = container.querySelector('[data-row="2"][data-col="1"]')
  expect(naCellMirror?.classList.contains('na')).toBe(true)

  // Identified cell (0,1): R = 0.4, must NOT carry the NA class and must carry a real fill.
  const identifiedCell = container.querySelector('[data-row="0"][data-col="1"]')
  expect(identifiedCell).toBeTruthy()
  expect(identifiedCell?.classList.contains('na')).toBe(false)
  const fill = identifiedCell?.getAttribute('fill')
  expect(fill).toBeTruthy()
  expect(fill).not.toBe('none')
})

it('tooltip names both topics in the pair', () => {
  const { container } = render(CorrelationHeatmap, { props: { correlation } })
  // topic_order = [0,1,2]; cell (0,1) pairs topic 0 with topic 1. With no bundle
  // in the test, labels fall back to the topic ids, joined by "×".
  const cell = container.querySelector('[data-row="0"][data-col="1"]')
  expect(cell?.getAttribute('data-tip')).toMatch(/0 × 1/)
})

it('clicking a cell selects its COLUMN topic', async () => {
  selectedPhenotypeId.set(null)
  const { container } = render(CorrelationHeatmap, { props: { correlation } })
  // cell (2,1): row topic order[2]=2, column topic order[1]=1 -> selects 1.
  const cell = container.querySelector('[data-row="2"][data-col="1"]') as SVGRectElement
  await fireEvent.click(cell)
  expect(get(selectedPhenotypeId)).toBe(1)
})
