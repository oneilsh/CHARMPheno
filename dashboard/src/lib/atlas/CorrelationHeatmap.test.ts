import { it, expect, afterEach } from 'vitest'
import { render, cleanup, fireEvent } from '@testing-library/svelte'
import { get } from 'svelte/store'
import CorrelationHeatmap from './CorrelationHeatmap.svelte'
import { selectedPhenotypeId, comparePair } from '../store'
import type { Correlation } from '../types'

afterEach(() => cleanup())

// Two blocks (background 0..1, cancer 2..3); (0,3)/(3,0) are an unidentified
// background×cancer pair to exercise the NA path.
const correlation: Correlation = {
  topic_order: [0, 1, 2, 3],
  block_labels: ['background', 'background', 'cancer', 'cancer'],
  R: [
    [1, 0.3, 0.1, 0.2],
    [0.3, 1, 0.4, 0.1],
    [0.1, 0.4, 1, 0.5],
    [0.2, 0.1, 0.5, 1],
  ],
  identified: [
    [true, true, true, false],
    [true, true, true, true],
    [true, true, true, true],
    [false, true, true, true],
  ],
  support: [
    [300, 200, 100, 20],
    [200, 300, 150, 120],
    [100, 150, 300, 200],
    [20, 120, 200, 300],
  ],
  reference_topic: null,
}

it('renders row and column block pickers with friendly block names', () => {
  const { getAllByRole } = render(CorrelationHeatmap, { props: { correlation } })
  const selects = getAllByRole('combobox') as HTMLSelectElement[]
  expect(selects.length).toBe(2)
  // background -> "All"; cancer stays "cancer"
  const opts = Array.from(selects[0].options).map((o) => o.textContent)
  expect(opts).toEqual(['All', 'cancer'])
})

it('defaults to All × All (the background block) on both axes', () => {
  const { container } = render(CorrelationHeatmap, { props: { correlation } })
  // background×background -> matrix rows/cols {0,1} -> 4 cells, all from that block
  const cells = container.querySelectorAll('rect.cell')
  expect(cells.length).toBe(4)
  const mrs = new Set(Array.from(cells).map((c) => c.getAttribute('data-mr')))
  expect(mrs).toEqual(new Set(['0', '1']))
})

it('clicking a cell selects its column topic', async () => {
  selectedPhenotypeId.set(null)
  const { container } = render(CorrelationHeatmap, { props: { correlation } })
  // default All × All -> background cells (matrix rows/cols {0,1})
  const cell = container.querySelector('rect.cell[data-mr="0"][data-mc="1"]') as SVGRectElement
  await fireEvent.click(cell)
  // column topic = order[1] = 1
  expect(get(selectedPhenotypeId)).toBe(1)
})

it('a cross-block selection surfaces the unidentified NA cell', async () => {
  const { container, getAllByRole } = render(CorrelationHeatmap, { props: { correlation } })
  const [rowSel, colSel] = getAllByRole('combobox') as HTMLSelectElement[]
  await fireEvent.change(rowSel, { target: { value: 'background' } })
  await fireEvent.change(colSel, { target: { value: 'cancer' } })
  // background(rows 0,1) × cancer(cols 2,3): cell (0,3) is unidentified
  const na = container.querySelector('rect.cell[data-mr="0"][data-mc="3"]')
  expect(na?.classList.contains('na')).toBe(true)
  expect(na?.getAttribute('fill')).toBe('var(--rule)')
})

it('in pair-select mode, clicking a cell sets the comparePair (row=A, col=B)', async () => {
  comparePair.set(null)
  const { container } = render(CorrelationHeatmap, { props: { correlation, pairSelect: true } })
  // default All × All -> background cells (matrix rows/cols {0,1})
  const cell = container.querySelector('rect.cell[data-mr="0"][data-mc="1"]') as SVGRectElement
  await fireEvent.click(cell)
  expect(get(comparePair)).toEqual({ a: 0, b: 1 })   // order[0]=0, order[1]=1
})

it('diagonal click (A===B) clears the pair', async () => {
  comparePair.set({ a: 9, b: 9 })
  const { container } = render(CorrelationHeatmap, { props: { correlation, pairSelect: true } })
  const cell = container.querySelector('rect.cell[data-mr="0"][data-mc="0"]') as SVGRectElement
  await fireEvent.click(cell)
  expect(get(comparePair)).toBeNull()
})
