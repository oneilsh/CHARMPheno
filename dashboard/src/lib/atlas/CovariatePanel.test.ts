import { describe, it, expect } from 'vitest'
import { initialValues, canInteract } from './covariate-panel'

const schema = { k: 20, unsupported: [] as string[],
  controls: [
    { name: 'age', type: 'continuous', range: [40, 90], default: 65 },
    { name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'] },
  ],
  design_columns: [] }

it('initialValues uses continuous default + categorical reference', () => {
  expect(initialValues(schema as any)).toEqual({ age: 65, sex: 'F' })
})
it('canInteract is false when unsupported non-empty', () => {
  expect(canInteract(schema as any)).toBe(true)
  expect(canInteract({ ...schema, unsupported: ['x'] } as any)).toBe(false)
})
