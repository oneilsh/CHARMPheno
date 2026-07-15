import type { CovariateSchema } from '../types'

export function initialValues(schema: CovariateSchema): Record<string, number | string> {
  const v: Record<string, number | string> = {}
  for (const c of schema.controls) {
    if (c.type === 'continuous') v[c.name] = c.default ?? 0
    else v[c.name] = c.reference ?? (c.levels?.[0] ?? '')
  }
  return v
}

export function canInteract(schema: CovariateSchema): boolean {
  return schema.unsupported.length === 0
}
