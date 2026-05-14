import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loadBundle } from './bundle'

describe('loadBundle', () => {
  beforeEach(() => {
    global.fetch = vi.fn((url: string) => {
      const stubs: Record<string, unknown> = {
        'data/model.json':         { K: 2, V: 3, alpha: [0.1, 0.1], beta: [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]] },
        'data/phenotypes.json':    { phenotypes: [], npmi_threshold: 0 },
        'data/vocab.json':         { codes: [] },
        'data/corpus_stats.json':  { corpus_size_docs: 10, mean_codes_per_doc: 5, k: 2, v: 3, v_full: 3 },
      }
      const key = Object.keys(stubs).find((k) => url.endsWith(k))!
      return Promise.resolve({ ok: true, json: () => Promise.resolve(stubs[key]) } as Response)
    }) as any
  })

  it('loads all four files', async () => {
    const b = await loadBundle('/')
    expect(b.model.K).toBe(2)
    expect(b.corpusStats.v_full).toBe(3)
  })
})
