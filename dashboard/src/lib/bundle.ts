import type {
  DashboardBundle, Model, PhenotypesBundle, VocabBundle, CorpusStats,
} from './types'

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`failed to load ${url}: ${r.status}`)
  return r.json() as Promise<T>
}

export async function loadBundle(baseUrl: string): Promise<DashboardBundle> {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
  const [model, phenotypes, vocab, corpusStats] = await Promise.all([
    fetchJson<Model>(`${base}data/model.json`),
    fetchJson<PhenotypesBundle>(`${base}data/phenotypes.json`),
    fetchJson<VocabBundle>(`${base}data/vocab.json`),
    fetchJson<CorpusStats>(`${base}data/corpus_stats.json`),
  ])
  return { model, phenotypes, vocab, corpusStats }
}
