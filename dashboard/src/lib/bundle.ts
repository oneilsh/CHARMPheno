import type {
  CohortManifest, DashboardBundle, Model, PhenotypesBundle, VocabBundle, CorpusStats,
  CovariateSchema, CovariateEffects, GatingSpec,
} from './types'

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`failed to load ${url}: ${r.status}`)
  return r.json() as Promise<T>
}

async function fetchJsonOptional<T>(url: string): Promise<T | undefined> {
  const r = await fetch(url)
  if (!r.ok) return undefined
  // A dev server (e.g. Vite) serves the SPA fallback index.html with a 200 for
  // a missing file under public/, so an absent optional bundle file arrives as
  // an HTML body rather than a 404. r.json() rejects on that body; treat any
  // non-JSON response as "absent" instead of failing the whole bundle load.
  try {
    return await (r.json() as Promise<T>)
  } catch {
    return undefined
  }
}

// Fetch the top-level cohort manifest. The manifest is the source of
// truth for which bundles are available — listed cohorts must each have
// a matching `data/<id>/` subdir of the four bundle files. A missing
// manifest is a fatal error rather than silently falling back to a
// single bundle, since that would mask deployment mistakes.
export async function loadManifest(baseUrl: string): Promise<CohortManifest> {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
  return fetchJson<CohortManifest>(`${base}data/manifest.json`)
}

export async function loadBundle(baseUrl: string, cohortId: string): Promise<DashboardBundle> {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
  const [model, phenotypes, vocab, corpusStats] = await Promise.all([
    fetchJson<Model>(`${base}data/${cohortId}/model.json`),
    fetchJson<PhenotypesBundle>(`${base}data/${cohortId}/phenotypes.json`),
    fetchJson<VocabBundle>(`${base}data/${cohortId}/vocab.json`),
    fetchJson<CorpusStats>(`${base}data/${cohortId}/corpus_stats.json`),
  ])
  const [covariateSchema, covariateEffects, gating] = await Promise.all([
    fetchJsonOptional<CovariateSchema>(`${base}data/${cohortId}/covariate_schema.json`),
    fetchJsonOptional<CovariateEffects>(`${base}data/${cohortId}/covariate_effects.json`),
    fetchJsonOptional<GatingSpec>(`${base}data/${cohortId}/gating.json`),
  ])
  return { model, phenotypes, vocab, corpusStats, covariateSchema, covariateEffects, gating }
}
