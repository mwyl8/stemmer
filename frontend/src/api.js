// Typed wrappers over the Stemmer backend API. Every endpoint here matches
// backend/app.py exactly — see that file (and backend/tests/test_app.py for
// worked examples) if the contract ever seems to disagree with this file.
//
// Requests are same-origin (`/jobs`, `/health`): the Vite dev server proxies
// them to the FastAPI backend (vite.config.js), and a production build would
// be served from the same origin as the API. No CORS needed either way.

export const MODES = ['music', 'video', 'full']
export const TIERS = ['fast', 'balanced', 'best']

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request(path, options) {
  let res
  try {
    res = await fetch(path, options)
  } catch (cause) {
    throw new ApiError(0, 'Could not reach the server — is the backend running?')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body && body.detail) detail = body.detail
    } catch {
      // error body wasn't JSON (e.g. a raw 500) — fall back to statusText
    }
    throw new ApiError(res.status, detail)
  }
  return res
}

/** Create a job from a local file. Returns {job_id}. */
export async function createJobFromFile(file, { mode, tier } = {}) {
  const form = new FormData()
  form.append('file', file)
  if (mode) form.append('mode', mode)
  if (tier) form.append('tier', tier)
  const res = await request('/jobs', { method: 'POST', body: form })
  return res.json()
}

/** Create a job from a public URL (YouTube/TikTok/Instagram). Returns {job_id}. */
export async function createJobFromUrl(url, { mode, tier } = {}) {
  const res = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, mode, tier }),
  })
  return res.json()
}

/**
 * {id, status, stage, progress_pct (int 0-100), mode, tier, stem_count, error,
 *  created_at, expires_at, submitted_at, stage_started_at, stage_timings
 *  ({stage: {started_at, ended_at}}), chunks_done, chunks_total,
 *  elapsed_seconds, eta_seconds, from_cache, stems}.
 */
export async function getJob(jobId) {
  const res = await request(`/jobs/${jobId}`)
  return res.json()
}

/** [{name, format, duration}, ...] */
export async function listStems(jobId) {
  const res = await request(`/jobs/${jobId}/stems`)
  return res.json()
}

/** Streamable URL for one stem — `format` is "mp3" (preview, default) or "wav" (download). */
export function stemUrl(jobId, name, format = 'mp3') {
  return `/jobs/${jobId}/stems/${encodeURIComponent(name)}?format=${format}`
}

/** URL for the zip-all-stems (wav) download. */
export function downloadAllUrl(jobId) {
  return `/jobs/${jobId}/download`
}

export async function deleteJob(jobId) {
  await request(`/jobs/${jobId}`, { method: 'DELETE' })
}

/** Terminal job statuses — polling should stop once status is one of these. */
export const TERMINAL_STATUSES = new Set(['done', 'error', 'expired'])

/** Stem names that count as "the vocal/speech stem" for the karaoke toggle,
 * across all three modes (music -> vocals; video -> speech; full -> both). */
export const KARAOKE_STEM_NAMES = new Set(['vocals', 'speech'])
