// "Recent jobs" (PRD Addendum §2.5) is deliberately client-only: there's no
// user/session model in this app (jobs are anonymous, ephemeral, TTL'd), so
// a server-side "list all jobs" endpoint would let any visitor enumerate
// every other visitor's job ids/content hashes -- a needless privacy hole.
// Instead the browser just remembers which job ids *it* created; the actual
// status/metadata for each is always re-fetched live from GET /jobs/{id},
// so this list is never a second source of truth, only an index of ids.

const STORAGE_KEY = 'stemmer.recentJobs'
const MAX_ENTRIES = 20

function readAll() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return [] // localStorage unavailable (private mode, quota, disabled) -- degrade to "no history"
  }
}

function writeAll(entries) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  } catch {
    // best-effort -- a failed write just means this entry won't show up next time
  }
}

/** Newest first. */
export function listRecentJobs() {
  return readAll()
}

/** Record (or bump to the front) a job this browser just created/opened.
 * `sourceLabel` is a short human label (filename, or the URL) for display. */
export function recordRecentJob({ id, mode, tier, sourceLabel }) {
  const rest = readAll().filter((entry) => entry.id !== id)
  const next = [{ id, mode, tier, sourceLabel: sourceLabel || null, recordedAt: new Date().toISOString() }, ...rest].slice(
    0,
    MAX_ENTRIES,
  )
  writeAll(next)
}

export function removeRecentJob(id) {
  writeAll(readAll().filter((entry) => entry.id !== id))
}
