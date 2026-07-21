import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJob } from '../api'
import { listRecentJobs, removeRecentJob } from '../lib/recentJobs'
import { jobRoute } from '../lib/routes'

/** "Recent jobs" (PRD Addendum §2.5) — reads the client-side index
 * (lib/recentJobs.js) and refreshes each entry's live status from the
 * server, since the local list only ever stores ids/labels, never status. */
export default function RecentJobsList() {
  const [entries, setEntries] = useState(() => listRecentJobs())
  const [statuses, setStatuses] = useState({}) // id -> job summary | null (gone/404) | undefined (loading)

  useEffect(() => {
    let cancelled = false
    entries.forEach((entry) => {
      getJob(entry.id)
        .then((job) => {
          if (!cancelled) setStatuses((prev) => ({ ...prev, [entry.id]: job }))
        })
        .catch(() => {
          if (!cancelled) setStatuses((prev) => ({ ...prev, [entry.id]: null }))
        })
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleForget(id) {
    removeRecentJob(id)
    setEntries(listRecentJobs())
  }

  if (entries.length === 0) return null

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-medium text-slate-300">Recent jobs</p>
      <ul className="flex flex-col gap-1.5">
        {entries.map((entry) => {
          const job = statuses[entry.id]
          const gone = job === null || job?.status === 'expired'
          const label = entry.sourceLabel || `Job ${entry.id.slice(0, 8)}`
          return (
            <li
              key={entry.id}
              className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2 text-sm"
            >
              {gone ? (
                <span className="truncate text-slate-500 line-through" title="This job has expired">
                  {label}
                </span>
              ) : (
                <Link to={jobRoute(entry.id)} className="focus-ring truncate text-indigo-300 hover:underline">
                  {label}
                </Link>
              )}
              <span className="shrink-0 whitespace-nowrap capitalize text-slate-500">
                {entry.mode} · {entry.tier}
                {job && !gone ? ` · ${job.status}` : ''}
              </span>
              <button
                type="button"
                onClick={() => handleForget(entry.id)}
                aria-label={`Remove ${label} from recent jobs`}
                className="focus-ring shrink-0 text-slate-500 hover:text-slate-300"
              >
                ✕
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
