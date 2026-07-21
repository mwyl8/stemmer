import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ModeTierPicker from './ModeTierPicker'
import { ApiError, rerunJob } from '../api'
import { recordRecentJob } from '../lib/recentJobs'
import { jobRoute } from '../lib/routes'

/** "Re-run this audio at a different mode/tier" (PRD Addendum §2.5) — no
 * re-upload, no re-fetch: POST /jobs/{id}/rerun reuses the source audio
 * pool.py already persisted alongside this job's stems. Collapsed by
 * default so it doesn't compete with the player for attention. */
export default function RerunPanel({ job }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState(job.mode)
  const [tier, setTier] = useState(job.tier)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const unchanged = mode === job.mode && tier === job.tier

  async function handleRerun() {
    setSubmitting(true)
    setError(null)
    try {
      const { job_id } = await rerunJob(job.id, { mode, tier, stem_count: job.stem_count })
      recordRecentJob({ id: job_id, mode, tier, sourceLabel: sourceLabelFor(job) })
      navigate(jobRoute(job_id))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not start the re-run.')
      setSubmitting(false)
    }
  }

  if (!job.has_original) return null

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="focus-ring self-start text-sm text-indigo-300 hover:underline"
      >
        Re-run this audio at a different mode/tier
      </button>
    )
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <p className="text-sm font-medium text-slate-200">Re-run without re-uploading</p>
      <ModeTierPicker mode={mode} onModeChange={setMode} tier={tier} onTierChange={setTier} disabled={submitting} />
      {error && (
        <p role="alert" className="text-sm text-red-300">
          {error}
        </p>
      )}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleRerun}
          disabled={submitting || unchanged}
          className="focus-ring rounded-lg bg-indigo-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          {submitting ? 'Starting…' : 'Re-run'}
        </button>
        <button
          type="button"
          onClick={() => setOpen(false)}
          disabled={submitting}
          className="focus-ring text-sm text-slate-400 hover:text-slate-200 disabled:opacity-50"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

function sourceLabelFor(job) {
  return `${job.mode} · re-run of a previous job`
}
