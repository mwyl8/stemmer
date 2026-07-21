import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import StemPlayer from './StemPlayer'
import DownloadBar from './DownloadBar'
import { formatDurationWords } from '../lib/format'

export default function ResultView({ job, onDelete }) {
  const [deleting, setDeleting] = useState(false)

  // job.stems has one row per (name, format) pair — dedupe to the stem
  // names themselves, in a stable order. Memoized on job.id so the array
  // reference only changes when the job itself changes, not on every
  // parent re-render (StemPlayer tears down and rebuilds the player
  // whenever this reference changes).
  const stemNames = useMemo(() => [...new Set(job.stems.map((s) => s.name))], [job.id])

  async function handleDelete() {
    if (!window.confirm('Delete this job\'s stems now? This cannot be undone.')) return
    setDeleting(true)
    await onDelete()
  }

  if (stemNames.length === 0) {
    return (
      <div role="alert" className="mx-auto flex max-w-xl flex-col items-center gap-2 px-4 py-24 text-center text-slate-400">
        <p>Job finished but produced no stems — this shouldn't happen; something went wrong server-side.</p>
        <Link to="/" className="focus-ring mt-2 inline-block rounded text-indigo-300 hover:underline">
          Start a new job
        </Link>
      </div>
    )
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-12">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-medium text-slate-100">Stems ready</h1>
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-slate-400">
            <span className="capitalize">
              {job.mode} mode — {job.tier} tier
              {job.stem_count === 6 ? ' — 6-stem' : ''}
            </span>
            <span className="text-slate-600">·</span>
            {job.from_cache ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-950/60 px-2 py-0.5 text-xs font-medium text-emerald-300">
                <span aria-hidden="true">⚡</span> returned instantly from cache
              </span>
            ) : (
              <span>took {formatDurationWords(job.elapsed_seconds)}</span>
            )}
          </p>
        </div>
        <Link to="/" className="focus-ring rounded text-sm text-indigo-300 hover:underline">
          New job
        </Link>
      </div>

      <StemPlayer jobId={job.id} stemNames={stemNames} />
      <DownloadBar jobId={job.id} stemNames={stemNames} />

      <button
        type="button"
        onClick={handleDelete}
        disabled={deleting}
        className="focus-ring self-start rounded text-sm text-red-400 transition-colors hover:text-red-300 disabled:opacity-50"
      >
        {deleting ? 'Deleting…' : 'Delete these stems now'}
      </button>
    </div>
  )
}
