import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import StemPlayer from './StemPlayer'
import DownloadBar from './DownloadBar'

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
      <div className="mx-auto max-w-xl px-4 py-24 text-center text-slate-400">
        Job finished but produced no stems — this shouldn't happen; something went wrong server-side.
      </div>
    )
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-12">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-medium text-slate-100">Stems ready</h1>
          <p className="mt-1 text-sm text-slate-400 capitalize">
            {job.mode} mode — {job.tier} tier
          </p>
        </div>
        <Link to="/" className="text-sm text-indigo-300 hover:underline">
          New job
        </Link>
      </div>

      <StemPlayer jobId={job.id} stemNames={stemNames} />
      <DownloadBar jobId={job.id} stemNames={stemNames} />

      <button
        type="button"
        onClick={handleDelete}
        disabled={deleting}
        className="self-start text-sm text-red-400 transition-colors hover:text-red-300 disabled:opacity-50"
      >
        {deleting ? 'Deleting…' : 'Delete these stems now'}
      </button>
    </div>
  )
}
