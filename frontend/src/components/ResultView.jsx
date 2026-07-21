import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import StemPlayer from './StemPlayer'
import DownloadBar from './DownloadBar'
import RerunPanel from './RerunPanel'
import { formatBitrate, formatDurationWords, formatSampleRate } from '../lib/format'
import { recordRecentJob } from '../lib/recentJobs'

export default function ResultView({ job, onDelete }) {
  const [deleting, setDeleting] = useState(false)
  const [linkCopied, setLinkCopied] = useState(false)

  // Recorded here (not just at submission) so opening a shared link also
  // adds it to *this* browser's recent-jobs list -- recentJobs.js is purely
  // a client-side index of ids, never a second source of truth (see its
  // module comment for why there's no server-side "list all jobs" instead).
  useEffect(() => {
    recordRecentJob({ id: job.id, mode: job.mode, tier: job.tier, sourceLabel: job.source_codec ? job.source_codec.toUpperCase() : null })
  }, [job.id, job.mode, job.tier, job.source_codec])

  // job.stems has one row per (name, format) pair — dedupe to the stem
  // names themselves, in a stable order. Memoized on job.id so the array
  // reference only changes when the job itself changes, not on every
  // parent re-render (StemPlayer tears down and rebuilds the player
  // whenever this reference changes).
  const stemNames = useMemo(() => [...new Set(job.stems.map((s) => s.name))], [job.id])

  // What the pipeline actually started from (Part A #1) — read off the
  // source file before normalization to 44.1kHz stereo WAV, so a low-bitrate
  // rip and a lossless upload don't look identical here. Parts can be null
  // (ffprobe couldn't read that field, or on a legacy job from before this
  // was tracked) — only show what's actually known.
  const sourceDetails = [
    job.source_codec?.toUpperCase(),
    formatBitrate(job.source_bitrate),
    formatSampleRate(job.source_sample_rate),
    job.source_channels === 1 ? 'mono' : job.source_channels === 2 ? 'stereo' : job.source_channels ? `${job.source_channels}ch` : null,
  ].filter(Boolean)

  async function handleDelete() {
    if (!window.confirm('Delete this job\'s stems now? This cannot be undone.')) return
    setDeleting(true)
    await onDelete()
  }

  async function handleCopyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setLinkCopied(true)
      setTimeout(() => setLinkCopied(false), 2000)
    } catch {
      // clipboard permission denied/unsupported -- the URL bar already has
      // the same shareable link, nothing else useful to do client-side
    }
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
          {sourceDetails.length > 0 && (
            <p className="mt-0.5 text-xs text-slate-500">
              Source: <span className="tabular-nums">{sourceDetails.join(' · ')}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button type="button" onClick={handleCopyLink} className="focus-ring rounded text-sm text-indigo-300 hover:underline">
            {linkCopied ? 'Link copied!' : 'Copy shareable link'}
          </button>
          <Link to="/" className="focus-ring rounded text-sm text-indigo-300 hover:underline">
            New job
          </Link>
        </div>
      </div>

      <StemPlayer jobId={job.id} stemNames={stemNames} hasOriginal={job.has_original} />
      <DownloadBar jobId={job.id} stemNames={stemNames} />
      <RerunPanel job={job} />

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
