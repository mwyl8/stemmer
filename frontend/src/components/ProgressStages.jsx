import ElapsedTimer from './progress/ElapsedTimer'
import StageTimings from './progress/StageTimings'
import CacheBanner from './progress/CacheBanner'
import { formatDurationWords } from '../lib/format'

const ALL_STAGES = [
  { key: 'downloading', label: 'Download' },
  { key: 'decoding', label: 'Decode' },
  { key: 'separating', label: 'Separate' },
  { key: 'encoding', label: 'Encode' },
]

// Modes that chain two separation passes (backend/separators/_two_stage_chain.py)
// — worth telling the user why progress won't reset partway through.
const CHAINED_MODE_LABELS = {
  full: 'Full',
  singing: 'Speech vs Singing',
  karaoke: 'Lead vs Backing Vocals',
}

/** Staged bar + everything PRD §2.4 asks the progress UI to show: a live
 * elapsed timer, a *real* progress_pct bar driven by chunks_done/total (not
 * a spinner), the ETA before separation begins, per-stage timings, and an
 * explicit cache-hit state. `skipDownload` — file uploads never pass
 * through the "downloading" stage (only URL jobs fetch anything). */
export default function ProgressStages({ job, skipDownload }) {
  if (job.from_cache) return <CacheBanner />

  const stages = skipDownload ? ALL_STAGES.filter((s) => s.key !== 'downloading') : ALL_STAGES
  const currentIndex = stages.findIndex((s) => s.key === job.stage)
  const showChunks = job.stage === 'separating' && job.chunks_total > 0
  const showEta = job.eta_seconds != null && job.status === 'running'

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          {stages.map((s, index) => {
            const state = currentIndex === -1 ? 'pending' : index < currentIndex ? 'done' : index === currentIndex ? 'active' : 'pending'
            return (
              <div key={s.key} className="flex-1">
                <div
                  className={`h-2 rounded-full transition-colors ${
                    state === 'done' ? 'bg-indigo-400' : state === 'active' ? 'bg-indigo-500/60' : 'bg-slate-800'
                  }`}
                />
                <p className={`mt-1.5 text-xs ${state === 'pending' ? 'text-slate-500' : 'text-slate-200'}`}>{s.label}</p>
              </div>
            )
          })}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-sm">
          <div className="flex items-center gap-3 text-slate-300">
            <span className="font-medium tabular-nums text-slate-100">{job.progress_pct}%</span>
            <span className="text-slate-500">·</span>
            <span>
              elapsed <ElapsedTimer elapsedSeconds={job.elapsed_seconds} running={job.status === 'running'} />
            </span>
            {showChunks && (
              <>
                <span className="text-slate-500">·</span>
                <span className="tabular-nums">
                  chunk {job.chunks_done}/{job.chunks_total}
                </span>
              </>
            )}
          </div>
          {showEta && <span className="text-slate-400">ETA ~{formatDurationWords(job.eta_seconds)}</span>}
        </div>

        {CHAINED_MODE_LABELS[job.mode] && job.stage === 'separating' && (
          <p className="text-xs text-slate-500">
            {CHAINED_MODE_LABELS[job.mode]} mode chains two separation passes — progress above is weighted across
            both, so it won't reset partway through.
          </p>
        )}
      </div>

      <StageTimings stages={stages} stageTimings={job.stage_timings} currentStage={job.stage} />
    </div>
  )
}
