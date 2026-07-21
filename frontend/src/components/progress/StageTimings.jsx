import { formatDurationWords } from '../../lib/format'

/** Per-stage timing table (PRD §2.4: "so a slow download is visibly
 * distinct from slow separation"). Reads job.stage_timings directly —
 * `{stage: {started_at, ended_at?}}`, written by backend/jobs.py on every
 * update_stage() call — no client-side computation of stage boundaries. */
export default function StageTimings({ stages, stageTimings, currentStage }) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
      {stages.map((s) => {
        const timing = stageTimings?.[s.key]
        const isCurrent = s.key === currentStage
        let value = '—'
        if (timing?.started_at && timing?.ended_at) {
          const secs = (new Date(timing.ended_at) - new Date(timing.started_at)) / 1000
          value = formatDurationWords(secs)
        } else if (timing?.started_at) {
          value = 'in progress'
        }
        return (
          <div key={s.key} className="flex flex-col gap-0.5">
            <dt className={isCurrent ? 'text-indigo-300' : 'text-slate-500'}>{s.label}</dt>
            <dd className={isCurrent ? 'font-medium text-indigo-200' : 'text-slate-400'}>{value}</dd>
          </div>
        )
      })}
    </dl>
  )
}
