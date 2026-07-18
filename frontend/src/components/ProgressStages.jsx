const ALL_STAGES = [
  { key: 'downloading', label: 'Download' },
  { key: 'decoding', label: 'Decode' },
  { key: 'separating', label: 'Separate' },
  { key: 'encoding', label: 'Encode' },
]

/** `skipDownload` — file uploads never pass through the "downloading" stage
 * (only URL jobs fetch anything), so JobPage hides that segment for them. */
export default function ProgressStages({ stage, progress, skipDownload }) {
  const stages = skipDownload ? ALL_STAGES.filter((s) => s.key !== 'downloading') : ALL_STAGES
  const currentIndex = stages.findIndex((s) => s.key === stage)

  return (
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
      <p className="text-sm text-slate-400">{Math.round(progress * 100)}%</p>
    </div>
  )
}
