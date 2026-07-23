import { MODES, TIERS } from '../api'

const MODE_LABELS = { music: 'Music', video: 'Video', full: 'Full', singing: 'Speech vs Singing', karaoke: 'Lead vs Backing Vocals' }
const MODE_HINTS = {
  music: 'vocals / drums / bass / other',
  video: 'speech / music / effects',
  full: 'speech / vocals / drums / bass / other',
  singing: 'spoken_speech / sung_vocals / instruments',
  karaoke: 'lead_vocal / backing_vocals / instruments',
}
const TIER_LABELS = { fast: 'Fast', balanced: 'Balanced', best: 'Best' }

function SegmentedGroup({ label, options, value, onChange, disabled, renderHint }) {
  return (
    <div>
      <p className="mb-1.5 text-sm font-medium text-slate-300">{label}</p>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const active = option === value
          return (
            <button
              key={option}
              type="button"
              disabled={disabled}
              onClick={() => onChange(option)}
              className={`rounded-lg border px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                active
                  ? 'border-indigo-400 bg-indigo-500/20 text-indigo-200'
                  : 'border-slate-700 text-slate-300 hover:border-slate-500'
              }`}
            >
              {renderHint ? renderHint(option) : option}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default function ModeTierPicker({ mode, onModeChange, tier, onTierChange, disabled }) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:gap-8">
      <SegmentedGroup
        label="Mode"
        options={MODES}
        value={mode}
        onChange={onModeChange}
        disabled={disabled}
        renderHint={(m) => (
          <span className="flex flex-col items-start">
            <span>{MODE_LABELS[m]}</span>
            <span className="text-[11px] font-normal text-slate-400">{MODE_HINTS[m]}</span>
          </span>
        )}
      />
      <SegmentedGroup
        label="Tier"
        options={TIERS}
        value={tier}
        onChange={onTierChange}
        disabled={disabled}
        renderHint={(t) => TIER_LABELS[t]}
      />
    </div>
  )
}
