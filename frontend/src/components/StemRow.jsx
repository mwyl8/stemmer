import VUMeter from './player/VUMeter'
import PanKnob from './player/PanKnob'

/** One control row per stem: color swatch, mute/solo, volume, pan, live VU
 * meter. The row itself is focusable (tabIndex=0) and reports focus back to
 * the player hook — that's what the M/S keyboard shortcuts (PRD §2.1)
 * target, so tabbing to a row and pressing M/S acts on exactly the stem
 * you're looking at, the same stem a screen reader would just have
 * announced. Purely controlled — useMultitrackPlayer owns the actual
 * mute/solo/volume/pan state and recomputes effective volume (solo
 * overrides everything else) whenever any row changes. */
export default function StemRow({
  index,
  name,
  volume,
  muted,
  solo,
  pan,
  color,
  isKaraokeTarget,
  isFocused,
  onFocus,
  onVolumeChange,
  onPanChange,
  onToggleMute,
  onToggleSolo,
  registerMeterEl,
}) {
  return (
    <div
      tabIndex={0}
      onFocus={onFocus}
      role="group"
      aria-label={`${name} stem controls`}
      className={`focus-ring flex flex-col gap-2 rounded-lg border bg-slate-900/60 px-3 py-2.5 transition-colors sm:flex-row sm:items-center sm:gap-3 ${
        isFocused ? 'border-indigo-500/60' : 'border-slate-800'
      }`}
    >
      <div className="flex items-center gap-2 sm:w-28 sm:shrink-0">
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: color.solid }} aria-hidden="true" />
        <span className="truncate text-sm font-medium capitalize text-slate-200" title={isKaraokeTarget ? `${name} (karaoke target)` : name}>
          {name}
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        <button
          type="button"
          onClick={onToggleMute}
          aria-pressed={muted}
          aria-label={`Mute ${name}`}
          className={`focus-ring h-7 w-7 shrink-0 rounded text-xs font-semibold transition-colors ${
            muted ? 'bg-red-500 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
          title="Mute (M)"
        >
          M
        </button>
        <button
          type="button"
          onClick={onToggleSolo}
          aria-pressed={solo}
          aria-label={`Solo ${name}`}
          className={`focus-ring h-7 w-7 shrink-0 rounded text-xs font-semibold transition-colors ${
            solo ? 'bg-amber-400 text-slate-900' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
          title="Solo (S)"
        >
          S
        </button>
      </div>

      <div className="flex flex-1 items-center gap-2">
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(volume * 100)}
          onChange={(e) => onVolumeChange(Number(e.target.value) / 100)}
          aria-label={`${name} volume`}
          className="focus-ring h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-slate-800 accent-indigo-400"
        />
        <span className="w-9 shrink-0 text-right text-xs tabular-nums text-slate-400">{Math.round(volume * 100)}%</span>
      </div>

      <PanKnob pan={pan} onChange={onPanChange} label={name} />

      <div className="w-full sm:w-24 sm:shrink-0">
        <VUMeter registerRef={(el) => registerMeterEl(index, el)} color={color.solid} />
      </div>
    </div>
  )
}
