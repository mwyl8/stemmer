/** One control row per stem: name, mute, solo, volume. Purely controlled —
 * StemPlayer owns the actual mute/solo/volume state and recomputes effective
 * per-track volume (solo overrides everything else) whenever any row changes. */
export default function StemRow({ name, volume, muted, solo, isKaraokeTarget, onVolumeChange, onToggleMute, onToggleSolo }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
      <span className="w-20 shrink-0 truncate text-sm font-medium capitalize text-slate-200" title={isKaraokeTarget ? `${name} (karaoke target)` : name}>
        {name}
      </span>
      <button
        type="button"
        onClick={onToggleMute}
        aria-pressed={muted}
        className={`h-7 w-7 shrink-0 rounded text-xs font-semibold transition-colors ${
          muted ? 'bg-red-500 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
        }`}
        title="Mute"
      >
        M
      </button>
      <button
        type="button"
        onClick={onToggleSolo}
        aria-pressed={solo}
        className={`h-7 w-7 shrink-0 rounded text-xs font-semibold transition-colors ${
          solo ? 'bg-amber-400 text-slate-900' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
        }`}
        title="Solo"
      >
        S
      </button>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(volume * 100)}
        onChange={(e) => onVolumeChange(Number(e.target.value) / 100)}
        className="flex-1 accent-indigo-400"
      />
      <span className="w-10 shrink-0 text-right text-xs text-slate-400">{Math.round(volume * 100)}%</span>
    </div>
  )
}
