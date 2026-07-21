import { formatTime } from '../../lib/format'

const RATE_LABELS = { 0.5: '0.5×', 0.75: '0.75×', 1: '1×', 1.25: '1.25×', 1.5: '1.5×', 2: '2×' }

function TransportButton({ onClick, disabled, label, children, active }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={`focus-ring flex h-9 min-w-9 items-center justify-center rounded-lg px-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        active ? 'bg-indigo-500/20 text-indigo-200' : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
      }`}
    >
      {children}
    </button>
  )
}

/** The full transport bar: play/pause, ±10s/±30s skip, jump to start/end,
 * current/duration readout, playback speed, zoom, and the A/B loop toggle
 * (PRD §2.1). Every control here goes through `player`'s public actions —
 * nothing touches audio elements directly, so sync is never at risk. */
export default function Transport({ player, hasKaraokeTarget, onToggleKaraoke }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <TransportButton onClick={player.jumpToStart} disabled={!player.ready} label="Jump to start">
        ⏮
      </TransportButton>
      <TransportButton onClick={() => player.skip(-30)} disabled={!player.ready} label="Back 30 seconds">
        −30s
      </TransportButton>
      <TransportButton onClick={() => player.skip(-10)} disabled={!player.ready} label="Back 10 seconds">
        −10s
      </TransportButton>

      <button
        type="button"
        onClick={player.togglePlay}
        disabled={!player.ready}
        aria-label={player.playing ? 'Pause' : 'Play'}
        className="focus-ring flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-indigo-500 text-lg font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {player.playing ? '⏸' : '▶'}
      </button>

      <TransportButton onClick={() => player.skip(10)} disabled={!player.ready} label="Forward 10 seconds">
        +10s
      </TransportButton>
      <TransportButton onClick={() => player.skip(30)} disabled={!player.ready} label="Forward 30 seconds">
        +30s
      </TransportButton>
      <TransportButton onClick={player.jumpToEnd} disabled={!player.ready} label="Jump to end">
        ⏭
      </TransportButton>

      <span className="min-w-24 select-none text-center font-mono text-sm text-slate-300" aria-live="off">
        {formatTime(player.currentTime)} / {formatTime(player.duration)}
      </span>

      <div className="mx-1 h-6 w-px bg-slate-800" />

      <label className="flex items-center gap-1.5 text-sm text-slate-400">
        Speed
        <select
          value={player.playbackRate}
          onChange={(e) => player.setPlaybackRate(Number(e.target.value))}
          disabled={!player.ready}
          className="focus-ring rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200 disabled:opacity-40"
        >
          {player.playbackRates.map((rate) => (
            <option key={rate} value={rate}>
              {RATE_LABELS[rate] ?? `${rate}×`}
            </option>
          ))}
        </select>
      </label>

      <div className="flex items-center gap-1">
        <TransportButton onClick={() => player.zoomBy(1 / 1.5)} disabled={!player.ready} label="Zoom out">
          −
        </TransportButton>
        <span className="w-10 select-none text-center text-xs text-slate-500">zoom</span>
        <TransportButton onClick={() => player.zoomBy(1.5)} disabled={!player.ready} label="Zoom in">
          +
        </TransportButton>
      </div>

      <TransportButton onClick={player.toggleLoopRegion} disabled={!player.ready} label="Toggle A/B loop region" active={player.loopEnabled}>
        {player.loopEnabled ? 'Loop on' : 'Loop'}
      </TransportButton>

      {hasKaraokeTarget && (
        <TransportButton onClick={onToggleKaraoke} disabled={!player.ready} label="Karaoke — mute vocals/speech">
          Karaoke
        </TransportButton>
      )}
    </div>
  )
}
