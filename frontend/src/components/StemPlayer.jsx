import useMultitrackPlayer from '../hooks/useMultitrackPlayer'
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts'
import { stemColor } from '../lib/stemPalette'
import Transport from './player/Transport'
import Minimap from './player/Minimap'
import StemRow from './StemRow'

/** One synced multitrack transport loading the mp3 *previews* only — never
 * the wavs, which stay download-only (PRD §2.4: avoid decoding large files
 * in-browser). `stemNames` should be a referentially stable array (see
 * ResultView's useMemo) so this doesn't tear down and rebuild the player —
 * and restart playback — on every unrelated re-render. */
export default function StemPlayer({ jobId, stemNames }) {
  const player = useMultitrackPlayer(jobId, stemNames)
  useKeyboardShortcuts(player.ready ? player : null)

  return (
    <div className="flex flex-col gap-4">
      <Transport player={player} hasKaraokeTarget={player.hasKaraokeTarget} onToggleKaraoke={player.toggleKaraoke} />

      <Minimap player={player} />

      <div className="flex flex-col gap-2">
        {player.tracks.map((t, index) => (
          <StemRow
            key={t.name}
            index={index}
            name={t.name}
            volume={t.volume}
            muted={t.muted}
            solo={t.solo}
            pan={t.pan}
            color={stemColor(t.name)}
            isKaraokeTarget={player.hasKaraokeTarget && (t.name === 'vocals' || t.name === 'speech')}
            isFocused={player.focusedIndex === index}
            onFocus={() => player.setFocusedIndex(index)}
            onVolumeChange={(v) => player.setTrackVolume(index, v)}
            onPanChange={(p) => player.setTrackPan(index, p)}
            onToggleMute={() => player.toggleMute(index)}
            onToggleSolo={() => player.toggleSolo(index)}
            registerMeterEl={player.registerMeterEl}
          />
        ))}
      </div>

      <div className="relative">
        <div
          ref={player.containerRef}
          className="scrollbar-thin overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-2"
          aria-label="Stem waveforms — click anywhere to seek"
        />
        {!player.ready && (
          <div className="absolute inset-0 flex flex-col justify-center gap-2 rounded-xl bg-slate-950 p-2" aria-hidden="true">
            {stemNames.map((name) => (
              <div key={name} className="skeleton h-16 w-full" />
            ))}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Shortcuts: <kbd className="rounded bg-slate-800 px-1">Space</kbd> play/pause ·{' '}
        <kbd className="rounded bg-slate-800 px-1">←/→</kbd> skip 10s (
        <kbd className="rounded bg-slate-800 px-1">Shift</kbd>+←/→ for 30s) ·{' '}
        <kbd className="rounded bg-slate-800 px-1">M</kbd>/<kbd className="rounded bg-slate-800 px-1">S</kbd> mute/solo the
        focused stem · <kbd className="rounded bg-slate-800 px-1">0</kbd>–<kbd className="rounded bg-slate-800 px-1">9</kbd>{' '}
        jump to 0–90%.
      </p>
    </div>
  )
}
