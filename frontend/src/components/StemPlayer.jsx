import { useCallback, useEffect, useRef, useState } from 'react'
import Multitrack from 'wavesurfer-multitrack'
import StemRow from './StemRow'
import { KARAOKE_STEM_NAMES, stemUrl } from '../api'

const WAVE_COLORS = [
  ['#818cf8', '#4338ca'],
  ['#34d399', '#047857'],
  ['#fbbf24', '#b45309'],
  ['#f472b6', '#be185d'],
  ['#60a5fa', '#1d4ed8'],
]

/** One synced multitrack transport (wavesurfer-multitrack) loading the mp3
 * *previews* only — never the wavs, which stay download-only (PRD §6: avoid
 * decoding large files in-browser). `stemNames` should be a referentially
 * stable array (see ResultView's useMemo) so this doesn't tear down and
 * rebuild the player — and restart playback — on every unrelated re-render. */
export default function StemPlayer({ jobId, stemNames }) {
  const containerRef = useRef(null)
  const multitrackRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [tracks, setTracks] = useState(() => stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false })))

  useEffect(() => {
    setReady(false)
    setPlaying(false)
    setTracks(stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false })))

    const instance = Multitrack.create(
      stemNames.map((name, index) => {
        const [waveColor, progressColor] = WAVE_COLORS[index % WAVE_COLORS.length]
        return {
          id: index,
          url: stemUrl(jobId, name, 'mp3'),
          volume: 1,
          draggable: false,
          options: { waveColor, progressColor, height: 56 },
        }
      }),
      {
        container: containerRef.current,
        minPxPerSec: 40,
        cursorWidth: 2,
        cursorColor: '#e2e8f0',
        trackBackground: '#0f172a',
        trackBorderColor: '#1e293b',
      },
    )
    multitrackRef.current = instance
    setReady(true)

    // The plugin doesn't expose play/pause events publicly (only drag/volume/
    // cue events) — poll isPlaying() so the button label can't drift out of
    // sync with reality (e.g. when playback reaches the end on its own).
    const syncPlaying = setInterval(() => setPlaying(instance.isPlaying()), 300)

    return () => {
      clearInterval(syncPlaying)
      instance.destroy()
      multitrackRef.current = null
    }
  }, [jobId, stemNames])

  const applyVolumes = useCallback((nextTracks) => {
    const instance = multitrackRef.current
    if (!instance) return
    const anySolo = nextTracks.some((t) => t.solo)
    nextTracks.forEach((t, index) => {
      const effective = anySolo ? (t.solo ? t.volume : 0) : t.muted ? 0 : t.volume
      instance.setTrackVolume(index, effective)
    })
  }, [])

  function updateTrack(index, patch) {
    setTracks((prev) => {
      const next = prev.map((t, i) => (i === index ? { ...t, ...patch } : t))
      applyVolumes(next)
      return next
    })
  }

  function togglePlay() {
    const instance = multitrackRef.current
    if (!instance) return
    instance.isPlaying() ? instance.pause() : instance.play()
    setPlaying(instance.isPlaying())
  }

  function toggleKaraoke() {
    setTracks((prev) => {
      const currentlyMuted = prev.some((t) => KARAOKE_STEM_NAMES.has(t.name) && t.muted)
      const next = prev.map((t) => (KARAOKE_STEM_NAMES.has(t.name) ? { ...t, muted: !currentlyMuted } : t))
      applyVolumes(next)
      return next
    })
  }

  const hasKaraokeTarget = stemNames.some((name) => KARAOKE_STEM_NAMES.has(name))

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          disabled={!ready}
          className="rounded-full bg-indigo-500 px-5 py-2 font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        {hasKaraokeTarget && (
          <button
            type="button"
            onClick={toggleKaraoke}
            className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 transition-colors hover:border-slate-500"
          >
            Karaoke (mute vocals)
          </button>
        )}
      </div>

      <div className="flex flex-col gap-2">
        {tracks.map((t, index) => (
          <StemRow
            key={t.name}
            name={t.name}
            volume={t.volume}
            muted={t.muted}
            solo={t.solo}
            isKaraokeTarget={KARAOKE_STEM_NAMES.has(t.name)}
            onVolumeChange={(v) => updateTrack(index, { volume: v })}
            onToggleMute={() => updateTrack(index, { muted: !t.muted })}
            onToggleSolo={() => updateTrack(index, { solo: !t.solo })}
          />
        ))}
      </div>

      <div ref={containerRef} className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950 p-2" />
    </div>
  )
}
