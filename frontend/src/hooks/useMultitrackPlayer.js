import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Multitrack from 'wavesurfer-multitrack'
import WaveSurfer from 'wavesurfer.js'
import SpectrogramPlugin from 'wavesurfer.js/plugins/spectrogram'
import { stemColor } from '../lib/stemPalette'
import { KARAOKE_STEM_NAMES, getStemPeaks, originalUrl, stemUrl } from '../api'

const MIN_PX_PER_SEC = 15
const MAX_PX_PER_SEC = 800
const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2]

/**
 * Owns one wavesurfer-multitrack instance and everything layered on top of
 * its public API: transport, per-track volume/mute/solo (native to the
 * library), plus several things it doesn't provide that PRD §2.1/§2.2/§2.3
 * ask for:
 *
 * - **Pan + VU meters**: wavesurfer-multitrack sets each track's gain via
 *   the underlying <audio> element's `.volume` (confirmed by reading the
 *   bundled source — there's no `createMediaElementSource` in it at all),
 *   and exposes its `audios` (HTMLAudioElement[]) and shared `audioContext`
 *   as plain instance properties (TypeScript's `private` is compile-time
 *   only — both are reachable at runtime). So: tap each track's element
 *   into a small Web Audio graph — `source -> StereoPannerNode ->
 *   AnalyserNode -> destination` — for pan and level metering, entirely
 *   alongside the library's own `.volume`-based gain, which keeps working
 *   unmodified (element volume/mute still attenuate before a
 *   MediaElementSourceNode per spec). iOS/iPadOS routes tracks through a
 *   different WebAudioPlayer wrapper internally (Safari's multi-<audio>
 *   limitations) — pan/VU are skipped there rather than risk breaking
 *   playback, since that wrapper isn't a real HTMLAudioElement.
 * - **A/B loop region**: implemented as a plain absolutely-positioned div
 *   appended into the multitrack's own scrollable inner wrapper (so it
 *   scrolls/zooms in lock-step with the waveforms), with two drag handles.
 *   Looping itself is just "if playing and past the region end, jump back
 *   to the start" — checked once a frame, in the same loop already reading
 *   `getCurrentTime()`.
 * - **Minimap**: the library has no multi-track-aware minimap, and
 *   wavesurfer.js's own Minimap plugin is built for a single instance. This
 *   hook exposes decoded peaks from one reference track plus a small
 *   scroll-fraction subscription so `<Minimap>` can draw its own overview +
 *   viewport box and drive the real scroll position directly.
 * - **Precomputed peaks (PRD Addendum §2.5)**: fetched from
 *   GET /jobs/{id}/stems/{name}/peaks and passed as each track's `peaks`
 *   option. wavesurfer.js treats a supplied `peaks` array exactly like
 *   decoded audio (see its `loadAudio`), so when both `peaks` and the
 *   duration it derives internally are present it skips the full blob
 *   fetch + AudioContext decode entirely and just points the underlying
 *   <audio> element at the mp3 preview URL for native streaming playback —
 *   the browser never decodes the full file just to draw a waveform.
 * - **Per-stem spectrogram toggle (PRD Addendum §2.2)**: the SpectrogramPlugin
 *   computes its own FFT from `this.wavesurfer.getDecodedData()` on
 *   whichever WaveSurfer instance it's registered on — real, dense
 *   time-domain PCM, sliced into windows via `Float32Array.subarray()`.
 *   The multitrack's own per-track instances are *not* a valid host: their
 *   `getDecodedData()` returns the precomputed `peaks` array from above
 *   (a sparse min/max-per-bucket downsample, not real samples), so an FFT
 *   window sliced from it would be spectral nonsense, not to mention that
 *   plain JS arrays (peaks arrive as JSON) don't even have `.subarray` —
 *   that's the "e3.subarray is not a function" crash this replaced.
 *   Wrapping the peaks in a Float32Array would silence the crash but still
 *   feed the FFT synthetic downsampled data, producing a spectrogram that
 *   *looks* plausible but doesn't correspond to the audio's real frequency
 *   content — worse than the crash, since it fails silently.
 *   So: each stem gets its own dedicated, lazily-created (only once
 *   actually toggled on) decode-only WaveSurfer, loading the same mp3
 *   preview with no `peaks` override — a real, full decode, exactly like
 *   the pre-peaks version of this player did for every stem. It renders
 *   into an invisible 1px host container (real layout, so plugin sizing
 *   reads a correct nonzero width, but nothing user-visible) plus a visible
 *   sibling container for the actual spectrogram canvas, both absolutely
 *   positioned over the track row. It's zoomed to fit its own full duration
 *   to the row's width once decoded, independent of the live waveform's
 *   current zoom/scroll — a deliberate scope cut (see the toggleSpectrogram
 *   comment) rather than wiring pixel-locked sync into a second instance.
 * - **A/B against the original (PRD §2.3)**: a second, plain
 *   `HTMLAudioElement` loaded with the source mixture's mp3
 *   (`GET /jobs/{id}/original`, `preload="none"` so it's never fetched
 *   unless the user actually toggles to it). Toggling swaps which side the
 *   transport controls act on and carries the current time/playing state
 *   across the switch, so it reads as "the same mix at a different volume"
 *   rather than a jump back to the start.
 * - **Master volume / mute-all / reset-mix (PRD §2.3)**: folded into the
 *   same effective-volume computation the per-track solo/mute logic
 *   already does, so a soloed track still respects master mute/volume.
 *
 * Playback itself never bypasses the library's public transport
 * (play/pause/seekTo/setTime) — only pan/VU touch the per-track audio
 * graph, and only downstream of the library's own gain — so every stem
 * stays sample-synced through skip/seek/zoom/loop/speed exactly as the
 * library's own drift-correcting sync loop (`startSync`) already
 * guarantees.
 */
export default function useMultitrackPlayer(jobId, stemNames, { hasOriginal = false } = {}) {
  const containerRef = useRef(null)
  const instanceRef = useRef(null)
  const audioNodesRef = useRef([]) // [{source, panner, analyser} | null] per stem index
  const meterElsRef = useRef([]) // [HTMLElement | null] per stem index, written to directly (no React state)
  const rafRef = useRef(null)
  const loopRef = useRef(null) // {start, end} in seconds, or null
  const loopEnabledRef = useRef(false)
  const vuMetersEnabledRef = useRef(true)
  const loopOverlayRef = useRef(null) // vanilla-DOM controller, see makeLoopOverlay()
  const scrollWrapperRef = useRef(null) // the library's own inner `overflow-x: scroll` div
  const scrollCleanupRef = useRef(null)
  const scrollSubscribersRef = useRef(new Set())
  const referencePeaksRef = useRef(null)
  const spectrogramNodesRef = useRef([]) // [{container, plugin} | null] per stem index, created lazily
  const originalAudioRef = useRef(null) // hidden <audio> for the A/B toggle
  const abModeRef = useRef('stems') // 'stems' | 'original' -- mirrors abMode state for use in rAF/callbacks
  const tracksRef = useRef([]) // mirrors `tracks` state -- lets master volume/mute/reset read it without a stale closure
  const masterVolumeRef = useRef(1)
  const masterMutedRef = useRef(false)

  const [ready, setReady] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playbackRate, setPlaybackRateState] = useState(1)
  const [minPxPerSec, setMinPxPerSec] = useState(MIN_PX_PER_SEC)
  const [loopRegion, setLoopRegionState] = useState(null)
  const [loopEnabled, setLoopEnabledState] = useState(false)
  const [vuMetersEnabled, setVuMetersEnabledState] = useState(true)
  const [focusedIndex, setFocusedIndex] = useState(stemNames.length > 0 ? 0 : null)
  const [tracks, setTracks] = useState(() => stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false, pan: 0 })))
  const [spectrogramStems, setSpectrogramStems] = useState(() => new Set())
  const [abMode, setAbModeState] = useState('stems')
  const [masterVolume, setMasterVolumeState] = useState(1)
  const [masterMuted, setMasterMutedState] = useState(false)

  useEffect(() => {
    tracksRef.current = tracks
  }, [tracks])

  const applyEffectiveVolumes = useCallback(
    (nextTracks, masterVol = masterVolumeRef.current, masterMute = masterMutedRef.current) => {
      const instance = instanceRef.current
      if (!instance) return
      const anySolo = nextTracks.some((t) => t.solo)
      nextTracks.forEach((t, index) => {
        const soloEffective = anySolo ? (t.solo ? t.volume : 0) : t.muted ? 0 : t.volume
        instance.setTrackVolume(index, masterMute ? 0 : soloEffective * masterVol)
      })
    },
    [],
  )

  // --- mount: build the Multitrack instance + the pan/VU tap + loop overlay ---
  useEffect(() => {
    if (!containerRef.current || stemNames.length === 0) return

    let cancelled = false
    setReady(false)
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setLoopRegionState(null)
    setLoopEnabledState(false)
    loopRef.current = null
    loopEnabledRef.current = false
    setVuMetersEnabledState(true)
    vuMetersEnabledRef.current = true
    referencePeaksRef.current = null
    spectrogramNodesRef.current = []
    setSpectrogramStems(new Set())
    abModeRef.current = 'stems'
    setAbModeState('stems')
    masterVolumeRef.current = 1
    masterMutedRef.current = false
    setMasterVolumeState(1)
    setMasterMutedState(false)
    const initialTracks = stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false, pan: 0 }))
    setTracks(initialTracks)
    tracksRef.current = initialTracks
    setFocusedIndex(0)

    const originalAudio = new Audio()
    originalAudio.preload = 'none' // never fetched unless the user actually toggles to it
    originalAudio.src = originalUrl(jobId)
    originalAudioRef.current = originalAudio

    let cleanupInner = () => {}

    ;(async () => {
      // Precomputed peaks (PRD Addendum §2.5) -- best-effort per stem; a
      // stem with no peaks yet (or a 404) just falls back to wavesurfer's
      // normal full decode instead of blocking the whole player on it.
      const peaksPerStem = await Promise.all(stemNames.map((name) => getStemPeaks(jobId, name).catch(() => null)))
      if (cancelled) return

      const instance = Multitrack.create(
        stemNames.map((name, index) => {
          const colors = stemColor(name)
          const peaks = peaksPerStem[index]
          return {
            id: index,
            url: stemUrl(jobId, name, 'mp3'),
            ...(peaks ? { peaks } : {}),
            volume: 1,
            draggable: false,
            options: { waveColor: colors.wave, progressColor: colors.progress, height: 64 },
          }
        }),
        {
          container: containerRef.current,
          minPxPerSec: MIN_PX_PER_SEC,
          cursorWidth: 2,
          cursorColor: '#e2e8f0',
          trackBackground: '#0f172a',
          trackBorderColor: '#1e293b',
          timelineOptions: { height: 20, insertPosition: 'beforebegin' },
        },
      )
      if (cancelled) {
        instance.destroy()
        return
      }
      instanceRef.current = instance

      const unsubCanplay = instance.once('canplay', () => {
        const total = instance.maxDuration || 0
        setDuration(total)
        setReady(true)

        // Fit the initial zoom to the container width so a short clip doesn't
        // open zoomed-out to nothing, and a long one doesn't open zoomed in
        // past the first few seconds.
        //
        // Multitrack's own 'canplay' fires the instant every per-track
        // WaveSurfer has been *constructed* (right after initAllWavesurfers,
        // synchronously) -- it does NOT wait for each one's own async decode
        // to finish setting its `decodedData`, which is what Multitrack's
        // own `.zoom()` requires from every track (it throws "No audio
        // loaded" for any track it isn't set on yet, aborting the whole
        // per-track loop partway through). Before peaks.js, every track did
        // a real fetch+AudioContext decode, which was slow enough that this
        // race basically never lost; peaks-based tracks resolve their
        // "decodedData" via a same-tick promise continuation (Decoder.
        // createBuffer, no network/decode at all), so the race is real now
        // -- reproducible on every load, not intermittent. Waiting for each
        // track's own 'ready' (proof `decodedData` is actually set) before
        // fitting zoom fixes the race at its source instead of catching the
        // resulting throw.
        const containerWidth = containerRef.current?.clientWidth || 800
        const fitPx = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, containerWidth / (total || 1)))
        const applyFitZoom = () => {
          if (cancelled) return
          instance.zoom(fitPx)
          setMinPxPerSec(fitPx)
        }
        const pendingDecodes = (instance.wavesurfers || [])
          .filter((trackWs) => !trackWs.getDecodedData())
          .map((trackWs) => new Promise((resolve) => trackWs.once('ready', resolve)))
        if (pendingDecodes.length === 0) applyFitZoom()
        else Promise.all(pendingDecodes).then(applyFitZoom)

        // The library's DOM structure (read from its bundled source): the
        // container it's given gets exactly one child, an `overflow-x:
        // scroll` wrapper, whose own child is a `position: relative` wrapper
        // holding every track row plus the cursor. That inner wrapper is
        // where a loop-region overlay needs to live to scroll/zoom in sync.
        const scrollWrapper = containerRef.current?.firstElementChild
        const innerWrapper = scrollWrapper?.firstElementChild
        scrollWrapperRef.current = scrollWrapper || null
        if (innerWrapper instanceof HTMLElement) {
          loopOverlayRef.current = makeLoopOverlay(innerWrapper, (nextRegion) => {
            loopRef.current = nextRegion
            setLoopRegionState(nextRegion)
          })
        }
        if (scrollWrapper instanceof HTMLElement) {
          const onScroll = () => scrollSubscribersRef.current.forEach((cb) => cb())
          scrollWrapper.addEventListener('scroll', onScroll)
          scrollCleanupRef.current = () => scrollWrapper.removeEventListener('scroll', onScroll)
        }

        // Reference peaks for the minimap — first track that actually decodes.
        const refWs = instance.wavesurfers?.[0]
        if (refWs) {
          const grabPeaks = () => {
            try {
              const decoded = refWs.getDecodedData?.()
              if (decoded) referencePeaksRef.current = decoded.getChannelData(0)
            } catch {
              // no decoded data yet / unsupported — minimap just shows a flat bar
            }
          }
          grabPeaks()
          refWs.on?.('ready', grabPeaks)
        }
      })

      // Pan + VU tap. Each element can only ever have createMediaElementSource
      // called on it once — safe here because this whole block runs exactly
      // once per fresh Multitrack instance (new <audio> elements every time).
      const attachAudioGraph = () => {
        const ctx = instance.audioContext
        if (!ctx) return
        audioNodesRef.current = stemNames.map((_, index) => {
          const el = instance.audios?.[index]
          if (!(el instanceof HTMLAudioElement)) return null // iOS WebAudioPlayer path — skip, don't risk playback
          try {
            const source = ctx.createMediaElementSource(el)
            const panner = ctx.createStereoPanner()
            const analyser = ctx.createAnalyser()
            analyser.fftSize = 256
            analyser.smoothingTimeConstant = 0.6
            source.connect(panner)
            panner.connect(analyser)
            analyser.connect(ctx.destination)
            return { source, panner, analyser }
          } catch {
            return null // element already tapped, or context in a bad state — degrade gracefully
          }
        })
      }
      // audios[] is populated asynchronously inside Multitrack's own init; the
      // 'canplay' event fires only after that resolves, so it's also the
      // right moment to attach the parallel Web Audio graph.
      const unsubForAudioGraph = instance.once('canplay', attachAudioGraph)

      // One shared rAF loop: throttled currentTime state, loop-region
      // enforcement, and direct (non-React) VU meter bar updates. Reads
      // from whichever side of the A/B toggle is currently active.
      //
      // VU reads are throttled to ~30fps (not the loop's own ~60fps) and can be
      // disabled entirely via vuMetersEnabledRef/setVuMetersEnabled — added per
      // Part A #4's audio-quality audit: this was the one candidate cause of
      // browser-player static that hadn't been ruled out by inspection alone
      // (AnalyserNodes are already created once in attachAudioGraph above, not
      // per-render, and fftSize is already the small end at 256), so the
      // throttle plus an explicit off switch makes it directly testable: if
      // crackle persists with meters off, the meters aren't the cause.
      let lastStatePush = 0
      let lastMeterUpdate = 0
      const METER_INTERVAL_MS = 1000 / 30
      const meterBuf = new Uint8Array(128)
      function frame(now) {
        const usingOriginal = abModeRef.current === 'original'
        const originalEl = originalAudioRef.current
        const t = usingOriginal ? (originalEl?.currentTime ?? 0) : (instance.getCurrentTime?.() ?? 0)
        const isPlaying = usingOriginal ? !(originalEl?.paused ?? true) : (instance.isPlaying?.() ?? false)

        if (!usingOriginal) {
          const region = loopRef.current
          if (loopEnabledRef.current && region && isPlaying && t >= region.end) {
            instance.setTime(region.start)
          }
        }

        if (now - lastStatePush > 80) {
          lastStatePush = now
          setCurrentTime(t)
          setPlaying(isPlaying)
        }

        if (now - lastMeterUpdate > METER_INTERVAL_MS) {
          lastMeterUpdate = now
          const metersOn = vuMetersEnabledRef.current && !usingOriginal
          audioNodesRef.current.forEach((nodes, index) => {
            const el = meterElsRef.current[index]
            if (!el) return
            let level = 0
            if (metersOn && nodes?.analyser) {
              nodes.analyser.getByteTimeDomainData(meterBuf)
              let sumSquares = 0
              for (let i = 0; i < meterBuf.length; i++) {
                const centered = (meterBuf[i] - 128) / 128
                sumSquares += centered * centered
              }
              level = Math.min(1, Math.sqrt(sumSquares / meterBuf.length) * 3.2) // RMS, gained up — meters read near-silent audio as visibly near-zero, not invisible
            }
            el.style.transform = `scaleX(${level})`
          })
        }

        rafRef.current = requestAnimationFrame(frame)
      }
      rafRef.current = requestAnimationFrame(frame)

      cleanupInner = () => {
        unsubCanplay?.()
        unsubForAudioGraph?.()
        if (rafRef.current) cancelAnimationFrame(rafRef.current)
        audioNodesRef.current.forEach((nodes) => {
          try {
            nodes?.source.disconnect()
            nodes?.panner.disconnect()
            nodes?.analyser.disconnect()
          } catch {
            // already torn down
          }
        })
        audioNodesRef.current = []
        spectrogramNodesRef.current.forEach((entry) => {
          try {
            entry?.unsubReady?.()
            entry?.decodeWs?.destroy() // also destroys the registered SpectrogramPlugin (WaveSurfer.destroy() does this itself)
            entry?.overlay?.remove()
          } catch {
            // already torn down
          }
        })
        spectrogramNodesRef.current = []
        loopOverlayRef.current?.destroy()
        loopOverlayRef.current = null
        scrollCleanupRef.current?.()
        scrollCleanupRef.current = null
        scrollWrapperRef.current = null
        // scrollSubscribersRef's Set identity never changes after useRef's
        // initializer — safe to read .current here, unlike a ref pointing at
        // a DOM node that could've been swapped out by the time this runs.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        scrollSubscribersRef.current.clear()
        instance.destroy()
        instanceRef.current = null
      }
    })()

    return () => {
      cancelled = true
      cleanupInner()
      originalAudioRef.current?.pause()
      originalAudioRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, stemNames])

  // --- transport ---
  const togglePlay = useCallback(() => {
    if (abModeRef.current === 'original') {
      const audio = originalAudioRef.current
      if (!audio) return
      if (audio.paused) audio.play().catch(() => {})
      else audio.pause()
      setPlaying(!audio.paused)
      return
    }
    const instance = instanceRef.current
    if (!instance) return
    if (instance.isPlaying()) instance.pause()
    else instance.play()
    setPlaying(instance.isPlaying())
  }, [])

  const seekFraction = useCallback((fraction) => {
    instanceRef.current?.seekTo(Math.min(1, Math.max(0, fraction)))
  }, [])

  const seekTime = useCallback(
    (seconds) => {
      if (abModeRef.current === 'original') {
        const audio = originalAudioRef.current
        if (!audio) return
        const clamped = Math.min(audio.duration || duration || 0, Math.max(0, seconds))
        audio.currentTime = clamped
        setCurrentTime(clamped)
        return
      }
      const instance = instanceRef.current
      if (!instance) return
      const clamped = Math.min(instance.maxDuration || 0, Math.max(0, seconds))
      instance.setTime(clamped)
      setCurrentTime(clamped)
    },
    [duration],
  )

  const getCurrentPlaybackTime = useCallback(() => {
    if (abModeRef.current === 'original') return originalAudioRef.current?.currentTime ?? 0
    return instanceRef.current?.getCurrentTime() ?? 0
  }, [])

  const skip = useCallback(
    (deltaSeconds) => {
      seekTime(getCurrentPlaybackTime() + deltaSeconds)
    },
    [seekTime, getCurrentPlaybackTime],
  )

  const jumpToStart = useCallback(() => seekTime(0), [seekTime])
  const jumpToEnd = useCallback(() => {
    const total = abModeRef.current === 'original' ? (originalAudioRef.current?.duration ?? duration) : (instanceRef.current?.maxDuration ?? duration)
    seekTime(Math.max(0, (total || 0) - 0.05))
  }, [seekTime, duration])
  const jumpToPercent = useCallback((pct) => seekFraction(pct / 100), [seekFraction])

  const cyclePlaybackRate = useCallback((direction) => {
    setPlaybackRateState((prev) => {
      const i = PLAYBACK_RATES.indexOf(prev)
      const next = PLAYBACK_RATES[Math.min(PLAYBACK_RATES.length - 1, Math.max(0, i + direction))]
      applyPlaybackRate(instanceRef.current, originalAudioRef.current, next)
      return next
    })
  }, [])

  const setPlaybackRate = useCallback((rate) => {
    setPlaybackRateState(rate)
    applyPlaybackRate(instanceRef.current, originalAudioRef.current, rate)
  }, [])

  const zoomBy = useCallback((factor) => {
    setMinPxPerSec((prev) => {
      const next = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, prev * factor))
      instanceRef.current?.zoom(next)
      loopOverlayRef.current?.reflow()
      return next
    })
  }, [])

  // --- A/B against the original (PRD §2.3) ---
  const setAbMode = useCallback((mode) => {
    if (mode === abModeRef.current) return
    const instance = instanceRef.current
    const originalAudio = originalAudioRef.current
    if (!instance || !originalAudio) return

    const wasPlaying = abModeRef.current === 'original' ? !originalAudio.paused : instance.isPlaying()
    const t = abModeRef.current === 'original' ? originalAudio.currentTime : instance.getCurrentTime()

    if (mode === 'original') {
      instance.pause()
      originalAudio.currentTime = t
      if (wasPlaying) originalAudio.play().catch(() => {})
    } else {
      originalAudio.pause()
      instance.setTime(t)
      if (wasPlaying) instance.play()
    }
    abModeRef.current = mode
    setAbModeState(mode)
    setPlaying(wasPlaying)
  }, [])

  const toggleAbMode = useCallback(() => {
    setAbMode(abModeRef.current === 'original' ? 'stems' : 'original')
  }, [setAbMode])

  // --- loop region ---
  const toggleLoopRegion = useCallback(() => {
    const overlay = loopOverlayRef.current
    const instance = instanceRef.current
    if (!overlay || !instance) return
    if (loopRef.current) {
      overlay.clear()
      loopRef.current = null
      setLoopRegionState(null)
      setLoopEnabledState(false)
      loopEnabledRef.current = false
      return
    }
    const total = instance.maxDuration || 1
    const start = instance.getCurrentTime()
    const end = Math.min(total, start + Math.max(2, total * 0.15))
    overlay.set(start, end, total)
    loopRef.current = { start, end }
    setLoopRegionState({ start, end })
    setLoopEnabledState(true)
    loopEnabledRef.current = true
  }, [])

  const setLoopEnabled = useCallback((enabled) => {
    setLoopEnabledState(enabled)
    loopEnabledRef.current = enabled
  }, [])

  const setVuMetersEnabled = useCallback((enabled) => {
    setVuMetersEnabledState(enabled)
    vuMetersEnabledRef.current = enabled
    if (!enabled) {
      // zero the bars immediately instead of waiting for them to decay via
      // the (now-skipped) analyser reads
      meterElsRef.current.forEach((el) => el && (el.style.transform = 'scaleX(0)'))
    }
  }, [])

  // --- per-track controls ---
  const setTrackVolume = useCallback(
    (index, volume) => {
      setTracks((prev) => {
        const next = prev.map((t, i) => (i === index ? { ...t, volume } : t))
        applyEffectiveVolumes(next)
        return next
      })
    },
    [applyEffectiveVolumes],
  )

  const setTrackPan = useCallback((index, pan) => {
    setTracks((prev) => prev.map((t, i) => (i === index ? { ...t, pan } : t)))
    const nodes = audioNodesRef.current[index]
    if (nodes?.panner) nodes.panner.pan.value = pan
  }, [])

  const toggleMute = useCallback(
    (index) => {
      setTracks((prev) => {
        const next = prev.map((t, i) => (i === index ? { ...t, muted: !t.muted } : t))
        applyEffectiveVolumes(next)
        return next
      })
    },
    [applyEffectiveVolumes],
  )

  const toggleSolo = useCallback(
    (index) => {
      setTracks((prev) => {
        const next = prev.map((t, i) => (i === index ? { ...t, solo: !t.solo } : t))
        applyEffectiveVolumes(next)
        return next
      })
    },
    [applyEffectiveVolumes],
  )

  const toggleKaraoke = useCallback(() => {
    setTracks((prev) => {
      const currentlyMuted = prev.some((t) => KARAOKE_STEM_NAMES.has(t.name) && t.muted)
      const next = prev.map((t) => (KARAOKE_STEM_NAMES.has(t.name) ? { ...t, muted: !currentlyMuted } : t))
      applyEffectiveVolumes(next)
      return next
    })
  }, [applyEffectiveVolumes])

  // --- master volume / mute-all / reset-mix (PRD §2.3) ---
  const setMasterVolume = useCallback(
    (volume) => {
      masterVolumeRef.current = volume
      setMasterVolumeState(volume)
      applyEffectiveVolumes(tracksRef.current, volume, masterMutedRef.current)
    },
    [applyEffectiveVolumes],
  )

  const setMasterMuted = useCallback(
    (muted) => {
      masterMutedRef.current = muted
      setMasterMutedState(muted)
      applyEffectiveVolumes(tracksRef.current, masterVolumeRef.current, muted)
    },
    [applyEffectiveVolumes],
  )

  const toggleMasterMuted = useCallback(() => setMasterMuted(!masterMutedRef.current), [setMasterMuted])

  const resetMix = useCallback(() => {
    const resetTracks = stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false, pan: 0 }))
    setTracks(resetTracks)
    tracksRef.current = resetTracks
    audioNodesRef.current.forEach((nodes) => {
      if (nodes?.panner) nodes.panner.pan.value = 0
    })
    masterVolumeRef.current = 1
    masterMutedRef.current = false
    setMasterVolumeState(1)
    setMasterMutedState(false)
    applyEffectiveVolumes(resetTracks, 1, false)
  }, [stemNames, applyEffectiveVolumes])

  // --- mix presets (PRD §2.3: save/load levels+pan+mute state) ---
  const getMixState = useCallback(
    () => ({
      tracks: tracksRef.current.map(({ name, volume, pan, muted, solo }) => ({ name, volume, pan, muted, solo })),
      masterVolume: masterVolumeRef.current,
      masterMuted: masterMutedRef.current,
    }),
    [],
  )

  const applyMixState = useCallback(
    (mixState) => {
      if (!mixState) return
      const byName = new Map((mixState.tracks || []).map((t) => [t.name, t]))
      const nextTracks = stemNames.map((name) => {
        const saved = byName.get(name)
        return saved
          ? { name, volume: saved.volume ?? 1, pan: saved.pan ?? 0, muted: !!saved.muted, solo: !!saved.solo }
          : { name, volume: 1, pan: 0, muted: false, solo: false }
      })
      setTracks(nextTracks)
      tracksRef.current = nextTracks
      nextTracks.forEach((t, index) => {
        const nodes = audioNodesRef.current[index]
        if (nodes?.panner) nodes.panner.pan.value = t.pan
      })
      const mv = mixState.masterVolume ?? 1
      const mm = !!mixState.masterMuted
      masterVolumeRef.current = mv
      masterMutedRef.current = mm
      setMasterVolumeState(mv)
      setMasterMutedState(mm)
      applyEffectiveVolumes(nextTracks, mv, mm)
    },
    [stemNames, applyEffectiveVolumes],
  )

  // --- per-stem spectrogram toggle (PRD Addendum §2.2) ---
  //
  // See the module docstring for why this can't reuse the multitrack's own
  // per-track wavesurfer instances (their decoded data is the peaks
  // substitute, not real PCM) — this creates a separate, real decode
  // instead. The spectrogram is fit to its own full duration at toggle-on
  // time rather than pixel-locked to the live waveform's current
  // zoom/scroll: keeping a second WaveSurfer instance's zoom/scroll in
  // lockstep with the visible one on every zoomBy()/scroll event is real
  // added complexity for what's fundamentally an inspection view, not part
  // of the synced transport — clicking the spectrogram still seeks the
  // shared transport (below), so it stays usable without that sync.
  const toggleSpectrogram = useCallback(
    (index) => {
      const ws = instanceRef.current?.wavesurfers?.[index]
      const name = stemNames[index]
      if (!ws || !name) return
      setSpectrogramStems((prev) => {
        const next = new Set(prev)
        const turningOn = !next.has(index)
        if (turningOn) next.add(index)
        else next.delete(index)

        let entry = spectrogramNodesRef.current[index]
        if (!entry) {
          const visibleWrapper = ws.getWrapper() // the real track row -- overlay lives inside it so it scrolls/positions in lockstep
          const overlay = document.createElement('div')
          overlay.setAttribute('style', 'position:absolute; inset:0; z-index:4; display:none;')
          visibleWrapper.appendChild(overlay)

          // Invisible (1px) host: WaveSurfer.getWrapper().offsetWidth is
          // what SpectrogramPlugin sizes its render off of, so this needs
          // real layout dimensions -- display:none would report 0 -- but
          // nothing about it should be visually perceptible.
          const hostContainer = document.createElement('div')
          hostContainer.setAttribute('style', 'width:100%; height:1px; overflow:hidden;')
          const specContainer = document.createElement('div')
          specContainer.setAttribute('style', 'width:100%; height:64px;')
          overlay.appendChild(hostContainer)
          overlay.appendChild(specContainer)

          const decodeWs = WaveSurfer.create({
            container: hostContainer,
            height: 1,
            waveColor: 'transparent',
            progressColor: 'transparent',
            cursorWidth: 0,
            interact: false,
            url: stemUrl(jobId, name, 'mp3'), // no `peaks` -- a real decode, deliberately
          })
          const plugin = SpectrogramPlugin.create({ container: specContainer, height: 64, labels: false, colorMap: 'roseus' })
          decodeWs.registerPlugin(plugin)
          plugin.on('click', (relativeX) => seekFraction(relativeX))
          // Fit-to-width zoom once decoded -- read the always-visible real
          // row's width, not hostContainer's (which may be mid-toggle-off,
          // i.e. display:none / 0 width, if the user is fast).
          const unsubReady = decodeWs.once('ready', (duration) => {
            const fitPx = Math.max(1, (visibleWrapper.clientWidth || 1) / (duration || 1))
            decodeWs.zoom(fitPx)
          })

          entry = { overlay, decodeWs, unsubReady }
          spectrogramNodesRef.current[index] = entry
        }
        entry.overlay.style.display = turningOn ? 'block' : 'none'
        return next
      })
    },
    [jobId, stemNames, seekFraction],
  )

  const registerMeterEl = useCallback((index, el) => {
    meterElsRef.current[index] = el
  }, [])

  const subscribeScroll = useCallback((cb) => {
    scrollSubscribersRef.current.add(cb)
    return () => scrollSubscribersRef.current.delete(cb)
  }, [])

  const getScrollViewport = useCallback(() => {
    const el = scrollWrapperRef.current
    const instance = instanceRef.current
    if (!el || !instance || !instance.maxDuration) return { startFrac: 0, endFrac: 1 }
    const pxTotal = Math.max(1, el.scrollWidth)
    return {
      startFrac: el.scrollLeft / pxTotal,
      endFrac: (el.scrollLeft + el.clientWidth) / pxTotal,
    }
  }, [])

  const setScrollFraction = useCallback((frac) => {
    const el = scrollWrapperRef.current
    if (!el) return
    const pxTotal = Math.max(1, el.scrollWidth)
    el.scrollLeft = Math.max(0, frac * pxTotal - el.clientWidth / 2)
  }, [])

  const hasKaraokeTarget = useMemo(() => stemNames.some((name) => KARAOKE_STEM_NAMES.has(name)), [stemNames])

  return {
    containerRef,
    ready,
    playing,
    currentTime,
    duration,
    playbackRate,
    playbackRates: PLAYBACK_RATES,
    minPxPerSec,
    minZoom: MIN_PX_PER_SEC,
    maxZoom: MAX_PX_PER_SEC,
    loopRegion,
    loopEnabled,
    vuMetersEnabled,
    setVuMetersEnabled,
    tracks,
    focusedIndex,
    hasKaraokeTarget,
    hasOriginal,
    abMode,
    setAbMode,
    toggleAbMode,
    masterVolume,
    masterMuted,
    setMasterVolume,
    setMasterMuted,
    toggleMasterMuted,
    resetMix,
    getMixState,
    applyMixState,
    spectrogramStems,
    toggleSpectrogram,
    setFocusedIndex,
    togglePlay,
    seekFraction,
    seekTime,
    skip,
    jumpToStart,
    jumpToEnd,
    jumpToPercent,
    setPlaybackRate,
    cyclePlaybackRate,
    zoomBy,
    toggleLoopRegion,
    setLoopEnabled,
    setTrackVolume,
    setTrackPan,
    toggleMute,
    toggleSolo,
    toggleKaraoke,
    registerMeterEl,
    subscribeScroll,
    getScrollViewport,
    setScrollFraction,
    getReferencePeaks: () => referencePeaksRef.current,
  }
}

function applyPlaybackRate(instance, originalAudio, rate) {
  const elements = [...(instance?.audios ?? []), originalAudio].filter((el) => el instanceof HTMLMediaElement)
  elements.forEach((el) => {
    el.playbackRate = rate
    // Keep pitch constant across speeds — without this a 2x speed sounds
    // like a chipmunk, which reads as broken rather than "fast forward".
    el.preservesPitch = true
    el.mozPreservesPitch = true
    el.webkitPreservesPitch = true
  })
}

/** Vanilla-DOM A/B loop region: a translucent band plus two drag handles,
 * appended into the multitrack's own inner (position: relative) wrapper so
 * it scrolls and zooms in lock-step with the waveforms without going
 * through React at all — consistent with how the library manages that
 * whole subtree itself. `pxPerSecGetter` isn't needed: positions are
 * derived from the wrapper's own current width vs the instance's
 * maxDuration, recomputed on every drag frame, so it stays correct across
 * zoom changes without a separate subscription. */
function makeLoopOverlay(wrapperEl, onChange) {
  const band = document.createElement('div')
  band.setAttribute(
    'style',
    'position:absolute; top:0; bottom:0; z-index:5; display:none; ' +
      'background:rgba(129,140,248,0.18); border-left:2px solid #818cf8; border-right:2px solid #818cf8; cursor:grab;',
  )
  const startHandle = document.createElement('div')
  const endHandle = document.createElement('div')
  for (const h of [startHandle, endHandle]) {
    h.setAttribute(
      'style',
      'position:absolute; top:0; bottom:0; width:8px; cursor:ew-resize; background:transparent;',
    )
  }
  startHandle.style.left = '-4px'
  endHandle.style.right = '-4px'
  band.appendChild(startHandle)
  band.appendChild(endHandle)
  wrapperEl.appendChild(band)

  let start = 0
  let end = 0

  function widthPx() {
    return wrapperEl.scrollWidth || wrapperEl.clientWidth || 1
  }
  function totalSeconds() {
    // Encoded implicitly by the ratio already drawn — recovered from the
    // last set() call via closure below instead of re-deriving here.
    return lastTotal || 1
  }
  let lastTotal = 1

  function render() {
    const px = widthPx()
    const total = totalSeconds()
    band.style.left = `${(start / total) * px}px`
    band.style.width = `${Math.max(4, ((end - start) / total) * px)}px`
    band.style.display = 'block'
  }

  function set(newStart, newEnd, total = lastTotal) {
    lastTotal = total || lastTotal
    start = Math.max(0, newStart)
    end = Math.max(start + 0.1, newEnd)
    render()
    onChange({ start, end })
  }

  function clear() {
    band.style.display = 'none'
  }

  /** Re-derive pixel position from the current wrapper width — call after
   * any zoom change so an already-set region doesn't go stale relative to
   * the now-resized waveform underneath it. */
  function reflow() {
    if (band.style.display !== 'none') render()
  }

  function pxToSeconds(px) {
    return (px / widthPx()) * lastTotal
  }

  function bindDrag(el, mode) {
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault()
      e.stopPropagation()
      const startX = e.clientX
      const origStart = start
      const origEnd = end
      function onMove(ev) {
        const deltaPx = ev.clientX - startX
        const deltaSec = pxToSeconds(deltaPx)
        if (mode === 'move') set(origStart + deltaSec, origEnd + deltaSec)
        else if (mode === 'start') set(origStart + deltaSec, origEnd)
        else set(origStart, origEnd + deltaSec)
      }
      function onUp() {
        window.removeEventListener('pointermove', onMove)
        window.removeEventListener('pointerup', onUp)
      }
      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
    })
  }
  bindDrag(band, 'move')
  bindDrag(startHandle, 'start')
  bindDrag(endHandle, 'end')

  return {
    set: (s, e, total) => set(s, e, total),
    clear,
    reflow,
    destroy: () => band.remove(),
  }
}
