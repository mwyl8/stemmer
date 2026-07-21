import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Multitrack from 'wavesurfer-multitrack'
import { stemColor } from '../lib/stemPalette'
import { KARAOKE_STEM_NAMES, stemUrl } from '../api'

const MIN_PX_PER_SEC = 15
const MAX_PX_PER_SEC = 800
const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2]

/**
 * Owns one wavesurfer-multitrack instance and everything layered on top of
 * its public API: transport, per-track volume/mute/solo (native to the
 * library), plus three things it doesn't provide that PRD §2.1/§2.2 ask for:
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
 *
 * Playback itself never bypasses the library's public transport
 * (play/pause/seekTo/setTime) — only pan/VU touch the per-track audio
 * graph, and only downstream of the library's own gain — so every stem
 * stays sample-synced through skip/seek/zoom/loop/speed exactly as the
 * library's own drift-correcting sync loop (`startSync`) already
 * guarantees.
 */
export default function useMultitrackPlayer(jobId, stemNames) {
  const containerRef = useRef(null)
  const instanceRef = useRef(null)
  const audioNodesRef = useRef([]) // [{source, panner, analyser} | null] per stem index
  const meterElsRef = useRef([]) // [HTMLElement | null] per stem index, written to directly (no React state)
  const rafRef = useRef(null)
  const loopRef = useRef(null) // {start, end} in seconds, or null
  const loopEnabledRef = useRef(false)
  const loopOverlayRef = useRef(null) // vanilla-DOM controller, see makeLoopOverlay()
  const scrollWrapperRef = useRef(null) // the library's own inner `overflow-x: scroll` div
  const scrollCleanupRef = useRef(null)
  const scrollSubscribersRef = useRef(new Set())
  const referencePeaksRef = useRef(null)

  const [ready, setReady] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playbackRate, setPlaybackRateState] = useState(1)
  const [minPxPerSec, setMinPxPerSec] = useState(MIN_PX_PER_SEC)
  const [loopRegion, setLoopRegionState] = useState(null)
  const [loopEnabled, setLoopEnabledState] = useState(false)
  const [focusedIndex, setFocusedIndex] = useState(stemNames.length > 0 ? 0 : null)
  const [tracks, setTracks] = useState(() => stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false, pan: 0 })))

  const applyEffectiveVolumes = useCallback((nextTracks) => {
    const instance = instanceRef.current
    if (!instance) return
    const anySolo = nextTracks.some((t) => t.solo)
    nextTracks.forEach((t, index) => {
      const effective = anySolo ? (t.solo ? t.volume : 0) : t.muted ? 0 : t.volume
      instance.setTrackVolume(index, effective)
    })
  }, [])

  // --- mount: build the Multitrack instance + the pan/VU tap + loop overlay ---
  useEffect(() => {
    if (!containerRef.current || stemNames.length === 0) return

    setReady(false)
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setLoopRegionState(null)
    setLoopEnabledState(false)
    loopRef.current = null
    loopEnabledRef.current = false
    referencePeaksRef.current = null
    const initialTracks = stemNames.map((name) => ({ name, volume: 1, muted: false, solo: false, pan: 0 }))
    setTracks(initialTracks)
    setFocusedIndex(0)

    const instance = Multitrack.create(
      stemNames.map((name, index) => {
        const colors = stemColor(name)
        return {
          id: index,
          url: stemUrl(jobId, name, 'mp3'),
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
    instanceRef.current = instance

    const unsubCanplay = instance.once('canplay', () => {
      const total = instance.maxDuration || 0
      setDuration(total)
      setReady(true)

      // Fit the initial zoom to the container width so a short clip doesn't
      // open zoomed-out to nothing, and a long one doesn't open zoomed in
      // past the first few seconds.
      const containerWidth = containerRef.current?.clientWidth || 800
      const fitPx = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, containerWidth / (total || 1)))
      instance.zoom(fitPx)
      setMinPxPerSec(fitPx)

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
    // enforcement, and direct (non-React) VU meter bar updates.
    let lastStatePush = 0
    const meterBuf = new Uint8Array(128)
    function frame(now) {
      const t = instance.getCurrentTime?.() ?? 0
      const isPlaying = instance.isPlaying?.() ?? false

      const region = loopRef.current
      if (loopEnabledRef.current && region && isPlaying && t >= region.end) {
        instance.setTime(region.start)
      }

      if (now - lastStatePush > 80) {
        lastStatePush = now
        setCurrentTime(t)
        setPlaying(isPlaying)
      }

      audioNodesRef.current.forEach((nodes, index) => {
        const el = meterElsRef.current[index]
        if (!el) return
        let level = 0
        if (nodes?.analyser) {
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

      rafRef.current = requestAnimationFrame(frame)
    }
    rafRef.current = requestAnimationFrame(frame)

    return () => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, stemNames])

  // --- transport ---
  const togglePlay = useCallback(() => {
    const instance = instanceRef.current
    if (!instance) return
    if (instance.isPlaying()) instance.pause()
    else instance.play()
    setPlaying(instance.isPlaying())
  }, [])

  const seekFraction = useCallback((fraction) => {
    instanceRef.current?.seekTo(Math.min(1, Math.max(0, fraction)))
  }, [])

  const seekTime = useCallback((seconds) => {
    const instance = instanceRef.current
    if (!instance) return
    const clamped = Math.min(instance.maxDuration || 0, Math.max(0, seconds))
    instance.setTime(clamped)
    setCurrentTime(clamped)
  }, [])

  const skip = useCallback(
    (deltaSeconds) => {
      const instance = instanceRef.current
      if (!instance) return
      seekTime(instance.getCurrentTime() + deltaSeconds)
    },
    [seekTime],
  )

  const jumpToStart = useCallback(() => seekTime(0), [seekTime])
  const jumpToEnd = useCallback(() => {
    const instance = instanceRef.current
    if (!instance) return
    seekTime(Math.max(0, (instance.maxDuration || 0) - 0.05))
  }, [seekTime])
  const jumpToPercent = useCallback((pct) => seekFraction(pct / 100), [seekFraction])

  const cyclePlaybackRate = useCallback((direction) => {
    setPlaybackRateState((prev) => {
      const i = PLAYBACK_RATES.indexOf(prev)
      const next = PLAYBACK_RATES[Math.min(PLAYBACK_RATES.length - 1, Math.max(0, i + direction))]
      applyPlaybackRate(instanceRef.current, next)
      return next
    })
  }, [])

  const setPlaybackRate = useCallback((rate) => {
    setPlaybackRateState(rate)
    applyPlaybackRate(instanceRef.current, rate)
  }, [])

  const zoomBy = useCallback((factor) => {
    setMinPxPerSec((prev) => {
      const next = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, prev * factor))
      instanceRef.current?.zoom(next)
      loopOverlayRef.current?.reflow()
      return next
    })
  }, [])

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
    tracks,
    focusedIndex,
    hasKaraokeTarget,
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

function applyPlaybackRate(instance, rate) {
  if (!instance) return
  instance.audios?.forEach((el) => {
    if (!(el instanceof HTMLMediaElement)) return
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
