import { useEffect, useRef, useState } from 'react'

/** A lightweight, self-drawn overview strip — wavesurfer.js's own Minimap
 * plugin is built for a single WaveSurfer instance and doesn't know about
 * wavesurfer-multitrack's synced tracks, so this reads one reference
 * track's decoded peaks (via useMultitrackPlayer.getReferencePeaks) and
 * draws its own: a static waveform silhouette, a viewport box you can drag
 * to pan the real (zoomed) view, and a click-anywhere-else to jump
 * playback there. */
export default function Minimap({ player }) {
  const canvasRef = useRef(null)
  const trackRef = useRef(null)
  const [viewport, setViewport] = useState({ startFrac: 0, endFrac: 1 })
  const [drawn, setDrawn] = useState(false)

  // Peaks arrive asynchronously (decoded after the 'ready' event on the
  // reference track) — poll briefly rather than plumbing a dedicated event
  // through the hook for what's a one-shot, non-critical draw.
  useEffect(() => {
    if (!player.ready) return
    let cancelled = false
    let attempts = 0
    function tryDraw() {
      if (cancelled) return
      const peaks = player.getReferencePeaks()
      const canvas = canvasRef.current
      if (peaks && canvas) {
        drawPeaks(canvas, peaks)
        setDrawn(true)
        return
      }
      if (attempts++ < 25) setTimeout(tryDraw, 200)
    }
    tryDraw()
    return () => {
      cancelled = true
    }
  }, [player, player.ready])

  useEffect(() => {
    const update = () => setViewport(player.getScrollViewport())
    update()
    const unsub = player.subscribeScroll(update)
    return unsub
  }, [player, player.minPxPerSec, player.ready])

  function handleTrackClick(event) {
    const rect = trackRef.current.getBoundingClientRect()
    const frac = (event.clientX - rect.left) / rect.width
    player.seekFraction(frac)
  }

  function handleViewportPointerDown(event) {
    event.stopPropagation()
    event.preventDefault()
    function onMove(moveEvent) {
      const rect = trackRef.current.getBoundingClientRect()
      const frac = (moveEvent.clientX - rect.left) / rect.width
      player.setScrollFraction(frac)
    }
    function onUp() {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const playFrac = player.duration > 0 ? player.currentTime / player.duration : 0

  return (
    <div
      ref={trackRef}
      onClick={handleTrackClick}
      className="relative h-10 w-full cursor-pointer overflow-hidden rounded-lg border border-slate-800 bg-slate-950"
      role="slider"
      aria-label="Track overview — click to jump, drag the highlighted box to pan"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(playFrac * 100)}
    >
      <canvas ref={canvasRef} width={1200} height={40} className="h-full w-full" />
      {!drawn && <div className="skeleton absolute inset-0" />}
      <div className="pointer-events-none absolute top-0 h-full w-px bg-slate-100" style={{ left: `${playFrac * 100}%` }} />
      <div
        onPointerDown={handleViewportPointerDown}
        className="absolute top-0 h-full cursor-grab rounded border border-indigo-400/70 bg-indigo-400/10 active:cursor-grabbing"
        style={{
          left: `${viewport.startFrac * 100}%`,
          width: `${Math.max(1, (viewport.endFrac - viewport.startFrac) * 100)}%`,
        }}
      />
    </div>
  )
}

function drawPeaks(canvas, peaks) {
  const ctx = canvas.getContext('2d')
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)
  const buckets = Math.min(width, 400)
  const samplesPerBucket = Math.max(1, Math.floor(peaks.length / buckets))
  ctx.fillStyle = '#475569'
  const mid = height / 2
  for (let b = 0; b < buckets; b++) {
    let max = 0
    const start = b * samplesPerBucket
    for (let i = start; i < start + samplesPerBucket && i < peaks.length; i++) {
      const v = Math.abs(peaks[i])
      if (v > max) max = v
    }
    const barHeight = Math.max(1, max * height)
    const x = (b / buckets) * width
    ctx.fillRect(x, mid - barHeight / 2, Math.max(1, width / buckets), barHeight)
  }
}
