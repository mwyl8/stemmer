import { useEffect, useRef, useState } from 'react'
import { formatDurationWords } from '../../lib/format'

/** Ticks locally between polls (the backend is only polled every ~1.2s —
 * see useJobPolling) so the "elapsed" readout looks live rather than
 * stepping visibly. Re-baselines from the server's own elapsed_seconds on
 * every poll response, so it can never drift far even across many ticks. */
export default function ElapsedTimer({ elapsedSeconds, running }) {
  const [display, setDisplay] = useState(elapsedSeconds)
  const baseRef = useRef({ value: elapsedSeconds, at: performance.now() })

  useEffect(() => {
    baseRef.current = { value: elapsedSeconds, at: performance.now() }
    setDisplay(elapsedSeconds)
  }, [elapsedSeconds])

  useEffect(() => {
    if (!running) return undefined
    const id = setInterval(() => {
      const { value, at } = baseRef.current
      setDisplay(value + (performance.now() - at) / 1000)
    }, 250)
    return () => clearInterval(id)
  }, [running])

  return <span className="tabular-nums">{formatDurationWords(display)}</span>
}
