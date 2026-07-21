/** "0:07", "1:23", "12:04:05" — matches the precision a transport readout
 * needs; hours only appear if the clip is actually that long. */
export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const total = Math.floor(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m)
  const ss = String(s).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

/** "44s", "1m 12s" — coarser than formatTime, for elapsed/ETA/summary text
 * where sub-second precision would just be noise. */
export function formatDurationWords(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0s'
  const total = Math.round(seconds)
  const m = Math.floor(total / 60)
  const s = total % 60
  if (m === 0) return `${s}s`
  return `${m}m ${s}s`
}

/** "320 kbps" from a bits/sec bitrate — null-safe for sources ffprobe
 * couldn't read (e.g. a codec with no fixed bitrate field). */
export function formatBitrate(bitsPerSecond) {
  if (!Number.isFinite(bitsPerSecond) || bitsPerSecond <= 0) return null
  return `${Math.round(bitsPerSecond / 1000)} kbps`
}

/** "44.1 kHz" from a sample rate in Hz. */
export function formatSampleRate(hz) {
  if (!Number.isFinite(hz) || hz <= 0) return null
  return `${(hz / 1000).toFixed(1)} kHz`
}
