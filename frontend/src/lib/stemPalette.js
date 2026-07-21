// Consistent stem -> color mapping across the whole app (waveforms, VU
// meters, color swatches). Keyed by the exact stem names the backend router
// produces (backend/separators/router.py, chained_sep.py, bandit_sep.py):
// music mode -> vocals/drums/bass/other(/guitar/piano for 6-stem); video
// mode -> speech/music/effects; full mode -> speech/vocals/drums/bass/other.
// A name not in this table (future stems) falls back to a stable cycle so
// the app never breaks, it just runs out of hand-picked hues.
const KNOWN_STEM_COLORS = {
  vocals: { wave: '#818cf8', progress: '#4338ca', solid: '#818cf8' }, // indigo
  speech: { wave: '#818cf8', progress: '#4338ca', solid: '#818cf8' }, // same role as vocals across modes
  drums: { wave: '#f472b6', progress: '#be185d', solid: '#f472b6' }, // pink
  bass: { wave: '#34d399', progress: '#047857', solid: '#34d399' }, // emerald
  other: { wave: '#94a3b8', progress: '#475569', solid: '#94a3b8' }, // slate
  music: { wave: '#94a3b8', progress: '#475569', solid: '#94a3b8' }, // Bandit's raw "music" stem, pre-Demucs
  effects: { wave: '#fbbf24', progress: '#b45309', solid: '#fbbf24' }, // amber
  guitar: { wave: '#60a5fa', progress: '#1d4ed8', solid: '#60a5fa' }, // blue
  piano: { wave: '#c084fc', progress: '#7e22ce', solid: '#c084fc' }, // purple
}

const FALLBACK_CYCLE = [
  { wave: '#2dd4bf', progress: '#0f766e', solid: '#2dd4bf' },
  { wave: '#fb923c', progress: '#c2410c', solid: '#fb923c' },
  { wave: '#a3e635', progress: '#4d7c0f', solid: '#a3e635' },
]

/** Returns {wave, progress, solid} — stable for a given name across renders
 * and across different stem lists (doesn't depend on array position, unlike
 * the old index-based palette this replaces). */
export function stemColor(name) {
  if (KNOWN_STEM_COLORS[name]) return KNOWN_STEM_COLORS[name]
  // Deterministic fallback so an unrecognized name still gets a *consistent*
  // color across the session, just not a hand-picked one.
  let hash = 0
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  return FALLBACK_CYCLE[hash % FALLBACK_CYCLE.length]
}
