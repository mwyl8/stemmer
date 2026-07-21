// Saved mix presets (PRD Addendum §2.3: levels/pan/mute state). Scoped by
// the exact set of stem names involved (sorted, joined) rather than by job
// id, so a preset made on one music-mode song is reusable on any other
// music-mode song -- the whole point of a *preset* is applying it beyond
// the one job it was made on. Client-side only, same reasoning as
// recentJobs.js: no user/session model exists to hang server-side presets
// off of.

const KEY_PREFIX = 'stemmer.mixPreset.'

function presetsKey(stemNames) {
  return KEY_PREFIX + [...stemNames].sort().join(',')
}

function readAll(stemNames) {
  try {
    const raw = localStorage.getItem(presetsKey(stemNames))
    const parsed = raw ? JSON.parse(raw) : {}
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function writeAll(stemNames, presets) {
  try {
    localStorage.setItem(presetsKey(stemNames), JSON.stringify(presets))
  } catch {
    // best-effort -- a failed write just means this preset won't be there next time
  }
}

/** {[name]: {tracks: [{name, volume, pan, muted, solo}], masterVolume, masterMuted}} */
export function listMixPresets(stemNames) {
  return readAll(stemNames)
}

export function saveMixPreset(stemNames, name, mixState) {
  const presets = readAll(stemNames)
  presets[name] = mixState
  writeAll(stemNames, presets)
}

export function deleteMixPreset(stemNames, name) {
  const presets = readAll(stemNames)
  delete presets[name]
  writeAll(stemNames, presets)
}
