import { useState } from 'react'
import { exportMix } from '../../api'
import { listMixPresets, saveMixPreset } from '../../lib/mixPresets'

/** Mixing & comparison (PRD Addendum §2.3): master volume/mute-all/reset,
 * the A/B-against-the-original toggle, saved mix presets, and exporting the
 * current mix as a downloadable wav. Everything here reads/writes through
 * `player` (useMultitrackPlayer) — this component owns no audio state of
 * its own except the preset name being typed and which preset is selected. */
export default function MixToolbar({ jobId, player, stemNames }) {
  const [presetName, setPresetName] = useState('')
  const [selectedPreset, setSelectedPreset] = useState('')
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState(null)

  const presets = listMixPresets(stemNames)
  const presetNames = Object.keys(presets)

  function handleSavePreset() {
    const name = presetName.trim()
    if (!name) return
    saveMixPreset(stemNames, name, player.getMixState())
    setPresetName('')
  }

  function handleLoadPreset(name) {
    setSelectedPreset(name)
    if (!name) return
    player.applyMixState(presets[name])
  }

  async function handleExport() {
    setExporting(true)
    setExportError(null)
    try {
      const mixState = player.getMixState()
      const blob = await exportMix(jobId, mixState)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${jobId}-mix.wav`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setExportError(err.message || 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Master
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(player.masterVolume * 100)}
            onChange={(e) => player.setMasterVolume(Number(e.target.value) / 100)}
            aria-label="Master volume"
            className="focus-ring h-1.5 w-28 cursor-pointer appearance-none rounded-full bg-slate-800 accent-indigo-400"
          />
          <span className="w-9 text-right text-xs tabular-nums text-slate-400">{Math.round(player.masterVolume * 100)}%</span>
        </label>

        <button
          type="button"
          onClick={player.toggleMasterMuted}
          aria-pressed={player.masterMuted}
          className={`focus-ring rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
            player.masterMuted ? 'bg-red-500 text-white' : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
          }`}
        >
          {player.masterMuted ? 'Unmute all' : 'Mute all'}
        </button>

        <button
          type="button"
          onClick={player.resetMix}
          className="focus-ring rounded-lg bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 transition-colors hover:bg-slate-700"
        >
          Reset mix
        </button>

        {player.hasOriginal && (
          <button
            type="button"
            onClick={player.toggleAbMode}
            aria-pressed={player.abMode === 'original'}
            title="A/B against the original source mixture"
            className={`focus-ring rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              player.abMode === 'original' ? 'bg-amber-400 text-slate-900' : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
            }`}
          >
            {player.abMode === 'original' ? 'Hearing: Original' : 'Hearing: Stems'}
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-slate-800 pt-3">
        <input
          type="text"
          value={presetName}
          onChange={(e) => setPresetName(e.target.value)}
          placeholder="Preset name"
          aria-label="New mix preset name"
          className="focus-ring w-36 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200 placeholder:text-slate-500"
        />
        <button
          type="button"
          onClick={handleSavePreset}
          disabled={!presetName.trim()}
          className="focus-ring rounded-lg bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Save mix preset
        </button>

        {presetNames.length > 0 && (
          <select
            value={selectedPreset}
            onChange={(e) => handleLoadPreset(e.target.value)}
            aria-label="Load a saved mix preset"
            className="focus-ring rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200"
          >
            <option value="">Load preset…</option>
            {presetNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        )}

        <div className="flex-1" />

        <button
          type="button"
          onClick={handleExport}
          disabled={exporting}
          className="focus-ring rounded-lg bg-indigo-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {exporting ? 'Exporting…' : 'Export custom mix (wav)'}
        </button>
      </div>
      {exportError && (
        <p role="alert" className="text-xs text-red-300">
          {exportError}
        </p>
      )}
    </div>
  )
}
