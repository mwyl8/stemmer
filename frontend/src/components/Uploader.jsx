import { useRef, useState } from 'react'

/** Drag-drop (or click-to-browse) file picker. Purely presentational — the
 * parent owns the actual File value and passes it back down as `file`. */
export default function Uploader({ file, onFileChange, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) onFileChange(dropped)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
        disabled
          ? 'cursor-not-allowed border-slate-800 opacity-50'
          : dragging
            ? 'cursor-pointer border-indigo-400 bg-indigo-500/10'
            : 'cursor-pointer border-slate-700 hover:border-slate-500'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*,.mp3,.wav,.mp4,.m4a"
        className="hidden"
        disabled={disabled}
        onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <>
          <p className="font-medium text-slate-100">{file.name}</p>
          <p className="text-sm text-slate-400">{(file.size / (1024 * 1024)).toFixed(1)} MB — click or drop to replace</p>
        </>
      ) : (
        <>
          <p className="font-medium text-slate-200">Drag & drop an audio or video file</p>
          <p className="text-sm text-slate-400">or click to browse — mp3, wav, mp4, m4a</p>
        </>
      )}
    </div>
  )
}
