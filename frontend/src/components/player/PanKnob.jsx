/** -1 (hard left) .. 0 (center) .. 1 (hard right). A plain range input reads
 * fine here — pan is a coarse, occasional adjustment, not something that
 * benefits from a custom knob widget — but centered visually with a tick. */
export default function PanKnob({ pan, onChange, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wide text-slate-500">Pan</span>
      <input
        type="range"
        min={-100}
        max={100}
        value={Math.round(pan * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        aria-label={label ? `${label} pan` : 'Pan'}
        className="focus-ring h-1.5 w-16 cursor-pointer appearance-none rounded-full bg-slate-800 accent-slate-300"
      />
    </div>
  )
}
