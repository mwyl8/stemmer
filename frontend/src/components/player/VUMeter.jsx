/** A live level meter driven imperatively — see useMultitrackPlayer's rAF
 * loop, which writes `style.transform` on the registered element directly,
 * bypassing React state entirely so ~60fps updates never trigger re-renders
 * up the tree. `registerRef` is `hook.registerMeterEl` bound to this row's
 * track index. */
export default function VUMeter({ registerRef, color }) {
  return (
    <div
      className="relative h-2 w-full overflow-hidden rounded-full bg-slate-800"
      role="meter"
      aria-label="Level"
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        ref={registerRef}
        className="h-full w-full origin-left rounded-full"
        style={{ backgroundColor: color, transform: 'scaleX(0)', transition: 'transform 60ms linear' }}
      />
    </div>
  )
}
