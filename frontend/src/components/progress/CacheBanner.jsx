/** "returned instantly from cache" (PRD §2.4) — content-hash cache hits skip
 * straight to done server-side, so a 0->100% bar would be actively
 * misleading here; this replaces it entirely rather than flashing one. */
export default function CacheBanner() {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-emerald-800 bg-emerald-950/40 px-4 py-3 text-sm text-emerald-300">
      <span aria-hidden="true">⚡</span>
      Returned instantly from cache — this exact audio was already separated at this mode/tier.
    </div>
  )
}
