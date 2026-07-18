/** Paste a public YouTube/TikTok/Instagram URL. Client-side validation is
 * intentionally light (non-empty + looks like a URL) — the backend is the
 * source of truth on host allowlists/private-IP rejection (fetch.py), so we
 * don't duplicate that logic here; a bad URL just surfaces the backend's
 * 400 error message after submit. */
export default function LinkInput({ url, onUrlChange, disabled }) {
  return (
    <div className="flex flex-col gap-1">
      <input
        type="url"
        inputMode="url"
        placeholder="https://www.youtube.com/watch?v=..."
        value={url}
        disabled={disabled}
        onChange={(e) => onUrlChange(e.target.value)}
        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
      />
      <p className="text-xs text-slate-500">YouTube, TikTok, or Instagram — public links only.</p>
    </div>
  )
}
