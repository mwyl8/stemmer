import { downloadAllUrl, stemUrl } from '../api'

/** Per-stem wav/mp3 links plus a "download all" zip (wav-only, per
 * GET /jobs/{id}/download's contract). Plain <a> tags — the browser handles
 * the download, no fetch/blob juggling needed. */
export default function DownloadBar({ jobId, stemNames }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-200">Downloads</p>
        <a
          href={downloadAllUrl(jobId)}
          className="rounded-lg bg-indigo-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-400"
        >
          Download all (zip)
        </a>
      </div>
      <div className="flex flex-col gap-1.5">
        {stemNames.map((name) => (
          <div key={name} className="flex items-center justify-between text-sm">
            <span className="capitalize text-slate-300">{name}</span>
            <div className="flex gap-3">
              <a href={stemUrl(jobId, name, 'wav')} download className="text-indigo-300 hover:underline">
                wav
              </a>
              <a href={stemUrl(jobId, name, 'mp3')} download className="text-indigo-300 hover:underline">
                mp3
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
