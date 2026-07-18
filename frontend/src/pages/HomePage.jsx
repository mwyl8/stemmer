import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Uploader from '../components/Uploader'
import LinkInput from '../components/LinkInput'
import ModeTierPicker from '../components/ModeTierPicker'
import { ApiError, createJobFromFile, createJobFromUrl } from '../api'

export default function HomePage() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')
  const [mode, setMode] = useState('music')
  const [tier, setTier] = useState('fast')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const hasFile = Boolean(file)
  const hasUrl = url.trim().length > 0
  const canSubmit = (hasFile || hasUrl) && !(hasFile && hasUrl) && !submitting

  function handleFileChange(nextFile) {
    setFile(nextFile)
    if (nextFile) setUrl('') // file and link are mutually exclusive
  }

  function handleUrlChange(nextUrl) {
    setUrl(nextUrl)
    if (nextUrl) setFile(null)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    setError(null)
    try {
      const { job_id } = hasFile
        ? await createJobFromFile(file, { mode, tier })
        : await createJobFromUrl(url.trim(), { mode, tier })
      navigate(`/jobs/${job_id}`, { state: { source: hasFile ? 'file' : 'url' } })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong submitting the job.')
      setSubmitting(false)
    }
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-8 px-4 py-16">
      <div>
        <h1 className="text-3xl font-semibold text-slate-100">Stemmer</h1>
        <p className="mt-1 text-slate-400">Split a song or video into stems — vocals, drums, bass, speech, and more.</p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <Uploader file={file} onFileChange={handleFileChange} disabled={submitting} />

        <div className="flex items-center gap-3 text-sm text-slate-500">
          <div className="h-px flex-1 bg-slate-800" />
          or paste a link
          <div className="h-px flex-1 bg-slate-800" />
        </div>

        <LinkInput url={url} onUrlChange={handleUrlChange} disabled={submitting} />

        <ModeTierPicker mode={mode} onModeChange={setMode} tier={tier} onTierChange={setTier} disabled={submitting} />

        {error && (
          <p role="alert" className="rounded-lg border border-red-800 bg-red-950/50 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-lg bg-indigo-500 px-4 py-3 font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          {submitting ? 'Submitting…' : 'Separate'}
        </button>
      </form>
    </div>
  )
}
