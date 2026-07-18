import { useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import ProgressStages from '../components/ProgressStages'
import ResultView from '../components/ResultView'
import useJobPolling from '../hooks/useJobPolling'
import { deleteJob } from '../api'

export default function JobPage() {
  const { jobId } = useParams()
  const location = useLocation()
  const { job, error, loading } = useJobPolling(jobId)
  const [deleted, setDeleted] = useState(false)

  const skipDownload = location.state?.source === 'file'

  async function handleDelete() {
    try {
      await deleteJob(jobId)
      setDeleted(true)
    } catch {
      // deleteJob failing here just means the next poll/manual refresh will
      // show the real state — nothing else useful to do client-side.
    }
  }

  if (loading) {
    return <CenteredMessage>Loading job…</CenteredMessage>
  }

  if (error) {
    return (
      <CenteredMessage>
        <p className="text-red-300">{error.status === 404 ? 'Job not found.' : error.message}</p>
        <Link to="/" className="mt-4 inline-block text-indigo-300 hover:underline">
          Start a new job
        </Link>
      </CenteredMessage>
    )
  }

  if (deleted || job.status === 'expired') {
    return (
      <CenteredMessage>
        <p>This job's stems have been deleted.</p>
        <Link to="/" className="mt-4 inline-block text-indigo-300 hover:underline">
          Start a new job
        </Link>
      </CenteredMessage>
    )
  }

  if (job.status === 'error') {
    return (
      <CenteredMessage>
        <p className="font-medium text-slate-100">Separation failed</p>
        <p className="mt-2 max-w-lg text-sm text-red-300">{job.error}</p>
        <Link to="/" className="mt-4 inline-block text-indigo-300 hover:underline">
          Start a new job
        </Link>
      </CenteredMessage>
    )
  }

  if (job.status === 'done') {
    return <ResultView job={job} onDelete={handleDelete} />
  }

  // queued | running
  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6 px-4 py-24">
      <div>
        <h1 className="text-xl font-medium text-slate-100">Separating your {job.mode} job…</h1>
        <p className="mt-1 text-sm text-slate-400">
          {job.tier} tier — this page updates on its own, no need to refresh.
        </p>
      </div>
      <ProgressStages stage={job.stage} progress={job.progress} skipDownload={skipDownload} />
    </div>
  )
}

function CenteredMessage({ children }) {
  return <div className="mx-auto flex max-w-xl flex-col items-center gap-2 px-4 py-24 text-center text-slate-300">{children}</div>
}
