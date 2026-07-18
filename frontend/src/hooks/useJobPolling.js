import { useEffect, useRef, useState } from 'react'
import { ApiError, TERMINAL_STATUSES, getJob } from '../api'

const POLL_INTERVAL_MS = 1200

/** Polls GET /jobs/{id} until the job reaches a terminal status
 * (done/error/expired), or the request 404s (job id doesn't exist). */
export default function useJobPolling(jobId) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const timerRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    setJob(null)
    setError(null)
    setLoading(true)

    async function poll() {
      try {
        const nextJob = await getJob(jobId)
        if (cancelled) return
        setJob(nextJob)
        setLoading(false)
        if (!TERMINAL_STATUSES.has(nextJob.status)) {
          timerRef.current = setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (err) {
        if (cancelled) return
        setError(err instanceof ApiError ? err : new ApiError(0, 'Lost connection to the server.'))
        setLoading(false)
      }
    }

    poll()
    return () => {
      cancelled = true
      clearTimeout(timerRef.current)
    }
  }, [jobId])

  return { job, error, loading }
}
