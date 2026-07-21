import { test, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Both bugs these tests guard against shipped because nothing exercised the
// *rendered* UI: the peaks-precompute change (server-side waveform peaks)
// and the spectrogram toggle were each individually correct in isolation,
// but the spectrogram plugin turned out to need real decoded PCM -- not the
// downsampled peaks array -- and only threw once actually clicked in a
// browser. The shareable-link bug was a route collision between the
// frontend's client-side router and the backend's own /jobs/{id} API that
// only manifests on a *fresh* page load (a pasted link, a new tab), which a
// SPA-internal navigation never exercises. Backend pytest and a frontend
// build/lint pass don't catch either class of bug -- this is the ONNX
// oracle diff-check's equivalent for the browser-rendered path.
//
// Note for whoever touches this file next: wavesurfer.js renders every
// waveform/spectrogram canvas inside a shadow root. `document.querySelectorAll
// ('canvas')` cannot see into it -- only Minimap.jsx's own light-DOM canvas
// shows up that way. `countNonBlankCanvases` below walks `shadowRoot`
// explicitly; don't swap it back to a plain querySelectorAll, or every
// assertion here silently stops observing anything wavesurfer renders.

const e2eDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(e2eDir, '../..')
const FIXTURE_MP3 = path.join(repoRoot, 'data', 'test.mp3')

async function submitJob(request, { seed = '' } = {}) {
  const res = await request.post('/jobs', {
    multipart: {
      file: {
        name: `test${seed}.mp3`,
        mimeType: 'audio/mpeg',
        buffer: readFileSync(FIXTURE_MP3),
      },
      mode: 'music',
      tier: 'fast',
    },
  })
  expect(res.ok(), `POST /jobs failed: ${res.status()} ${await res.text()}`).toBeTruthy()
  return (await res.json()).job_id
}

async function waitForJobDone(request, jobId, timeoutMs = 150_000) {
  const deadline = Date.now() + timeoutMs
  let last
  while (Date.now() < deadline) {
    const res = await request.get(`/jobs/${jobId}`)
    last = await res.json()
    if (last.status === 'done' || last.status === 'error') return last
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  throw new Error(`job ${jobId} did not reach a terminal status in time (last: ${JSON.stringify(last)})`)
}

/** Counts canvases with real, non-blank pixel content, walking through
 * shadow roots -- wavesurfer.js's own renderer attaches its wrapper (and
 * every per-track waveform canvas) inside a shadow root, so a plain
 * `document.querySelectorAll('canvas')` only ever finds canvases that live
 * directly in the light DOM (e.g. Minimap.jsx's own canvas) and silently
 * misses everything wavesurfer renders -- including the spectrogram
 * overlay's canvas, since it's appended as a descendant of a wavesurfer
 * track's shadow-rooted wrapper. That blind spot cost real debugging time
 * here: a passing "canvas count increased" check needs `shadowRoot` walked
 * explicitly, or it silently rots into a check that can never observe
 * wavesurfer content at all. */
async function countNonBlankCanvases(page) {
  return page.evaluate(() => {
    const canvases = []
    function collect(root) {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT)
      let node = walker.currentNode
      do {
        if (node.tagName === 'CANVAS') canvases.push(node)
        if (node.shadowRoot) collect(node.shadowRoot)
      } while ((node = walker.nextNode()))
    }
    collect(document.body)

    let nonBlank = 0
    for (const canvas of canvases) {
      if (canvas.width < 2 || canvas.height < 2) continue
      const ctx = canvas.getContext('2d')
      if (!ctx) continue
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
      let sum = 0
      for (let i = 0; i < data.length; i += 4) sum += data[i] + data[i + 1] + data[i + 2] + data[i + 3]
      if (sum > 0) nonBlank++
    }
    return nonBlank
  })
}

test('spectrogram toggle renders real content with no console errors', async ({ page, request }) => {
  const jobId = await submitJob(request)
  const job = await waitForJobDone(request, jobId)
  expect(job.status, `job errored: ${job.error}`).toBe('done')

  const consoleErrors = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => consoleErrors.push(err.message))

  await page.goto(`/results/${jobId}`)
  await expect(page.getByRole('heading', { name: 'Stems ready' })).toBeVisible()

  const firstRow = page.getByRole('group', { name: /stem controls/i }).first()
  const spectrogramButton = firstRow.getByRole('button', { name: /toggle spectrogram/i })
  await expect(spectrogramButton).toBeVisible()

  // The button renders as soon as the job data loads, well before the
  // multitrack player itself finishes its own async setup (peaks fetch +
  // Multitrack.create) -- clicking before that instance exists is a silent
  // no-op (toggleSpectrogram bails when instanceRef.current has no
  // wavesurfer yet), which reads as "the click did nothing" rather than a
  // failure. Wait for the player's own waveforms to actually render first,
  // the same readiness bar a real user's eyes would use.
  await expect
    .poll(() => countNonBlankCanvases(page), { timeout: 30_000, message: 'expected the player to finish loading and render its waveforms' })
    .toBeGreaterThan(0)

  const baselineNonBlank = await countNonBlankCanvases(page)

  await spectrogramButton.click()
  await expect(spectrogramButton).toHaveAttribute('aria-pressed', 'true')

  // Real decode + FFT takes a moment -- poll rather than a fixed sleep.
  await expect
    .poll(() => countNonBlankCanvases(page), {
      timeout: 30_000,
      message: 'expected a newly-created, actually-rendered (non-blank) spectrogram canvas',
    })
    .toBeGreaterThan(baselineNonBlank)

  expect(consoleErrors, `unexpected browser console/page errors:\n${consoleErrors.join('\n')}`).toEqual([])

  await request.delete(`/jobs/${jobId}`)
})

test('shareable link renders the result page on a fresh load with no localStorage', async ({ browser, request }) => {
  const jobId = await submitJob(request, { seed: '-share' })
  const job = await waitForJobDone(request, jobId)
  expect(job.status, `job errored: ${job.error}`).toBe('done')

  // A private/incognito-equivalent context: no storage carried over from
  // any other test or session, and localStorage explicitly disabled --
  // recentJobs.js/mixPresets.js must degrade to "no history" rather than
  // throwing, and the page itself must not depend on it to render at all.
  const context = await browser.newContext({ storageState: undefined })
  await context.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      get() {
        throw new DOMException('localStorage disabled for this test', 'SecurityError')
      },
    })
  })
  const page = await context.newPage()
  const consoleErrors = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => consoleErrors.push(err.message))

  // The actual regression: a fresh top-level navigation (what opening a
  // pasted link does), not an in-app client-side transition. Before the
  // /results rename this hit the /jobs/{id} route, which the dev proxy (and
  // any prod reverse-proxy following the same "API prefix -> backend, else
  // -> SPA" convention) forwards straight to the backend, returning raw
  // JSON instead of the SPA shell.
  await page.goto(`/results/${jobId}`, { waitUntil: 'load' })

  expect(page.url()).toContain(`/results/${jobId}`)
  await expect(page.getByRole('heading', { name: 'Stems ready' })).toBeVisible()
  await expect(page.getByText(/copy shareable link/i)).toBeVisible()

  expect(consoleErrors, `unexpected browser console/page errors:\n${consoleErrors.join('\n')}`).toEqual([])

  await context.close()
  await request.delete(`/jobs/${jobId}`)
})
