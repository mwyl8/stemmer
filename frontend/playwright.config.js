import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(frontendDir, '..')

// Real end-to-end smoke tests against the actual running app (backend +
// frontend), not a mocked DOM -- see e2e/spectrogram.spec.js's header for
// why: the bugs these guard against (spectrogram crash, shareable-link
// route collision) only ever showed up in a rendered browser hitting the
// real dev server, never in the unit/API test suites.
export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: 'list',
  webServer: [
    {
      command: 'uv run uvicorn backend.app:app --port 8000',
      cwd: repoRoot,
      url: 'http://localhost:8000/health',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      command: 'npm run dev',
      cwd: frontendDir,
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
  use: {
    baseURL: 'http://localhost:5173',
  },
})
