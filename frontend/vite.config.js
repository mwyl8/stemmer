import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Dev-only proxy so the Vite dev server can call the FastAPI backend
      // without touching CORS on the backend at all (Phase 5 is frontend-only).
      '/jobs': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
