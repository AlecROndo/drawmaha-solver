import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /rung1 on the project's Vercel deployment (see /vercel.json).
  base: '/rung1/',
  server: {
    // The exploit tab runs REAL CFR rather than replaying a committed solve —
    // its input is a strategy you type, so there is no finite set of solves to
    // ship. `uv run kuhn-exploit-server` answers it; proxying keeps the page on
    // a same-origin path so CORS never enters the picture.
    proxy: { '/api': 'http://localhost:8000' },
  },
  // `vite preview` is a separate server with its own config, and it is the one
  // a production-build screenshot goes through — without this the exploit tab
  // 404s there while working fine in dev.
  preview: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
