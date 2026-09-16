import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /rung2 on the project's Vercel deployment (see /vercel.json).
  base: '/rung2/',
  server: {
    // Exploit runs are REAL CFR rather than a committed solve — the input is
    // a strategy you edit, so there is no finite set of solves to ship.
    // `uv run leduc-exploit-server` answers on 8001 (rung 1's server owns
    // 8000); proxying keeps the page same-origin so CORS never enters.
    proxy: { '/api': 'http://localhost:8001' },
  },
  // `vite preview` is a separate server with its own config, and it is the
  // one a production-build screenshot goes through — without this the
  // response panel 404s there while working fine in dev.
  preview: {
    proxy: { '/api': 'http://localhost:8001' },
  },
})
