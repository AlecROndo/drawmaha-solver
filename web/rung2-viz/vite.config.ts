import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /rung2 on the project's Vercel deployment (see /vercel.json).
  // No proxy blocks: this page renders a committed solve and runs no server —
  // the exploit tab's equivalent arrives with the Leduc exploiter PR.
  base: '/rung2/',
})
