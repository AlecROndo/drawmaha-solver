import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /rung3 on the project's Vercel deployment (see /vercel.json).
  // The deal pack under public/deals/ is fetched relative to this base.
  base: '/rung3/',
})
