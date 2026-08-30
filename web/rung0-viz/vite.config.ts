import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /rung0 on the project's Vercel deployment (see /vercel.json).
  base: '/rung0/',
})
