import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Served at /solver on the project's Vercel deployment (see /vercel.json).
  base: '/solver/',
})
