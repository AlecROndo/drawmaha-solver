import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // Served at /rung3 on the project's Vercel deployment (see /vercel.json).
  // The deal pack under public/deals/ is fetched relative to this base.
  base: '/rung3/',
  // The Learn tab imports the range interface's source from ../solver-viz.
  // Those files import `react`; `dedupe` resolves it from this app's own
  // node_modules, so the build does not depend on the sibling having been
  // installed (the Vercel build installs the apps one after another) and the
  // page never bundles two Reacts. `fs.allow` lets the dev server read the
  // sibling's files.
  resolve: { dedupe: ['react', 'react-dom'] },
  server: { fs: { allow: ['..'] } },
})
