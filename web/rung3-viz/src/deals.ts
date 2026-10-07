/**
 * Fetching the deal pack. The pack is static files under `public/deals/` —
 * a manifest and chunks of 100 deals, about 1 MB raw and 110 KB over the
 * wire each — written by `uv run minidraw-deals`. The first chunk is enough
 * to sit down; later chunks are fetched as the queue runs low.
 */

import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

import { addDeals, freshSession, runningLow, type Deal, type Manifest, type Session } from './play'

const BASE = `${import.meta.env.BASE_URL}deals/`

export async function loadManifest(fetcher: typeof fetch = fetch): Promise<Manifest> {
  const response = await fetcher(`${BASE}index.json`)
  if (!response.ok) throw new Error(`the deal pack's manifest did not load (${response.status})`)
  return (await response.json()) as Manifest
}

export async function loadChunk(name: string, fetcher: typeof fetch = fetch): Promise<Deal[]> {
  const response = await fetcher(`${BASE}${name}`)
  if (!response.ok) throw new Error(`deal chunk ${name} did not load (${response.status})`)
  return (await response.json()) as Deal[]
}

export interface PackState {
  manifest: Manifest | null
  /** how many chunks have been folded into the session */
  loaded: number
  error: string | null
}

/**
 * The session and the pack together: fetches the manifest and chunk 0 on
 * mount and seats the player, then fetches the next chunk whenever the
 * queue is short. The session itself is ordinary React state the panel
 * updates with the pure transitions in `play.ts`.
 */
export function useSession(): [Session, Dispatch<SetStateAction<Session>>, PackState] {
  const [session, setSession] = useState<Session>(() => freshSession([]))
  const [pack, setPack] = useState<PackState>({ manifest: null, loaded: 0, error: null })
  const fetching = useRef(false)

  useEffect(() => {
    let cancelled = false
    loadManifest()
      .then(async (manifest) => {
        const first = await loadChunk(manifest.chunks[0])
        if (cancelled) return
        setSession(() => freshSession(first))
        setPack({ manifest, loaded: 1, error: null })
      })
      .catch((error: Error) => {
        if (!cancelled) setPack((p) => ({ ...p, error: error.message }))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const low = runningLow(session)
  useEffect(() => {
    const manifest = pack.manifest
    if (!manifest || !low || fetching.current || pack.loaded >= manifest.chunks.length) return
    fetching.current = true
    const name = manifest.chunks[pack.loaded]
    loadChunk(name)
      .then((deals) => {
        setSession((s) => addDeals(s, deals))
        setPack((p) => ({ ...p, loaded: p.loaded + 1 }))
      })
      .catch((error: Error) => setPack((p) => ({ ...p, error: error.message })))
      .finally(() => {
        fetching.current = false
      })
  }, [low, pack.manifest, pack.loaded])

  return [session, setSession, pack]
}
