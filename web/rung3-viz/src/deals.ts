/**
 * Fetching the deal pack. The pack is static files under `public/deals/` —
 * a manifest and chunks of 100 deals, about 1 MB raw and 110 KB over the
 * wire each — written by `uv run minidraw-deals`. The first chunk is enough
 * to sit down; later chunks are fetched as the queue runs low.
 */

import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

import { addDeals, freshSession, runningLow, validateDeal, type Deal, type Manifest, type Session } from './play'

const BASE = `${import.meta.env.BASE_URL}deals/`

/** The deals chunk `index` must hold: `chunk` each, the last one whatever is left. */
export function expectedDeals(manifest: Manifest, index: number): number {
  const full = manifest.chunks.length - 1
  return index < full ? manifest.chunk : manifest.deals - manifest.chunk * full
}

/**
 * Refuse a chunk the manifest did not describe, or a deal the grammar cannot
 * play. The manifest and the chunks are separate files, so a cache can hand
 * the browser one export's manifest with another's chunks; the count catches
 * a re-export of a different size, and `validateDeal` catches a node the
 * export missed. Neither can tell two same-sized exports apart — that is what
 * the digest in the manifest is for, and it is shown, not checked.
 * `checkManifest` only holds the manifest to its own arithmetic; whether the
 * files on the wire match it is this check's job, chunk by chunk.
 */
export function checkChunk(manifest: Manifest, index: number, deals: Deal[]): void {
  const name = manifest.chunks[index]
  const expected = expectedDeals(manifest, index)
  if (deals.length !== expected) throw new Error(`${name} holds ${deals.length} deals; the manifest says ${expected}`)
  deals.forEach((deal, i) => {
    const why = validateDeal(deal)
    if (why) throw new Error(`${name}, deal ${i}: ${why}`)
  })
}

export async function loadManifest(fetcher: typeof fetch = fetch): Promise<Manifest> {
  const response = await fetcher(`${BASE}index.json`)
  if (!response.ok) throw new Error(`the deal pack's manifest did not load (${response.status})`)
  const manifest = (await response.json()) as Manifest
  checkManifest(manifest)
  return manifest
}

/** What `write_pack` names a chunk, and all a chunk name is allowed to be before it goes into a URL. */
const CHUNK_NAME = /^pack-\d{2}\.json$/

/**
 * Refuse a manifest whose own arithmetic does not close: `deals` in chunks
 * of `chunk` must name exactly ceil(deals / chunk) chunks, and at least one,
 * each a plain `pack-NN.json`. `expectedDeals` divides by this and
 * `loadChunk` fetches by it, so it is checked before anything is.
 */
export function checkManifest(manifest: Manifest): void {
  const { deals, chunk, chunks } = manifest
  const named = Array.isArray(chunks) ? chunks.length : 0
  const whole = Number.isInteger(deals) && deals > 0 && Number.isInteger(chunk) && chunk > 0
  if (!whole || named !== Math.ceil(deals / chunk)) {
    throw new Error(`the manifest describes ${deals} deals in chunks of ${chunk} but names ${named} chunks`)
  }
  const odd = chunks.find((name) => typeof name !== 'string' || !CHUNK_NAME.test(name))
  if (odd !== undefined) throw new Error(`the manifest names a chunk that is not a pack-NN.json: ${String(odd)}`)
}

export async function loadChunk(name: string, fetcher: typeof fetch = fetch): Promise<Deal[]> {
  const response = await fetcher(`${BASE}${name}`)
  if (!response.ok) throw new Error(`deal chunk ${name} did not load (${response.status})`)
  return (await response.json()) as Deal[]
}

/**
 * What sitting down needs: the manifest, and chunk 0 checked against it.
 * The `useSession` effects are thin wrappers over this and `loadNext`, so
 * the whole load-and-refuse path runs under a fake fetcher in tests and the
 * hook's own job is only to route the result or the error into state.
 */
export async function loadFirst(fetcher: typeof fetch = fetch): Promise<{ manifest: Manifest; deals: Deal[] }> {
  const manifest = await loadManifest(fetcher)
  const deals = await loadChunk(manifest.chunks[0], fetcher)
  checkChunk(manifest, 0, deals)
  return { manifest, deals }
}

/** Chunk `index`, checked against the manifest, for a queue that is running low. */
export async function loadNext(manifest: Manifest, index: number, fetcher: typeof fetch = fetch): Promise<Deal[]> {
  const deals = await loadChunk(manifest.chunks[index], fetcher)
  checkChunk(manifest, index, deals)
  return deals
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
  // Whether the panel is still on the page when a fetch lands. A ref rather
  // than a per-effect `cancelled` flag because the fetch below is keyed by
  // `fetching`, not by the effect run that started it: under StrictMode the
  // first run's fetch is the one that completes, and it must still count.
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    loadFirst()
      .then(({ manifest, deals }) => {
        if (cancelled) return
        setSession(() => freshSession(deals))
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
    const index = pack.loaded
    loadNext(manifest, index)
      .then((deals) => {
        if (!mounted.current) return
        setSession((s) => addDeals(s, deals))
        setPack((p) => ({ ...p, loaded: p.loaded + 1 }))
      })
      .catch((error: Error) => {
        if (mounted.current) setPack((p) => ({ ...p, error: error.message }))
      })
      .finally(() => {
        fetching.current = false
      })
  }, [low, pack.manifest, pack.loaded])

  return [session, setSession, pack]
}
