/**
 * The frozen strategy's export as the browser reads it: the manifest, the
 * three key tables, and the per-point mix chunks fetched on demand.
 *
 * Terms. A *public point* is one of the 141 spots a decision can stand on —
 * the betting lines, both draw counts, how many board cards are out and whose
 * turn it is — and nothing about any card (`enumeration.public_decision_points`).
 * A *private key* is the acting player's own (hole, thrown, board) after the
 * joint suit relabelling (`canonical.ts`); a *shape* is the size class a key
 * belongs to — `b1d0` for one board card, `b2d0` / `b2d1` for two board cards
 * with none / one card thrown — and each shape has one *key table* listing
 * its keys in the Python exporter's sorted order. A *chunk* is one point's
 * strategy: one row per key of the point's shape, one byte per legal action,
 * each byte a probability times `scale` (250), each row summing to exactly
 * `scale` by construction. The ledger at (point, key) is therefore row
 * `keyIndex(shape, key)` of chunk `point(id)`.
 *
 * `range_export.py` writes everything here; the format is the "Export format"
 * section of `docs/superpowers/plans/2026-10-07-solver-minidrawmaha.md`.
 *
 * Chunks are gzip-compressed and inflated with the platform's
 * `DecompressionStream`, which every browser this app targets and Node 20+
 * provide. There is no zlib fallback: a bundle that imports `node:zlib` would
 * break the page, and vitest's node environment has the stream API, so a
 * missing `DecompressionStream` is a broken runtime and throws as one.
 */

import { assertMiniCard, type MiniCard } from './cards'
import { keyString, type Picture } from './canonical'

export type Shape = 'b1d0' | 'b2d0' | 'b2d1'

export interface PointMeta {
  id: number
  player: 0 | 1
  boardCards: 1 | 2
  /** the `DrawSignal.count`s present: [] in round 1 and at P0's draw, [p0] at P1's draw, [p0, p1] in round 2 */
  draws: number[]
  /** one `line_symbol` string per betting round begun */
  lines: string[]
  /** how many cards the player to act has already thrown: picks the shape */
  discards: 0 | 1
  /** true at the two draw decisions */
  draw: boolean
  /** the ledger's columns in `legal_actions()` order: 'f' 'c' 'p' at a bet, 'n' 'l' 'm' 't' at a draw */
  actions: string[]
  shape: Shape
  file: string
  /** rows in the chunk = keys in the shape */
  n: number
}

export interface ShapeMeta {
  file: string
  n: number
  /** bytes per key row: hole (3) + thrown (d) + board (bc) */
  stride: number
}

export interface MiniIndex {
  strategy: { sha256: string; rule: string; column: string; iteration: number; workers: number; seed: number }
  deck: { ranks: string; suits: string }
  /** a byte in a chunk is probability × scale */
  scale: number
  shapes: Record<string, ShapeMeta>
  points: PointMeta[]
}

/** What `MiniData` fetches with; `fetch` in the app, a file reader in tests. */
export type Fetcher = (url: string) => Promise<Response>

const defaultFetcher: Fetcher = (url) => fetch(url)

async function fetchOk(fetcher: Fetcher, url: string): Promise<Response> {
  const res = await fetcher(url)
  if (!res.ok) throw new Error(`fetching ${url} failed: ${res.status} ${res.statusText}`)
  return res
}

/** The board-card and thrown-card counts a shape name encodes, checked against its stride. */
function shapeCounts(name: string, stride: number): { boardCards: number; discards: number } {
  const m = /^b(\d)d(\d)$/.exec(name)
  if (!m) throw new Error(`${name} is not a key shape (want b<boardCards>d<discards>)`)
  const boardCards = Number(m[1])
  const discards = Number(m[2])
  if (3 + discards + boardCards !== stride) {
    throw new Error(`shape ${name} has stride ${stride}, but hole(3) + thrown(${discards}) + board(${boardCards}) is ${3 + discards + boardCards}`)
  }
  return { boardCards, discards }
}

/**
 * One key table as a lookup: `keyString` of each row → its index. Rows are
 * read in file order, which is the exporter's sorted order, so the index is
 * the chunk row.
 */
function indexKeys(name: string, meta: ShapeMeta, bytes: Uint8Array): Map<string, number> {
  const { boardCards, discards } = shapeCounts(name, meta.stride)
  if (bytes.byteLength !== meta.n * meta.stride) {
    throw new Error(`key table ${name} has ${bytes.byteLength} bytes, want ${meta.n} × ${meta.stride} = ${meta.n * meta.stride}`)
  }
  const map = new Map<string, number>()
  for (let row = 0; row < meta.n; row++) {
    const at = row * meta.stride
    const hole: MiniCard[] = [bytes[at], bytes[at + 1], bytes[at + 2]]
    const thrown: MiniCard[] = []
    for (let k = 0; k < discards; k++) thrown.push(bytes[at + 3 + k])
    const board: MiniCard[] = []
    for (let k = 0; k < boardCards; k++) board.push(bytes[at + 3 + discards + k])
    for (const c of [...hole, ...thrown, ...board]) assertMiniCard(c)
    const key = keyString({ hole, thrown, board })
    if (map.has(key)) throw new Error(`key table ${name} lists ${key} twice (rows ${map.get(key)} and ${row})`)
    map.set(key, row)
  }
  return map
}

/** Inflate a gzip body with the platform stream API. */
async function gunzip(body: Blob): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('this runtime has no DecompressionStream; the strategy chunks cannot be inflated')
  }
  const inflated = body.stream().pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(inflated).arrayBuffer())
}

/**
 * The export, loaded: the manifest and all three key tables up front (they
 * are small and every lookup needs them), the chunks lazily and once each.
 */
export class MiniData {
  readonly index: MiniIndex
  private readonly base: string
  private readonly fetcher: Fetcher
  private readonly keys: Map<string, Map<string, number>>
  private readonly chunks = new Map<number, Promise<Uint8Array>>()

  private constructor(base: string, fetcher: Fetcher, index: MiniIndex, keys: Map<string, Map<string, number>>) {
    this.base = base
    this.fetcher = fetcher
    this.index = index
    this.keys = keys
  }

  /**
   * Fetch `${base}index.json` and the key table of every shape it names.
   * `base` is the directory the exporter wrote to, with its trailing slash
   * (`/solver/mini/`). Throws on a failed fetch, a key table whose length
   * disagrees with its manifest, or a duplicated key.
   */
  static async load(base: string, fetcher: Fetcher = defaultFetcher): Promise<MiniData> {
    const index = (await (await fetchOk(fetcher, `${base}index.json`)).json()) as MiniIndex
    if (!Number.isInteger(index.scale) || index.scale <= 0) throw new Error(`index.json has no positive integer scale: ${String(index.scale)}`)
    if (!Array.isArray(index.points) || index.points.length === 0) throw new Error('index.json lists no public points')
    const names = Object.keys(index.shapes)
    const tables = await Promise.all(
      names.map(async (name) => {
        const meta = index.shapes[name]
        const bytes = new Uint8Array(await (await fetchOk(fetcher, `${base}${meta.file}`)).arrayBuffer())
        return [name, indexKeys(name, meta, bytes)] as const
      }),
    )
    return new MiniData(base, fetcher, index, new Map(tables))
  }

  /** The chunk row of a canonical key in its shape. Throws naming the key when the shape lacks it. */
  keyIndex(shape: string, key: Picture): number {
    const table = this.keys.get(shape)
    if (table === undefined) throw new Error(`${shape} is not a key shape in this export (have ${[...this.keys.keys()].join(', ')})`)
    const row = table.get(keyString(key))
    if (row === undefined) throw new Error(`key ${keyString(key)} is not in shape ${shape}`)
    return row
  }

  /** How many keys a shape has. */
  shapeSize(shape: string): number {
    const table = this.keys.get(shape)
    if (table === undefined) throw new Error(`${shape} is not a key shape in this export`)
    return table.size
  }

  /**
   * The strategy at public point `id`: `n × actions.length` bytes, row-major,
   * fetched and inflated once and cached. Throws when the inflated length
   * disagrees with the manifest, before any mix is read from it.
   */
  point(id: number): Promise<Uint8Array> {
    const cached = this.chunks.get(id)
    if (cached !== undefined) return cached
    const meta = this.index.points[id]
    if (meta === undefined || meta.id !== id) throw new Error(`public point ${id} is not in this export`)
    const pending = (async () => {
      const body = await (await fetchOk(this.fetcher, `${this.base}${meta.file}`)).blob()
      const bytes = await gunzip(body)
      const want = meta.n * meta.actions.length
      if (bytes.byteLength !== want) {
        throw new Error(`point ${id} (${meta.file}) inflated to ${bytes.byteLength} bytes, want ${meta.n} × ${meta.actions.length} = ${want}`)
      }
      return bytes
    })()
    // A failed fetch must not poison the cache: the next call retries.
    pending.catch(() => this.chunks.delete(id))
    this.chunks.set(id, pending)
    return pending
  }
}
