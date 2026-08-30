/** Seeded PRNG (mulberry32). Same seed → identical sequence, so runs are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Sample an index from a distribution using one uniform draw. */
export function sampleIndex(dist: readonly number[], rng: () => number): number {
  const x = rng()
  let acc = 0
  for (let i = 0; i < dist.length - 1; i++) {
    acc += dist[i]
    if (x < acc) return i
  }
  return dist.length - 1
}
