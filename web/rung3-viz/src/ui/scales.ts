/**
 * The small arithmetic every chart on the research page shares: log scales
 * over a pixel range, and the way hands and chips are printed. Kept out of
 * the component files so fast refresh has only components to refresh.
 */

/** A log10 scale from [lo, hi] onto [a, b]. */
export const logScale = (lo: number, hi: number, a: number, b: number) => (v: number) =>
  a + (Math.log10(Math.min(Math.max(v, lo), hi) / lo) / Math.log10(hi / lo)) * (b - a)

/** A linear scale from [lo, hi] onto [a, b]. */
export const linScale = (lo: number, hi: number, a: number, b: number) => (v: number) =>
  a + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (b - a)

/** 100000 → 100k, 2000000 → 2M. */
export const fmtHands = (n: number): string => (n >= 1e6 ? `${n / 1e6}M` : `${n / 1e3}k`)

/** The page's unit: chips per hundred hands, to one decimal. */
export const per100 = (chipsPerHand: number, digits = 1): string => (chipsPerHand * 100).toFixed(digits)

/** A ratio under one as a spoken fraction: 0.117 → "an eighth" (the floor of its reciprocal, so the wording never overstates). */
export const fraction = (ratio: number): string => {
  const n = Math.floor(1 / ratio)
  const words: Record<number, string> = { 2: 'a half', 3: 'a third', 4: 'a quarter', 5: 'a fifth', 6: 'a sixth', 7: 'a seventh', 8: 'an eighth', 9: 'a ninth', 10: 'a tenth' }
  return words[n] ?? `a ${n}th`
}
