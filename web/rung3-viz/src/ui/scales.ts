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

/** The pot chart's axes (`PotChart.tsx`); kept here so `deep.test.ts` can hold every plotted point inside them, since the scales clamp rather than fail. */
export const POT_DOMAIN = { hands: [1e5, 5e6], per100: [7, 100] } as const

/** The equal-hands ruler's axes (`EqualHandsChart.tsx`): hands per seat, and the table's grade in chips per hundred hands. `deep.test.ts` holds every mark inside them, because the scales clamp rather than fail and a mark past an edge would draw on it in silence. */
export const EQUAL_DOMAIN = { hands: [1e5, 5e6], per100: [20, 400] } as const

/**
 * Labels that end at the same x, stacked top to bottom: each y is pushed down
 * until it clears the one above by `gap`; if the stack then runs past
 * `floor` (the lowest y a label may sit at), the whole stack slides up by the
 * excess. Returns the ys in the input order.
 */
export const stackLabels = (ys: number[], gap: number, floor = Infinity): number[] => {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  for (let k = 1; k < order.length; k++) order[k].y = Math.max(order[k].y, order[k - 1].y + gap)
  const excess = order.length ? order[order.length - 1].y - floor : 0
  if (excess > 0) for (const o of order) o.y -= excess
  const out = new Array<number>(ys.length)
  for (const { y, i } of order) out[i] = y
  return out
}

/** 100000 → 100k, 2000000 → 2M. */
export const fmtHands = (n: number): string => (n >= 1e6 ? `${n / 1e6}M` : `${n / 1e3}k`)

/**
 * Training hours at a mark, from its wall clock in seconds, as the prose
 * prints them: one decimal under ten hours, whole from ten up. The branch is
 * taken on the rounded value, so 9.97 hours reads 10, never 10.0.
 */
export const hoursOf = (wallClockS: number): string => {
  const h = wallClockS / 3600
  const tenths = h.toFixed(1)
  return Number(tenths) < 10 ? tenths : h.toFixed(0)
}

/** The same with its unit, the way a figure prints it: 0.4 h, 21 h. */
export const hoursAt = (wallClockS: number): string => `${hoursOf(wallClockS)} h`

/** The page's unit: chips per hundred hands, to one decimal. */
export const per100 = (chipsPerHand: number, digits = 1): string => (chipsPerHand * 100).toFixed(digits)

/** A ratio under one as a spoken fraction: 0.117 → "an eighth" (the floor of its reciprocal, so the wording never overstates). */
export const fraction = (ratio: number): string => {
  const n = Math.floor(1 / ratio)
  const words: Record<number, string> = { 2: 'a half', 3: 'a third', 4: 'a quarter', 5: 'a fifth', 6: 'a sixth', 7: 'a seventh', 8: 'an eighth', 9: 'a ninth', 10: 'a tenth' }
  return words[n] ?? `a ${n}th`
}
