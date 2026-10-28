/**
 * Which entry of a contents rail the reader is at. `tops` are the sections'
 * tops in viewport pixels, in page order; `line` is the reading line, so a
 * section has been reached once its top is at or above it; `atEnd` says the
 * page is scrolled to its end, where the last section counts as reached even
 * if it is too short for its top to climb to the line. Returns the index of
 * the last section reached, or -1 above the first.
 */
export function currentSection(tops: readonly number[], line: number, atEnd: boolean): number {
  if (atEnd && tops.length > 0) return tops.length - 1
  let at = -1
  tops.forEach((top, i) => {
    if (top <= line) at = i
  })
  return at
}
