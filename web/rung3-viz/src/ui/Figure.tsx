/**
 * A figure on the research page: the drawing at column width, then a
 * caption that states the finding and a source line that names the file
 * every number in it was read from. No window chrome: the page is a
 * publication, and a figure in a publication is a figure, not an app.
 */

export function Figure({
  n,
  id,
  caption,
  source,
  label,
  children,
}: {
  /** the figure's number in page order */
  n: number
  id?: string
  /** one sentence, the finding */
  caption: React.ReactNode
  /** the file or files the numbers come from, and the axis or unit if it helps */
  source: React.ReactNode
  label: string
  children: React.ReactNode
}) {
  return (
    <figure className="figure" id={id ?? `fig-${n}`} aria-label={label}>
      <div className="figure-body">{children}</div>
      <figcaption>
        <b>Figure {n}.</b> {caption}
        <span className="source">{source}</span>
      </figcaption>
    </figure>
  )
}
