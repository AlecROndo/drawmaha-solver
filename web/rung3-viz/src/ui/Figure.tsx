import { Figure as SiteFigure } from './site'

/**
 * A figure on the research page, in the site's figure block (`Figure` in
 * `site.tsx`): its number, a bold title that states the finding, a sub-title
 * that names the measure and the run, the drawing, a caption of at most two
 * sentences, and a source line naming the file every number in it was read
 * from. No frame: a top hairline and the page's paper. The anchor is
 * `fig-<n>` unless the chapter names one.
 */

export function Figure({
  n,
  id,
  title,
  sub,
  caption,
  source,
  label,
  children,
}: {
  /** the figure's number in page order */
  n: number
  id?: string
  /** the finding, in one line */
  title: React.ReactNode
  /** the measure, the unit and the run */
  sub: React.ReactNode
  /** at most two sentences */
  caption?: React.ReactNode
  /** the file or files the numbers come from */
  source: React.ReactNode
  label: string
  children: React.ReactNode
}) {
  return (
    <SiteFigure
      n={`Figure ${n}`}
      id={id ?? `fig-${n}`}
      title={title}
      sub={sub}
      label={label}
      className="pubfig"
      caption={
        <>
          {caption}
          <span className="fig-src">Source: {source}</span>
        </>
      }
    >
      {children}
    </SiteFigure>
  )
}
