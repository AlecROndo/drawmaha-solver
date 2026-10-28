import { Squiggle } from './site'

/**
 * The page's masthead, in the register of a research announcement: a
 * category eyebrow, a title that states the finding, the date and the
 * author, and a summary. One masthead for the three tabs; the Research tab
 * adds its key numbers and contents under it. The ASCII object the rung
 * once opened on is gone: the page is a document, not a toy.
 */

export function Masthead({
  eyebrow,
  title,
  date,
  byline,
  tags,
  children,
}: {
  eyebrow: React.ReactNode
  title: React.ReactNode
  date: string
  byline: string
  tags: string[]
  children?: React.ReactNode
}) {
  return (
    <header className="masthead">
      <p className="eyebrow">
        <span className="dot done" />
        {eyebrow}
      </p>
      <h1>{title}</h1>
      <Squiggle />
      <p className="meta">
        <time>{date}</time>
        <span className="sep">·</span>
        {byline}
        <span className="sep">·</span>
        {tags.join(', ')}
      </p>
      {children}
    </header>
  )
}
