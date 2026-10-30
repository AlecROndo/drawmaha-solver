/**
 * The research page's title block, in the register of a lab's research
 * announcement: an eyebrow naming the field and the date, a serif title that
 * states the finding, a dek of two or three declarative sentences, a meta row
 * (the author, the size of the document, the doors to its data and code),
 * and the results strip on a top rule: three numbers, each with what it
 * measures and the chapter that shows it. Left-aligned to 900 px; the views
 * sit under it.
 */

export interface Result {
  /** the number as printed, unit-free: "6.1", "4×" */
  value: string
  /** what the number measures, in one sentence */
  label: React.ReactNode
  /** the chapter that shows it */
  href: string
  chapter: string
}

export function Masthead({
  eyebrow,
  title,
  dek,
  meta,
  results,
}: {
  eyebrow: React.ReactNode
  title: React.ReactNode
  dek: React.ReactNode
  meta: React.ReactNode[]
  results: Result[]
}) {
  return (
    <header className="masthead">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="lede dek">{dek}</p>
      <ul className="meta">
        {meta.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
      <div className="results" role="list" aria-label="Results">
        {results.map((r) => (
          <div key={r.value} role="listitem">
            <p className="v">{r.value}</p>
            <p className="l">
              {r.label} · <a href={r.href}>Chapter {r.chapter}</a>
            </p>
          </div>
        ))}
      </div>
    </header>
  )
}
