/**
 * The chrome every page of the site shares, on paper: the top bar with the
 * turning chip as the home mark, the five product surfaces and one call to
 * action; the page head (eyebrow, title, lede, an optional controls slot); the
 * figure a chart sits in (number, finding title, sub-title, the figure, a
 * caption); and the footer that closes the page on the chip again, with the
 * project line and three columns of doors.
 *
 * Deliberately duplicated in each visualizer rather than extracted to a
 * package: the apps deploy independently and a shared build step would buy
 * ~300 lines at the cost of a workspace. Keep the five copies identical; a
 * change here is a change in all five (`tests/test_shared_chrome.py`).
 *
 * `Panel`, `Hero` and `Squiggle` are kept as thin aliases over `Figure` and
 * `PageHead` so every app compiles while its own surface is restyled; each
 * app's restyle replaces its calls and the aliases go when the last one does.
 */

import { useEffect, useRef } from 'react'
import { mountChip } from './mark'

const HOME = 'https://drawmaha.app'
const REPO = 'https://github.com/AlecROndo/drawmaha-solver'

/** The surfaces the nav names, in order; `here` is the one the page is. */
export type Surface = 'research' | 'solver' | 'benchmarks' | 'play' | 'rules' | 'home'

const NAV: { id: Surface; label: string; href: string }[] = [
  { id: 'research', label: 'Research', href: `${HOME}/research` },
  { id: 'solver', label: 'Solver', href: `${HOME}/solver` },
  { id: 'benchmarks', label: 'Benchmarks', href: `${HOME}/benchmarks` },
  { id: 'play', label: 'Play', href: `${HOME}/play` },
  { id: 'rules', label: 'Rules', href: `${HOME}/rules` },
]

/**
 * The chip, turning: the face in the accent, the pattern in ink. `font` is the
 * cell size in px: 1.5 both in the bar and where it closes the page, fine
 * enough that the sectors read.
 */
export function Mark({ font, speed }: { font: number; speed?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!ref.current) return
    return mountChip(ref.current, { font, speed, mark: '#8e2038', ink: '#16140f' })
  }, [font, speed])
  return <canvas ref={ref} aria-hidden />
}

/** The top bar: the chip as the way home, the five surfaces, one way into the product. */
export function TopBar({ here }: { here: Surface }) {
  return (
    <header className="top">
      <div className="wrap">
        <a className="chip-home" href={HOME} aria-label="Drawmaha home">
          <Mark font={1.5} />
          <span className="word">
            Drawmaha <small>Solver</small>
          </span>
        </a>
        <nav className="nav" aria-label="Site">
          {NAV.map((item) => (
            <a key={item.id} className="navbtn" href={item.href} aria-current={item.id === here ? 'page' : undefined}>
              {item.label}
            </a>
          ))}
        </nav>
        <a className="btn primary" href={`${HOME}/solver`}>
          Open the solver
        </a>
      </div>
    </header>
  )
}

/** The footer: the chip, the project in one paragraph, and three columns of doors. */
export function Footer() {
  return (
    <footer>
      <div className="wrap">
        <div className="foot-cols">
          <div className="foot-about">
            <a className="chip-end" href={HOME} aria-label="Drawmaha home">
              <Mark font={1.5} speed={0.32} />
            </a>
            <p>
              A research project by Alec Ondo. A heads-up pot-limit Drawmaha solver, built one solved game at a time,
              each graded against a known answer. Python, NumPy and torch; training on Modal; the site on Vercel. The code,
              the graders and the data behind every figure are public.
            </p>
          </div>
          <div>
            <h4>Product</h4>
            <ul>
              <li>
                <a href={`${HOME}/solver`}>Range explorer</a>
              </li>
              <li>
                <a href={`${HOME}/play`}>Play the solver</a>
              </li>
              <li>
                <a href={`${HOME}/rules`}>Rules of Drawmaha</a>
              </li>
              <li>
                <a href={`${HOME}/releases`}>Releases</a>
              </li>
            </ul>
          </div>
          <div>
            <h4>Research</h4>
            <ul>
              <li>
                <a href={`${HOME}/research`}>Mini-drawmaha, solved</a>
              </li>
              <li>
                <a href={`${HOME}/learn`}>Learn</a>
              </li>
              <li>
                <a href={`${HOME}/benchmarks`}>Benchmarks</a>
              </li>
              <li>
                <a href={`${REPO}/blob/main/paper/drawmaha-ml-survey.pdf`}>ML survey (PDF)</a>
              </li>
            </ul>
          </div>
          <div>
            <h4>Resources</h4>
            <ul>
              <li>
                <a href={REPO}>GitHub</a>
              </li>
              <li>
                <a href={`${REPO}/tree/main/figures`}>Figure data</a>
              </li>
              <li>
                <a href={`${REPO}/releases`}>Strategy packs</a>
              </li>
            </ul>
          </div>
        </div>
        <div className="foot-line">
          <span>© 2026 Alec Ondo · Built in public</span>
          <a href={REPO}>github.com/AlecROndo/drawmaha-solver</a>
        </div>
      </div>
    </footer>
  )
}

/**
 * The page head: an eyebrow naming the surface, a serif title that states
 * what the page is or finds, a lede of one or two declarative sentences, and
 * an optional controls slot (a segmented control, a select) on the right.
 */
export function PageHead({
  eyebrow,
  title,
  lede,
  controls,
  children,
}: {
  eyebrow?: React.ReactNode
  title: React.ReactNode
  lede?: React.ReactNode
  controls?: React.ReactNode
  children?: React.ReactNode
}) {
  return (
    <header className="page-head">
      <div className="page-head-copy">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
        {children}
      </div>
      {controls && <div className="controls">{controls}</div>}
    </header>
  )
}

/**
 * A figure: its number, a bold title that states the finding, a sub-title
 * naming the measure and the run, the figure itself, and a caption of at most
 * two sentences. No frame: a top hairline and the page's paper.
 */
export function Figure({
  n,
  id,
  title,
  sub,
  caption,
  wide,
  className,
  label,
  children,
}: {
  n?: string
  /** anchor target, for a figure the contents rail links straight to */
  id?: string
  title?: React.ReactNode
  sub?: React.ReactNode
  caption?: React.ReactNode
  wide?: boolean
  className?: string
  label?: string
  children: React.ReactNode
}) {
  return (
    <figure id={id} className={['fig', wide ? 'wide' : '', className ?? ''].filter(Boolean).join(' ')} aria-label={label}>
      {(n || title || sub) && (
        <div className="fig-head">
          {n && <span className="fig-n">{n}</span>}
          {title && <div className="fig-title">{title}</div>}
          {sub && <div className="fig-sub">{sub}</div>}
        </div>
      )}
      <div className="fig-body">{children}</div>
      {caption && <figcaption className="fig-cap">{caption}</figcaption>}
    </figure>
  )
}

/* ---------- aliases kept while the surfaces are restyled ------------------ */

/** The old figure window, now a `Figure`: `k` and `live` have no place on paper, `say` is the sub-title. */
export function Panel({
  n,
  id,
  k: _k,
  title,
  say,
  wide,
  className,
  label,
  live: _live,
  children,
}: {
  n?: string
  id?: string
  k?: string
  title?: string
  say?: React.ReactNode
  wide?: boolean
  className?: string
  label?: string
  live?: boolean
  children: React.ReactNode
}) {
  return (
    <Figure n={n} id={id} title={title} sub={say} wide={wide} className={className} label={label}>
      {children}
    </Figure>
  )
}

/** The old hero, now a `PageHead`: the ASCII object and its dot are gone, the copy stays. */
export function Hero({
  eyebrow,
  done: _done,
  title,
  icon: _icon,
  iconLabel: _iconLabel,
  children,
}: {
  eyebrow: React.ReactNode
  done?: boolean
  title: React.ReactNode
  icon?: string
  iconLabel?: string
  children?: React.ReactNode
}) {
  return (
    <PageHead eyebrow={eyebrow} title={title}>
      {children}
    </PageHead>
  )
}

/** The red squiggle under a headline is gone from the paper system; the alias renders nothing. */
export function Squiggle() {
  return null
}
