/**
 * The chrome every page of the site shares, in the cover page's theme: the
 * top bar with the turning chip as the home mark and the Rungs menu, the hero
 * (copy beside the rung's own ASCII object), the window a figure sits in, and
 * the footer that closes the page on the chip again.
 *
 * Deliberately duplicated in each visualizer rather than extracted to a
 * package: the apps deploy independently and a shared build step would buy
 * ~300 lines at the cost of a workspace. Keep the three copies identical; a
 * change here is a change in all three.
 */

import { useEffect, useRef, useState } from 'react'
import { mountChip } from './mark'
import { mountObject, type IconName } from './ascii'

const HOME = 'https://drawmaha.app'

/** The five rungs as the Rungs menu lists them, and what each one links to. */
const RUNGS = [
  { n: 0, name: 'Rock-paper-scissors', href: `${HOME}/rung0`, state: 'done', acts: [['Analysis', `${HOME}/rung0`]] },
  { n: 1, name: 'Kuhn', href: `${HOME}/rung1`, state: 'done', acts: [['Analysis', `${HOME}/rung1`]] },
  {
    n: 2,
    name: 'Leduc',
    href: `${HOME}/rung2`,
    state: 'done',
    acts: [
      ['Analysis', `${HOME}/rung2`],
      ['Solver', `${HOME}/rung2#walk`],
      ['Trainer', `${HOME}/rung2#play`],
    ],
  },
  { n: 3, name: 'Mini-Drawmaha', href: `${HOME}/#rung3`, state: 'now', acts: [['Analysis', `${HOME}/#rung3`]] },
  { n: 4, name: 'Full Drawmaha', href: `${HOME}/#rung4`, state: 'todo', acts: [['Analysis', `${HOME}/#rung4`]] },
] as const

/** The chip, turning. `font` is the cell size: 3 for the mark in the bar, 4 for the one that closes the page. */
export function Mark({ font, speed }: { font: number; speed?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!ref.current) return
    return mountChip(ref.current, { font, speed })
  }, [font, speed])
  return <canvas ref={ref} aria-hidden />
}

/** The rung's object, raymarched into glyphs; turns slowly, faster under the mouse. */
export function RungObject({ icon, label }: { icon: IconName; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!ref.current) return
    return mountObject(ref.current, icon)
  }, [icon])
  return <canvas ref={ref} className="object" role="img" aria-label={label} />
}

/**
 * The Rungs menu: a rope ladder pulled open leftward from its button. Opens on
 * hover (250 ms), click, tap or keyboard focus; Escape and a click outside
 * close it. On a narrow screen the CSS turns it into a list under the button.
 */
function RungsMenu({ here }: { here: number }) {
  const root = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const pinned = useRef(false)
  const hoverT = useRef(0)
  const leaveT = useRef(0)

  const fit = (isOpen: boolean) => {
    const m = menu.current
    if (!m) return
    if (matchMedia('(max-width:1240px)').matches) {
      m.style.width = ''
      return
    }
    const cells = m.querySelectorAll<HTMLElement>('.rcell')
    const a = cells[0]
    const z = cells[cells.length - 1]
    m.style.width = isOpen ? `${z.offsetLeft + z.offsetWidth - a.offsetLeft + 10}px` : '0px'
  }
  const set = (v: boolean, pin = false) => {
    pinned.current = v && pin
    setOpen(v)
    fit(v)
  }

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (open && root.current && !root.current.contains(e.target as Node)) set(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        set(false)
        root.current?.querySelector<HTMLButtonElement>('.rungs-btn')?.focus()
      }
    }
    const onResize = () => {
      if (open) fit(true)
    }
    document.addEventListener('click', onClick)
    document.addEventListener('keydown', onKey)
    addEventListener('resize', onResize)
    return () => {
      document.removeEventListener('click', onClick)
      document.removeEventListener('keydown', onKey)
      removeEventListener('resize', onResize)
    }
  }, [open])

  return (
    <div
      ref={root}
      className={open ? 'rungs open' : 'rungs'}
      onMouseEnter={() => {
        clearTimeout(leaveT.current)
        if (!open) hoverT.current = window.setTimeout(() => set(true, false), 250)
      }}
      onMouseLeave={() => {
        clearTimeout(hoverT.current)
        if (open && !pinned.current) leaveT.current = window.setTimeout(() => set(false), 280)
      }}
      onBlur={(e) => {
        if (open && e.relatedTarget && !root.current?.contains(e.relatedTarget as Node)) set(false)
      }}
    >
      <button
        className="navbtn rungs-btn"
        type="button"
        aria-expanded={open}
        aria-controls="rungsMenu"
        aria-haspopup="true"
        onClick={() => {
          clearTimeout(hoverT.current)
          if (open && pinned.current) set(false)
          else set(true, true)
        }}
        onFocus={(e) => {
          if (e.target.matches(':focus-visible') && !open) set(true, true)
        }}
      >
        Rungs
      </button>
      <div ref={menu} className="rungs-menu" id="rungsMenu" role="group" aria-label="The five rungs">
        {RUNGS.map((rung, i) => (
          <div
            key={rung.n}
            className={['rcell', rung.state, rung.n === here ? 'here' : ''].filter(Boolean).join(' ')}
            style={{ '--i': RUNGS.length - 1 - i, '--j': i } as React.CSSProperties}
          >
            <a className="rmain" href={rung.href} aria-current={rung.n === here ? 'page' : undefined}>
              <b>{rung.n}</b>
              <span>{rung.name}</span>
            </a>
            <div className="racts">
              {rung.acts.map(([label, href]) => (
                <a key={label} href={href}>
                  {label}
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** The top bar: the chip as the way home, the site nav, one way into the product. */
export function TopBar({ here }: { here: number }) {
  return (
    <header className="top">
      <div className="wrap">
        <a className="chip-home" href={HOME} aria-label="Drawmaha home">
          <Mark font={3} />
          <span className="word">
            Drawmaha<small>Solver · Deep CFR</small>
          </span>
        </a>
        <nav className="nav" aria-label="Site">
          <a className="navbtn" href={`${HOME}/rules`}>
            Rules
          </a>
          <RungsMenu here={here} />
        </nav>
        {here === 2 ? (
          <a className="btn primary" href="#play">
            Play the solver →
          </a>
        ) : (
          <a className="btn primary" href={`${HOME}/rung2`}>
            Open the solver →
          </a>
        )}
      </div>
    </header>
  )
}

/** The page closes on the chip, turning slower, and the site's short list of doors. */
export function Footer() {
  return (
    <footer>
      <div className="wrap">
        <a className="chip-end" href={HOME} aria-label="Drawmaha home">
          <Mark font={4} speed={0.32} />
        </a>
        <span className="tag">Each rung is checked against a known answer before we climb.</span>
        <div className="links">
          <a href={HOME}>Home</a>
          <a href={`${HOME}/rules`}>Rules</a>
          <a href={`${HOME}/cover`}>Cover</a>
          <a href={`${HOME}/rung0`}>/rung0</a>
          <a href={`${HOME}/rung1`}>/rung1</a>
          <a href={`${HOME}/rung2`}>/rung2</a>
        </div>
      </div>
    </footer>
  )
}

/** The red squiggle under a headline: the one hand-drawn mark on the page. */
export function Squiggle() {
  return (
    <svg className="squig" viewBox="0 0 190 12" aria-hidden>
      <path d="M2 7 C 12 1, 20 12, 30 7 S 48 1, 58 7 S 76 12, 86 7 S 104 1, 114 7 S 132 12, 142 7 S 160 1, 170 7 S 184 12, 188 7" />
    </svg>
  )
}

/**
 * The hero: the eyebrow says which rung and where it stands, the headline
 * states the finding, the lede says what is on the page, and the rung's own
 * object turns beside it.
 */
export function Hero({
  eyebrow,
  done,
  title,
  icon,
  iconLabel,
  children,
}: {
  eyebrow: React.ReactNode
  /** a finished rung's dot is yellow; the rung in progress keeps the pink one */
  done?: boolean
  title: React.ReactNode
  icon: IconName
  iconLabel: string
  children?: React.ReactNode
}) {
  return (
    <div className="hero">
      <div className="hero-copy">
        <p className="eyebrow">
          <span className={done ? 'dot done' : 'dot'} />
          {eyebrow}
        </p>
        <h1>{title}</h1>
        <Squiggle />
        {children}
      </div>
      <div className="hero-obj">
        <RungObject icon={icon} label={iconLabel} />
      </div>
    </div>
  )
}

/**
 * A figure's window: a title bar with the figure's number and field label, then
 * the pane with a serif title that states the finding, a mono sentence saying
 * what is drawn, and the figure itself.
 */
export function Panel({
  n,
  id,
  k,
  title,
  say,
  wide,
  className,
  label,
  live,
  children,
}: {
  n?: string
  /** anchor target, for a panel the nav links straight to */
  id?: string
  k?: string
  title?: string
  say?: React.ReactNode
  wide?: boolean
  className?: string
  label?: string
  /** a running figure gets the live dot in its title bar */
  live?: boolean
  children: React.ReactNode
}) {
  return (
    <section
      id={id}
      className={['panel', wide ? 'wide' : '', className ?? ''].filter(Boolean).join(' ')}
      aria-label={label}
    >
      <div className="titlebar">
        <span className="dots">
          <i />
          <i />
          <i />
        </span>
        {k && <span className="k">{k}</span>}
        {live && (
          <span className="live">
            <i />
            live
          </span>
        )}
        {n && <span className="no">{n}</span>}
      </div>
      <div className="pane">
        {title && <h3>{title}</h3>}
        {say && <p className="say">{say}</p>}
        {children}
      </div>
    </section>
  )
}
