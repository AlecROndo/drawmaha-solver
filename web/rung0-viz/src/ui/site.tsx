/**
 * The chrome every page of the site shares: the ladder drawn as a nav, the
 * identity rail down the left, and the paper panel a figure sits on.
 *
 * Deliberately duplicated in each visualizer rather than extracted to a
 * package: the two apps deploy independently and a shared build step would buy
 * ~120 lines at the cost of a workspace. When a third rung ships, extract.
 */

/** The five rungs, and how far the project has actually climbed. */
const RUNGS = [
  { n: 0, name: 'Rung 0', sub: 'rock-paper-scissors', href: '/rung0', done: true },
  { n: 1, name: 'Rung 1', sub: 'kuhn poker', href: '/rung1', done: true },
  { n: 2, name: 'Rung 2', sub: 'leduc', href: null, done: false },
  { n: 3, name: 'Rung 3', sub: 'mini-drawmaha', href: null, done: false },
  { n: 4, name: 'Rung 4', sub: 'full drawmaha', href: null, done: false },
]

/**
 * The validation ladder as the site's nav.
 *
 * The project's whole thesis is that it climbs one rung at a time, each
 * checked against a known answer, so the nav is that line: a station per rung,
 * filled where the rung is done, and a solid segment running only as far as
 * the climb has actually got. "Two of five complete" is the picture rather
 * than a caption under it.
 */
export function LadderLine({ here }: { here: number }) {
  return (
    <nav className="line" aria-label="The validation ladder">
      <div className="track">
        <div className="bar" />
        <div className="bar done" />
        <ol>
          {RUNGS.map((rung) => {
            const state = [rung.done ? 'done' : 'todo', rung.n === here ? 'here' : '']
              .filter(Boolean)
              .join(' ')
            const label = (
              <>
                {rung.name}
                <span className="sub">{rung.sub}</span>
              </>
            )
            return (
              <li key={rung.n} className={state}>
                <span className="dot" />
                {rung.href ? (
                  <a href={rung.href} aria-current={rung.n === here ? 'page' : undefined}>
                    {label}
                  </a>
                ) : (
                  <span className="stop-name">{label}</span>
                )}
              </li>
            )
          })}
        </ol>
      </div>
    </nav>
  )
}

/** The suit mark, drawn monoline like everything else. */
function Mark() {
  return (
    <svg className="mark" viewBox="0 0 46 46" aria-hidden>
      <g fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="23" cy="23" r="22" />
        <path d="M23 11c-4.4 4.6-8 8-8 11.6a8 8 0 0 0 16 0C31 19 27.4 15.6 23 11Z" />
        <path d="M23 30.6V35M19.4 35h7.2" />
      </g>
    </svg>
  )
}

/** Chips and a couple of cards: monoline, one stroke weight, lightly hatched. */
function Sketch() {
  return (
    <svg className="sketch" viewBox="0 0 150 62" aria-hidden>
      <g fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
        <ellipse cx="34" cy="46" rx="20" ry="7" />
        <path d="M14 46v-7M54 46v-7" />
        <ellipse cx="34" cy="39" rx="20" ry="7" />
        <path d="M14 39v-7M54 39v-7" />
        <ellipse cx="34" cy="32" rx="20" ry="7" />
        <path d="M22 31.4a13 6 0 0 0 24 0" strokeOpacity=".55" />
        <ellipse cx="96" cy="50" rx="15" ry="5.4" />
        <path d="M81 50v-5.5M111 50v-5.5" />
        <ellipse cx="96" cy="44.5" rx="15" ry="5.4" />
        <rect x="112" y="16" width="24" height="33" rx="3" transform="rotate(11 124 32)" />
        <rect x="104" y="14" width="24" height="33" rx="3" transform="rotate(-4 116 30)" />
        <path d="M113 27.5l3.6-4 3.6 4-3.6 4.2z" />
        <path d="M6 57h138" strokeOpacity=".4" />
      </g>
    </svg>
  )
}

/**
 * The identity rail: who this is, where the climb stands, and one way in. It
 * is sticky because it is the page's fixed point — everything to its right is
 * one rung's worth of evidence.
 */
export function IdentityRail({ now, next }: { now: string; next: string }) {
  return (
    <aside className="rail">
      <Mark />
      <h1>
        Drawmaha
        <br />
        Solver.
      </h1>
      <p className="quote">“Each rung is checked against a known answer before we climb.”</p>
      <dl>
        <div>
          <dt>now</dt>
          <dd>{now}</dd>
        </div>
        <div>
          <dt>next</dt>
          <dd>{next}</dd>
        </div>
        <div>
          <dt>method</dt>
          <dd>Deep CFR</dd>
        </div>
      </dl>
      <span className="spacer" />
      <Sketch />
      <a className="btn" href="/rung1#play">
        Play the solver →
      </a>
    </aside>
  )
}

/** The hand-drawn rule under a headline — the one imperfect mark on the page. */
export function Squiggle() {
  return (
    <svg className="squiggle" viewBox="0 0 232 9" aria-hidden>
      <path
        d="M2 6.2c14-5 28 3.4 42-.6s28-4.6 42 .4 28 4 42-.8 28-4 42 1 28 3.4 60-1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  )
}

/**
 * A figure's paper panel: numbered top-right, a mono field label, a serif
 * title that states the finding, and a mono sentence saying what is drawn.
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
  children: React.ReactNode
}) {
  return (
    <section
      id={id}
      className={['panel', wide ? 'wide' : '', className ?? ''].filter(Boolean).join(' ')}
      aria-label={label}
    >
      {n && <span className="no">{n}</span>}
      {k && <span className="k">{k}</span>}
      {title && <h3>{title}</h3>}
      {say && <p className="say">{say}</p>}
      {children}
    </section>
  )
}
