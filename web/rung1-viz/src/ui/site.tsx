/**
 * The chrome every page of the site shares: the nav, the band at the foot, and
 * the three-part head a figure carries.
 *
 * Deliberately duplicated in each visualizer rather than extracted to a
 * package: the two apps deploy independently and a shared build step would buy
 * ~60 lines at the cost of a workspace. When a third rung ships, extract.
 */

const NAV_LINKS = [
  { href: '/', label: 'Ladder' },
  { href: '/rung0', label: 'Rung 0' },
  { href: '/rung1', label: 'Rung 1' },
]

/** Wordmark left, text links right, one ink button. No bottom border. */
export function Nav({ here }: { here: string }) {
  return (
    <nav className="top">
      <a className="wordmark" href="/">
        drawmaha solver
      </a>
      <div className="links">
        <ul>
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <a href={link.href} className={link.href === here ? 'here' : undefined}>
                {link.label}
              </a>
            </li>
          ))}
        </ul>
        <a className="btn" href="/rung1#play">
          Play against CFR <span aria-hidden>→</span>
        </a>
      </div>
    </nav>
  )
}

/**
 * The rungs, with the two that exist linked and the three that do not stated
 * as their status rather than dressed up as links to nowhere.
 */
export function Footer() {
  return (
    <footer className="band">
      <div className="wrap g12">
        <div className="about">
          <a className="wordmark" href="/">
            drawmaha solver
          </a>
          <p className="tiny">
            A Deep-CFR solver for heads-up pot-limit Drawmaha, built on a validation ladder. This
            site is served until the rung-4 dashboard exists.
          </p>
        </div>
        <div className="rungs">
          <h3>Rungs</h3>
          <ul>
            <li>
              <a href="/rung0">0 · Rock-paper-scissors</a>
            </li>
            <li>
              <a href="/rung1">1 · Kuhn poker</a>
            </li>
            <li className="pending">2 · Leduc poker — next</li>
            <li className="pending">3 · Mini-drawmaha — pending</li>
            <li className="pending">4 · Full drawmaha — pending</li>
          </ul>
        </div>
      </div>
      <div className="wrap">
        <p className="tiny">
          Code, figures and writeups live in AlecROndo/drawmaha-solver (private).
        </p>
      </div>
    </footer>
  )
}

/**
 * A figure's head: mono figure number, sans title that states the finding, and
 * a serif sentence explaining what is being drawn. The three type roles in the
 * order the eye takes them.
 */
export function FigureHead({
  n,
  title,
  children,
}: {
  n: string
  title: string
  children?: React.ReactNode
}) {
  return (
    <div className="figtitle">
      <span className="label">{n}</span>
      <h3 className="fig">{title}</h3>
      {children && <p className="serif dim">{children}</p>}
    </div>
  )
}
