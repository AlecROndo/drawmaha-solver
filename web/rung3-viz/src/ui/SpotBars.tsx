import { CATEGORY_NAME, type SpotFigure } from '../research'
import { inkFor } from '../inks'
import { Panel } from './site'

/**
 * What the frozen strategy does, by inner category: one stacked bar per
 * category, one figure per spot. The bars average each spot's strategy over
 * every physical deal of the category with equal weight, so they show what
 * the solution does holding that hand, not how often the line happens.
 */

const pct = (x: number) => (x * 100 >= 9.5 ? `${(x * 100).toFixed(0)}` : x * 100 >= 0.95 ? `${(x * 100).toFixed(0)}` : '')

function Spot({ fig }: { fig: SpotFigure }) {
  return (
    <div className="spot">
      <span className="k">{fig.title}</span>
      <ol className="catbars">
        {fig.rows.map((row) => (
          <li key={row.category}>
            <span className="cat">{CATEGORY_NAME[row.category]}</span>
            <span className="mix" role="img" aria-label={fig.columns.map((c, i) => `${c} ${(row.mix[i] * 100).toFixed(0)}%`).join(', ')}>
              {row.mix.map((v, i) => (
                <i key={fig.columns[i]} style={{ flex: `${v} 0 0`, background: inkFor(fig.columns[i]) }}>
                  {v > 0.12 && <small>{pct(v)}</small>}
                </i>
              ))}
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

export function SpotBars({
  n,
  k,
  title,
  say,
  label,
  figures,
}: {
  n: string
  k: string
  title: string
  say: React.ReactNode
  label: string
  figures: SpotFigure[]
}) {
  const columns = figures[figures.length - 1].columns
  return (
    <Panel n={n} wide k={k} title={title} say={say} label={label}>
      <div className="legend">
        {columns.map((c) => (
          <span key={c} className="item">
            <span className="swatch" style={{ background: inkFor(c) }} />
            {c}
          </span>
        ))}
        {figures[0].columns.length !== columns.length && (
          <span className="item">
            <span className="swatch" style={{ background: inkFor(figures[0].columns[1]) }} />
            {figures[0].columns[1]}
          </span>
        )}
      </div>
      <div className="spots">
        {figures.map((fig) => (
          <Spot key={fig.key} fig={fig} />
        ))}
      </div>
    </Panel>
  )
}
