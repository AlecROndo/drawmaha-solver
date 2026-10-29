import { CATEGORY_NAME, type SpotFigure } from '../research'
import { inkFor } from '../inks'

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
                <i key={fig.columns[i]} style={{ flex: `${v} 0 0`, background: inkFor(fig.columns[i]), color: fig.columns[i] === 'throw low' ? 'var(--ink)' : 'var(--paper)' }}>
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

export function SpotBars({ figures }: { figures: SpotFigure[] }) {
  // one legend entry per colour, in first-seen order: the check and the fold
  // share the quiet blue and never meet in one spot, so they share an entry
  const keys: { ink: string; words: string[] }[] = []
  for (const c of figures.flatMap((f) => f.columns)) {
    const ink = inkFor(c)
    const at = keys.find((k) => k.ink === ink)
    if (!at) keys.push({ ink, words: [c] })
    else if (!at.words.includes(c)) at.words.push(c)
  }
  return (
    <>
      <div className="legend">
        {keys.map((k) => (
          <span key={k.ink} className="item">
            <span className="swatch sq" style={{ background: k.ink }} />
            {k.words.join(' / ')}
          </span>
        ))}
      </div>
      <div className="spots">
        {figures.map((fig) => (
          <Spot key={fig.key} fig={fig} />
        ))}
      </div>
    </>
  )
}
