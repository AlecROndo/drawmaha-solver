/**
 * The two equities as a plane: outer equity across, inner equity up, the
 * filtered range's density drawn as cells (ink = how many holdings, colour
 * = what most of them do) so a million holdings do not become soup. The
 * corners have names a player can remember — scoop, Omaha only, hand only,
 * air — and the held hand, when five cards are placed, is a ring.
 */

import type { Cell } from '../engine/aggregate'
import { dominant } from './Grid'

export function Plane({ cells, bins, hand }: { cells: Cell[]; bins: number; hand: { inner: number; outer: number } | null }) {
  const W = 300
  const pad = 14
  const inner = W - pad * 2
  const cw = inner / bins
  const maxShare = Math.max(1e-9, ...cells.map((c) => c.share))
  const colour = (k: 'f' | 'c' | 'p') => (k === 'p' ? 'var(--pot)' : k === 'f' ? 'var(--fold)' : 'var(--call)')
  return (
    <div className="plane">
      <div className="lbl">the two equities</div>
      <svg viewBox={`0 0 ${W} ${W}`} width={W} height={W} aria-label="holdings placed by outer equity across and inner equity up; ink is how many, colour is what most do">
        <rect x={pad} y={pad} width={inner} height={inner} fill="var(--well)" stroke="var(--iv-10)" />
        <line x1={pad + inner / 2} x2={pad + inner / 2} y1={pad} y2={pad + inner} stroke="var(--iv-10)" />
        <line x1={pad} x2={pad + inner} y1={pad + inner / 2} y2={pad + inner / 2} stroke="var(--iv-10)" />
        {cells.map((c, idx) => {
          if (c.n === 0) return null
          const x = idx % bins
          const y = Math.floor(idx / bins)
          const a = Math.pow(c.share / maxShare, 0.45)
          return (
            <rect
              key={idx}
              x={pad + x * cw + 0.75}
              y={pad + inner - (y + 1) * cw + 0.75}
              width={cw - 1.5}
              height={cw - 1.5}
              rx={1.5}
              fill={colour(dominant(c))}
              opacity={0.1 + 0.9 * a}
            />
          )
        })}
        {hand && (
          <circle cx={pad + hand.outer * inner} cy={pad + inner - hand.inner * inner} r={5} fill="none" stroke="var(--ivory)" strokeWidth={2} />
        )}
        <text x={pad + inner - 2} y={pad + 11} textAnchor="end" className="quad">
          scoop
        </text>
        <text x={pad + 3} y={pad + 11} className="quad">
          hand only
        </text>
        <text x={pad + inner - 2} y={pad + inner - 4} textAnchor="end" className="quad">
          omaha only
        </text>
        <text x={pad + 3} y={pad + inner - 4} className="quad">
          air
        </text>
      </svg>
      <div className="plane-axis">
        <span>inner ↑</span>
        <span>outer →</span>
      </div>
    </div>
  )
}
