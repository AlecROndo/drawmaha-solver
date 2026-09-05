import {
  CARDS,
  CARD_SYMBOL,
  DECISION_NODES,
  EDGES,
  NODE_H,
  NODE_W,
  TERMINAL_H,
  TERMINAL_NODES,
  TERMINAL_W,
  TREE_H,
  TREE_W,
  actionFrequency,
  actionLabel,
  infosetKey,
  type Card,
} from '../kuhn'

/**
 * The whole game and the whole strategy in one picture.
 *
 * Kuhn's tree is four decision nodes and five terminals, so it fits on screen
 * intact — and every infoset can be drawn where it actually sits. Each node
 * carries three bars, one per card the acting player might hold, showing how
 * often they bet there. The twelve bars are the twelve information sets: this
 * is the strategy table, laid out as the tree it belongs to.
 *
 * The hairline on each bar is whatever answer the caller is grading against —
 * the closed-form equilibrium on the solve tab, the exact best response on the
 * exploit tab. Watching the fills walk onto their hairlines is the point of
 * both, which is why this component takes its numbers as props instead of
 * reaching for one particular solve.
 */

const CARD_VAR = ['var(--jack)', 'var(--queen)', 'var(--king)']

const BAR_X = 40
const BAR_W = 110
const BAR_H = 6
const ROW_H = 18
/** First bar's baseline, below the node's caption. */
const ROW_TOP = 36

export interface GameTreeProps {
  index: number
  /** infoset label ("K", "Jpb", ...) to P(bet) at each checkpoint */
  bet: Record<string, number[]>
  /** where each bar belongs, or null when no ground truth applies */
  hairline: Record<string, number> | null
  /** infosets the visitor has pinned, label to P(bet) — exploit tab only */
  locked?: Record<string, number>
  /** provided only when rows are clickable; absence makes the tree read-only */
  onToggleLock?: (key: string) => void
}

interface NodeCardProps extends GameTreeProps {
  node: (typeof DECISION_NODES)[number]
}

function NodeCard({ node, index, bet, hairline, locked, onToggleLock }: NodeCardProps) {
  const left = node.cx - NODE_W / 2
  const top = node.cy - NODE_H / 2
  return (
    <g>
      <rect className="node-card" x={left} y={top} width={NODE_W} height={NODE_H} rx={8} />
      <text className="node-caption" x={left + 14} y={top + 18}>
        {node.caption.toUpperCase()}
      </text>
      {CARDS.map((card, row) => {
        const key = infosetKey(card as Card, node.key)
        const p = bet[key][index]
        const exact = hairline?.[key]
        const pinned = locked?.[key]
        const y = top + ROW_TOP + row * ROW_H
        return (
          <g
            key={key}
            className={onToggleLock ? 'row-lockable' : undefined}
            onClick={onToggleLock ? () => onToggleLock(key) : undefined}
            // A row that announces itself as a button has to behave like one:
            // reachable by Tab, and fired by Enter or Space. Without this the
            // lock gesture is mouse-only and the role is a promise the tree
            // does not keep.
            tabIndex={onToggleLock ? 0 : undefined}
            onKeyDown={
              onToggleLock
                ? (event) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return
                    // Space would otherwise scroll the page out from under the
                    // row that was just activated.
                    event.preventDefault()
                    onToggleLock(key)
                  }
                : undefined
            }
            role={onToggleLock ? 'button' : undefined}
            aria-pressed={onToggleLock ? pinned !== undefined : undefined}
            aria-label={onToggleLock ? `Lock ${key}` : undefined}
          >
            {/* Hit area: the bar alone is 6px tall, too small to click at. */}
            {onToggleLock && (
              <rect x={left + 8} y={y - 6} width={NODE_W - 16} height={ROW_H} fill="transparent" />
            )}
            <text className="node-letter" x={left + 14} y={y + 6} fill={CARD_VAR[card]}>
              {CARD_SYMBOL[card]}
            </text>
            <rect x={left + BAR_X} y={y} width={BAR_W} height={BAR_H} rx={1.5} fill="var(--panel-hair)" />
            <rect
              x={left + BAR_X}
              y={y}
              width={Math.max(0, p * BAR_W)}
              height={BAR_H}
              rx={1.5}
              fill={CARD_VAR[card]}
              // A locked bar is held, not learned. Hollowing it keeps the two
              // populations legible at a glance: solid bars are what CFR
              // decided, outlined bars are what you dictated.
              fillOpacity={pinned === undefined ? 1 : 0.28}
              stroke={pinned === undefined ? undefined : CARD_VAR[card]}
              strokeWidth={pinned === undefined ? undefined : 1}
            />
            {exact !== undefined && (
              <line
                x1={left + BAR_X + exact * BAR_W}
                x2={left + BAR_X + exact * BAR_W}
                y1={y - 3}
                y2={y + BAR_H + 3}
                stroke="var(--panel-mark)"
                strokeWidth={1.5}
              />
            )}
            {/* Locked rows keep showing their number — it is the thing being
                set — and say "locked" by going muted next to the hollow bar,
                rather than by a glyph competing with the digits. */}
            <text
              className="node-value"
              x={left + BAR_X + BAR_W + 10}
              y={y + 6}
              fill={pinned === undefined ? undefined : 'var(--panel-dim)'}
            >
              {(pinned ?? p).toFixed(2)}
            </text>
          </g>
        )
      })}
    </g>
  )
}

export function GameTree(props: GameTreeProps) {
  const { index, bet } = props
  const betAt = (card: Card, history: string) => bet[infosetKey(card, history)][index]
  const centre = (key: string) =>
    DECISION_NODES.find((n) => n.key === key) ?? TERMINAL_NODES.find((n) => n.key === key)!
  const isDecision = (key: string) => DECISION_NODES.some((n) => n.key === key)

  return (
    <div className="tree-wrap">
      <div className="tree">
        <svg viewBox={`0 0 ${TREE_W} ${TREE_H}`} role="img" aria-label="Kuhn poker game tree">
          {EDGES.map((edge) => {
            const from = centre(edge.from)!
            const to = centre(edge.to)!
            const p = actionFrequency(edge.from, edge.act, betAt)
            // The tree reads left to right, so an edge leaves a node's right
            // face and arrives at the next one's left face.
            const x1 = from.cx + NODE_W / 2
            const x2 = to.cx - (isDecision(edge.to) ? NODE_W : TERMINAL_W) / 2
            return (
              <g key={`${edge.from}-${edge.act}`}>
                <path
                  d={`M ${x1} ${from.cy} C ${x1 + 40} ${from.cy}, ${x2 - 40} ${to.cy}, ${x2} ${to.cy}`}
                  fill="none"
                  stroke="var(--panel-hair)"
                  strokeWidth={1 + 4 * p}
                  strokeLinecap="round"
                />
                <text
                  className="edge-label"
                  x={(x1 + x2) / 2}
                  y={(from.cy + to.cy) / 2 - 7}
                  textAnchor="middle"
                >
                  {actionLabel(edge.act, edge.from)}
                </text>
              </g>
            )
          })}
          {TERMINAL_NODES.map((node) => (
            <g key={node.key}>
              <rect
                className="terminal"
                x={node.cx - TERMINAL_W / 2}
                y={node.cy - TERMINAL_H / 2}
                width={TERMINAL_W}
                height={TERMINAL_H}
                rx={8}
              />
              <text className="terminal-text" x={node.cx - TERMINAL_W / 2 + 12} y={node.cy + 4}>
                {node.key} · {node.text}
              </text>
            </g>
          ))}
          {DECISION_NODES.map((node) => (
            <NodeCard key={node.key || 'root'} node={node} {...props} />
          ))}
        </svg>
      </div>
    </div>
  )
}

/** What the bars and ticks mean, sitting under the tree inside the plate. */
export function TreeLegend({ hairlineIs }: { hairlineIs: string | null }) {
  return (
    <div className="legend" style={{ margin: '18px 0 0' }}>
      {CARDS.map((card) => (
        <span className="item" key={card}>
          <span className="dot" style={{ background: CARD_VAR[card] }} />
          {['jack', 'queen', 'king'][card]}
        </span>
      ))}
      <span className="item" style={{ marginLeft: 'auto' }}>
        bar = P(bet), reads as P(call) when facing a bet
        {hairlineIs === null ? '' : ` · tick = ${hairlineIs}`}
      </span>
    </div>
  )
}
