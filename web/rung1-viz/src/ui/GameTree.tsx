import {
  CARDS,
  CARD_SYMBOL,
  DECISION_NODES,
  EDGES,
  NODE_H,
  NODE_W,
  TERMINAL_NODES,
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
// Leaves room for the value text: a bar at 1.00 must not run under it, or the
// hairline lands on top of the digits.
const BAR_W = 54
const ROW_H = 14

export interface GameTreeProps {
  index: number
  /** infoset label ("K", "Jpb", ...) to P(bet) at each checkpoint */
  bet: Record<string, number[]>
  /** where each bar belongs, or null when no ground truth applies */
  hairline: Record<string, number> | null
  heading: string
  sub: string
  /** infosets the visitor has pinned, label to P(bet) — exploit tab only */
  locked?: Record<string, number>
  /** provided only when rows are clickable; absence makes the tree read-only */
  onToggleLock?: (key: string) => void
}

interface NodeCardProps extends Omit<GameTreeProps, 'heading' | 'sub'> {
  node: (typeof DECISION_NODES)[number]
}

function NodeCard({ node, index, bet, hairline, locked, onToggleLock }: NodeCardProps) {
  const left = node.cx - NODE_W / 2
  const top = node.cy - NODE_H / 2
  return (
    <g>
      <rect className="node-card" x={left} y={top} width={NODE_W} height={NODE_H} rx={3} />
      <text className="node-caption" x={left + 10} y={top + 15}>
        {node.caption}
      </text>
      {CARDS.map((card, row) => {
        const key = infosetKey(card as Card, node.key)
        const p = bet[key][index]
        const exact = hairline?.[key]
        const pinned = locked?.[key]
        const y = top + 28 + row * ROW_H
        return (
          <g
            key={key}
            className={onToggleLock ? 'row-lockable' : undefined}
            onClick={onToggleLock ? () => onToggleLock(key) : undefined}
            role={onToggleLock ? 'button' : undefined}
            aria-label={onToggleLock ? `Lock ${key}` : undefined}
          >
            {/* Hit area: the bar alone is 6px tall, too small to click at. */}
            {onToggleLock && (
              <rect x={left + 4} y={y - 4} width={NODE_W - 8} height={ROW_H} fill="transparent" />
            )}
            <text className="node-letter" x={left + 10} y={y + 5} fill={CARD_VAR[card]}>
              {CARD_SYMBOL[card]}
            </text>
            <rect x={left + BAR_X} y={y} width={BAR_W} height={6} rx={1} fill="var(--grid)" />
            <rect
              x={left + BAR_X}
              y={y}
              width={Math.max(0, p * BAR_W)}
              height={6}
              rx={1}
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
                y1={y - 2}
                y2={y + 8}
                stroke="var(--ink-2)"
                strokeWidth={1}
              />
            )}
            {/* Locked rows keep showing their number — it is the thing being
                set — and say "locked" by going muted next to the hollow bar,
                rather than by a glyph competing with the digits. */}
            <text
              className="node-value"
              x={left + NODE_W - 9}
              y={y + 6}
              textAnchor="end"
              fill={pinned === undefined ? undefined : 'var(--muted)'}
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
  const { index, bet, heading, sub, onToggleLock } = props
  const betAt = (card: Card, history: string) => bet[infosetKey(card, history)][index]
  const centre = (key: string) =>
    DECISION_NODES.find((n) => n.key === key) ?? TERMINAL_NODES.find((n) => n.key === key)!
  const isDecision = (key: string) => DECISION_NODES.some((n) => n.key === key)

  return (
    <section className="panel wide" aria-label="Game tree and strategy">
      <h2>{heading}</h2>
      <p className="sub">{sub}</p>
      <p className="card-legend">
        {CARDS.map((card) => (
          <span className="chip" key={card}>
            <span className="dot" style={{ background: CARD_VAR[card] }} />
            {CARD_SYMBOL[card]}
          </span>
        ))}
        {props.hairline && (
          <span className="chip">
            <span className="tick" /> target
          </span>
        )}
        <span className="chip">
          {onToggleLock
            ? 'click any row to lock that spot'
            : 'edge width = how often that action is taken, over the hands that reach it'}
        </span>
      </p>
      <div className="tree-wrap">
        <div className="tree">
          <svg viewBox="0 0 640 500" role="img" aria-label="Kuhn poker game tree">
            {EDGES.map((edge) => {
              const from = centre(edge.from)!
              const to = centre(edge.to)!
              const p = actionFrequency(edge.from, edge.act, betAt)
              const y1 = from.cy + NODE_H / 2
              const y2 = isDecision(edge.to) ? to.cy - NODE_H / 2 : to.cy - 11
              return (
                <g key={`${edge.from}-${edge.act}`}>
                  <path
                    d={`M ${from.cx} ${y1} C ${from.cx} ${y1 + 26}, ${to.cx} ${y2 - 26}, ${to.cx} ${y2}`}
                    fill="none"
                    stroke="var(--axis)"
                    strokeWidth={1 + 5 * p}
                    strokeLinecap="round"
                    opacity={0.55}
                  />
                  <text
                    className="edge-label"
                    x={(from.cx + to.cx) / 2}
                    y={(y1 + y2) / 2 + 3}
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
                  x={node.cx - 54}
                  y={node.cy - 11}
                  width={108}
                  height={22}
                  rx={11}
                />
                <text className="terminal-text" x={node.cx} y={node.cy + 4} textAnchor="middle">
                  {node.text}
                </text>
              </g>
            ))}
            {DECISION_NODES.map((node) => (
              <NodeCard key={node.key || 'root'} node={node} {...props} />
            ))}
          </svg>
        </div>
      </div>
    </section>
  )
}
