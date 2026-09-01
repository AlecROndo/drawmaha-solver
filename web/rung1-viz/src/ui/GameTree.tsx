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
import { SOLVE } from './useSolve'

/**
 * The whole game and the whole strategy in one picture.
 *
 * Kuhn's tree is four decision nodes and five terminals, so it fits on screen
 * intact — and every infoset can be drawn where it actually sits. Each node
 * carries three bars, one per card the acting player might hold, showing how
 * often they bet there. The twelve bars are the twelve information sets: this
 * is the strategy table, laid out as the tree it belongs to.
 *
 * The hairline on each bar is the closed-form answer at the alpha this run
 * found. Watching the fills walk onto their hairlines is the solve.
 */

const CARD_VAR = ['var(--jack)', 'var(--queen)', 'var(--king)']

const BAR_X = 40
// Leaves room for the value text: a bar at 1.00 must not run under it, or the
// closed-form hairline lands on top of the digits.
const BAR_W = 54
const ROW_H = 14

/** P(bet) at one infoset, at the checkpoint being shown. */
const betAt = (index: number) => (card: Card, history: string) =>
  SOLVE.bet[infosetKey(card, history)][index]

function NodeCard({ node, index }: { node: (typeof DECISION_NODES)[number]; index: number }) {
  const left = node.cx - NODE_W / 2
  const top = node.cy - NODE_H / 2
  return (
    <g>
      <rect
        className="node-card"
        x={left}
        y={top}
        width={NODE_W}
        height={NODE_H}
        rx={3}
      />
      <text className="node-caption" x={left + 10} y={top + 15}>
        {node.caption}
      </text>
      {CARDS.map((card, row) => {
        const key = infosetKey(card as Card, node.key)
        const p = SOLVE.bet[key][index]
        const exact = SOLVE.closedForm[key]
        const y = top + 28 + row * ROW_H
        return (
          <g key={key}>
            <text
              className="node-letter"
              x={left + 10}
              y={y + 5}
              fill={CARD_VAR[card]}
            >
              {CARD_SYMBOL[card]}
            </text>
            <rect
              x={left + BAR_X}
              y={y}
              width={BAR_W}
              height={6}
              rx={1}
              fill="var(--grid)"
            />
            <rect
              x={left + BAR_X}
              y={y}
              width={Math.max(0, p * BAR_W)}
              height={6}
              rx={1}
              fill={CARD_VAR[card]}
            />
            {/* Where the closed form says this bar belongs. */}
            <line
              x1={left + BAR_X + exact * BAR_W}
              x2={left + BAR_X + exact * BAR_W}
              y1={y - 2}
              y2={y + 8}
              stroke="var(--ink-2)"
              strokeWidth={1}
            />
            <text className="node-value" x={left + NODE_W - 9} y={y + 6} textAnchor="end">
              {p.toFixed(2)}
            </text>
          </g>
        )
      })}
    </g>
  )
}

export function GameTree({ index }: { index: number }) {
  const bet = betAt(index)
  const centre = (key: string) =>
    DECISION_NODES.find((n) => n.key === key) ?? TERMINAL_NODES.find((n) => n.key === key)!
  const isDecision = (key: string) => DECISION_NODES.some((n) => n.key === key)

  return (
    <section className="panel wide" aria-label="Game tree and strategy">
      <h2>Fig. 1 · The game, with the strategy drawn on it</h2>
      <p className="sub">
        four decision nodes × three possible cards = the twelve information sets · bar is
        P(bet), which reads as P(call) at the two nodes facing a bet · hairline is the closed
        form
      </p>
      <p className="card-legend">
        {CARDS.map((card) => (
          <span className="chip" key={card}>
            <span className="dot" style={{ background: CARD_VAR[card] }} />
            {CARD_SYMBOL[card]}
          </span>
        ))}
        <span className="chip">
          <span className="tick" /> closed form
        </span>
        <span className="chip">
          edge width = how often that action is taken, over the hands that reach it
        </span>
      </p>
      <div className="tree-wrap">
        <div className="tree">
          <svg viewBox="0 0 640 500" role="img" aria-label="Kuhn poker game tree">
            {EDGES.map((edge) => {
              const from = centre(edge.from)!
              const to = centre(edge.to)!
              const p = actionFrequency(edge.from, edge.act, bet)
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
                <text
                  className="terminal-text"
                  x={node.cx}
                  y={node.cy + 4}
                  textAnchor="middle"
                >
                  {node.text}
                </text>
              </g>
            ))}
            {DECISION_NODES.map((node) => (
              <NodeCard key={node.key || 'root'} node={node} index={index} />
            ))}
          </svg>
        </div>
      </div>
    </section>
  )
}
