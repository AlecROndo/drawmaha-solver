import { CARDS, CARD_NAME, CARD_SYMBOL, DECISIONS, infosetKey, type Card } from '../kuhn'
import { Figure } from './site'
import { SOLVE } from './useSolve'

/**
 * The answer sheet: CFR's strategy at all twelve information sets, spot by
 * spot, against Kuhn's closed form.
 *
 * The tree draws the same twelve numbers as bars; this is the version that
 * can be read to the third decimal. Rows are the three cards, columns the four
 * spots where somebody acts, and each cell is CFR's average at the playhead
 * over the closed form. The closed form is a one-parameter family in α, the
 * jack's opening bluff rate, so it is instantiated at the α this run found —
 * which makes the jack's opening cell agree by construction, and every other
 * cell a real test.
 */

const LAST = SOLVE.iterations.length - 1

const SPOT_NAME: Record<(typeof DECISIONS)[number], string> = {
  '': 'Opening',
  p: 'Checked to',
  b: 'Facing a bet',
  pb: 'Check, then a bet',
}

/** The largest |CFR − closed form| over the twelve spots at one checkpoint. */
const largestGap = (index: number): number =>
  Math.max(...Object.keys(SOLVE.closedForm).map((key) => Math.abs(SOLVE.bet[key][index] - SOLVE.closedForm[key])))

const FINAL_GAP = largestGap(LAST)

const signed = (v: number, digits: number) => (v < 0 ? `−${Math.abs(v).toFixed(digits)}` : v.toFixed(digits))

export function AnswerSheet({ index }: { index: number }) {
  const t = SOLVE.iterations[index]
  const gap = largestGap(index)
  return (
    <Figure
      n="Figure 3"
      title={`At 100,000 iterations all twelve spots sit within ${FINAL_GAP.toFixed(3)} of the closed form`}
      sub={
        <>
          P(bet), or P(call) facing a bet, at iteration <b>{t.toLocaleString()}</b>: CFR's average over the closed form
        </>
      }
      caption={`The closed form is a family in α, the jack's opening bluff rate; it is read at the α = ${SOLVE.alpha.toFixed(3)} this run found, so the jack's opening cell agrees by construction and the other eleven are the test.`}
      label="Answer sheet: CFR's strategy against the closed form"
    >
      <table className="hair sheet">
        <thead>
          <tr>
            <th>Card</th>
            {DECISIONS.map((h) => (
              <th key={h} className="r">
                {SPOT_NAME[h]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CARDS.map((card) => (
            <tr key={card}>
              <td>
                <span className={`card c-${CARD_SYMBOL[card].toLowerCase()}`}>{CARD_SYMBOL[card]}</span>{' '}
                <span className="spot">{CARD_NAME[card]}</span>
              </td>
              {DECISIONS.map((h) => {
                const key = infosetKey(card as Card, h)
                return (
                  <td key={h} className="r cell">
                    <span className="cfr">{SOLVE.bet[key][index].toFixed(3)}</span>
                    <span className="exact">{SOLVE.closedForm[key].toFixed(3)}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="stats">
        <div className="stat">
          <dt>Largest gap now</dt>
          <dd>{gap.toFixed(4)}</dd>
        </div>
        <div className="stat">
          <dt>Game value at 100k</dt>
          <dd>
            {signed(SOLVE.gameValue, 5)} <small>vs −1/18</small>
          </dd>
        </div>
      </dl>
    </Figure>
  )
}
