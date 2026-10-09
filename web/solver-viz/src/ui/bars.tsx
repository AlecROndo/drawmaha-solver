/**
 * The small marks every view of the range shares: a card face, the
 * fold/call/pot mix bar and its key, and the number formats.
 *
 * A "mix" is the solver's action distribution over a region of the range —
 * fold, check/call, pot — and it is drawn the same way everywhere on the
 * page: fold in the quiet ink, call in yellow (the thing you are pointing
 * at), pot in pink (the aggressive act), left to right, with a 2 px gap of
 * surface between bands so adjacent fills never need a border.
 */

import type { Mix } from '../engine/aggregate'
import type { GameUI } from './game'

export const pct = (x: number): string => `${(x * 100).toFixed(x * 100 >= 10 || x === 0 ? 0 : 1)}%`
export const pct0 = (x: number): string => `${Math.round(x * 100)}%`
export const count = (n: number): string => n.toLocaleString('en-US')

/** Cards sorted high to low so a hand reads the way a player fans it. */
export const byRankDesc = (cards: readonly number[]): number[] => [...cards].sort((a, b) => b - a)

/** One card face, spelled by the game. `size` is `sm` for a hand in a row of text, `lg` for the holding strip. */
export function CardFace({ game, card, size = 'sm', ghost }: { game: GameUI; card?: number; size?: 'sm' | 'lg'; ghost?: boolean }) {
  if (card === undefined || ghost) return <span className={`cf ${size} ghost`} aria-hidden />
  return <span className={`cf ${size}${game.isRed(card) ? ' red' : ''}`}>{game.label(card)}</span>
}

/** A fanned hand, high card first. */
export function Hand({ game, cards, size = 'sm' }: { game: GameUI; cards: readonly number[]; size?: 'sm' | 'lg' }) {
  return (
    <span className="hand">
      {byRankDesc(cards).map((c) => (
        <CardFace key={c} game={game} card={c} size={size} />
      ))}
    </span>
  )
}

/** The stacked mix bar. `h` is its height in px; the bands are proportional. */
export function MixBar({ mix, h = 8, className }: { mix: Mix; h?: number; className?: string }) {
  return (
    <span className={['mixbar', className ?? ''].filter(Boolean).join(' ')} style={{ height: h }} aria-hidden>
      <i className="f" style={{ width: `${mix.f * 100}%` }} />
      <i className="c" style={{ width: `${mix.c * 100}%` }} />
      <i className="p" style={{ width: `${mix.p * 100}%` }} />
    </span>
  )
}

/** The three numbers under a mix bar. `node` decides whether the yellow band is a check or a call. */
export function MixKey({ mix, node }: { mix: Mix; node: 'facing' | 'open' }) {
  return (
    <span className="mixkey">
      {node === 'facing' && (
        <span className="f">
          <i />
          fold <b>{pct0(mix.f)}</b>
        </span>
      )}
      <span className="c">
        <i />
        {node === 'facing' ? 'call' : 'check'} <b>{pct0(mix.c)}</b>
      </span>
      <span className="p">
        <i />
        pot <b>{pct0(mix.p)}</b>
      </span>
    </span>
  )
}

/** The action a mix band stands for, in the node's own words. */
export const actionWord = (key: 'f' | 'c' | 'p', node: 'facing' | 'open'): string =>
  key === 'f' ? 'fold' : key === 'c' ? (node === 'facing' ? 'call' : 'check') : 'pot'
