import { describe, expect, test } from 'vitest'
import { newLedger, strategy, averageStrategy, update } from './ledger'
import { utilityVs, PAPER } from './game'

describe('ledger', () => {
  test('fresh ledger plays uniform and has uniform average', () => {
    const L = newLedger()
    expect(strategy(L)).toEqual([1 / 3, 1 / 3, 1 / 3])
    expect(averageStrategy(L)).toEqual([1 / 3, 1 / 3, 1 / 3])
  })

  test('worked example: fresh ledger, opponent plays paper', () => {
    // u = (−1, 0, +1), ⟨σ,u⟩ = 0, increments = (−1, 0, +1), next σ = (0, 0, 1).
    const L = newLedger()
    const u = utilityVs(PAPER)
    const trace = update(L, u)

    expect(trace.expectedUtility).toBeCloseTo(0, 12)
    expect(trace.increments).toEqual([-1, 0, 1])
    expect(L.R).toEqual([-1, 0, 1])
    expect(L.n).toBe(1)
    expect(strategy(L)).toEqual([0, 0, 1])
  })

  test('regret baseline is expected utility ⟨σ,u⟩, not the sampled action utility', () => {
    // Force a non-uniform σ = (0, 0, 1), then face rock: u = (0, 1, −1).
    // Expected-utility baseline: ⟨σ,u⟩ = −1 → increments (1, 2, 0).
    // A sampled-action variant could not produce increment 0 for scissors here.
    const L = newLedger()
    update(L, utilityVs(PAPER)) // σ becomes (0, 0, 1)
    const trace = update(L, [0, 1, -1])
    expect(trace.expectedUtility).toBeCloseTo(-1, 12)
    expect(trace.increments).toEqual([1, 2, 0])
    expect(L.R).toEqual([0, 2, 1])
  })

  test('strategy falls back to uniform when no regret is positive', () => {
    const L = newLedger()
    L.R = [-2, -1, 0]
    expect(strategy(L)).toEqual([1 / 3, 1 / 3, 1 / 3])
  })

  test('strategy sum S accumulates the acting σ and average is S/n', () => {
    const L = newLedger()
    update(L, utilityVs(PAPER)) // acted with uniform σ
    expect(L.S).toEqual([1 / 3, 1 / 3, 1 / 3])
    update(L, utilityVs(PAPER)) // acted with (0,0,1)
    expect(L.S).toEqual([1 / 3, 1 / 3, 1 / 3 + 1])
    expect(averageStrategy(L)).toEqual([1 / 6, 1 / 6, (1 / 3 + 1) / 2])
  })
})
