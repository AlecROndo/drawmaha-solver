import { describe, expect, test } from 'vitest'
import { PAYOFF, ROCK, PAPER, SCISSORS, utilityVs, exploitability } from './game'

describe('PAYOFF', () => {
  test('matches the spec matrix (rock loses to paper, beats scissors, ...)', () => {
    expect(PAYOFF).toEqual([
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ])
  })

  test('is zero-sum antisymmetric', () => {
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) expect(PAYOFF[a][b] + PAYOFF[b][a]).toBe(0)
  })
})

describe('utilityVs', () => {
  test('returns the column PAYOFF[:, b] — payoff of each own action vs revealed b', () => {
    expect(utilityVs(PAPER)).toEqual([-1, 0, 1])
    expect(utilityVs(ROCK)).toEqual([0, 1, -1])
    expect(utilityVs(SCISSORS)).toEqual([1, -1, 0])
  })
})

describe('exploitability', () => {
  test('uniform strategy is unexploitable (Nash): exactly 0', () => {
    expect(exploitability([1 / 3, 1 / 3, 1 / 3])).toBeCloseTo(0, 12)
  })

  test('pure rock is fully exploitable: 1', () => {
    expect(exploitability([1, 0, 0])).toBe(1)
  })

  test('pure paper and pure scissors are also fully exploitable', () => {
    expect(exploitability([0, 1, 0])).toBe(1)
    expect(exploitability([0, 0, 1])).toBe(1)
  })

  test('is never negative (best responder can always pick the max)', () => {
    expect(exploitability([0.4, 0.3, 0.3])).toBeGreaterThanOrEqual(0)
  })
})
