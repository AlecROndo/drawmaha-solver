import { describe, expect, test } from 'vitest'
import { Engine } from './engine'
import { exploitability, PAPER, ROCK, SCISSORS } from './game'
import { averageStrategy } from './ledger'

describe('Engine self-play (acceptance §6.2)', () => {
  test('100k iterations: average within 0.01 of 1/3 each, exploitability ≤ 0.005', () => {
    const e = new Engine({ mode: 'self-play', seed: 42 })
    e.step(100_000)
    const avg = averageStrategy(e.ledgers[0])
    for (const p of avg) expect(Math.abs(p - 1 / 3)).toBeLessThanOrEqual(0.01)
    expect(exploitability(avg)).toBeLessThanOrEqual(0.005)
    // Player 1 learns too.
    const avg1 = averageStrategy(e.ledgers[1])
    expect(exploitability(avg1)).toBeLessThanOrEqual(0.005)
  })
})

describe('Engine vs fixed opponent (acceptance §6.3)', () => {
  test('vs 50% rock / 25% paper / 25% scissors: average → pure paper, EV ≈ +0.24', () => {
    const e = new Engine({
      mode: 'vs-fixed',
      seed: 42,
      fixedDist: [0.5, 0.25, 0.25],
    })
    e.step(100_000)
    const avg = averageStrategy(e.ledgers[0])
    expect(avg[PAPER]).toBeGreaterThanOrEqual(0.99)
    const evPerRound = e.chips / e.iteration
    expect(evPerRound).toBeGreaterThan(0.22)
    expect(evPerRound).toBeLessThan(0.27)
  })
})

describe('Engine determinism (acceptance §6.4)', () => {
  test('same seed → bit-identical trajectory', () => {
    const a = new Engine({ mode: 'self-play', seed: 7 })
    const b = new Engine({ mode: 'self-play', seed: 7 })
    a.step(10_000)
    // Different batching must not change the trajectory.
    for (let i = 0; i < 100; i++) b.step(100)
    expect(a.ledgers[0].R).toEqual(b.ledgers[0].R)
    expect(a.ledgers[0].S).toEqual(b.ledgers[0].S)
    expect(a.ledgers[1].R).toEqual(b.ledgers[1].R)
    expect(a.iteration).toBe(b.iteration)
  })

  test('different seed → different trajectory', () => {
    const a = new Engine({ mode: 'self-play', seed: 7 })
    const b = new Engine({ mode: 'self-play', seed: 8 })
    a.step(1000)
    b.step(1000)
    expect(a.ledgers[0].R).not.toEqual(b.ledgers[0].R)
  })
})

describe('Engine vs-you mode', () => {
  test('playUserAction scores chips from the bot side and updates the bot ledger on the user action', () => {
    const e = new Engine({ mode: 'vs-you', seed: 1 })
    const r = e.playUserAction(ROCK)
    expect([ROCK, PAPER, SCISSORS]).toContain(r.botAction)
    expect(r.userAction).toBe(ROCK)
    expect(e.iteration).toBe(1)
    expect(e.ledgers[0].n).toBe(1)
    // Bot observed rock: its regret toward paper must be the top increment.
    expect(e.ledgers[0].R[PAPER]).toBeGreaterThanOrEqual(e.ledgers[0].R[ROCK])
    expect(e.ledgers[0].R[PAPER]).toBeGreaterThanOrEqual(e.ledgers[0].R[SCISSORS])
  })

  test('bot exploits a pure-rock user habit', () => {
    const e = new Engine({ mode: 'vs-you', seed: 1 })
    for (let i = 0; i < 500; i++) e.playUserAction(ROCK)
    const avg = averageStrategy(e.ledgers[0])
    expect(avg[PAPER]).toBeGreaterThan(0.9)
    expect(e.chips).toBeGreaterThan(0) // bot is winning
  })
})

describe('Decimated history (spec §5)', () => {
  test('keeps all of the first 1k, every 10th to 10k, every 100th to 100k', () => {
    const e = new Engine({ mode: 'self-play', seed: 3 })
    e.step(100_000)
    const ts = e.history.map((h) => h.t)
    const inRange = (lo: number, hi: number) => ts.filter((t) => t > lo && t <= hi)
    expect(inRange(0, 1000)).toHaveLength(1000)
    expect(inRange(1000, 10_000)).toHaveLength(900)
    expect(inRange(10_000, 100_000)).toHaveLength(900)
    expect(ts[ts.length - 1]).toBe(100_000)
    // History carries what the charts need.
    const last = e.history[e.history.length - 1]
    expect(last.avg).toHaveLength(3)
    expect(last.cur).toHaveLength(3)
    expect(last.exploit).toBeGreaterThanOrEqual(0)
  })
})

describe('Reset / reproducibility of runs', () => {
  test('a fresh engine with the same seed reproduces the same history', () => {
    const a = new Engine({ mode: 'self-play', seed: 99 })
    a.step(5000)
    const b = new Engine({ mode: 'self-play', seed: 99 })
    b.step(5000)
    expect(a.history).toEqual(b.history)
  })
})
