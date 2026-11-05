---
title: Full Drawmaha, the target game
description: The game the project exists to solve, what changes from mini-drawmaha, and how a solve will be graded once no exact grader can run.
---

Everything so far — four solved games and two solves of mini-drawmaha — exists so that a method can meet full Drawmaha already checked. This page states the target, measures how much bigger it is than anything solved, and lays out what the solve needs that mini-drawmaha did not. Nothing here is built yet except the range interface at [`/solver`](/solver), which runs on the full game with an illustrative action mix.

## The game: a bomb pot with one draw and a pot that pays twice

Standard Drawmaha, heads-up, as the project pins it (README, "The target game"):

- Each player is dealt **5 hole cards**.
- It is a **bomb pot**: both players ante and the flop comes out with **no preflop betting**. The order is ante → **flop** (3 cards) → betting → **one draw** (discard 0–5, replace) → **turn** → betting → **river** → betting. Three betting rounds, not four.
- **The draw-1 rule.** A player who draws exactly one card gets it dealt **face up**; they may keep it or reject it and take the next card face down.
- **The split pot.** Half goes to the best **Omaha** hand (exactly 2 hole cards + 3 board cards), half to the best **five-card poker hand made of the hole cards alone**.

The solve's v1 defaults, all configurable: pot-limit with the action set {fold, check/call, pot}; 100 big blinds effective; both players ante, no blinds. The bet-size menu is a config, so a multi-size solve is a re-solve, not a rewrite. [The Rules page](/rules) walks a hand through both readings.

## What changes from mini-drawmaha, and by how much

Mini-drawmaha was designed to keep the two things that make Drawmaha Drawmaha — a draw, and one pot read two ways — on a deck small enough to solve exactly.

| | Mini-drawmaha | Full Drawmaha |
| --- | --- | --- |
| Deck | 15 cards: ranks 2–6, three suits | 52 cards |
| Hole cards | 3 | 5 |
| Board | 2 cards, one per round | 5 cards: flop, turn, river |
| Betting rounds | 2 | 3 |
| Draw | throw at most one card, replaced face down | throw 0–5 (32 discard subsets); a one-card draw is dealt face up |
| The two halves | best 3 held (inner); best 2 held + both board cards (outer) | best 5 held (draw half); best 2 held + 3 board (Omaha half) |
| Stacks | 26 chips, ante 1 | 100 big blinds |
| Holdings a player can be dealt | 455 (three of fifteen) | 2,598,960, against 1,326 in hold'em and 270,725 in Omaha |
| Infosets | 6,220,050 — storable as a table | far beyond any table |
| Exact grader | yes: a best-response walk over the whole game | no |

Two of those rows end the methods that solved mini-drawmaha. **The table cannot be stored**, which is why rung 4 existed: Deep CFR replaces the 6.2M ledgers with a network, and the full game has no table to replace. **The exact grader cannot run**, and that one is harder, because the grader is what every result so far has been trusted on.

Suit isomorphism helps, but less than it seems: the 2,598,960 five-card holdings collapse to 134,459 suit-isomorphic classes before the deal (`docs/research/05-abstraction-bucketing.md`), and the factor shrinks as the board and the kept cards fix suits.

## What carries over unchanged

- **The algorithm.** External-sampling Deep CFR with linear weighting, ÷ √pot regret targets, reservoir memories and a from-scratch refit — the configuration the [run log](wiki:research/run-log) converged on. The rule that came out of rungs 3 and 4 holds: anything sampled uses linear weighting, and DCFR or CFR+ need a reason to be tried again ([regret rules](wiki:theory/regret-rules)).
- **Π as the answer.** Mini-drawmaha could keep an exact average in a table; the full game cannot. The policy net Π — fit at each mark on the stored strategy samples — is the only average it can keep, and on mini-drawmaha Π was always more exploitable than the exact average (0.103 against 0.068 at 20M hands). How good Π can be made is a full-game question that mini-drawmaha can still answer cheaply.
- **The rule that a grader never shares code with what it grades.** At full scale the grader is an exploiter rather than a walk, but it must still be written against the rules alone.
- **The range interface.** [`/solver`](/solver) already samples 50,000 five-card holdings for a flop, classifies each twice (the five cards alone, and the best two with the board), and draws the range as a grid of inner rows × outer columns. The composition is real to sampling error; the action mix is a smooth function of each hand's two equities and is labelled illustrative. [`web/solver-viz/src/engine/policy.ts`](repo:web/solver-viz/src/engine/policy.ts) is the one file that changes when the network exists.

## What has to be built

1. **An engine for the full game**: deck, streets, the draw with its face-up rule, both scorers, terminal payoffs including odd chips and double ties.
2. **An encoder.** Mini-drawmaha's net reads 136 features; the full game needs a representation of five hole cards, the discard, a five-card board, three rounds of betting, and both players' public draw counts. Which cards were kept matters, not only how many — draw-game solvers found discard blockers change strategies.
3. **A discard head.** Thirty-two keep/discard subsets as a flat output, or per-card logits, and how regret matching behaves over them, is an open decision ([open questions](wiki:research/open-questions)).
4. **Training at scale.** The refit is already about 95% of an iteration on mini-drawmaha, so the full game's cost is mostly fitting a bigger net on more rows.
5. **An evaluation that needs no exact grader** — below.

## How a full-game solve will be graded: pre-registered before training

Training loss is not a progress meter in self-play; the only honest question is what an adversary wins. The README's protocol, fixed before any full-game code exists:

- **A duplicate-dealt head-to-head checkpoint ladder.** Each checkpoint plays the next with the cards dealt twice, seats swapped, so card luck cancels and what remains is skill. It shows progress, not distance from equilibrium.
- **A trained RL exploiter against the frozen net**, reported as a **lower bound on exploitability**. A best response found by training is never better than the true one, so whatever it wins, the true exploitability is at least that.
- **Units**: total chips across both pot halves, per hand, in big blinds, with per-half EV and scoop rate as split-pot diagnostics.

On mini-drawmaha both ends of this can be checked against the exact grader first — how close a trained exploiter comes to the true best response there is a measurement that can be made today.

## Where the plans live

Full Drawmaha has no design document yet. The rung-4 spec, [`docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md`](repo:docs/superpowers/specs/2026-10-07-deep-cfr-m1-design.md), lists the full game's engine and encoder as out of scope for its milestone, and its PR table is the live record for current work. The range interface was planned in [`docs/superpowers/plans/2026-10-07-solver-range-interface.md`](repo:docs/superpowers/plans/2026-10-07-solver-range-interface.md); the ten design studies it was placed from are under `docs/research/range-ui-studies/`. Per the docs convention, a change of direction gets a new dated spec — the full-game solve will start with one.

:::cards
- [Where the project stands](wiki:research/status) What is solved today.
- [Open questions](wiki:research/open-questions) The decisions the full game forces.
- [The range interface](/solver) The full game's interface, built ahead of its network.
- [The rules](/rules) Drawmaha, one hand read two ways.
:::
