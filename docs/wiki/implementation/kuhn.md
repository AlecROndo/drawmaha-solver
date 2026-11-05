---
title: Kuhn poker
description: The first game tree — twelve ledgers, vanilla CFR's two-direction walk, an exact meter over 64 pure strategies, and an exploit finder built from one extra parameter.
---

Kuhn poker is the first game with a tree, and so the first where the ledger's utilities have to be *computed*. The deck has three cards, J < Q < K. Both players ante 1, each gets one card, and one round of betting follows with a single bet size of 1. The whole state of a solve is 12 ledgers of 4 numbers each. Its equilibrium has a closed form, so every number the solver produces can be checked on paper.

## Eight files; the meter imports only the rules

| File | What it is for |
| --- | --- |
| [`game.py`](repo:src/drawmaha_solver/kuhn/game.py) | The rules: deal, betting, the tree, infosets, payoffs. No solver code. |
| [`infoset_table.py`](repo:src/drawmaha_solver/kuhn/infoset_table.py) | The solver's memory: twelve ledgers, pre-allocated |
| [`cfr.py`](repo:src/drawmaha_solver/kuhn/cfr.py) | Vanilla CFR, the recursive walk that feeds the ledgers |
| [`exploitability.py`](repo:src/drawmaha_solver/kuhn/exploitability.py) | The exact meter; imports `game.py` only |
| [`exploiter.py`](repo:src/drawmaha_solver/kuhn/exploiter.py) | Lock a strategy and let CFR find the most that beats it |
| [`analysis.py`](repo:src/drawmaha_solver/kuhn/analysis.py) | `kuhn-analysis`: one solve, three figures, the answer sheet, `--json` for the site |
| [`play.py`](repo:src/drawmaha_solver/kuhn/play.py) | `kuhn-play`: hands against the solved equilibrium |
| [`exploit_server.py`](repo:src/drawmaha_solver/kuhn/exploit_server.py) | `kuhn-exploit-server`: the local endpoint behind `/rung1#exploit` |

## The rules: four decisions, two actions, twelve infosets

`game.py` gives a solver its vocabulary:

- **`Action`** is PASS or BET at every node, so a ledger is always two wide. **`action_label`** renders the poker word for display: PASS is a check with nothing to call and a fold facing a bet, and BET is a bet or a call.
- **`KuhnState`** is one node of the tree: the deal plus the actions so far. It is frozen, so **`apply`** returns a new node and a walker never has to undo a move. **`returns`** pays in chips with the ante as 1 and sums to zero.
- **`InfoSet`** is what the player to act knows: their own card and the public history, and deliberately *not* the opponent's card. Two deals that differ only in the opponent's card must collapse to one equal, equally hashing key.
- **`all_infosets`** lists the 12 = 4 decision nodes × 3 cards. `DEALS` lists the six equally likely deals; chance acts only once, at the root.

## The memory: a 4 × 3 grid of ledgers

**`new_infoset_table`** builds a `dict[InfoSet, RegretMatcher]` with all twelve entries, so no code path can invent a thirteenth. **`current_strategy`** and **`average_strategy`** read the whole table at once. **`format_strategy_grid`** prints P(BET) as the 4 × 3 grid, with one row per decision node and one column per card. Moving down a column is the same hand in a different spot. Moving across a row is the same spot with a different hand, which is exactly what the opponent cannot see.

## The walk: reach goes down, values come up

`cfr.py` is the only new idea of the game. **`walk`** recurses through the tree carrying two quantities in opposite directions:

- **Down: reach.** `reach[p]` is the product of player p's own action probabilities along the path, and nothing else.
- **Up: values.** What the node is worth to each seat, as a plain conditional expectation with no reach weighting.

The weights are applied once, when a ledger is banked: regret at the counterfactual reach, the average at the acting player's own reach (see [The regret ledger](wiki:implementation/ledger)). `walk` returns the value to *both* seats, so nothing is ever negated on the way up. **`run_iteration`** walks the tree once per deal. **`train`** repeats that for the requested number of iterations and returns the table. The full derivation is in [CFR](wiki:theory/cfr).

:::note
There is no seed. Vanilla CFR enumerates the whole tree and never samples, so two runs agree bit for bit, and the committed `figures/rung1/` doubles as a regression fingerprint for `cfr.py`. A change that moves any number shows up as a dirty working tree.
:::

## The meter: all 64 pure strategies, exactly

`exploitability.py` measures what a perfect adversary wins, `(BR₀ + BR₁) / 2`, which is zero exactly at Nash. Each player owns 6 infosets with 2 actions, so **`best_response_value`** enumerates all 2⁶ = 64 pure strategies for the responder and keeps the best one. Mixed play is handled by weighting both branches at every node, and the hidden card by enumerating all six deals at ⅙, so every number is an exact expectation. **`best_response`** returns the counter-strategy itself, **`expected_value`** is what each seat earns when both play a given profile, and **`exploitability`** averages the two best responses.

Enumerating pure strategies *over infosets* is what keeps the adversary honest. A pure strategy fixes one action per infoset before any card is dealt, so an opponent who sees your card cannot be written down. A test pins that a "peeking" meter would score the true equilibrium at 0.278 chips a hand.

## The exploit finder: CFR against something that cannot move

`exploiter.py` asks the opposite question, "what beats *this*?", with no new algorithm. **`LockedStrategy`** pins some infosets to probabilities you supply. They never bank an update, but their probabilities still drive the reach, so they still enter the learner's counterfactual weight. Minimising regret against a player who cannot move back converges to a **best response** instead of to Nash. **`exploit`** runs that training and samples **`Checkpoint`**s along the way. **`ExploitResult`** carries the grade: **`gap`** is how far CFR still is from the exact ceiling the meter computes. **`locked_seat`** tells the two cases apart. A whole locked seat has a ground-truth ceiling. A partial lock leaves the rest of that seat learning, which gives a constrained equilibrium rather than a best response, so no ceiling is claimed.

## Three commands: solve, play, serve the Exploit tab

| Command | What it does |
| --- | --- |
| `uv run kuhn-analysis` | **`run`** solves to 100k iterations, measuring both candidate answers at 60 log-spaced checkpoints (**`Trajectory`**); three figures go to `figures/rung1/`; **`to_json`** / **`write_json`** export the solve the `/rung1` page draws |
| `uv run kuhn-play` | **`main`** solves, then **`play_hand`** deals hands against the *average* strategy, seats alternating because the first seat is worth −1/18; **`bot_action`** samples only from the bot's own infoset |
| `uv run kuhn-exploit-server` | stdlib HTTP; **`solve`** turns a POST of locked spots (**`parse_locked`**) into an exploit run and returns the trace; **`nash_value_against`** adds what equilibrium play earns against the same lock |

The answer, graded: 0.00063 chips a hand after 100k iterations, game value −0.05555 against −1/18, with the jack bluffing at α = 0.221 and the king opening at 0.661 ≈ 3α.

![Only the average strategy converges](fig:rung1/exploitability.png)

:::cards
- [CFR](wiki:theory/cfr) The walk's derivation and the two reaches.
- [Exploitability](wiki:theory/exploitability) Why the meter must be independent, and why it maximises over infosets, not deals.
- [Leduc poker](wiki:implementation/leduc) The next game: chance in the middle of the tree.
:::
