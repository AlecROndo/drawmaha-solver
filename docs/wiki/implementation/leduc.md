---
title: Leduc poker
description: Chance in the middle of the tree, ledgers of varying width, a best-response recursion that replaces enumeration, and OpenSpiel as a frozen referee.
---

Leduc poker has six cards, two each of J, Q and K. Each player gets one, a betting round follows, then one public board card, then a second betting round at double the stake. Two things change from Kuhn, and between them they are the whole game. The deck now acts *in the middle* of the hand, so the walk meets a chance node mid-tree. And spots no longer all offer the same actions, so ledgers are two or three wide depending on where they sit. Leduc has 288 infosets against Kuhn's 12, and its equilibrium has no closed form, so ground truth comes from an outside implementation.

## Files

| File | What it is for |
| --- | --- |
| [`game.py`](repo:src/drawmaha_solver/leduc/game.py) | The rules: two rounds, the board, legal actions, infosets keyed by rank |
| [`infoset_table.py`](repo:src/drawmaha_solver/leduc/infoset_table.py) | 288 ledgers, each sized by its own legal actions |
| [`cfr.py`](repo:src/drawmaha_solver/leduc/cfr.py) | Vanilla CFR with a chance branch inside the walk |
| [`exploitability.py`](repo:src/drawmaha_solver/leduc/exploitability.py) | The exact meter, now a recursion rather than an enumeration |
| [`exploiter.py`](repo:src/drawmaha_solver/leduc/exploiter.py) | Lock a strategy, find its best response — Kuhn's module at Leduc's scale |
| [`analysis.py`](repo:src/drawmaha_solver/leduc/analysis.py) | `leduc-analysis`: one solve, three figures, `--json` for `/rung2` |
| [`play.py`](repo:src/drawmaha_solver/leduc/play.py) | `leduc-play`: hands against the solved strategy |
| [`exploit_server.py`](repo:src/drawmaha_solver/leduc/exploit_server.py) | `leduc-exploit-server`: the endpoint behind `/rung2`'s exploit panel |
| [`scripts/leduc/generate_referee.py`](repo:scripts/leduc/generate_referee.py) | Interrogates OpenSpiel once and writes `tests/leduc/referee.json` |

## The rules: three actions, never three choices

- **`Action`** is FOLD, CALL or RAISE, but **`legal_actions_for`** makes folding illegal with nothing to call and raising illegal once the round's two-raise cap is spent. A node is therefore 2 or 3 wide, and `legal_actions()` is the authority on which.
- **`LeducState`** is one node: deal, board, both rounds' betting. **`kind`** reads which of the three things it is (**`NodeKind`**: decision, chance or terminal). At a chance node **`chance_outcomes`** lists each possible board card at probability ¼ and **`apply_chance`** turns it. **`contributions`** and **`returns`** do the chip accounting.
- **`InfoSet`** is the actor's own *rank*, the board's rank and both betting lines. Suits are dropped because Leduc has no flushes, so a seat's two jacks are the same hand. **`all_infosets`** yields 18 in round 1 and 270 in round 2.
- **`hand_strength`** ranks a private card against a board. A pair with the board beats everything, so a jack is trash until a jack lands.

## The memory: width is per key

**`new_infoset_table`** sizes each ledger by `len(infoset.legal_actions())`: 192 are two wide and 96 three wide. Entry k of a ledger belongs to `legal_actions()[k]`, *not* to `Action` value k. Column 0 means CALL at the open and FOLD facing a raise. Every loop over a ledger's columns pairs `enumerate(legal_actions())` with the ledger, and nothing else keeps a column's meaning in step with the rules. **`format_strategy_tables`** prints the 288 spots as four tables: round 1, then round 2 per board rank.

## The walk gains a chance branch

`cfr.py`'s **`walk`** now has three branches, and the chance branch does two different things:

- **For values, it averages.** Nobody chooses the board, so a chance node's value is the plain probability-weighted average of its children. Every round-1 regret above it depends on this average.
- **For weights, it multiplies a receipt.** Counterfactual reach includes chance, so the walk threads π_c down: 1/30 once the private cards are out, and 1/120 once the board lands. It rides in the regret weight only. The strategy weight is still the player's own reach.

One iteration walks 9,450 nodes in about 50 ms, so `leduc-analysis` defaults to 2,000 iterations, roughly two minutes. As with Kuhn there is no seed, and `figures/rung2/` is a regression fingerprint.

## The meter: sum over hidden worlds first, then take one max

A Leduc seat has about 6 × 10⁵¹ pure strategies, so Kuhn's enumeration is dead. **`best_response_value`** runs the recursion that was hiding inside the enumeration:

```eq
BRᵢ(I) = maxₐ Σ_{h ∈ I} π_c(h) · π₋ᵢ(h) · v(h · a)
```

Each piece is a part of the problem. I is the responder's infoset. h ranges over the hidden worlds it could be, which are the opponent's possible cards. π_c(h) · π₋ᵢ(h) is how often the deck and the frozen opponent produce that world. v(h · a) is what action a then earns there. The order is the point. The sum over worlds happens *inside* and the max *outside*, because one action has to be chosen for the whole infoset. Put the max inside and it picks a different action per world: that is an adversary who peeks at face-down cards, and it over-reports exploitability. **`best_response`** returns the counter-strategy, and **`expected_value`** and **`exploitability`** are as in Kuhn. See [Exploitability](wiki:theory/exploitability).

## The referee: OpenSpiel, interrogated once

`generate_referee.py` is the only file in the repo that imports OpenSpiel, and nothing imports it. It is run by hand:

```bash
uv run --with open_spiel --with cvxpy --with ecos --no-project \
    python3 scripts/leduc/generate_referee.py
```

It writes `tests/leduc/referee.json`: the node census, the collapse of OpenSpiel's 936 suited infosets onto our 288 ranked ones, the payoffs, the path probabilities, the sequence-form LP's game value and OpenSpiel's own CFR curve. `tests/leduc/test_referee.py` re-derives every fact from our code with no OpenSpiel present. OpenSpiel's wheels are large and the Vercel build must not pull them, so it is deliberately not a dependency.

## Entry points

| Command | What it does |
| --- | --- |
| `uv run leduc-analysis` | **`run`** solves and samples **`Trajectory`** checkpoints; three figures go to `figures/rung2/`: exploitability, game value against the LP's −0.0856, and three poker facts no equilibrium can miss (the nuts check-raise, a beaten hand folds to a raise, the king opens for a raise) |
| `uv run leduc-play` | hands against the solved strategy; **`turn_board`** draws the board at the deck's own odds, seats alternate (the first seat is worth about −0.086) |
| `uv run leduc-exploit-server` | as Kuhn's, with two changes for scale: runs capped at 2,000 iterations, and the Nash baseline is *read* from the committed `web/rung2-viz/src/data/solve.json` (**`nash_profile`**) instead of re-solved at boot; **`row_of`** shapes one spot's distribution for the wire |

The answer: 0.011 chips a hand, game value −0.0874 against the LP's −0.0856.

![Exploitability of the average falls; the current strategy does not](fig:rung2/exploitability.png)

:::cards
- [Exploitability](wiki:theory/exploitability) Sum first, max second, and why.
- [Kuhn poker](wiki:implementation/kuhn) The same five layers without chance mid-tree.
- [Mini-drawmaha rules](wiki:implementation/minidrawmaha-rules) The game that ends the tabular line.
:::
