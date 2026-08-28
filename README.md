# Drawmaha Solver

A neural solver for heads-up pot-limit Drawmaha — a split-pot draw/Omaha hybrid poker variant with no existing solver — built with Deep CFR, plus a GTOWizard-style dashboard for exploring the solved game tree.

## The game

Standard Drawmaha, heads-up:

- Each player is dealt **5 hole cards**
- Preflop betting → **flop** → betting → **one draw** (discard 0–5, replace) → **turn** → betting → **river** → betting
- **Draw-1 rule:** drawing exactly one card gets dealt **face up** (public); the drawer may keep it or reject it and take the next card face down
- **Split pot:** half to the best Omaha hand (exactly 2 hole + 3 board), half to the best 5-card poker hand made of the hole cards themselves

Solve parameters (v1 defaults, all configurable): pot-limit with action set {fold, check/call, pot}; 100bb effective stacks; standard HU blinds. Bet-sizing menus are a config so multi-size/custom solves are a re-solve, not a rewrite.

## Approach: the validation ladder

Each rung adds one layer and is checked against a known answer before climbing:

| Rung | Game | What it proves |
|------|------|----------------|
| 0 | Rock-paper-scissors | regret-matching ledger math |
| 1 | Kuhn poker | tabular vanilla CFR vs. the known exact equilibrium |
| 2 | Leduc poker | CFR with a board, vs. published benchmarks |
| 3 | Mini-drawmaha | split-pot scoring, draw actions, face-up draw-1 rule (tabular MCCFR) |
| 4 | Full drawmaha | Deep CFR — neural nets replace the regret tables |

The dashboard queries the rung-4 average-policy network: navigate the tree node by node, see range-level strategy aggregates (bucketed — 2.6M starting hands don't fit in a grid) with drill-down to exact hands. A trainer mode (quiz spots against the solver) is a later thin layer on the same query API. PPO self-play is a planned extension chapter: train a PPO agent and measure the CFR solution exploiting it.

## Status

Design/grilling phase — decisions above are locked; dashboard scope, evaluation story, and stack are being settled next.
