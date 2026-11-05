---
title: Abstraction — how mini-drawmaha shrinks full Drawmaha
description: Why the repo solves a smaller game first, what was cut and what was kept, and the one lossless collapse it allows.
---

Every CFR-family solver needs something per information set, and real poker has astronomically many. The literature's bridge is **abstraction**: build a smaller game whose solution says something about the big one. Mini-drawmaha is this repo's abstraction. It is not a bucketing of full Drawmaha's hands. It is a separate, smaller game that keeps every *idea* full Drawmaha adds over Leduc. It's small enough to solve exactly, so a method can be graded on it before it meets a game where no grade is possible.

## Mini-drawmaha keeps the ideas and cuts the size

| | Full Drawmaha (the target) | Mini-drawmaha (rung 3–4) |
| --- | --- | --- |
| Deck | 52 cards | 15 cards: ranks 2–6 in three suits |
| Hole cards | 5 each | 3 each |
| Start | bomb pot: everyone antes, no preflop betting | the same: ante 1 each, no blinds, no preflop betting |
| Board before the draw | flop, 3 cards | 1 card |
| Betting rounds | three (flop, turn, river) | two |
| The draw | discard 0–5 and replace; draw-1 is dealt **face up** (keep it or take the next face down) | throw at most **one**, replaced face down; P1 learns only *whether* P0 drew |
| Inner half of the pot | best 5-card hand of the hole cards alone | best 3-card hand of the hole cards alone |
| Outer half of the pot | Omaha: exactly 2 hole + 3 board | exactly 2 hole + both board cards |
| Betting | pot-limit, {fold, check/call, pot}, 100bb stacks | the same action set, 26-chip stacks |
| Size | far beyond any table | 6,220,050 infosets, 85.1 billion decision nodes |

What survives the shrink is the list of things Leduc does not have, each of which breaks a naive solver differently:

- **The chance node depends on the action.** Stand pat and the deck never acts; throw a card and it deals one. A throw is two nodes, the decision and then the replacement.
- **Thrown cards leave the deck for good, and only their owner knows which.** `remaining_deck()` is derived from the state, never counted down, so a discard can never be redealt.
- **The pot pays twice, by two rules over the same cards.** Dealt trips is the inner half's top category and can bring only a pair outside. Strength is two-dimensional.
- **Suits carry information.** Both halves have flushes, so Leduc's trick of throwing suits away is no longer lossless.

### Every size number was chosen for a reason

- **Three suits** because with two, a rank holds two cards and the inner half loses trips entirely.
- **Five ranks** because at four, a run of four distinct ranks *is* the whole rank set, and high card and plain flush would have nowhere to exist.
- **26-chip stacks** because from the 2-chip ante pot an uncapped pot-raise war commits 2 → 8 → 26. The stack buys a bet, a raise and an all-in shove (truncated to the 25 left behind the ante), and nothing deeper. Round 1 then has 13 complete lines, and round 2 opens in one of four (pot, stacks) states, so the infoset key needs no explicit chip counts.
- **Hand rankings by measured rarity in this deck**, not 52-card convention. A four-card straight (25.11%) ranks *below* two pair (16.38%) and trips (9.89%), because a run of four out of five ranks is easy to make. Outer high card is the rarest outer result (0.74%) and still ranks last, by definition.

:::note
Mini-drawmaha has no referee. No engine plays Drawmaha, so every rule is a decision, argued in the rung-3 plan (kept outside the repo, `~/Desktop/Claude/poker/rung3-delta/rung3-delta-plan.md`, §3–§4). The ground truth is the exact best-response walk. Changing the deck, the draw cap, the betting tree or the infoset key means re-reading that plan first.
:::

## The one collapse allowed is relabelling suits, jointly

Within the game, the only abstraction the solver uses is **lossless**: two positions that differ only in which physical suit is which are the same position. `canonical()` relabels the three suits to the lexicographically smallest image of the whole visible picture. That takes the 455 dealt hands down to 95 classes, a 4.79× saving that costs nothing.

:::warning
**"Jointly" is the correctness trap.** Canonicalising the hole and the board *separately* forgets whether they share a suit, which is exactly what a flush is. Two genuinely different positions would hash to one key, and the solve would be quietly wrong rather than loudly broken. The draw adds a second trap: a throw action names a card by **position** ("throw the lowest"), and the positions are ordered by the canonical labels, so "throw the lowest" means the same card in every deal that shares an infoset.
:::

The same lossless step scales to the target: the survey's notes brute-force full Drawmaha's 2,598,960 five-card hands down to exactly **134,459** canonical classes (19.3×) (`docs/research/05-abstraction-bucketing.md`).

## Counting infosets by factorising the key

85 billion nodes can't be walked to count infosets, but an infoset factorises into two independent halves:

```eq
infoset = public point × private key
6,220,050 = Σ over the 141 public points of  (private keys at that point's shape)
```

- **A public point** is the betting so far, both draw counts, how many board cards are out, and whose turn it is. There are **141**, and none depends on a card, so one representative deal walks them all.
- **A private key** is the actor's own (hole, discarded, board) after the joint relabelling, with the board kept in the order it came. There are 970 or 10,170 with one board card out (by whether the actor has thrown), and 10,170 or 100,400 with two.

Keeping the board's order doubles the round-2 half: a sorted board would have given 3,142,290 infosets, merging histories the player can tell apart. The census splits 3,106,630 for P0 and 3,113,420 for P1, and `tests/minidrawmaha/census.json` holds it.

## What the literature does at full scale, and where Deep CFR sits

The survey's abstraction lane (`docs/research/05-abstraction-bucketing.md`) traces a ratchet of answers to "which hands should share a strategy?":

| Layer | Idea | Cost |
| --- | --- | --- |
| Lossless | suit isomorphism | none, provably exact |
| Expectation buckets | rank hands by equity E[HS] and split into percentiles | hides shape: a made hand and a live draw can share an E[HS] |
| Distribution-aware | k-means on equity *histograms* under earth mover's distance | imperfect recall can trade convergence guarantees for size |
| Potential-aware | cluster on histograms over *next-round* buckets | built bottom-up, so trajectory matters |

Two warnings carry over. Refining an abstraction can **increase** real-game exploitability (Waugh et al. 2009), so an abstraction is validated by exploitability or head-to-head, never assumed. And Drawmaha's split pot makes strength two-dimensional, so a scalar equity is worse than usual.

Deep CFR claims to "obviate the need for abstraction". The survey's synthesis is narrower: nets replace the **clustering**, not the **representation**. The net's input is itself an abstraction. Rung 4's 136 features are four card groups of 23 floats each (multi-hot over the deck, plus rank counts and suit counts, a hand-built stand-in for learned card embeddings), plus 44 floats of public state. Two keys that share everything the player can see share a vector, and nothing in the vector names the table row. A net handed key ids would just be a table with extra steps.

:::cards
- [Mini-drawmaha rules in code](wiki:implementation/minidrawmaha-rules) `cards.py`, `hands.py`, `game.py`, `enumeration.py`, `features.py`.
- [Full Drawmaha](wiki:research/full-drawmaha) What the target game needs that mini-drawmaha does not test.
- [Deep CFR](wiki:theory/deep-cfr) The method meant to carry the solve to full scale.
:::
