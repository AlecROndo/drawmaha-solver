---
title: Mini-drawmaha rules
description: The 15-card bomb pot with one draw and a split pot — the deck and its suit relabelling, both hand rankings, the state machine, the 6,220,050-infoset enumeration and the 136-float features the nets read.
---

Mini-drawmaha is a shrunken Drawmaha that keeps the two things the full game is made of, **a draw** and **a split pot**, on a deck small enough that the exact answer can still be computed. It is the last game where "how far is this strategy from optimal?" has an exact answer, which is why Deep CFR is graded here before it meets the real game. No engine implements Drawmaha, so every rule is a decision, and [`rules/`](repo:src/drawmaha_solver/minidrawmaha/rules) has no outside referee. It holds no solver code either: the exact grader imports this package and nothing else.

## The hand, start to finish

1. Both players ante **1** from a **26**-chip stack. It is a bomb pot: no blinds, no position asymmetry and no preflop betting.
2. Three private cards each, then **board card 1**, face up.
3. **Betting round 1.** Pot-limit, so the actions are fold, check/call and pot, with the bet capped at the stack.
4. **The draw.** P0 throws at most one card, then P1 does, knowing *whether* P0 drew but never what. The replacement is dealt face down; the full game's face-up draw-one rule is not played here.
5. **Board card 2**, then **betting round 2**, then the showdown.
6. **The pot splits in half.** The *inner* half goes to the best 3-card hand made of the three private cards alone. The *outer* half goes to the best 4-card hand made of exactly two private cards plus both board cards. Winning both halves is a scoop.

## Files

| File | What it is for |
| --- | --- |
| [`cards.py`](repo:src/drawmaha_solver/minidrawmaha/rules/cards.py) | The 15-card deck and the one collapse allowed: relabelling suits jointly |
| [`hands.py`](repo:src/drawmaha_solver/minidrawmaha/rules/hands.py) | Both halves of the pot as orderable scores |
| [`game.py`](repo:src/drawmaha_solver/minidrawmaha/rules/game.py) | The hand as a state machine, the `InfoSet`, the chips |
| [`enumeration.py`](repo:src/drawmaha_solver/minidrawmaha/rules/enumeration.py) | Every infoset, all 6,220,050, reached through 141 public points |
| [`features.py`](repo:src/drawmaha_solver/minidrawmaha/rules/features.py) | Every infoset as 136 floats, the view Deep CFR's nets read |

## `cards.py`: five ranks, three suits, one relabelling

**`Card`** is a rank 0–4 (printed 2–6) and a suit 0–2. Three suits is forced: with two, a rank holds only two cards and the inner half loses trips. The fifth rank gives the outer half a complete ranking. **`hand_symbol`** and **`parse_cards`** convert to and from text such as `"6h 2c"`.

Suits carry information here, because both halves of the pot have flushes. The table may forget which physical suit is which, but never how the suits *relate*. **`canonical`** relabels the three suits to the lexicographically smallest image of all the visible groups *jointly*, and **`canonical_relabelling`** returns the relabelling it used. **`relabel`** applies one. Doing the hole and the board separately would forget whether they share a suit, which is exactly what a flush is. The joint relabelling takes the 455 possible hands down to 95 classes at no cost.

## `hands.py`: rankings by measured rarity in this deck

**`inner_score`** scores the best 3-card hand of the three held cards. **`outer_score`** takes the best of the three ways to choose exactly two hole cards and scores each with **`four_card_score`** against both board cards. Each returns a **`Score`**: the category (**`InnerCategory`**, six of them; **`OuterCategory`**, seven), then what it is made of, so plain tuple comparison decides who wins.

Both orders are measured rarity *in this 15-card deck*, not 52-card convention, which makes two rows look wrong when they are correct. A four-card straight (25.11%) ranks *below* two pair and trips, because a run of four from five ranks is easy to make. Outer high card is the rarest outer result (0.74%) and still ranks last, because high card is the absence of a made hand.

## `game.py`: the state machine

- **`Action`** holds every decision in one enum: FOLD, CHECK_CALL and POT for betting, and four throws (none, low, mid, top) for the draw. Values are not ledger indices. **`throw_count`** is the public half of a throw. **`action_label`** and **`line_symbol`** are for display.
- **`MiniState`** is one node: both holes, the discards, the board and the betting. Its **`kind`** (**`NodeKind`**) is decision, chance or terminal. The chance node depends on the action: a player who stands pat gets no replacement deal. **`chance_outcomes`** and **`apply_chance`** turn board cards and deal replacements; **`legal_actions`** is betting or the four throws; **`returns`** pays half the pot on each hand, using **`pot_shares`**.
- **`ChipState`** and **`chip_state`** replay the betting to report the pot, each player's contribution and what is owed. **`betting_legal_actions`** applies pot-limit.
- **`InfoSet`** is what the actor can see, suits relabelled: **`canonical_picture`** of (hole, discarded, board *in the order it came*). The board order is kept on purpose, because round 1 was played against the first card alone. **`DrawSignal`** is what the table learns of an opponent's draw: the count and nothing else. **`draw_order`** lays the actor's cards out as low/mid/top so a throw names one card.
- **`random_deal`** deals uniformly over all 100,100 deals.

:::warning
The key is still slightly coarser than the real infoset. With a rank held twice, "throw the low one" picks a physical card by the relabelling's tie-break, and about one traversal in a thousand banks the same row twice. The grader therefore takes its max per *physical* state, never per table key: grouping by key under-reports by about 0.012 chips a hand. [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) covers what this does to the regret rules.
:::

## `enumeration.py`: 6,220,050 infosets from two small halves

The concrete tree has **85,129,644,600** decision nodes, so it cannot be walked. A position does factorise into two small, independent halves:

- **A public point** (**`PublicPoint`**): the betting so far, both draw counts, how many board cards are out and whose turn it is. Its **`width`** is the ledger width at that point. **`public_tree`** builds the whole public game, 141 decisions, 36 chance nodes and 178 terminals, from one representative deal, because no public point depends on a card. **`merged`** combines two routes into the same public spot by adding their weights, and refuses if the arrivals disagree about who acts. **`public_decision_points`** lists the 141 in walk order.
- **A private key** (**`private_keys`**): the canonical (hole, discarded, board), sorted. A decision stands on one of three shapes: 970 keys with one board card out, 10,170 with two out before a throw, and 100,400 with two out after one.

**`all_infosets`** is two nested loops over those halves and yields each infoset exactly once, 6,220,050 in all. Streaming them takes about fifteen seconds.

## `features.py`: what a net sees instead of a key id

A net handed a key id would just be a table with extra steps, so **`encode`** turns an infoset into the 136 floats the player could *see*:

| Part | Floats | Built by |
| --- | --- | --- |
| hole, discard, board card 1, board card 2 | 4 × 23 = 92 | **`card_group`**: a multi-hot over the 15 cards, plus 5 rank counts and 3 suit counts (a pair reads as a 2) |
| both players' draw signals | 6 | in **`public_row`** |
| the betting lines, 2 rounds × 5 slots × 3 actions | 30 | |
| chips | 3 | |
| seat and stage | 5 | |

The rank and suit counts repeat what the multi-hot already says, on purpose. They are a hand-built stand-in for learned rank and suit embeddings, and they hand a single unit "any pair" directly. Because the vector splits where the key splits, **`private_features`** and **`public_features`** build each half once. **`layout`** returns the whole-game **`Layout`**, which places each table row: **`locate_rows`** gives its public point and position, and **`features`** gathers its 136 floats. **`legal_gather`** maps each ledger column to the net's 7-wide head.

## The census: a stand-in for a referee

With no engine to ask, [`scripts/minidrawmaha/generate_census.py`](repo:scripts/minidrawmaha/generate_census.py) counts what the enumerator yields, one `InfoSet` at a time, and writes `tests/minidrawmaha/census.json`. `tests/minidrawmaha/test_census.py` rebuilds every number a second, different way. A change to the deck, the throw cap or the betting tree then fails loudly instead of shipping a solver for a different game. The rules were decided in a plan that lives outside the repo, at `~/Desktop/Claude/poker/rung3-delta/rung3-delta-plan.md`. Check it before changing a ranking, a count or a rule.

:::cards
- [Abstraction](wiki:theory/abstraction) Why a shrunken game is the right test bed.
- [Mini-drawmaha solver](wiki:implementation/minidrawmaha-solver) The table and the sampled walk over these rules.
- [Deep CFR package](wiki:implementation/deepcfr) Where the 136 features go.
:::
