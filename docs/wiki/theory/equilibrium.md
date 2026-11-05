---
title: Nash equilibrium in two-player zero-sum games
description: What the solver is trying to find, why it is the right target for heads-up poker, and how the repo measures distance from it.
---

Every game in this repo is two-player and zero-sum: whatever one seat wins, the other loses. In that setting there is one well-defined thing a solver can aim for — a **Nash equilibrium**, a pair of strategies where neither seat can gain by changing its own play while the other holds still. This page says what that object is, why it is the right target for heads-up Drawmaha, and what it does *not* promise, using the solved games the repo grades against.

## A strategy is a probability for every action at every spot

Poker is a game of **imperfect information**: you see your own cards and the betting, not your opponent's cards. So a strategy cannot depend on the full state of the deal. It depends on an **information set** (an *infoset*, or *spot*): everything the acting player can see at a decision.

- In Kuhn poker an infoset is the actor's own card plus the public betting history — `K` with history `pb` (check, bet) is one spot. Two deals that differ only in the opponent's card are the *same* infoset, and must map to one strategy entry.
- A **behavioural strategy** σᵢ for seat i assigns, at each of its infosets I, a probability σᵢ(I, a) to every legal action a. Probabilities at one infoset sum to 1.
- A **strategy profile** σ = (σ₀, σ₁) is one strategy per seat.

The expected payoff of a profile is uᵢ(σ): average over the deck's deals and both seats' action probabilities, in chips per hand. Zero-sum means u₀(σ) = −u₁(σ) for every σ.

## Equilibrium means no unilateral improvement

```eq
u₀(σ₀*, σ₁*) ≥ u₀(σ₀, σ₁*)   for every σ₀
u₁(σ₀*, σ₁*) ≥ u₁(σ₀*, σ₁)   for every σ₁
```

- **σ₀\*, σ₁\*** are the equilibrium strategies.
- The first line says: holding P1 at σ₁\*, nothing P0 can switch to does better than σ₀\*. The second says the same for P1.
- Neither line mentions what the opponent *should* do — each is about one seat deviating alone.

### Why this is the right target for heads-up poker

In a two-player zero-sum game the minimax theorem makes the equilibrium payoff a single number, the **game value** v. Every equilibrium earns exactly v for P0, and playing *any* equilibrium strategy guarantees at least v whatever the opponent does. That guarantee is the reason to want one: an equilibrium strategy cannot lose more than the game's built-in edge, against anyone.

The repo's own game values:

| Game | Game value to the first seat | Source |
| --- | --- | --- |
| Rock-paper-scissors | 0 | the uniform Nash (⅓, ⅓, ⅓) |
| Kuhn poker | −1/18 ≈ −0.05556 | closed form; the solver reaches −0.05555 |
| Leduc poker | −0.08561 | OpenSpiel's sequence-form LP, frozen in `tests/leduc/referee.json` |
| Mini-drawmaha | about −0.0906 in self-play of the frozen answer | the README's rung-3 answer, read from the released `rung3-lcfr-2m` strategy |

The negative values all have the same cause: the second seat acts last. In mini-drawmaha P1 acts last in both betting rounds and in the draw, and that is worth about a tenth of an ante.

## The equilibrium can be a family, not a point

Kuhn poker shows that "the" equilibrium is often a set. The first player's equilibria form a one-parameter family over α ∈ [0, ⅓]:

- with the jack, P0 bluffs (opens with a bet) with probability α;
- with the king, P0 value-bets with probability 3α;
- with the queen facing a check–bet, P0 calls with probability α + ⅓.

Every α in the range earns exactly −1/18. The solver lands wherever its iterates take it — α = 0.221 and K-open = 0.661 ≈ 3α in the README's run — and the ratio is discovered, never supplied. That is why `tests/kuhn` pins the nine infosets that are the same in every equilibrium and the two invariants (K-open = 3α, Qpb = α + ⅓), and never a hard-coded α.

:::note
A family of equilibria is why a strategy is never graded by comparing its numbers to "the answer". It is graded by **exploitability** — what a perfect adversary wins against it — which is zero for every member of the family and positive for everything else. See [Exploitability](wiki:theory/exploitability).
:::

## Nash is unbeatable, not punishing

An equilibrium guarantees v; it does not collect what a weak opponent leaves on the table. The Kuhn exploit tab measures the gap. Take a P1 that plays equilibrium everywhere except two leaks — it calls a bet with the queen 90% of the time instead of a third, and folds the king 40% of the time:

| | chips/hand to P0 |
| --- | ---: |
| exact best response against the leaky P1 | +0.183334 |
| CFR's counter-strategy, 20k iterations | +0.183328 |
| equilibrium play against the same leak | −0.011502 |

Equilibrium play still loses a little (the seat is worth −1/18 at equilibrium, so the leaks have only partly offset it). The best response wins 0.195 chips/hand more. Equilibrium play doesn't make mistakes of its own, but it doesn't exploit yours either. Exploiting a known opponent means computing a **best response**, which is a different object (see [Exploitability](wiki:theory/exploitability) and [Kuhn](wiki:implementation/kuhn)).

## The unit: chips per hand, with the ante worth 1

| Quantity | Unit |
| --- | --- |
| Payoffs, game values, exploitability | chips per hand, ante = 1 (per round for rock-paper-scissors) |
| The research page (`/research`) | chips per **hundred** hands (the per-hand files × 100, in one place) |
| Full Drawmaha (planned) | big blinds per hand, total across both halves of the pot |

:::cards
- [Regret matching](wiki:theory/regret-matching) The learning rule whose average converges to equilibrium.
- [Exploitability](wiki:theory/exploitability) How distance from equilibrium is measured exactly.
- [The method](wiki:overview/method) The graded games, and why each one comes before the next.
:::
