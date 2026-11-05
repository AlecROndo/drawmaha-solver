---
title: Method
description: The validation ladder, the pre-registered evaluation protocol, and what each rung found.
---

The full game is too large to solve on the first try, and a solver that is wrong does not announce it: a swapped weight or a peeking grader still produces a strategy, a falling curve and a confident number. The project's answer is to never take a step that cannot be checked. Each **rung** is a game that adds exactly one new idea to the rung below it, and each is graded against an answer that was known before the solver ran.

## Each rung adds one idea and is graded before the next

| Rung | Game | What it adds | Graded against |
| --- | --- | --- | --- |
| 0 | Rock-paper-scissors | the regret-matching ledger itself | the uniform Nash, closed form |
| 1 | Kuhn poker | a game tree, hidden cards, CFR's two reach weights | the closed-form equilibrium family; all 64 pure strategies of each seat enumerated |
| 2 | Leduc poker | chance in the middle of the hand; actions that differ by spot | an OpenSpiel solve, frozen as `tests/leduc/referee.json` |
| 3 | Mini-drawmaha, 6.2M infosets | sampling (MCCFR), parallel lockstep, four regret rules raced | the exact best-response walk; no outside engine plays this game |
| 4 | Mini-drawmaha | a neural network in place of the regret tables (Deep CFR) | rung 3's grader and rung 3's frozen answer |
| next | Full Drawmaha | Deep CFR at scale | a head-to-head checkpoint ladder and a trained exploiter |

The learning rule never changes as the games grow. `src/drawmaha_solver/regret_matching.py` is the same ledger at every rung; what changes is how the utilities it is fed are computed — a payoff column in rock-paper-scissors, a tree walk in Kuhn and Leduc, a sampled walk in mini-drawmaha, a network's output in Deep CFR. See [The ledger](wiki:implementation/ledger).

## Exploitability is the only grade

The evaluation protocol was fixed in the README before the results existed ("Evaluation protocol (pre-registered)"). The reasoning: in self-play, training loss is not a progress meter. Both players move together, so a loss can fall while the pair stays easy to beat. The only honest question is what an adversary wins.

```eq
exploitability(σ) = (BR₀(σ₁) + BR₁(σ₀)) / 2
```

- **BR₀(σ₁)** is what seat 0 wins per hand by playing its best response against seat 1's strategy, and **BR₁(σ₀)** the same for seat 1.
- **Why the average of two**: the game is zero-sum, so at an equilibrium the two best responses are the game value and its negative and the sum is exactly zero. Any leak in either seat's strategy pushes one of them up, so the average is positive everywhere else.
- **What moves it**: a strategy that leaks more chips to a perfect opponent in either seat raises it; it cannot be lowered by playing well against a weak opponent. That is what makes it a meter and not a score.

How it is computed at each rung:

- **Toys (rungs 0–2)**: exactly, and checked against Kuhn's closed form and an OpenSpiel solve of Leduc.
- **Mini-drawmaha (rungs 3–4)**: the exact best-response walk in `src/drawmaha_solver/minidrawmaha/exploitability.py` as ground truth, plus the Deep CFR versus tabular gap.
- **Full scale**: a duplicate-dealt head-to-head ladder of checkpoints plus a trained reinforcement-learning exploiter against the frozen network, reported as a *lower bound* on exploitability, because no exact walk is possible there.

Units are total chips across both pot halves, per hand, with per-half EV and scoop rate kept as split-pot diagnostics. The grader never imports the solver — see [Onboarding](wiki:overview/onboarding#a-grader-never-shares-code-with-what-it-grades).

:::warning
The **current** strategy is not the answer. At rungs 1 and 2 its exploitability sits near 0.2 and between 0.75 and 1.48 chips/hand for the whole run while the **average** strategy's falls steadily (the README's Kuhn and Leduc tables). Every surface that plays a strategy reads an average, never the final iterate.
:::

## Results, rung by rung

From the results table at the top of `README.md`; every number there is recomputed by a test from a committed file.

| Rung | Game | Method | Result |
| --- | --- | --- | --- |
| 0 | Rock-paper-scissors | regret matching | 0.0009 chips/round exploitable after 100k rounds |
| 1 | Kuhn poker | vanilla CFR | 0.00063 chips/hand; game value −0.05555 against −1/18 |
| 2 | Leduc poker | vanilla CFR | 0.011 chips/hand; game value −0.0874 against −0.0856 |
| 3 | Mini-drawmaha | lockstep MCCFR, four regret rules | LCFR wins: 0.061 chips/hand at 20M hands per seat |
| 4 | Mini-drawmaha | Deep CFR | 0.068 chips/hand at 10M hands per seat, where LCFR is at 0.126; still 0.068 at 20M, where LCFR has reached 0.061 |
| next | Full Drawmaha | Deep CFR at scale | not started; the range interface is built ahead of it |

What each rung taught, in a sentence:

- **Rung 0** — the ledger converges to the uniform Nash in self-play and to pure paper against a 50%-rock opponent (+0.24/round against a best response of +0.25).
- **Rung 1** — CFR rediscovers Kuhn's bluffing ratio unprompted: the jack bluffs at α = 0.221 and the king value-bets at 0.661 ≈ 3α. Swap the two reach weights and the run still "converges", to a king that folds 18% of the time.
- **Rung 2** — a jack on a jack board checks and raises rather than leading, a check-raise nobody encoded.
- **Rung 3** — on 6.2M infosets, LCFR's linear average beats vanilla, CFR+ and DCFR at every checkpoint of the race; it is frozen as the release `rung3-lcfr-2m`. See [Regret rules](wiki:theory/regret-rules).
- **Rung 4** — at 1M hands per seat the best net is about a tenth as exploitable as the table (0.108 against 1.131), but the nets flatten past 1M hands and the table passes the ÷ pot net between 10M and 20M. One seed; see [Status](wiki:research/status).

![Exploitability by hands per seat for the four regret rules](fig:rung3/race.png)

:::cards
- [Status](wiki:research/status) Where the project stands now, and what is running.
- [Run log](wiki:research/run-log) Every Deep CFR arm and its grades.
- [Exploitability](wiki:theory/exploitability) How the exact grader works, and why its order of operations matters.
:::
