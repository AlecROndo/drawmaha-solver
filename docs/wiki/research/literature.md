---
title: The literature, mapped
description: The nine research notes the ML survey was written from — the question each answers, its key papers, and what the solver took from it.
---

Before any solver code existed, a literature sweep (2026-08-28) split the field into eight lanes, each a question a Drawmaha solver has to answer, and a ninth note audited what the eight left out. The notes live under [`docs/research/`](repo:docs/research) and are finished reference: they are not updated as the code moves. The 26-page survey written from them is [`paper/drawmaha-ml-survey.pdf`](repo:paper/drawmaha-ml-survey.pdf).

Read a lane when the code you are about to touch rests on its idea. Each note opens with a lane narrative — the story, the builder's options as a decision tree, the pitfalls — then one section per paper.

## One row per lane

| Note | The question it answers | Key papers | What the project took from it |
| --- | --- | --- | --- |
| [01 · CFR foundations](repo:docs/research/01-cfr-foundations.md) | Why does iterating a local update rule produce a Nash equilibrium in an adversarial game? | Hart & Mas-Colell 2000 (regret matching); Zinkevich et al. 2007 (CFR); Lanctot et al. 2009 (MCCFR); CFR+ 2014; Cepheus 2015 | The whole tabular stack: the shared regret-matching ledger, vanilla CFR on Kuhn and Leduc, external-sampling MCCFR on mini-drawmaha, and the four-rule race. The note recommended DCFR; the race found LCFR beats it under sampling. |
| [02 · Neural CFR](repo:docs/research/02-neural-cfr.md) | How do you replace CFR's regret table with a regression when the game is too big to store? | Deep CFR 2019; Single Deep CFR 2019; DREAM 2020; ESCHER 2022; Double Neural CFR; ReCFR | Rung 4's design: external sampling (the project owns its simulator), reservoir memories, a from-scratch refit each iteration, t-weighted strategy samples, and grading the exact (SD-CFR-style) average beside the distilled Π so distillation error is not blamed on CFR. |
| [03 · Search and value nets](repo:docs/research/03-search-value-nets.md) | Can test-time search be made sound when you do not know which state you are in? | Burch et al. 2014 (decomposition); DeepStack 2017; safe and nested subgame solving 2017; Libratus 2018; depth-limited solving 2018; Pluribus 2019 | Not yet used. It frames the phase after a blueprint — a depth-limited resolve at the draw boundary — and its warning: Drawmaha ranges are 2,598,960-dimensional, so range-vector value nets do not port without bucketing. |
| [04 · RL and equilibrium](repo:docs/research/04-rl-equilibrium.md) | Why does naive self-play RL fail at mixed strategies, and what fixes it? | NFSP 2016; PSRO 2017; Pipeline PSRO 2020; Perolat et al. 2020 (recurrence); DeepNash / R-NaD 2022; MMD 2022 | That the last iterate orbits and the time average converges — the same reason the site queries averages, never the final iterate. It is the background for the planned PPO chapter. |
| [05 · Abstraction and bucketing](repo:docs/research/05-abstraction-bucketing.md) | How do you build a smaller game whose solution transfers back? | Gilpin & Sandholm 2007 (lossless abstraction); Waugh et al. 2009 (pathologies); Ganzfried & Sandholm 2013–14 (action translation, EMD imperfect recall) | Mini-drawmaha *is* a hand-designed abstraction that keeps the draw and the split pot; its infoset key relabels suits jointly, the lossless-abstraction idea this lane opens with. See [abstraction](wiki:theory/abstraction). |
| [06 · Evaluation](repo:docs/research/06-evaluation.md) | How do you measure a strategy when training loss says nothing? | Johanson et al. 2011 (fast best response); local best response 2017; AIVAT 2018; Cepheus; Deep CFR's evaluation | The project's first rule: grade by exploitability, computed by a best-response walk that shares no code with the solver, and at full scale a head-to-head ladder plus a trained exploiter as a lower bound. See [exploitability](wiki:theory/exploitability). |
| [07 · Variant games](repo:docs/research/07-variant-games.md) | What do solved draw and Omaha variants teach about Drawmaha's two blow-ups — hand count and the draw round? | Ostroumov's Draw Solver; Omaha hi-lo jam/fold 2015; Poker-CNN 2015; GTO Wizard PLO 2026; MonkerSolver docs; Zadeh 1977 | That the split pot is the cheap part (it changes only terminal payoffs), that discard identity matters for blockers, that infoset keys must hold both players' draw counts — and the load-bearing **absence finding**: no Drawmaha solver, bot or paper existed as of 2026-08. |
| [08 · Engineering practice](repo:docs/research/08-engineering-practice.md) | What do you actually build with, and where do projects break? | Neller & Lanctot 2013 (CFR tutorial); OpenSpiel 2019; PokerRL and the Deep CFR reference code 2019; RLCard 2019 | OpenSpiel as the Leduc referee (frozen in `tests/leduc/referee.json`), the Deep CFR hyperparameter anchors (Adam 1e-3, clip 1, batch 10K, 4K steps, 10K traversals), and the validation-ladder habit of checking every piece on a game with a known answer. |
| [09 · Gaps audit](repo:docs/research/09-gaps-audit.md) | What did the eight lanes leave out that the survey had to cover or scope out? | nineteen items, from Kuhn's theorem and sequence-form LP to discard encoding and multiway play | The open engineering decisions for full Drawmaha, sorted on [open questions](wiki:research/open-questions). |

## Three findings from the sweep that shaped the plan

- **Average, never last.** Lanes 01, 02 and 04 arrive at the same fact from three directions: in a two-player zero-sum game the time-averaged strategy converges to equilibrium while the current one cycles. Kuhn's exploitability table shows it on the smallest game ([Kuhn](wiki:implementation/kuhn)), and it is why every Deep CFR mark grades the average.
- **The grader is the product.** Lane 06's one-sentence summary — self-play produces no labels, so the only principled question is what a perfect adversary wins — is why each game in the project was solved against a known answer before the next was attempted.
- **Nobody had done this.** Lane 07 searched papers, code, the commercial solver market and forums for any Drawmaha equilibrium work and found none. The game is documented and dealt online; no solver plays it.

## Caveats the notes carry about themselves

Each lane ends with its own verification caveats: which papers were fetched in full, which hyperparameters were read from summaries, which venue labels come from background knowledge. One correction is worth knowing before quoting the notes: lane 01 gives the two-player deal space as C(52,5)² ≈ 6.7 × 10¹², but the hands are disjoint, so the right count is C(52,5) × C(47,5) ≈ 4.0 × 10¹² (the audit's own note).

:::cards
- [ML survey (PDF)](repo:paper/drawmaha-ml-survey.pdf) The 26-page survey assembled from these notes.
- [Open questions](wiki:research/open-questions) The audit's items, sorted into decisions.
- [CFR](wiki:theory/cfr) The algorithm lane 01 explains, as this project implements it.
- [Deep CFR](wiki:theory/deep-cfr) Lane 02's algorithm, as rung 4 runs it.
:::
