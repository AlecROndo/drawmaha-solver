---
title: Open questions
description: The Deep CFR arms nobody has run, the decisions full Drawmaha forces, and the gaps the literature audit found.
---

Two lists feed this page. The first is the run log's "Not yet run" — experiments on mini-drawmaha whose answer would change a default. The second is [`docs/research/09-gaps-audit.md`](repo:docs/research/09-gaps-audit.md), written when the literature survey was assembled, which lists nineteen things the survey had to cover or scope out. Many of the audit's items are now *engineering decisions* for full Drawmaha rather than gaps in a paper; they are sorted that way below.

## On mini-drawmaha: what would move the 0.068 floor

The best Deep CFR net flattens at 0.068 chips/hand from 10M hands, and the table passes it by 20M ([where the project stands](wiki:research/status)). Every comparison so far is one seed, so a difference under about 10% is suggestive, not settled. The untried arms, each with the question it answers:

| Arm | The question it answers |
| --- | --- |
| **A second seed** of any arm | How big run-to-run noise really is. The only estimate today is two raw runs 0.005 apart that also differ in learning rate. |
| **A bigger net, or a cross-entropy Π, to t = 2,000 under ÷ pot** | Is 0.068 a capacity floor (a bigger net moves it) or a fitting floor (a better objective moves it)? |
| **÷ √pot at learning rate 1e-3** | The two single-seed gains (√pot targets, the full rate) have not been tried together — and it is the pairing the default config now runs, so the default is an ungraded config. |
| **A raw-target arm with a looser gradient clip** | Separates the pot weighting from the clipping regime it changes: dividing targets shrinks the gradients, so the clip of 1.0 bites less. |
| **The √pot arm past t = 200** | Does it flatten where ÷ pot did (0.068) or elsewhere? |
| **The tᵞ sample-weight arm** (γ = 0 vanilla, γ = 2 quadratic) | Linear is γ = 1; is it the best weighting for a sampled, refit-from-scratch learner? |
| **A Modal run of the ÷ pot config** | A reproducibility check of the Mac run, not a new arm — rows do not depend on the machine, only the refit's arithmetic order does. |

:::note
Whichever arm wins, releasing it is separate work: the best graded object today (0.0679 at 10M hands) is frozen under `runs/` but not released, so `/solver`, Learn and Play still show older strategies. See [releases](wiki:ops/releases).
:::

## On full Drawmaha: decisions the game forces before training

These come from the audit and are open because no code for the full game exists yet.

- **How the net outputs a discard.** A five-card hand has 32 keep/discard subsets. A flat 32-way head, per-card keep/discard logits, or something factorised — and how regret matching behaves over a combinatorial action space, with masking for illegal actions — is an encoding decision with no citation anywhere in the survey (audit item 8).
- **How the non-card state is encoded.** Betting history, pot and stack normalisation, position, and both players' public draw counts all have to enter the net's input. Mini-drawmaha's 136-feature encoder is the starting point, not the answer (item 9).
- **How hands are indexed.** Suit isomorphism collapses 2,598,960 five-card holdings to 134,459 classes before the deal, but the factor shrinks as board and kept cards fix suits. The standard perfect-indexing algorithm (Waugh 2013) is not yet used (item 7).
- **The exact rule variant.** Drawmaha has sub-variants (draw timing, number of betting rounds, odd-chip rules). The project pins one — see [full Drawmaha](wiki:research/full-drawmaha) — and every node count and street-boundary claim depends on that choice (item 3).
- **Units and diagnostics.** Hold'em calibration anchors (mbb/g, the always-fold baseline) do not transfer to a split-pot bomb pot. The pre-registered protocol reports total chips across both halves, per-half EV and scoop rate, and the terminal-utility table for odd chips and double ties still has to be written (item 16).
- **What is served.** Between "trained average-policy net" and "strategy shown or played" sits purification and thresholding — rounding near-zero probabilities before serving — which the project does not do today (item 12).

## Techniques the project has not used, and would consider

| Technique | Why it could matter here | Audit item |
| --- | --- | --- |
| Regret-based pruning (Brown & Sandholm 2015, 2017) | skips negative-regret subtrees; a major speedup in the Libratus and Pluribus blueprints | 4 |
| Predictive / optimistic CFR+ (Farina, Kroer & Sandholm 2021) | the tabular state of the art past DCFR, with faster rates | 5 |
| Public chance sampling / vector-form CFR (Johanson et al. 2012) | walk the public tree vectorised over private hands — the trick behind commercial-solver speed, and the shape the exact grader already has | 6 |
| Warm-starting CFR from a strategy (Brown & Sandholm 2016) | start full-game training from a shrunken variant's solution | 13 |
| Value nets and depth-limited resolving (DeepStack, Libratus) | the search phase after a blueprint; needs a way to generate counterfactual-value data and to update ranges through a hidden discard | 10, 11 |
| NeuRD, ARMAC, QPG/RPG | the bridges between policy gradients and CFR; relevant to the planned PPO chapter | 17 |

## Scope statements, decided or to be decided

- **Heads-up only.** Every guarantee the project relies on — any Nash strategy is safe, the game has one value — is a two-player zero-sum fact. Drawmaha is usually dealt multiway in mixed games, where none of that holds (audit item 18). The project solves heads-up.
- **Equilibrium, not exploitation.** The solver plays Nash, which is unbeatable but declines to punish mistakes; [Kuhn's exploit tab](/rung1) measures exactly how much that leaves on the table. Opponent modelling and safe exploitation (Restricted Nash Response) are out of scope (item 14).
- **Benchmarking a first-of-its-kind agent.** No outside Drawmaha bot exists, so head-to-head baselines must be homegrown: a duplicate-dealt ladder between independently seeded checkpoints plus a trained exploiter, reported as a lower bound on exploitability (item 19). Real-money play against people on poker sites is a terms-of-service violation and not a goal.
- **Rake.** Whether the solver targets raked or rake-free play is not stated anywhere; today's games are rake-free.

:::cards
- [The run log](wiki:research/run-log) Every arm already run, and its eight findings.
- [Full Drawmaha](wiki:research/full-drawmaha) The target game and the plan for it.
- [Literature map](wiki:research/literature) The nine notes the audit was written against.
:::
