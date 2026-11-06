---
title: Distilling the table into the net
description: The LCFR table copied straight into the rung-4 net by reach-weighted cross-entropy, so the 0.068 plateau splits into what the net cannot hold — at most a third at 4 × 256 — and what Deep CFR's training did not find.
---

Deep CFR's ÷ pot arm, a 4 × 256 net, flattens at an exact-average exploitability of 0.0685 from 10M hands per seat while the tabular LCFR run keeps falling, to 0.0157 at iteration 10,000,000 (100M hands per seat). Two very different things could hold the net there: it cannot **represent** a better strategy, or Deep CFR's **training** (sampled regrets, reservoir memories, from-scratch refits, a learned average) never finds one it could. No Deep CFR arm can tell them apart, because every arm trains the net the same way. This page is the experiment that does, run on 2026-11-05; the numbers are [`figures/rung4/distill/results.json`](repo:figures/rung4/distill/results.json) and the full write-up is finding 9 of the [run log](wiki:research/run-log).

## The method: fit the net to the table, then grade the copy

Take Deep CFR out of the picture. The **teacher** is the table's linear average strategy at iteration 10,000,000, one distribution over the legal actions at every one of the 6,220,050 infosets. The **student** is the rung-4 net itself ([`deepcfr/net.py`](repo:src/drawmaha_solver/deepcfr/net.py): 136 features in, seven outputs, ReLU layers of a given width and depth). [`deepcfr/distill.py`](repo:src/drawmaha_solver/deepcfr/distill.py) fits the student to the teacher by supervised learning over every infoset and freezes the result as a strategy the exact grader reads. If the copy grades near the teacher, the net can hold such a strategy and the plateau is a training effect; if it grades near the plateau, capacity is the story.

Three choices make the fit mean something:

- **The target is the average strategy, not the regrets.** The average is what CFR's guarantee is about and what the Deep CFR run was graded on. Copying regrets would copy the current iterate (0.1911 for the same run), a different and worse object.
- **The loss is cross-entropy on a softmax over the legal columns.** Illegal columns are held at −10⁹ before the softmax, so they carry no probability and get no gradient. The loss is smallest where the softmax *is* the teacher's row, which a net reaches only where it can represent that row (an exact zero in the teacher is approached, never reached). The [Π loss sweep](wiki:implementation/deepcfr-training) had already shown cross-entropy copies a distribution better than squared error.
- **Each row weighs what arrives at it.** A best response exploits a spot in proportion to how often the player's own choices and the deck bring the player there, and a spot the player's own play never reaches cannot be exploited at all. So a row's weight is its seat's own reach under the teacher (the same `π^q(I)` the [exact average](wiki:implementation/deepcfr) accumulates with) times the number of physical deals its canonical key stands for (6 over the size of the key's suit stabiliser). Rows are drawn in proportion to that weight, which minimises the weighted objective at far lower variance than weighting uniform draws.

There is no held-out split for the headline: the exact grader scores every infoset, so the whole-game grade is the test and memorising is the goal.

## The result: at most a third is capacity at 4 × 256, the rest training, and the capacity part goes away one doubling later

Every arm in the table is 100,000 Adam steps of 16,384 rows under a cosine schedule from 1e-3, one seed, on this Mac's GPU; the longer fits are under the ablations. `err` is the reach-weighted mean absolute error per column against the teacher, measured on the frozen file.

| arm | net | differs by | err | exploitability | × teacher |
|---|---|---|---|---|---|
| teacher | the LCFR table | — | 0 | 0.0157 | 1.0× |
| `3x64` | 3 × 64 | size | 0.1147 | 0.1842 | 11.7× |
| `3x128` | 3 × 128 | size | 0.0904 | 0.1015 | 6.5× |
| `4x256` | 4 × 256 | **the Deep CFR net** | 0.0450 | **0.0389** | 2.5× |
| `4x512` | 4 × 512 | size | 0.0061 | **0.0175** | 1.1× |
| `4x1024` | 4 × 1024 | size | 0.0010 | **0.0158** | 1.0× |
| Deep CFR | 4 × 256 | trained by Deep CFR, 20M hands | — | 0.0685 | 4.4× |

![Each copy of the table against its parameter count](fig:rung4/distill/by_size.png)

**The three gaps at 4 × 256.** After 100,000 steps the copy grades 0.0389 against the teacher's 0.0157 and Deep CFR's 0.0685, which puts what the net cannot hold at +0.0233 (44% of the plateau) and what its training did not find at +0.0295 (56%). The copy's grade only bounds what the net can hold from above, and the 400,000-step copy grades 0.0331, so the first part is at most 33% (+0.0174) and the second at least 67%; the figure draws that tighter split. Neither part is negligible: a wider net trained the same way would still carry the training gap, and a better training procedure with this net could get no lower than what the net can hold, which these fits place at or below 0.0331. The two shares are a reading of grades along one axis, not a decomposition of causes: exploitability is not additive in a strategy's errors.

![At most a third of the 4 × 256 plateau is what the net cannot hold](fig:rung4/distill/gaps.png)

**One doubling of the width removes almost all of the representation part.** The 4 × 512 copy grades 0.0175, within 0.0019 of the teacher, and its error against the teacher falls from 0.0450 to 0.0061. At 4 × 1024 the copy grades 0.0158, the teacher to four places. Every Deep CFR size that was ever run on this game (3 × 64 to 4 × 256) sits on the steep part of this curve.

## What the ablations add

![Loss and reach-weighted error by step](fig:rung4/distill/curves.png)

- **The 4 × 256 floor is partly the fit's.** 25,000 steps grade 0.0560 against 100,000's 0.0389, the headline's error curve was still easing down when the schedule ran out, and 400,000 steps grade 0.0331: the representation share of the plateau moves from 44% to 33% with a four-times longer fit, while the 4 × 512 copy got further in a quarter of the steps. The copy's grade is an *upper bound* on the net's floor for this reason and one more: cross-entropy copies the teacher's probabilities, it does not minimise exploitability, and another near-equilibrium may be easier to represent.
- **The reach weight is worth 2.6× in the grade.** Rows drawn uniformly over the 6.2M infosets, everything else equal, grade 0.0995 against 0.0389. The uniform draw gives every infoset the same share of the steps whatever its weight, so the rows the strategy's own play reaches most are fitted no better than rows it never reaches, and the best response finds them; which rows absorbed the lost steps was not measured.
- **Trained on a random half of the game, this net does not transfer to the other half.** A copy trained on a random half of the rows grades 0.1046: its error on the rows it saw is 0.0228, *better* than the full copy's 0.0450, so it is not short of steps there, but on the rows it never saw 0.1185, and the best response goes there. A tenth grades 0.3457. What this does not separate is why: the 136 features, the architecture, or the teacher itself — an average strategy that is irregular across neighbouring infosets, as an average built from few samples at rarely visited rows may be, would defeat any model that infers. For full Drawmaha, which cannot be enumerated and where every spot is in effect unseen, transfer is the number that matters, and on one seed and one split this net on these features does not have it.

![A copy trained on a share of the infosets](fig:rung4/distill/fraction.png)

## What it means for the next arm

A wider net trained by Deep CFR the same way is capped near its own copy, so the larger and more durable gain is in the training: the sampling, the memories and the average. A 4 × 512 copy at 0.0175 is the floor that training would have to approach. Caveats: one seed of everything; the teacher's own error (0.0157) is in every copy; the weight is own reach under the teacher, not under a best response.

:::cards
- [Run log](wiki:research/run-log) Finding 9 in the digest, beside the eight Deep CFR findings it explains.
- [Deep CFR package](wiki:implementation/deepcfr) The net, the features and the exact average the weight reuses.
- [Exploitability](wiki:theory/exploitability) The meter every copy is graded by.
:::
