# Rung 4 distillation: how much of the 0.068 plateau is the net's capacity?

Date: 2026-11-05. Decided in conversation with Alec; this file is the record.

## The question

The ÷ pot Deep CFR arm (4 × 256, 16,000-step refits, lr 5e-4) flattens at an
exact-average exploitability of 0.0685 from about 10M hands per seat, while
the tabular LCFR run keeps falling (0.126 at 10M hands, 0.061 at 20M, and
0.0157 at 100M hands, iteration 10,000,000 of the 20M run). Two things could
hold the net at 0.068: the net itself cannot represent a better strategy
(**capacity**), or Deep CFR's way of training it — sampled regrets, reservoir
memories, from-scratch refits — never finds one (**training**).

## The method: supervised distillation

Separate the two by taking Deep CFR out of the picture. Fit the same net
architecture directly to a known good strategy and grade the copy:

- **Teacher.** The LCFR table's linear average at iteration 10,000,000
  (`iter-010000000.npz` of the paused 20M run, seed 0, W = 10, so 100M hands
  per seat). Graded 0.015685 on 2026-11-05.
- **Student.** The rung-4 net (`deepcfr/net.py`: 136 features in, 7 outputs,
  ReLU MLP of `width` × `depth`), the 4 × 256 of the Deep CFR arm as the
  headline, smaller and larger sizes beside it.
- **Target.** The teacher's **average strategy**, one distribution over the
  legal actions at every one of the 6,220,050 infosets. Not regrets: regrets
  would copy the current iterate (0.191 at 20M hands), a different
  experiment.
- **Loss.** Cross-entropy of the student's softmax over the legal columns
  against the teacher's distribution, illegal columns masked out.
- **Weight.** Each row by the probability mass that arrives at it by the
  player's own choices under the teacher: own reach through the predecessor
  table (`deepcfr/average.py`), times the number of physical deals the
  canonical key stands for. A row the teacher never reaches by its own play
  has weight 0 and is never trained on. The weight is applied by sampling
  rows in proportion to it, which minimises the same objective at lower
  variance than weighting uniform draws.
- **Data.** Every infoset, no held-out split for the headline grade: the
  exact grader scores every infoset, so the whole-game grade IS the test and
  memorising is the goal.
- **Grade.** The student frozen as a `FrozenStrategy` (softmax at the legal
  columns of every key), saved, reloaded, and graded by the exact
  best-response walk, as every strategy in this repo is.

## Reading the result

| gap | between | measures |
|---|---|---|
| representation | teacher 0.0157 vs its 4 × 256 copy | what the net cannot hold |
| training | the copy vs Deep CFR's 0.0685 at the same size | what Deep CFR's training loses |
| both | teacher vs Deep CFR | the whole plateau |

A copy near 0.068 says capacity is the whole story (check it against a
larger net before concluding); a copy well below says training is.

## Arms

- Sizes: 3 × 64, 3 × 128, 4 × 256 (headline), 4 × 512, 4 × 1024.
- Ablations at 4 × 256: fewer steps; unweighted rows (uniform over infosets).
- Generalisation: 4 × 256 trained on a random 10% and 50% of the infosets,
  graded on the whole game, with the error on seen and unseen rows apart.

## Caveats the report must state

- The copy's grade is an **upper bound** on the net's floor: cross-entropy
  matches the teacher's probabilities, it does not minimise exploitability,
  and another near-equilibrium may be easier to represent.
- The teacher's own error (0.0157) is in every copy.
- One seed of everything.
- Hands: lockstep iteration × 10 = hands per seat (teacher 100M); Deep CFR
  mark t × 10,000 = hands per seat (t = 2,000 is 20M).

## Constraints

Train on this Mac (`mps`), never on Modal. Read the teacher only; never touch
`resume.npz` or the paused trainer. Grader shares no code with the student.
`uv run pytest` passes without torch. Results under `figures/rung4/distill/`
with the command that wrote them. Report as an arXiv-style preprint PDF in
`/Users/alec/Desktop/Claude/poker/rung4-distill/`, no rotated figures.
