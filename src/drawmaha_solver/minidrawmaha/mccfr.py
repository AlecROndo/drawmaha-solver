"""External-sampling MCCFR: rung 2's walk with the deck and the opponent sampled.

Vanilla CFR walks the whole tree every iteration. Leduc's tree is small enough
for that; mini-drawmaha's is 85.1 billion decision nodes, so the walk does not
run slowly here, it does not run. External sampling keeps rung 2's recursion —
terminal, chance and decision branches, utilities indexed by position in
`legal_actions()`, ledgers banked in place on the way back up — and changes
what two of those branches do:

- **The deck is sampled, not averaged.** A chance node picks ONE outcome by
  its probability and follows it.
- **The opponent is sampled, not enumerated.** At the opponent's spot the walk
  picks ONE action from the opponent's current mix and follows it.

The traverser's own spots are still enumerated: every legal action is walked,
because regret is a comparison between the actions and a comparison needs all
of them. One traversal per seat per iteration, P0 then P1, each against a
freshly dealt root.

**The weight is the sample.** Rung 2 multiplied the regret bank by π₋ᵢ·π_c —
how often the deck and the opponent bring play to this spot. Under sampling,
that product is exactly the probability the walk arrives here at all, so the
update lands with that FREQUENCY instead of that COEFFICIENT, and the regret
weight at the traverser's spot is exactly 1:

    at my spot:     R(I, ·) += 1 · (u − ⟨σ, u⟩)
    at their spot:  S(I, ·) += t · σ(I, ·)

`u[k]` is what my k-th action earned down the sampled path; `⟨σ, u⟩` is what my
current mix earned on it. Their difference is one unbiased sample of rung 2's
counterfactual regret. Multiply π₋ᵢ·π_c in on top and the probability is
applied twice: nothing crashes, and the opponent's reach — which differs
across the worlds of one spot, because the opponent's cards differ — is
squared, so the worlds they rarely produce are under-trained and the solve
converges somewhere else. On Leduc that walk stalls at an exploitability of
0.14–0.20 where this one reaches 0.05, and is indistinguishable from it for
the first few hundred thousand iterations. (The chance half is inert in both
games: every chance probability at one public point is the same number, so it
rescales a ledger uniformly. In mini-drawmaha that is because the deck deals
combinations of the stub uniformly and the stub's size is public; a test walks
many worlds through one public line to pin it.) That is the one way to get
this file wrong by pattern-matching `leduc/cfr.py`, so `traverse` has no reach
parameter to thread.

The strategy sum is banked at the OPPONENT's spots, where the walk samples from
σ anyway: arriving there already happened with the opponent's own reach, so
adding σ once per arrival reproduces the own-reach weighting rung 2 wrote out
by hand. The traverser's own spots bank no strategy — they were enumerated,
not played, and counting them would credit every action with having been
chosen. The weight `t` makes the AVERAGE linear: later, better iterates count
more in the answer. The regrets are not weighted — they still accumulate at
weight 1 — so this is vanilla regret with a linearly weighted average, not
Linear CFR, which weights both. Weighting the regrets by `t` too, or CFR+'s
regret clipping, are the plan's levers if convergence stalls; neither has been
measured against sampling noise here.

Linear weighting makes `t` part of a run's state, which is why training takes a
`Solve` — table, RNG, root sampler and iteration count together — rather than a
bare table: a table handed back alone would restart `t` at 1 and weight every
later strategy as if the run were young. The RNG is consumed in one fixed order
(root deal, then every pick in recursion order), so a seed reproduces a run bit
for bit, a run trained in chunks equals one trained straight, and a checkpoint
restores the random stream along with the ledgers.

The walk is written against a nine-member state protocol, not against
`MiniState`, and both `LeducState` and `MiniState` satisfy it unchanged. That is
load-bearing: mini-drawmaha has no referee, so the proof that this walk is right
is the same walk run on Leduc and graded against the LP value. Only the root
deal is game-specific — Leduc's is a constant tuple, mini-drawmaha's is
`random_deal` — so it is passed in.

One known cost, accepted by the plan: a subtree behind an opponent action whose
current probability is exactly zero receives no samples until that action's
regret turns positive again. It costs nothing while the action stays at zero,
and row 6's exact meter is what would show a stale spot that matters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from drawmaha_solver.minidrawmaha.game import random_deal
from drawmaha_solver.minidrawmaha.infoset_table import new_infoset_table
from drawmaha_solver.regret_matching import RegretMatcher

# ---------------------------------------------------------------------------
# What the walk reads
# ---------------------------------------------------------------------------

class State(Protocol):
    """The nine members a walk reads off a node. `LeducState` and `MiniState` both have them.

    A chance outcome is opaque: a single card in Leduc, a tuple of cards in
    mini-drawmaha. The walk hands back to `apply_chance` exactly what
    `chance_outcomes` handed it, and never looks inside.
    """

    def is_terminal(self) -> bool: ...
    def is_chance_node(self) -> bool: ...
    def chance_outcomes(self) -> Sequence[tuple[Any, float]]: ...
    def apply_chance(self, outcome: Any) -> State: ...
    @property
    def current_player(self) -> int: ...
    def legal_actions(self) -> Sequence[Any]: ...
    def apply(self, action: Any) -> State: ...
    def infoset(self) -> Hashable: ...
    def returns(self) -> tuple[float, float]: ...

# Deals one root. The private deal sits outside both state machines — Leduc
# enumerates 30 of them, mini-drawmaha samples from 100,100 — so each game
# supplies its own. Mini-drawmaha's is `random_deal`.
RootSampler = Callable[[np.random.Generator], State]

# One ledger per infoset. Mini-drawmaha's `InfoSetTable` and Leduc's are both
# dicts of `RegretMatcher`; a lookup that misses raises, which is the table's
# claim that the walk reached a spot the census says does not exist.
Table = Mapping[Hashable, RegretMatcher]

# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Solve:
    """Everything a run needs to continue exactly where it stopped.

    `iteration` is the index t of the last COMPLETED iteration, so the next one
    banks at t = iteration + 1. `seed` is provenance only; the live random
    stream is `rng`. Mutable on purpose: `train` advances it in place, and a
    copy would be a second solve silently diverging from the first.

    The answer is `average_strategy(solve.table)`; the table's current strategy
    cycles and is not the solution.
    """

    table: Table
    rng: np.random.Generator
    deal: RootSampler
    seed: int
    iteration: int = 0

def new_solve(
    seed: int, *, table: Table | None = None, deal: RootSampler = random_deal
) -> Solve:
    """A fresh run at iteration 0, seeded so that it can be reproduced.

    `table=None` allocates the whole mini-drawmaha table — 1.77 GB and about
    fourteen seconds. Pass a table to train a slice or another game; it must
    hold every key the walk can reach from `deal`'s roots, since a miss raises.
    """
    return Solve(
        table=new_infoset_table() if table is None else table,
        rng=np.random.default_rng(seed),
        deal=deal,
        seed=seed,
    )

def train(solve: Solve, iterations: int) -> Solve:
    """Run `iterations` more iterations on `solve`, in place, and return it.

    There is no progress callback: a caller wanting checkpoints trains in
    chunks, and chunks are guaranteed to equal one straight run. A callback
    that touched the RNG would break that without a sound.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    for _ in range(iterations):
        run_iteration(solve)
    return solve

def run_iteration(solve: Solve) -> None:
    """One iteration: a fresh deal walked with P0 traversing, then another with P1.

    Both traversals bank at the same t. Each seat gets its own deal rather than
    sharing one, so the two samples are independent.
    """
    t = solve.iteration + 1
    for traverser in (0, 1):
        traverse(solve.deal(solve.rng), solve.table, traverser, solve.rng, t)
    solve.iteration = t

# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def traverse(
    state: State, table: Table, traverser: int, rng: np.random.Generator, t: int
) -> float:
    """Chips to `traverser` from `state` along one sampled path, banking on the way up.

    Returns the traverser's value only. The opponent's value is not computed on
    a sampled path, and returning a pair would invite somebody to read it.

    The returned number is a sample, not an expectation: it is unbiased for the
    traverser's value under the current strategies, and noisy.
    """
    if state.is_terminal():
        return state.returns()[traverser]

    # The deck: one outcome, by its probability, and no ledger — the deck has
    # no regrets. Its probability is NOT carried down; having been sampled is
    # already how it enters.
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        picked = _pick(rng, [probability for _, probability in outcomes])
        return traverse(state.apply_chance(outcomes[picked][0]), table, traverser, rng, t)

    ledger = table[state.infoset()]
    actions = state.legal_actions()
    sigma = ledger.strategy()

    # My spot: walk every action, bank their regret at weight exactly 1. u[k]
    # belongs to actions[k], which is the ledger's column k.
    if state.current_player == traverser:
        utilities = np.array(
            [traverse(state.apply(action), table, traverser, rng, t) for action in actions]
        )
        ledger.update(utilities, regret_weight=1.0, strategy_weight=0.0)
        return float(sigma @ utilities)

    # Their spot: bank the mix they are about to be sampled from, weighted by
    # t, and follow the one action it picks. Zero utilities at zero regret
    # weight leave their regret untouched — it moves only when they traverse.
    ledger.update(np.zeros(len(actions)), regret_weight=0.0, strategy_weight=float(t))
    return traverse(state.apply(actions[_pick(rng, sigma)]), table, traverser, rng, t)

def _pick(rng: np.random.Generator, probabilities: Sequence[float]) -> int:
    """An index drawn with probability `probabilities[k]`, from one uniform number.

    Not `rng.choice(p=...)`, which validates and normalises its input on every
    call: 2.5 µs a draw against 0.3 µs here. Measured over whole traversals
    that is a fifth of Leduc's cost (0.145 → 0.117 ms) and a twentieth of
    mini-drawmaha's (2.6 → 2.5 ms), whose time goes to building states.

    A rounding tail (the probabilities summing to 1 − 1e-16 and the draw landing
    past them) falls back to the last action with positive probability, never
    to the last action outright: regret matching zeroes actions exactly, and a
    zero-probability action must never be followed.
    """
    threshold = rng.random()
    cumulative = 0.0
    fallback = -1
    for index, probability in enumerate(probabilities):
        if probability > 0.0:
            cumulative += probability
            if threshold < cumulative:
                return index
            fallback = index
    # A distribution with no positive entry has nothing to sample; the ledger
    # and both games guarantee one, so reaching here is a caller bug.
    if fallback < 0:
        raise ValueError(f"no action has positive probability: {list(probabilities)}")
    return fallback

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_solve(solve: Solve, path: Path) -> None:
    """Write a run to one `.npz`: both accumulators flat, plus t and the RNG.

    Flat arrays rather than a pickle because the full table is 1.77 GB of
    Python objects but only 116 MB of numbers; pickling three million ledgers
    is slower than rebuilding them. Every ledger's regret, and separately every
    ledger's strategy sum, are laid end to end in the table's own order, with
    each ledger's width alongside so `load_solve` can slice them back apart and
    refuse a table they do not fit.

    Raises on an empty table, which has nothing to restore, and on a generator
    other than the PCG64 `new_solve` and `load_solve` build, whose state could
    not be restored.
    """
    ledgers = list(solve.table.values())
    if not ledgers:
        raise ValueError("the table is empty; there is nothing to checkpoint")
    _require_restorable(solve.rng.bit_generator.state)
    with Path(path).open("wb") as file:
        np.savez(
            file,
            regret=np.concatenate([ledger.cumulative_regret for ledger in ledgers]),
            strategy_sum=np.concatenate([ledger.strategy_sum for ledger in ledgers]),
            widths=_widths(solve.table),
            iteration=np.int64(solve.iteration),
            seed=np.int64(solve.seed),
            rng_state=np.str_(json.dumps(solve.rng.bit_generator.state)),
        )

def load_solve(path: Path, *, table: Table, deal: RootSampler = random_deal) -> Solve:
    """Pour a checkpoint into `table` and return the run, ready to continue.

    The caller allocates `table` — the same game and the same keys, in the
    same order, as the table that was saved. Its ledgers are overwritten in
    place. Training the returned solve continues the saved random stream, so a
    resumed run equals an uninterrupted one bit for bit.

    The widths are the fingerprint: a table with another ledger count or any
    ledger of another width raises before anything is written. A table of the
    same shape but different keys cannot be told apart from the numbers alone,
    which is why the order contract is the caller's. The random stream is
    restored before any ledger is written too, so a checkpoint whose stream
    cannot be restored leaves the table untouched.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        widths = saved["widths"]
        _validate_shape(widths, table)
        rng_state = json.loads(str(saved["rng_state"]))
        _require_restorable(rng_state)
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        regret, strategy_sum = saved["regret"], saved["strategy_sum"]
        ends = np.cumsum(widths)
        for ledger, end, width in zip(table.values(), ends, widths, strict=True):
            ledger.cumulative_regret[:] = regret[end - width : end]
            ledger.strategy_sum[:] = strategy_sum[end - width : end]
        return Solve(
            table=table,
            rng=rng,
            deal=deal,
            seed=int(saved["seed"]),
            iteration=int(saved["iteration"]),
        )

def _require_restorable(rng_state: Mapping[str, Any]) -> None:
    """Refuse a random stream a checkpoint cannot carry, with a message that says so.

    Only PCG64 is carried: it is what `new_solve` seeds and what `load_solve`
    rebuilds, and its state is plain integers, so it survives JSON. Another
    generator's state holds arrays, and would fail deep inside NumPy or JSON.
    """
    name = rng_state.get("bit_generator")
    if name != "PCG64":
        raise ValueError(
            f"a checkpoint carries only a PCG64 random stream, not {name!r}"
        )

def _widths(table: Table) -> np.ndarray:
    return np.fromiter(
        (ledger.n_actions for ledger in table.values()), dtype=np.int64, count=len(table)
    )

def _validate_shape(widths: np.ndarray, table: Table) -> None:
    """Refuse a table the checkpoint's ledgers do not fit, before writing into it."""
    if len(widths) != len(table):
        raise ValueError(
            f"the checkpoint holds {len(widths)} ledgers, the table {len(table)}"
        )
    mismatched = np.flatnonzero(widths != _widths(table))
    if mismatched.size:
        first = int(mismatched[0])
        raise ValueError(
            f"ledger {first} is {widths[first]} wide in the checkpoint and "
            f"{list(table.values())[first].n_actions} wide in the table "
            f"({mismatched.size} width mismatches in all)"
        )
