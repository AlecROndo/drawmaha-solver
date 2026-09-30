"""The four regret rules and the averaging columns: what one visit banks, and how.

`mccfr.py` walks the tree and, at every one of the traverser's spots, measures
one sample of counterfactual regret — `r = u − ⟨σ, u⟩`, what each action earned
down the sampled path minus what the current mix earned. What happens to `r`
next is the REGRET RULE, and it is the only thing that differs between the four
algorithms this rung compares (Brown & Sandholm 2019, arXiv 1809.04040, for all
but the first):

- **vanilla** — `R += r`. Every iteration counts the same.
- **LCFR** (linear CFR) — `R += t·r`. Iteration t counts t times as much as
  iteration 1, so the early iterations, played from strategies that were mostly
  noise, fade: after T iterations the first tenth carries 1% of the weight.
  Equivalent to discounting the whole sum by t/(t+1) every iteration, which is
  why it needs no bookkeeping of its own.
- **CFR+** — `R += r`, then `R ← max(R, 0)`. A regret that goes negative is
  floored at zero instead of digging a hole, so an action that was bad early
  and good now starts being played the moment it is good, not only after it
  has climbed out. (The other half of CFR+, alternating updates, is already how
  `run_iteration` works: P0 traverses, then P1.) Brown & Sandholm report that
  the floor interacts badly with sampling — a single noisy sample can wipe a
  balance that took many iterations to build — so it is measured here, not
  assumed to win.
- **DCFR** (discounted CFR, α = 1.5, β = 0) — `R += r`, then every POSITIVE
  entry is multiplied by t^1.5/(t^1.5 + 1) and every NEGATIVE one by 1/2.
  Positive regret fades slowly (the factor tends to 1: 0.5 at t = 1, 0.97 at
  t = 1,000); negative regret is halved every iteration, so a mistake is
  forgotten within a few dozen iterations — a softer CFR+.

A rule is a function `bank(ledger, r, t)` and nothing else. The AVERAGE is not
the rule's business: every rule banks the same strategies into the same
averages, because what is averaged is what was played, whichever rule chose it.

**DCFR, discounted lazily.** Written as the paper writes it, DCFR multiplies
EVERY one of the traverser's regrets every iteration, visited or not. Under
sampling a row is visited once in thousands of iterations, and touching all six
million rows per iteration is out of the question — so a row is discounted only
when it is next visited, all the missed iterations at once. That is exact, not
an approximation, for one reason: a row nobody visits receives r = 0, and
multiplying by a positive factor never changes a sign, so every entry stays
positive or negative for the whole stretch and the per-sign factors simply
multiply. A row last banked at iteration s (its `stamp`) and visited again at t
owes, for iterations s+1 … t−1:

    positive entries:  ∏ₖ k^1.5/(k^1.5 + 1) = exp(L(t−1) − L(s))
    negative entries:  (1/2)^(t−1−s)

then the visit adds its r and applies iteration t's own factor by the NEW sign.
L(n) = −Σ_{k≤n} log(1 + k^−1.5) is the running log of the positive factors; a
difference of two L's is the product over just the stretch between them.

Reading a strategy needs no catch-up at all. `strategy()` normalises the
positive entries, every positive entry of a row owes the same factor, and a
common factor cancels in the normalisation. So the walk reads σ from the stored
numbers exactly as it does for the other three rules; only `settled_regret`,
for a reader who wants regret magnitudes, applies what is owed.

**L without a table the length of the run.** A table of L over every iteration
would be 8 bytes an iteration — 400 MB per process at fifty million, more than
the packed table itself. It is not needed, because L converges: the k-th term
is about k^−1.5, and those sum to a finite total. So L is exact from a 65,536-
entry table for the first stretch, and beyond it the tail — how much of the
remaining sum is still to come — has a short closed form (Euler–Maclaurin on
Σ log(1 + k^−1.5), expanded as Σ k^−1.5 − Σ k^−3/2 + …) whose first omitted
term is below 10^−17. The tests check both halves against brute-force sums out
to fifty million iterations. The lazy form is also no less precise than the
eager one: eager DCFR rounds once per iteration of a gap too.

**The averaging columns.** A run's primary average — `strategy_sum`, the one
`average_strategy()` reads — is LINEAR (weight t) under every rule, as it was
before rules existed. Beside it a run may bank further averages of the very
same strategies under other weights, one `extra_sums` row each: UNIFORM
(weight 1, the textbook CFR average) and QUADRATIC (weight t², DCFR's own
γ = 2 average). They are free in a way a second rule is not: the average is
never read during training, so a column cannot change the play it averages,
whereas two rules on one run would each be learning from strategies the other
chose. Which columns a run banks is its own choice, recorded in its checkpoint.

Only `cumulative_regret`, `stamp` and — for the columns — `extra_sums` and
`strategy_sum` are written, always in place (see `regret_matching`'s ledger
contract): a ledger may be a view into a packed table's buffer.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable
from enum import StrEnum

import numpy as np

from drawmaha_solver.regret_matching import RegretMatcher

# What a rule is: bank one traverser visit's regret `r` into `ledger` at
# iteration `t`, in place.
BankRegret = Callable[[RegretMatcher, np.ndarray, int], None]

class RegretRule(StrEnum):
    """The four rules a run can train under. The value is what a checkpoint records."""

    VANILLA = "vanilla"
    CFR_PLUS = "cfr+"
    LCFR = "lcfr"
    DCFR = "dcfr"

class Average(StrEnum):
    """The averaging weights a run can bank: weight 1, t or t² on iteration t."""

    UNIFORM = "uniform"
    LINEAR = "linear"
    QUADRATIC = "quadratic"

# The strategy sum every ledger already has. It is always linear, so a run
# asks for extra columns only among the others.
PRIMARY_AVERAGE = Average.LINEAR

_EXPONENT = {Average.UNIFORM: 0, Average.LINEAR: 1, Average.QUADRATIC: 2}

# DCFR's negative-regret factor: β = 0 makes t^β/(t^β + 1) = 1/2 at every t.
_NEGATIVE_FACTOR = 0.5

# ---------------------------------------------------------------------------
# The four rules
# ---------------------------------------------------------------------------

def bank_vanilla(ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
    """`R += r`: every iteration at the same weight."""
    ledger.cumulative_regret += regret

def bank_linear(ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
    """LCFR, `R += t·r`: iteration t weighs t times iteration 1."""
    ledger.cumulative_regret += t * regret

def bank_cfr_plus(ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
    """CFR+, `R ← max(R + r, 0)`: the SUM is floored, after the add.

    Flooring the increment instead would let a positive balance only grow, and
    an action that turned bad could never lose the lead it built.
    """
    cumulative = ledger.cumulative_regret
    cumulative += regret
    np.maximum(cumulative, 0.0, out=cumulative)

def bank_discounted(ledger: RegretMatcher, regret: np.ndarray, t: int) -> None:
    """DCFR: settle the iterations missed since the stamp, add, discount iteration t by sign.

    The stamp must be behind t. Equal would mean a second bank into the row in
    the same iteration, which perfect recall rules out — the traverser's own
    history is in the key, so one traversal reaches a spot of theirs at most
    once — and which the lazy form could not represent, because iteration t's
    factor has already been applied. Ahead would mean the row came from another
    run. Both raise rather than discount backwards.
    """
    stamp = int(ledger.stamp[0])
    if stamp >= t:
        raise ValueError(
            f"the row is stamped at iteration {stamp} and cannot be banked at {t}"
        )
    cumulative = ledger.cumulative_regret
    _discount(cumulative, since=stamp, until=t - 1)
    cumulative += regret
    positive = t**1.5 / (t**1.5 + 1.0)
    cumulative *= np.where(cumulative > 0.0, positive, _NEGATIVE_FACTOR)
    ledger.stamp[0] = t

RULES: dict[RegretRule, BankRegret] = {
    RegretRule.VANILLA: bank_vanilla,
    RegretRule.CFR_PLUS: bank_cfr_plus,
    RegretRule.LCFR: bank_linear,
    RegretRule.DCFR: bank_discounted,
}

def settled_regret(ledger: RegretMatcher, t: int) -> np.ndarray:
    """A DCFR row's regret as of the end of iteration t, as a copy; the ledger is untouched.

    What the stored numbers would be had every missed iteration been applied
    eagerly. The walk never needs it (see the module docstring on reads); a
    reader of regret MAGNITUDES does. For the other three rules the stamp stays
    0 and — since they are only defined from iteration 1 — this would apply
    discounts they never had, so it is for DCFR rows only.
    """
    stamp = int(ledger.stamp[0])
    if stamp > t:
        raise ValueError(
            f"the row is stamped at iteration {stamp} and cannot be settled to {t}"
        )
    settled = ledger.cumulative_regret.copy()
    _discount(settled, since=stamp, until=t)
    return settled

def _discount(regret: np.ndarray, *, since: int, until: int) -> None:
    """Apply DCFR's factors for iterations since+1 … until to `regret`, in place, by sign."""
    if until <= since:
        return
    positive = math.exp(cumulative_log_discount(until) - cumulative_log_discount(since))
    # Past ~1,075 missed iterations this underflows to exactly 0.0, which is
    # the right answer: a negative regret halved a thousand times is gone.
    negative = _NEGATIVE_FACTOR ** (until - since)
    regret *= np.where(regret > 0.0, positive, negative)

# ---------------------------------------------------------------------------
# The averaging columns
# ---------------------------------------------------------------------------

def validate_averages(averages: Iterable[str]) -> tuple[Average, ...]:
    """The extra columns a run asked for, as `Average`s, refused if they cannot be banked.

    Refuses an unknown name, a name twice (two rows banking identical numbers
    is a bug, not a feature) and `linear`, which is the primary strategy sum
    every ledger already has.
    """
    columns = []
    for name in averages:
        try:
            column = Average(name)
        except ValueError:
            known = ", ".join(repr(a.value) for a in Average if a is not PRIMARY_AVERAGE)
            raise ValueError(f"unknown averaging column {name!r}; known: {known}") from None
        if column is PRIMARY_AVERAGE:
            raise ValueError(
                f"{column.value!r} is the primary strategy sum, not an extra column"
            )
        if column in columns:
            raise ValueError(f"averaging column {column.value!r} is asked for twice")
        columns.append(column)
    return tuple(columns)

def extra_weights(averages: tuple[Average, ...], t: int) -> np.ndarray | None:
    """Iteration t's weight for each extra column, shaped (k, 1) to scale σ into `extra_sums`.

    None when the run banks no extra columns, so the walk can skip the work
    outright rather than add an empty array on every visit.
    """
    if not averages:
        return None
    return np.array([[float(t) ** _EXPONENT[column]] for column in averages])

def column_average(
    ledger: RegretMatcher, column: Average, averages: tuple[Average, ...]
) -> np.ndarray:
    """One ledger's average strategy under `column`, uniform when nothing was banked.

    `averages` is the run's extra columns in their row order; `linear` reads the
    primary sum. A column the run did not bank raises instead of reading
    another column's row.
    """
    if column is PRIMARY_AVERAGE:
        return ledger.average_strategy()
    if column not in averages:
        raise ValueError(
            f"this run banks no {column.value!r} average; its columns are "
            f"{[a.value for a in (PRIMARY_AVERAGE, *averages)]}"
        )
    banked = ledger.extra_sums[averages.index(column)]
    total = banked.sum()
    if total <= 0.0:
        return np.full(ledger.n_actions, 1.0 / ledger.n_actions)
    return banked / total

# ---------------------------------------------------------------------------
# DCFR's running log-discount
# ---------------------------------------------------------------------------

# Where the exact table stops and the tail formula takes over. At 2^16 the
# formula's first omitted term is ~1e-18, and the table is 512 KB.
_TABLE_END = 2**16

def cumulative_log_discount(n: int) -> float:
    """L(n) = −Σ_{k ≤ n} log(1 + k^−1.5): the log of DCFR's positive factors through iteration n.

    L(0) = 0. From a table up to 2^16, and from the table's last entry minus
    the tail formula beyond it. Measured against brute-force sums out to fifty
    million, both are within one unit in the last place (4.4e-16).
    """
    if n < 0:
        raise ValueError(f"an iteration index is non-negative, got {n}")
    head = _head_table()
    if n <= _TABLE_END:
        return head[n]
    return head[_TABLE_END] - (_tail(_TABLE_END + 1) - _tail(n + 1))

@functools.cache
def _head_table() -> list[float]:
    """L(0) … L(2^16), summed with Neumaier compensation.

    Compensated because a plain running sum rounds once per term and 65,536
    roundings reach ~1e-12 in the worst case; this keeps the table at the
    precision of its last bit. Built once, on DCFR's first use (~30 ms).
    """
    table = [0.0]
    total = compensation = 0.0
    for k in range(1, _TABLE_END + 1):
        term = -math.log1p(k**-1.5)
        moved = total + term
        if abs(total) >= abs(term):
            compensation += (total - moved) + term
        else:
            compensation += (term - moved) + total
        total = moved
        table.append(total + compensation)
    return table

def _tail(a: int) -> float:
    """Σ_{k ≥ a} log(1 + k^−1.5), for a past the table, from Euler–Maclaurin.

    log(1 + x) = x − x²/2 + …, so the sum is ζ(1.5, a) − ζ(3, a)/2 + ζ(4.5, a)/3
    − …, and each Hurwitz ζ(p, a) = Σ_{k≥a} k^−p has the Euler–Maclaurin
    expansion a^(1−p)/(p−1) + a^−p/2 + p·a^(−p−1)/12 − …. Kept: every term
    above 1e-17 at a = 2^16; the largest dropped one, ζ(4.5, a)/3 ≈ a^−3.5/10.5,
    is ~1e-18.
    """
    x = float(a)
    zeta_three_halves = 2.0 * x**-0.5 + 0.5 * x**-1.5 + 0.125 * x**-2.5
    zeta_three = 0.5 * x**-2 + 0.5 * x**-3
    return zeta_three_halves - 0.5 * zeta_three
