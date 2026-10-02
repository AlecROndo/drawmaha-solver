"""Rung 3's analysis: which regret rule solved mini-drawmaha, and what the answer plays.

Rungs 1 and 2 solved their game inside this script. Rung 3 cannot: the race
that settles it trained four regret rules for 2,000,000 lockstep iterations
each on Modal (W = 10 hands per seat per iteration, so 20M hands per seat),
and grading one checkpoint with the exact best-response walk takes about two
minutes. So the work is split in two:

* `--grade-runs DIR` grades every checkpoint under `DIR/<rule>/` that
  `figures/rung3/grades.json` does not already hold, in every averaging column
  the run banked, and adds the result. This is the slow, rare half, and it
  needs the checkpoints (555 MB each), which live on the Modal Volume.
* Without it, the script reads the committed `grades.json` and the frozen
  strategy (`strategy.py`, fetched from the release) and draws the figures.
  That half takes about two minutes, and every number it prints is
  reproducible from two committed or pinned files.

`grades.json` maps a key to one grade (both best responses, their mean — the
exploitability — and P0's self-play value, in chips per hand). A run's key is
`rule/iteration/column`: the rule's directory name on the Volume, the
iteration zero-padded to nine digits so the keys sort in training order, and
the averaging column the strategy was read from. The one other key is
`uniform_random`, the reference line: every infoset played uniformly.

The four figures:

1. **The race.** Exploitability against hands sampled, one line per rule. LCFR
   and vanilla — the two rules that keep negative regret — finish 3–4× below
   CFR+ and DCFR, which floor it or discount it away. CFR+ and DCFR are
   drawn in one colour because, under sampling, they treat negative regret
   the same way: a row is revisited every ~37,000 iterations at the median,
   and DCFR's per-iteration halving takes a negative regret to exactly zero
   long before that, which is CFR+'s floor. They are not one algorithm —
   DCFR also discounts positive regret and the two average differently — so
   the curves lie close (24.8 against 25.0 per 100), not identically.
2. **The averaging columns at 2M.** The same strategies, averaged under
   different weights. Linear beats uniform for both vanilla and CFR+.
3. **Round-1 betting**, by what the dealt hand already is on its own (its
   inner category): how often P0 bets out, how often P1 bets when checked
   to, and how often P1 folds to a bet. P0 leads 19–40% of every category
   and slowplays trips (21%); P1 checked to bets trips and straight flushes
   97% of the time; and facing a bet P1 folds a pair (39%) more often than
   high card (28%).
4. **The draw**, after round 1 is checked through: how often each player
   throws a card, by inner category. High cards and pairs throw 88–98% of
   the time, flushes and better at most 10%. P1 throws more often after
   P0 stands pat (a straight: 41% against 32%).

Every figure's title is read off the data it draws, so a regrade that moves a
number moves the title with it; the readings in this docstring are the
committed data's.

Figures 3 and 4 average the strategy over every physical deal of a category
with equal weight: they answer "holding this, what does the solution do", not
"how often does this line happen". A spot's reach is not in them, and neither
is the outer half: the category is the three held cards alone, so a gap
between two categories can be a mixture of the board textures their deals
meet rather than a property of the category. The readings above are what the
strategy does, not tested reasons for it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.cards import DECK, Card
from drawmaha_solver.minidrawmaha.exploitability import (
    best_response_value,
    expected_value,
)
from drawmaha_solver.minidrawmaha.game import (
    Action,
    DrawSignal,
    InfoSet,
    canonical_picture,
)
from drawmaha_solver.minidrawmaha.hands import InnerCategory, inner_score
from drawmaha_solver.minidrawmaha.lockstep import read_lockstep
from drawmaha_solver.minidrawmaha.regret_rules import (
    PRIMARY_AVERAGE,
    Average,
    RegretRule,
    validate_averages,
)
from drawmaha_solver.minidrawmaha.strategy import (
    STRATEGY_SHA256,
    StrategyInfo,
    fetch_strategy,
    from_checkpoint,
    load_strategy,
    uniform_strategy,
)
from drawmaha_solver.plotting import (
    BASELINE,
    CATEGORICAL,
    MUTED,
    SECONDARY,
    legend,
    new_axes,
    save,
)

# ---------------------------------------------------------------------------
# Files, keys and the four rules
# ---------------------------------------------------------------------------

FIGURES = Path("figures/rung3")
GRADES = FIGURES / "grades.json"

# The reference line's key in `grades.json`: the only key that names no run.
UNIFORM_RANDOM = "uniform_random"

# One entry of `grades.json`. A run's entry also carries its `workers` and
# `seed`, which the reference line has no use for.
Grade = dict[str, float]

@dataclass(frozen=True, slots=True)
class _Rule:
    """One regret rule as the analysis reads and draws it.

    `trains` is what the rule's checkpoints record, which is not the directory
    name (`cfrplus` holds runs that record `cfr+`). `keeps_negative_regret`
    splits the race's verdict: vanilla and LCFR carry negative regret forward,
    CFR+ floors it and DCFR halves it to zero between a row's visits.
    """

    trains: RegretRule
    label: str
    keeps_negative_regret: bool
    colour: str
    linestyle: str | tuple[int, tuple[int, int]]
    linewidth: float
    zorder: int

# CFR+ and DCFR are one family (both drop negative regret) and their curves
# lie almost on top of each other, so they share CATEGORICAL[1]'s hue in two
# shades: DCFR goes wide underneath in a light tint, CFR+ dashed on top in a
# dark shade. In one shade the dashes vanish into the wide line beneath them.
_FAMILY_LIGHT = "#f4b298"  # CATEGORICAL[1] mixed half and half with the surface
_FAMILY_DARK = "#a44924"  # CATEGORICAL[1] at 70% brightness

# Keyed by the directory name the Volume and `grades.json` both use, in the
# order the legend and the answer sheet list them.
RULES: dict[str, _Rule] = {
    "vanilla": _Rule(RegretRule.VANILLA, "vanilla", True, CATEGORICAL[0], "-", 1.8, 3),
    "lcfr": _Rule(RegretRule.LCFR, "LCFR", True, CATEGORICAL[2], "-", 1.8, 3),
    "cfrplus": _Rule(RegretRule.CFR_PLUS, "CFR+", False, _FAMILY_DARK, (0, (4, 3)), 1.8, 4),
    "dcfr": _Rule(RegretRule.DCFR, "DCFR", False, _FAMILY_LIGHT, "-", 4.0, 3),
}

COLUMN_COLOURS = {
    Average.LINEAR: CATEGORICAL[0],
    Average.UNIFORM: CATEGORICAL[1],
    Average.QUADRATIC: CATEGORICAL[2],
}

@dataclass(frozen=True, slots=True)
class _Point:
    """One run entry of `grades.json`, its key parsed and checked."""

    rule: str
    iteration: int
    column: Average
    grade: Grade

    @property
    def hands(self) -> int:
        """Hands sampled per seat: `workers` hands each lockstep iteration."""
        return self.iteration * int(self.grade["workers"])

@dataclass(frozen=True, slots=True)
class _Spot:
    """A place in the tree the readouts visit with every deal: who acts, after what."""

    player: int
    draws: tuple[DrawSignal, ...]
    betting: tuple[tuple[Action, ...], ...]

_CHECK, _BET = Action.CHECK_CALL, Action.POT

# Round 1's three spots. Columns: check, bet for the first two; fold, call,
# raise facing the bet.
ROUND_ONE_SPOTS = {
    "p0_open": _Spot(player=0, draws=(), betting=((),)),
    "p1_checked_to": _Spot(player=1, draws=(), betting=((_CHECK,),)),
    "p1_facing_bet": _Spot(player=1, draws=(), betting=((_BET,),)),
}

# The draw after a checked-through round 1. P1 sees how many cards P0 threw,
# so its row is read after each answer. Columns: stand pat, throw low, mid, top.
DRAW_SPOTS = {
    "p0": _Spot(player=0, draws=(), betting=((_CHECK, _CHECK),)),
    "p1_after_pat": _Spot(player=1, draws=(DrawSignal(0),), betting=((_CHECK, _CHECK),)),
    "p1_after_draw": _Spot(player=1, draws=(DrawSignal(1),), betting=((_CHECK, _CHECK),)),
}

# The inner categories each figure title reads its claim off.
_MADE = ("trips", "straight_flush")
_WEAK = ("high_card", "pair")
_STRONG = ("flush", "straight_flush", "trips")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Grade what is new, then draw the four figures and write the answer sheet.

    Steps:
      1. With `--grade-runs DIR`, grade every checkpoint column `--grades` lacks.
      2. Read the grades and load the frozen strategy (`--strategy`, else the
         release, checked against `STRATEGY_SHA256`).
      3. Grade the frozen strategy and read its round-1 and draw spots.
      4. Draw the four figures into `--out`.
      5. Write `answer_sheet.json` beside them and print its summary.
    """
    args = _parse_args(argv)

    # Step 1: grade new checkpoints
    if args.grade_runs is not None:
        grade_runs(args.grade_runs, args.grades)

    # Step 2: read the inputs
    grades = json.loads(args.grades.read_text())
    strategy = load_strategy(args.strategy or fetch_strategy())

    # Step 3: read the frozen strategy
    print("grading the frozen strategy...", flush=True)
    frozen = grade(strategy)
    round_one = round_one_frequencies(strategy)
    draw = draw_frequencies(strategy)

    # Step 4: draw the figures
    fig_race(grades, args.out / "race.png")
    fig_columns(grades, args.out / "averaging_columns.png")
    fig_round_one(round_one, args.out / "round_one_betting.png")
    fig_draw(draw, args.out / "draw.png")

    # Step 5: write and print the answer sheet
    sheet = answer_sheet(
        grades=grades, info=strategy.info, frozen=frozen, round_one=round_one, draw=draw
    )
    (args.out / "answer_sheet.json").write_text(json.dumps(sheet, indent=2) + "\n")
    print(f"wrote {(args.out / 'answer_sheet.json').resolve()}")
    print_summary(sheet)

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rung-3 analysis")
    parser.add_argument("--grades", type=Path, default=GRADES)
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument(
        "--grade-runs", type=Path, help="grade DIR/<rule>/iter-*.npz into --grades first"
    )
    parser.add_argument("--strategy", type=Path, help="a frozen strategy .npz")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Grading checkpoints into grades.json
# ---------------------------------------------------------------------------

def grade_runs(runs: Path, grades_path: Path) -> None:
    """Grade every checkpoint and column not yet in `grades_path`, saving after each.

    Reads `runs/<rule>/iter-*.npz` for each rule in `RULES`; any other
    directory under `runs` (the Volume keeps a `grades/` there) is not a run
    and is left alone. Each checkpoint is graded in its primary column and
    every extra column it banked. A key already in the file is never redone,
    and the file is rewritten after each grade, so an interrupted pass keeps
    what it finished and the next one resumes.

    Also grades the uniform-random profile once, as the `uniform_random`
    reference line the race is drawn against, so every entry in the committed
    file has a producer here.

    Refuses a `runs` that is not a directory — a mistyped path would otherwise
    grade nothing and look like a pass with nothing new — and a checkpoint
    whose recorded rule is not the one its directory names, which would bank
    one rule's grade under another's line.
    """
    if not runs.is_dir():
        raise FileNotFoundError(f"--grade-runs {runs} is not a directory")
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else {}

    def bank(key: str, value: Grade) -> None:
        grades[key] = value
        grades_path.parent.mkdir(parents=True, exist_ok=True)
        grades_path.write_text(json.dumps(grades, indent=2, sort_keys=True) + "\n")

    if UNIFORM_RANDOM not in grades:
        print(f"grading {UNIFORM_RANDOM}...", flush=True)
        bank(UNIFORM_RANDOM, grade(uniform_strategy()))
    for name, rule in RULES.items():
        for checkpoint in sorted((runs / name).glob("iter-*.npz")):
            run = read_lockstep(checkpoint)
            if RegretRule(run["rule"]) is not rule.trains:
                raise ValueError(
                    f"{checkpoint} was trained under {run['rule']!r}, "
                    f"not the {rule.trains.value!r} its directory names"
                )
            for column in (PRIMARY_AVERAGE, *validate_averages(run["averages"])):
                key = grade_key(name, run["iteration"], column)
                if key in grades:
                    continue
                print(f"grading {key}...", flush=True)
                bank(key, {
                    **grade(from_checkpoint(checkpoint, column)),
                    "workers": run["workers"],
                    "seed": run["seed"],
                })

def grade_key(rule: str, iteration: int, column: Average | str) -> str:
    """The `grades.json` key for one rule's checkpoint read in one averaging column."""
    return f"{rule}/{int(iteration):09d}/{Average(column).value}"

def grade(strategies: Mapping) -> Grade:
    """Both best responses, their mean, and P0's value in self-play, in chips/hand.

    `strategies` is any profile the exact grader takes: a `Mapping` from
    `InfoSet` to its probability row. This is the two-minute call.
    """
    br = [best_response_value(strategies, responder=seat) for seat in (0, 1)]
    return {
        "br0": br[0],
        "br1": br[1],
        "exploitability": (br[0] + br[1]) / 2,
        "value_p0": expected_value(strategies)[0],
    }

# ---------------------------------------------------------------------------
# Reading the frozen strategy, spot by spot
# ---------------------------------------------------------------------------

def round_one_frequencies(strategies: Mapping) -> dict[str, dict[str, list[float]]]:
    """Round-1 strategies by inner category, averaged over every physical deal.

    Keyed by spot (`ROUND_ONE_SPOTS`), then by category name in `InnerCategory`
    order; each value is the mean probability row.
    """
    return _spot_frequencies(strategies, ROUND_ONE_SPOTS)

def draw_frequencies(strategies: Mapping) -> dict[str, dict[str, list[float]]]:
    """Draw strategies after a checked-through round 1, by inner category.

    Keyed by spot (`DRAW_SPOTS`), then by category name in `InnerCategory`
    order; each value is the mean probability row.
    """
    return _spot_frequencies(strategies, DRAW_SPOTS)

def _spot_frequencies(
    strategies: Mapping, spots: Mapping[str, _Spot]
) -> dict[str, dict[str, list[float]]]:
    rows: dict[str, dict[InnerCategory, list[np.ndarray]]] = {name: {} for name in spots}
    for hole, board in _holes_and_boards():
        category = inner_score(hole).category
        for name, spot in spots.items():
            rows[name].setdefault(category, []).append(_row(strategies, spot, hole, board))
    # Every category is dealt (455 holes cover all six), so indexing each one
    # is safe, and a category that went missing would raise here.
    return {
        name: {c.name.lower(): np.mean(found[c], axis=0).tolist() for c in InnerCategory}
        for name, found in rows.items()
    }

def _holes_and_boards() -> Iterator[tuple[tuple[Card, ...], tuple[Card]]]:
    """Every physical (hole, first board card) pair: 455 holes × 12 boards."""
    for hole in combinations(DECK, 3):
        for card in DECK:
            if card not in hole:
                yield hole, (card,)

def _row(
    strategies: Mapping, spot: _Spot, hole: tuple[Card, ...], board: tuple[Card]
) -> np.ndarray:
    hole, discarded, board = canonical_picture(hole, (), board)
    key = InfoSet(
        player=spot.player, hole=hole, discarded=discarded, board=board,
        draws=spot.draws, betting=spot.betting,
    )
    return np.asarray(strategies[key], dtype=np.float64)

# ---------------------------------------------------------------------------
# Reading grades.json
# ---------------------------------------------------------------------------

def series(
    grades: Mapping[str, Grade], rule: str, *, column: Average | str = PRIMARY_AVERAGE
) -> list[tuple[int, float]]:
    """(hands per seat, exploitability) for one rule and column, in training order."""
    return sorted(
        (point.hands, point.grade["exploitability"])
        for point in _points(grades)
        if point.rule == rule and point.column == column
    )

def _points(grades: Mapping[str, Grade]) -> Iterator[_Point]:
    """Every run entry, skipping the reference line and refusing any key it cannot read."""
    for key, value in grades.items():
        if key == UNIFORM_RANDOM:
            continue
        parts = key.split("/")
        if len(parts) != 3:
            raise ValueError(f"{key!r} is neither {UNIFORM_RANDOM!r} nor rule/iteration/column")
        rule, iteration, column = parts
        if rule not in RULES:
            raise ValueError(f"{key!r} names no known rule; expected one of {list(RULES)}")
        if column not in {average.value for average in Average}:
            raise ValueError(
                f"{key!r} names no averaging column; expected one of {[a.value for a in Average]}"
            )
        yield _Point(rule=rule, iteration=int(iteration), column=Average(column), grade=value)

def _workers(grades: Mapping[str, Grade]) -> int:
    """The lockstep width every run shares; the figures' hand counts assume one."""
    widths = {int(point.grade["workers"]) for point in _points(grades)}
    if len(widths) != 1:
        raise ValueError(f"the runs disagree on hands per iteration: {sorted(widths)}")
    return widths.pop()

def _in_millions(hands: int) -> str:
    return f"{hands / 1e6:g}M"

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_race(grades: Mapping[str, Grade], out: Path) -> None:
    """Exploitability against hands sampled, one line per rule, over the uniform line.

    Refuses grades without the `uniform_random` reference line, which
    `grade_runs` always banks. The title claims that keeping negative regret
    wins only when every rule that keeps it finishes below every rule that
    does not.
    """
    if UNIFORM_RANDOM not in grades:
        raise ValueError(f"no {UNIFORM_RANDOM!r} line to draw the race against; "
                         "`--grade-runs` banks it")
    lines = {name: series(grades, name) for name in RULES}
    fig, ax = new_axes(_race_title(lines))
    for name, rule in RULES.items():
        hands, value = zip(*lines[name])
        ax.plot(hands, np.array(value) * 100, color=rule.colour, linestyle=rule.linestyle,
                linewidth=rule.linewidth, marker="o", markersize=3.5, label=rule.label,
                zorder=rule.zorder)
    ax.axhline(grades[UNIFORM_RANDOM]["exploitability"] * 100, color=BASELINE,
               linewidth=1.2, linestyle="--", label="uniform random play")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"hands sampled per seat ({_workers(grades)} per lockstep iteration)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("exploitability (chips per 100 hands, ante 1)", color=MUTED, fontsize=9)
    legend(ax, loc="lower left")
    save(fig, out)

def _race_title(lines: Mapping[str, list[tuple[int, float]]]) -> str:
    last = {name: line[-1] for name, line in lines.items()}
    final = {name: value * 100 for name, (_, value) in last.items()}

    def standings(keeps: bool) -> list[str]:
        names = [name for name, rule in RULES.items() if rule.keeps_negative_regret is keeps]
        return sorted(names, key=final.__getitem__)

    keepers, flooring = standings(True), standings(False)
    wins = max(final[name] for name in keepers) < min(final[name] for name in flooring)
    verdict = "Keeping negative regret wins" if wins else "Keeping negative regret does not win outright"
    hands = {count for count, _ in last.values()}
    at = f"at {_in_millions(hands.pop())} hands" if len(hands) == 1 else "at each run's last checkpoint"

    def listed(names: list[str]) -> str:
        return " and ".join(f"{RULES[name].label} {final[name]:.1f}" for name in names)

    return f"{verdict}: {listed(keepers)} vs {listed(flooring)} chips/100 {at}"

def fig_columns(grades: Mapping[str, Grade], out: Path) -> None:
    """Every averaging column at the race's last iteration, as bars.

    Refuses rather than draw a rule whose last checkpoint is missing, since its
    bars would then compare a shorter run against the others. The title's
    verdicts are read off the bars, each column against its rule's linear one.
    """
    points = list(_points(grades))
    last = max(point.iteration for point in points)
    behind = sorted({p.rule for p in points} - {p.rule for p in points if p.iteration == last})
    if behind:
        raise ValueError(f"{behind} have no checkpoint at iteration {last:,}")
    at = {(p.rule, p.column): p.grade["exploitability"] * 100
          for p in points if p.iteration == last}
    fig, ax = new_axes(
        "Averaging is a free choice at train time:\n" + "; ".join(_column_verdicts(at)),
        figsize=(8, 3.8),
    )
    bars = sorted(at.items(), key=lambda bar: bar[1])
    labels = [f"{RULES[rule].label} · {column.value}" for (rule, column), _ in bars]
    values = [value for _, value in bars]
    ax.barh(labels, values, color=[COLUMN_COLOURS[column] for (_, column), _ in bars],
            height=0.6)
    for y, value in enumerate(values):
        ax.text(value + 0.4, y, f"{value:.1f}", va="center", color=SECONDARY, fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=BASELINE, linewidth=0.5)
    ax.set_xlabel(f"exploitability at {_in_millions(last * _workers(grades))} hands "
                  "(chips per 100 hands)", color=MUTED, fontsize=9)
    save(fig, out)

def _column_verdicts(at: Mapping[tuple[str, Average], float]) -> list[str]:
    """One `winner beats loser for <rules>` clause per column pairing, in `RULES` order."""
    verdicts: dict[tuple[Average, Average], list[str]] = {}
    for name, rule in RULES.items():
        for column in Average:
            if column is PRIMARY_AVERAGE or (name, column) not in at:
                continue
            linear = at[name, PRIMARY_AVERAGE]
            pair = (PRIMARY_AVERAGE, column) if linear < at[name, column] else (column, PRIMARY_AVERAGE)
            verdicts.setdefault(pair, []).append(rule.label)
    return [f"{winner.value} beats {loser.value} for {' and '.join(labels)}"
            for (winner, loser), labels in verdicts.items()]

def fig_round_one(freqs: Mapping[str, Mapping[str, list[float]]], out: Path) -> None:
    """P0's lead, P1's bet when checked to and P1's fold to a bet, by inner category."""
    made = min(freqs["p1_checked_to"][c][1] for c in _MADE)
    fold = {c: freqs["p1_facing_bet"][c][0] for c in _WEAK}
    fig, ax = new_axes(
        f"Round 1: checked to, P1 bets trips and straight flushes {made:.0%};\n"
        f"facing a bet it folds a pair {fold['pair']:.0%}, high card {fold['high_card']:.0%}",
        figsize=(8.5, 4.5),
    )
    categories = list(freqs["p0_open"])
    _category_bars(ax, categories, (
        ("P0 bets out", CATEGORICAL[0], [freqs["p0_open"][c][1] for c in categories]),
        ("P1 bets when checked to", CATEGORICAL[2],
         [freqs["p1_checked_to"][c][1] for c in categories]),
        ("P1 folds to a bet", CATEGORICAL[1], [freqs["p1_facing_bet"][c][0] for c in categories]),
    ))
    ax.set_ylabel("frequency, every deal of the category weighted equally", color=MUTED, fontsize=9)
    legend(ax, loc="upper left")
    save(fig, out)

def fig_draw(freqs: Mapping[str, Mapping[str, list[float]]], out: Path) -> None:
    """How often each player throws a card after round 1 checks through, by inner category."""
    thrown = {name: {c: 1 - row[0] for c, row in by_category.items()}
              for name, by_category in freqs.items()}
    weak = [thrown[name][c] for name in thrown for c in _WEAK]
    strong = max(thrown[name][c] for name in thrown for c in _STRONG)
    straight = {name: thrown[name]["straight"] for name in ("p1_after_pat", "p1_after_draw")}
    fig, ax = new_axes(
        f"The draw: high cards and pairs throw {min(weak):.0%}–{max(weak):.0%}, "
        f"flushes and better at most {strong:.0%};\n"
        f"P1's straight throws {straight['p1_after_pat']:.0%} after a pat P0, "
        f"{straight['p1_after_draw']:.0%} after a draw",
        figsize=(8.5, 4.5),
    )
    categories = list(freqs["p0"])
    _category_bars(ax, categories, tuple(
        (label, colour, [thrown[name][c] for c in categories])
        for name, label, colour in (
            ("p0", "P0 draws", CATEGORICAL[0]),
            ("p1_after_pat", "P1 draws, P0 stood pat", CATEGORICAL[2]),
            ("p1_after_draw", "P1 draws, P0 drew one", CATEGORICAL[1]),
        )
    ))
    ax.set_ylabel("frequency of throwing a card (round 1 checked through)", color=MUTED, fontsize=9)
    legend(ax, loc="upper right")
    save(fig, out)

def _category_bars(
    ax, categories: list[str], bars: Sequence[tuple[str, str, list[float]]]
) -> None:
    """Three bars per inner category, side by side, on a 0–1 frequency axis."""
    positions = np.arange(len(categories))
    width = 0.27
    for offset, (label, colour, values) in zip((-width, 0, width), bars, strict=True):
        ax.bar(positions + offset, values, width, color=colour, label=label)
    ax.set_xticks(positions, [c.replace("_", " ") for c in categories])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("what the three dealt cards make on their own (inner half)",
                  color=MUTED, fontsize=9)

# ---------------------------------------------------------------------------
# The answer sheet
# ---------------------------------------------------------------------------

def answer_sheet(
    *,
    grades: Mapping[str, Grade],
    info: StrategyInfo,
    frozen: Grade,
    round_one: dict[str, dict[str, list[float]]],
    draw: dict[str, dict[str, list[float]]],
) -> dict:
    """Every number the figures and the README quote, as plain JSON.

    `race_final_linear` is each rule's last linear-column point; `frozen_strategy`
    is the frozen strategy's grade, where it came from and the digest it is
    pinned to; `columns` names the entries of each readout's rows.
    """
    final = {}
    for name, rule in RULES.items():
        hands, value = series(grades, name)[-1]
        final[rule.label] = {"hands_per_seat": hands, "exploitability": value}
    return {
        "race_final_linear": final,
        "frozen_strategy": {
            **frozen,
            "rule": info.rule,
            "column": info.column,
            "iteration": info.iteration,
            "workers": info.workers,
            "seed": info.seed,
            "sha256": STRATEGY_SHA256,
        },
        "round_one": round_one,
        "draw": draw,
        "columns": {
            "round_one": {"p0_open": ["check", "bet"], "p1_checked_to": ["check", "bet"],
                          "p1_facing_bet": ["fold", "call", "raise"]},
            "draw": ["stand pat", "throw low", "throw mid", "throw top"],
        },
    }

def print_summary(sheet: dict) -> None:
    """The frozen strategy's grade and the race's final standings, for the terminal."""
    frozen = sheet["frozen_strategy"]
    print(f"\nrung 3: {frozen['rule'].upper()} after {frozen['iteration']:,} iterations "
          f"({frozen['workers']} hands per seat each), {frozen['column']} average\n")
    print(f"  exploitability            {frozen['exploitability']:.6f} chips/hand "
          f"({frozen['exploitability'] * 100:.2f} per 100)")
    print(f"  best responses            BR_0 {frozen['br0']:+.6f}   BR_1 {frozen['br1']:+.6f}")
    print(f"  game value to P0          {frozen['value_p0']:+.6f} (self-play)")
    print("\n  the race, linear average, at the last checkpoint:")
    for rule, value in sheet["race_final_linear"].items():
        print(f"    {rule:<8} {value['exploitability'] * 100:6.2f} per 100 "
              f"at {value['hands_per_seat']:,} hands")

if __name__ == "__main__":
    main()
