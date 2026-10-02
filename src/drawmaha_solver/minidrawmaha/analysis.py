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
   the time, flushes and better almost never. P1 throws more often after
   P0 stands pat (a straight: 41% against 32%).

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
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.cards import DECK
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
    validate_averages,
)
from drawmaha_solver.minidrawmaha.strategy import (
    STRATEGY_SHA256,
    fetch_strategy,
    from_checkpoint,
    load_strategy,
    uniform_strategy,
)
from drawmaha_solver.plotting import (
    BASELINE,
    CATEGORICAL,
    INK,
    MUTED,
    SECONDARY,
    legend,
    new_axes,
    save,
)

FIGURES = Path("figures/rung3")
GRADES = FIGURES / "grades.json"

# Directory name on the Volume -> how the figures name the rule.
RULES = {"vanilla": "vanilla", "lcfr": "LCFR", "cfrplus": "CFR+", "dcfr": "DCFR"}

# Keep-negative rules in their own colours; the two that throw negative regret
# away share one, solid and dashed, because they trace the same curve.
STYLE = {
    "vanilla": (CATEGORICAL[0], "-"),
    "lcfr": (CATEGORICAL[2], "-"),
    "dcfr": (CATEGORICAL[1], "-"),
    "cfrplus": (CATEGORICAL[1], (0, (4, 3))),
}

CATEGORIES = tuple(InnerCategory)

Grade = dict[str, float]

# The reference line in `grades.json`: every infoset played uniformly.
UNIFORM_RANDOM = "uniform_random"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rung-3 analysis")
    parser.add_argument("--grades", type=Path, default=GRADES)
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument(
        "--grade-runs", type=Path, help="grade DIR/<rule>/iter-*.npz into --grades first"
    )
    parser.add_argument("--strategy", type=Path, help="a frozen strategy .npz")
    args = parser.parse_args(argv)

    if args.grade_runs is not None:
        grade_runs(args.grade_runs, args.grades)
    grades = json.loads(args.grades.read_text())

    strategy = load_strategy(args.strategy or fetch_strategy())
    print("grading the frozen strategy...", flush=True)
    frozen = grade(strategy)
    round_one = round_one_frequencies(strategy)
    draw = draw_frequencies(strategy)

    fig_race(grades, args.out / "race.png")
    fig_columns(grades, args.out / "averaging_columns.png")
    fig_round_one(round_one, args.out / "round_one_betting.png")
    fig_draw(draw, args.out / "draw.png")

    sheet = answer_sheet(grades, strategy, frozen, round_one, draw)
    (args.out / "answer_sheet.json").write_text(json.dumps(sheet, indent=2) + "\n")
    print(f"wrote {(args.out / 'answer_sheet.json').resolve()}")
    print_summary(sheet)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade(strategies: Mapping) -> Grade:
    """Both best responses, their mean, and P0's value in self-play, in chips/hand."""
    br = [best_response_value(strategies, responder=seat) for seat in (0, 1)]
    return {
        "br0": br[0],
        "br1": br[1],
        "exploitability": (br[0] + br[1]) / 2,
        "value_p0": expected_value(strategies)[0],
    }


def grade_runs(runs: Path, grades_path: Path) -> None:
    """Grade every checkpoint and column not yet in `grades_path`, saving after each.

    Also grades the uniform-random profile once, as the `uniform_random`
    reference line the race is drawn against, so every entry in the committed
    file has a producer here.
    """
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else {}

    def bank(key: str, value: Grade) -> None:
        grades[key] = value
        grades_path.parent.mkdir(parents=True, exist_ok=True)
        grades_path.write_text(json.dumps(grades, indent=2, sort_keys=True) + "\n")

    if UNIFORM_RANDOM not in grades:
        print(f"grading {UNIFORM_RANDOM}...", flush=True)
        bank(UNIFORM_RANDOM, grade(uniform_strategy()))
    for rule in RULES:
        for checkpoint in sorted((runs / rule).glob("iter-*.npz")):
            run = read_lockstep(checkpoint)
            columns = (PRIMARY_AVERAGE, *validate_averages(run["averages"]))
            for column in columns:
                key = grade_key(rule, run["iteration"], column)
                if key in grades:
                    continue
                print(f"grading {key}...", flush=True)
                bank(key, {
                    **grade(from_checkpoint(checkpoint, column)),
                    "workers": run["workers"],
                    "seed": run["seed"],
                })


def grade_key(rule: str, iteration: int, column: Average | str) -> str:
    return f"{rule}/{int(iteration):09d}/{Average(column).value}"


def _runs(grades: Mapping[str, Grade]):
    """(rule, iteration, column, grade) for every run entry, skipping reference lines."""
    for key, value in grades.items():
        if "/" not in key:
            continue
        rule, iteration, column = key.split("/")
        if rule not in RULES:
            raise ValueError(f"{key!r} names no known rule; expected one of {list(RULES)}")
        yield rule, int(iteration), column, value


def series(grades: Mapping[str, Grade], rule: str, column: str = "linear"):
    """(hands per seat, exploitability) for one rule and column, by iteration."""
    return sorted(
        (iteration * value["workers"], value["exploitability"])
        for name, iteration, col, value in _runs(grades)
        if name == rule and col == column
    )


def _workers(grades: Mapping[str, Grade]) -> int:
    """The lockstep width every run shares; the figures' hand counts assume one."""
    widths = {value["workers"] for *_, value in _runs(grades)}
    if len(widths) != 1:
        raise ValueError(f"the runs disagree on hands per iteration: {sorted(widths)}")
    return widths.pop()


def _hands(count: int) -> str:
    return f"{count / 1e6:g}M"


# ---------------------------------------------------------------------------
# Reading the strategy
# ---------------------------------------------------------------------------


def _holes_and_boards():
    """Every physical (hole, first board card) pair: 455 holes × 12 boards."""
    for hole in combinations(DECK, 3):
        for card in DECK:
            if card not in hole:
                yield hole, (card,)


def _row(strategies: Mapping, player: int, hole, board, draws, betting) -> np.ndarray:
    hole, discarded, board = canonical_picture(hole, (), board)
    key = InfoSet(
        player=player, hole=hole, discarded=discarded, board=board,
        draws=draws, betting=betting,
    )
    return np.asarray(strategies[key], dtype=np.float64)


def _by_category(rows: dict[InnerCategory, list[np.ndarray]]) -> dict[str, list[float]]:
    return {
        category.name.lower(): np.mean(rows[category], axis=0).tolist()
        for category in CATEGORIES
        if rows[category]
    }


def round_one_frequencies(strategies: Mapping) -> dict[str, dict[str, list[float]]]:
    """Round-1 strategies by inner category, averaged over every physical deal.

    Three spots: P0's open (check, bet), P1 checked to (check, bet), and P1
    facing a bet (fold, call, raise).
    """
    x, p = Action.CHECK_CALL, Action.POT
    spots = {"p0_open": (0, ((),)), "p1_checked_to": (1, ((x,),)), "p1_facing_bet": (1, ((p,),))}
    rows = {name: defaultdict(list) for name in spots}
    for hole, board in _holes_and_boards():
        category = inner_score(hole).category
        for name, (player, betting) in spots.items():
            rows[name][category].append(_row(strategies, player, hole, board, (), betting))
    return {name: _by_category(found) for name, found in rows.items()}


def draw_frequencies(strategies: Mapping) -> dict[str, dict[str, list[float]]]:
    """Draw strategies after a checked-through round 1, by inner category.

    P0 draws first; P1's row is taken after P0 stood pat and after P0 drew
    one, because P1 sees that count. Columns: stand pat, throw low, mid, top.
    """
    x = Action.CHECK_CALL
    betting = ((x, x),)
    spots = {
        "p0": (0, ()),
        "p1_after_pat": (1, (DrawSignal(0),)),
        "p1_after_draw": (1, (DrawSignal(1),)),
    }
    rows = {name: defaultdict(list) for name in spots}
    for hole, board in _holes_and_boards():
        category = inner_score(hole).category
        for name, (player, draws) in spots.items():
            rows[name][category].append(_row(strategies, player, hole, board, draws, betting))
    return {name: _by_category(found) for name, found in rows.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_race(grades: Mapping[str, Grade], out: Path) -> None:
    last = {rule: series(grades, rule)[-1] for rule in RULES}
    final = {rule: value * 100 for rule, (_, value) in last.items()}
    hands = {count for count, _ in last.values()}
    at = f"at {_hands(hands.pop())} hands" if len(hands) == 1 else "at each run's last checkpoint"
    fig, ax = new_axes(
        f"Keeping negative regret wins: LCFR {final['lcfr']:.1f} and vanilla "
        f"{final['vanilla']:.1f} vs CFR+ {final['cfrplus']:.1f} and DCFR "
        f"{final['dcfr']:.1f} chips/100 {at}"
    )
    for rule, label in RULES.items():
        colour, style = STYLE[rule]
        hands, value = zip(*series(grades, rule))
        # DCFR wide underneath, CFR+ dashed on top: the two curves coincide.
        width = 3.2 if rule == "dcfr" else 1.8
        ax.plot(hands, np.array(value) * 100, color=colour, linestyle=style,
                linewidth=width, marker="o", markersize=3.5, label=label,
                zorder=4 if rule == "cfrplus" else 3)
    if UNIFORM_RANDOM in grades:
        ax.axhline(grades[UNIFORM_RANDOM]["exploitability"] * 100, color=BASELINE,
                   linewidth=1.2, linestyle="--", label="uniform random play")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"hands sampled per seat ({_workers(grades)} per lockstep iteration)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("exploitability (chips per 100 hands, ante 1)", color=MUTED, fontsize=9)
    legend(ax, loc="lower left")
    save(fig, out)


def fig_columns(grades: Mapping[str, Grade], out: Path) -> None:
    """Every averaging column at the race's last iteration, as bars.

    Refuses rather than draw a rule whose last checkpoint is missing, since its
    bars would then compare a shorter run against the others. The title's
    verdicts are read off the bars, each column against its rule's linear one.
    """
    runs = list(_runs(grades))
    last = max(iteration for _, iteration, _, _ in runs)
    behind = sorted({rule for rule, *_ in runs} - {rule for rule, i, _, _ in runs if i == last})
    if behind:
        raise ValueError(f"{behind} have no checkpoint at iteration {last:,}")
    at = {(rule, column): value["exploitability"] * 100
          for rule, iteration, column, value in runs if iteration == last}
    bars = sorted(((RULES[rule], column, value) for (rule, column), value in at.items()),
                  key=lambda bar: bar[2])
    verdicts = defaultdict(list)  # "winner beats loser" -> the rules it holds for
    for (rule, column), value in sorted(at.items()):
        if column == "linear" or (rule, "linear") not in at:
            continue
        pair = ("linear", column) if at[rule, "linear"] < value else (column, "linear")
        verdicts[pair].append(RULES[rule])
    findings = [f"{winner} beats {loser} for {' and '.join(rules)}"
                for (winner, loser), rules in verdicts.items()]
    fig, ax = new_axes(
        "Averaging is a free choice at train time:\n" + "; ".join(findings),
        figsize=(8, 3.8),
    )
    colours = {"linear": CATEGORICAL[0], "uniform": CATEGORICAL[1], "quadratic": CATEGORICAL[2]}
    labels = [f"{rule} · {column}" for rule, column, _ in bars]
    values = [value for _, _, value in bars]
    ax.barh(labels, values, color=[colours[column] for _, column, _ in bars], height=0.6)
    for y, value in enumerate(values):
        ax.text(value + 0.4, y, f"{value:.1f}", va="center", color=SECONDARY, fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=BASELINE, linewidth=0.5)
    hands = _hands(last * _workers(grades))
    ax.set_xlabel(f"exploitability at {hands} hands (chips per 100 hands)", color=MUTED, fontsize=9)
    save(fig, out)


def _category_axis(ax, categories) -> np.ndarray:
    positions = np.arange(len(categories))
    ax.set_xticks(positions, [c.replace("_", " ") for c in categories])
    ax.set_ylim(0, 1.0)
    return positions


def fig_round_one(freqs, out: Path) -> None:
    categories = list(freqs["p0_open"])
    monsters = min(freqs["p1_checked_to"][c][1] for c in ("trips", "straight_flush"))
    fold = {c: freqs["p1_facing_bet"][c][0] for c in ("pair", "high_card")}
    fig, ax = new_axes(
        f"Round 1: checked to, P1 bets trips and straight flushes {monsters:.0%};\n"
        f"facing a bet it folds a pair {fold['pair']:.0%}, high card {fold['high_card']:.0%}",
        figsize=(8.5, 4.5),
    )
    positions = _category_axis(ax, categories)
    width = 0.27
    p0_bet = [freqs["p0_open"][c][1] for c in categories]
    p1_bet = [freqs["p1_checked_to"][c][1] for c in categories]
    p1_fold = [freqs["p1_facing_bet"][c][0] for c in categories]
    ax.bar(positions - width, p0_bet, width, color=CATEGORICAL[0], label="P0 bets out")
    ax.bar(positions, p1_bet, width, color=CATEGORICAL[2], label="P1 bets when checked to")
    ax.bar(positions + width, p1_fold, width, color=CATEGORICAL[1], label="P1 folds to a bet")
    ax.set_xlabel("what the three dealt cards make on their own (inner half)", color=MUTED, fontsize=9)
    ax.set_ylabel("frequency, every deal of the category weighted equally", color=MUTED, fontsize=9)
    legend(ax, loc="upper left")
    save(fig, out)


def fig_draw(freqs, out: Path) -> None:
    categories = list(freqs["p0"])
    straight = {name: 1 - freqs[name]["straight"][0] for name in ("p1_after_pat", "p1_after_draw")}
    fig, ax = new_axes(
        "The draw: pairs and high cards almost always throw, flushes and better stand pat;\n"
        f"P1's straight throws {straight['p1_after_pat']:.0%} after a pat P0, "
        f"{straight['p1_after_draw']:.0%} after a draw",
        figsize=(8.5, 4.5),
    )
    positions = _category_axis(ax, categories)
    width = 0.27
    for offset, (name, label, colour) in zip(
        (-width, 0, width),
        (("p0", "P0 draws", CATEGORICAL[0]),
         ("p1_after_pat", "P1 draws, P0 stood pat", CATEGORICAL[2]),
         ("p1_after_draw", "P1 draws, P0 drew one", CATEGORICAL[1])),
    ):
        drawn = [1 - freqs[name][c][0] for c in categories]
        ax.bar(positions + offset, drawn, width, color=colour, label=label)
    ax.set_xlabel("what the three dealt cards make on their own (inner half)", color=MUTED, fontsize=9)
    ax.set_ylabel("frequency of throwing a card (round 1 checked through)", color=MUTED, fontsize=9)
    legend(ax, loc="upper right")
    save(fig, out)


# ---------------------------------------------------------------------------
# The answer sheet
# ---------------------------------------------------------------------------


def answer_sheet(grades, strategy, frozen: Grade, round_one, draw) -> dict:
    final = {}
    for rule in RULES:
        hands, value = series(grades, rule)[-1]
        final[RULES[rule]] = {"hands_per_seat": hands, "exploitability": value}
    return {
        "race_final_linear": final,
        "frozen_strategy": {
            **frozen,
            "rule": strategy.info.rule,
            "column": strategy.info.column,
            "iteration": strategy.info.iteration,
            "workers": strategy.info.workers,
            "seed": strategy.info.seed,
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
