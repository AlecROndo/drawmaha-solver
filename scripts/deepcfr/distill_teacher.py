"""Distil the LCFR table into the rung-4 net, arm by arm, and grade every copy exactly.

    caffeinate -i uv run --extra deep python scripts/deepcfr/distill_teacher.py \\
        <lcfr-run>/runs/lcfr/iter-010000000.npz \\
        --out runs/distill --results figures/rung4/distill/results.json

`<lcfr-run>` is the directory of the LCFR run whose iteration-10,000,000
checkpoint is the teacher (`teacher.checkpoint` in the results file); it
is not in the repository. That command, with resumes, wrote
`figures/rung4/distill/results.json` on 2026-11-05; the last resume was
the same command with `--arms 4x256-400k`. `--arms a,b,c` runs a subset; an arm
already in the file is skipped, so an interrupted pass resumes.
`--steps N` overrides every arm's step count (a smoke run writes to a
scratch `--results`, never the committed one). Each invocation that writes
the file is appended to its `_commands`, oldest first; the committed file
predates that list and holds only the last one, the 400k resume, under
`_command`, which a later run moves to the head of `_commands`.

**The question** is `deepcfr/distill.py`'s: the ÷ pot Deep CFR arm, a
4 × 256 net, flattens at exact-average exploitability 0.068 while the
tabular LCFR run keeps falling. Fitting the same net straight to the
table's average strategy, by reach-weighted cross-entropy over every
infoset, and grading the copy, splits the plateau into what the net cannot
hold (teacher vs copy) and what Deep CFR's training loses (copy vs Deep
CFR).

**The arms.** `ARMS` names each by depth × width and a suffix for what
differs from the headline `4x256`: `-25k` / `-400k` fewer or more steps,
`-unweighted` rows drawn uniformly instead of by reach, `-f0.1` / `-f0.5`
trained on a random tenth / half of the rows and graded on the whole game.
Every arm is 100,000 Adam steps of 16,384 rows under a cosine schedule
from 1e-3 unless its suffix says otherwise. The headline's error curve was
still easing down when the schedule ran out (0.0452 → 0.0450 over the
last 10,000 steps, the learning rate 2.5e-13 at the last one), so the `-400k` arm asks how much of the copy's gap to
the teacher is optimisation rather than capacity, and `-25k` what fewer
steps would have cost.

**What is recorded per arm**: its config, the parameter count, the fit's
seconds, the curve (`distill`'s entries), the frozen file's name and
SHA-256, the reach-weighted error against the teacher of the strategy
reloaded from that file (`rw_error`, with `rw_error_seen` and
`rw_error_unseen` for a fraction arm), and the exact grade of that same
reloaded strategy (`br0`, `br1`, `exploitability`, `value_p0`, `grade_s`).
`teacher` holds the teacher's own grade, measured here once (two minutes),
and `deep_cfr` the committed grades of the Deep CFR arm this experiment
explains, each labelled with hands per seat read from its file: the
teacher's is iteration × workers (W = 10 hands per lockstep iteration),
Deep CFR's is the grades file's own field (10,000 walks per iteration).

The teacher checkpoint is only read, through `strategy.from_checkpoint`,
which loads one averaging column and nothing else of the 555 MB file.
torch is imported inside the fit, so this file imports without it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from drawmaha_solver.deepcfr.distill import (
    DistillConfig,
    distill,
    reach_weighted_error,
    row_weights,
    student_strategy,
    teacher_targets,
    training_rows,
)
from drawmaha_solver.minidrawmaha.answer import analysis
from drawmaha_solver.minidrawmaha.answer.strategy import (
    FrozenStrategy,
    StrategyInfo,
    from_checkpoint,
    load_strategy,
    save_strategy,
    sha256_of,
)
from drawmaha_solver.minidrawmaha.rules.features import FEATURE_WIDTH, HEAD_WIDTH

# The Deep CFR arm this experiment explains: the ÷ pot 4 × 256 run to the
# race's endpoint, graded at t = 2,000 (`figures/AGENTS.md`).
DEEP_GRADES = Path("figures/rung4/pot_1-256x4-s16000-lr5e-4.json")
DEEP_T = 2_000
HEADLINE = "4x256"
RESULTS = Path("figures/rung4/distill/results.json")

_HEADLINE = DistillConfig(width=256, depth=4, device="mps")
ARMS: dict[str, DistillConfig] = {
    "3x64": replace(_HEADLINE, width=64, depth=3),
    "3x128": replace(_HEADLINE, width=128, depth=3),
    HEADLINE: _HEADLINE,
    "4x512": replace(_HEADLINE, width=512),
    "4x1024": replace(_HEADLINE, width=1024),
    "4x256-25k": replace(_HEADLINE, steps=25_000),
    "4x256-400k": replace(_HEADLINE, steps=400_000, eval_every=20_000),
    "4x256-unweighted": replace(_HEADLINE, weighted=False),
    "4x256-f0.5": replace(_HEADLINE, train_fraction=0.5),
    "4x256-f0.1": replace(_HEADLINE, train_fraction=0.1),
}
# The headline first, the biggest last, so a stopped pass has the answer.
ORDER = (
    "4x256", "3x64", "3x128", "4x512", "4x256-25k", "4x256-unweighted",
    "4x256-f0.5", "4x256-f0.1", "4x1024", "4x256-400k",
)

# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------

def parameter_count(*, width: int, depth: int) -> int:
    """Weights and biases of `Net(width, depth)`: 136 → width, depth − 1 times width → width, width → 7."""
    return (FEATURE_WIDTH * width + width) + (depth - 1) * (width * width + width) + (width * HEAD_WIDTH + HEAD_WIDTH)

def references(teacher: StrategyInfo, teacher_grade: dict, checkpoint: Path, deep_grades: Path) -> dict:
    """The two things every copy is read against, each labelled with hands per seat from its own file."""
    deep = json.loads(deep_grades.read_text())
    exact = deep[f"deep/{DEEP_T:05d}/exact"]
    config = exact["config"]
    return {
        "teacher": {
            "checkpoint": checkpoint.name,
            "rule": teacher.rule,
            "column": teacher.column,
            "iteration": teacher.iteration,
            "seed": teacher.seed,
            "workers": teacher.workers,
            "hands_per_seat": teacher.iteration * teacher.workers,
            **teacher_grade,
        },
        "deep_cfr": {
            "file": str(deep_grades),
            "t": DEEP_T,
            "hands_per_seat": exact["hands_per_seat"],
            "width": config["width"],
            "depth": config["depth"],
            "parameters": parameter_count(width=config["width"], depth=config["depth"]),
            "pot_power": config["pot_power"],
            **{
                column: {
                    key: deep[f"deep/{DEEP_T:05d}/{column}"][key]
                    for key in ("br0", "br1", "exploitability", "value_p0")
                }
                for column in ("exact", "pi", "current")
            },
        },
    }

def arm_record(
    config: DistillConfig,
    *,
    curve: list[dict],
    fit_s: float,
    grade: dict,
    grade_s: float,
    strategy_file: str,
    strategy_sha256: str,
    errors: dict[str, float],
) -> dict:
    """One arm's line in the results file: config, size, the fit, the file, the errors and the grade."""
    return {
        **asdict(config),
        "parameters": parameter_count(width=config.width, depth=config.depth),
        "fit_s": round(fit_s, 1),
        "curve": curve,
        "strategy_file": strategy_file,
        "strategy_sha256": strategy_sha256,
        **errors,
        **grade,
        "grade_s": round(grade_s, 1),
    }

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def record_command(held: dict, command: str) -> None:
    """Append `command` to the file's `_commands`, oldest first, so a resume keeps the provenance of every arm before it.

    A file written before the list existed holds its last command alone
    under `_command`; that one moves to the head of the list.
    """
    commands = held.pop("_commands", [])
    if "_command" in held:
        commands.insert(0, held.pop("_command"))
    commands.append(command)
    held["_commands"] = commands

def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def _errors(student: FrozenStrategy, targets, weights, config: DistillConfig) -> dict[str, float]:
    """The reloaded student's reach-weighted error over every row, and over the seen and unseen rows of a fraction arm."""
    errors = {"rw_error": reach_weighted_error(student, targets, weights)}
    seen = training_rows(config, len(weights))
    if seen is not None:
        mask = weights.copy()
        mask[:] = 0.0
        mask[seen] = weights[seen]
        errors["rw_error_seen"] = reach_weighted_error(student, targets, mask)
        errors["rw_error_unseen"] = reach_weighted_error(student, targets, weights - mask)
    return errors

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("teacher", type=Path, help="a lockstep checkpoint; its linear average is the teacher")
    parser.add_argument("--out", type=Path, default=Path("runs/distill"), help="where the frozen copies go")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--arms", default="", help="comma-separated subset of ARMS; default every arm in ORDER")
    parser.add_argument("--steps", type=int, default=None, help="override every arm's steps (smoke runs)")
    parser.add_argument("--device", default="mps", choices=("cpu", "mps"))
    args = parser.parse_args(argv)
    arms = args.arms.split(",") if args.arms else list(ORDER)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    held = json.loads(args.results.read_text()) if args.results.exists() else {}
    record_command(held, " ".join(["uv run --extra deep python", "scripts/deepcfr/distill_teacher.py", *(argv or sys.argv[1:])]))

    started = time.perf_counter()
    teacher = from_checkpoint(args.teacher)
    targets = teacher_targets(teacher)
    weights = row_weights(teacher)
    _log(
        f"teacher {args.teacher.name}: {teacher.info.rule} {teacher.info.column} at iteration "
        f"{teacher.info.iteration:,}, W = {teacher.info.workers}; dataset built in {time.perf_counter() - started:.1f} s; "
        f"{int((weights > 0).sum()):,} of {len(weights):,} rows carry weight"
    )
    if "teacher" not in held:
        graded_at = time.perf_counter()
        grade = analysis.grade(teacher)
        grade["grade_s"] = round(time.perf_counter() - graded_at, 1)
        held.update(references(teacher.info, grade, args.teacher, DEEP_GRADES))
        _write(args.results, held)
        _log(f"teacher graded: exploitability {grade['exploitability']:.6f} in {grade['grade_s']:.0f} s")
    held.setdefault("arms", {})

    for arm in arms:
        if arm in held["arms"]:
            _log(f"{arm}: already in {args.results}, skipped")
            continue
        config = replace(ARMS[arm], device=args.device)
        if args.steps is not None:
            config = replace(config, steps=args.steps, eval_every=min(config.eval_every, args.steps))
        _log(f"{arm}: {config}")
        fit_started = time.perf_counter()
        net, curve = distill(targets, weights, config, log=lambda line, arm=arm: _log(f"{arm}  {line}"))
        fit_s = time.perf_counter() - fit_started
        path = args.out / f"{arm}.npz"
        save_strategy(student_strategy(net, teacher=teacher.info), path)
        # Everything measured below is of the file, so the number recorded
        # is the one a player loading it would get.
        student = load_strategy(path)
        errors = _errors(student, targets, weights, config)
        _log(f"{arm}: frozen to {path} ({path.stat().st_size / 1e6:.1f} MB); rw_error {errors['rw_error']:.5f}; grading")
        graded_at = time.perf_counter()
        grade = analysis.grade(student)
        grade_s = time.perf_counter() - graded_at
        held["arms"][arm] = arm_record(
            config, curve=curve, fit_s=fit_s, grade=grade, grade_s=grade_s,
            strategy_file=str(path), strategy_sha256=sha256_of(path), errors=errors,
        )
        _write(args.results, held)
        _log(f"{arm}: exploitability {grade['exploitability']:.6f} (BR0 {grade['br0']:+.6f}, BR1 {grade['br1']:+.6f}); fit {fit_s:.0f} s, grade {grade_s:.0f} s")
    _log("done")

def _write(path: Path, held: dict) -> None:
    path.write_text(json.dumps(held, indent=2) + "\n")

if __name__ == "__main__":
    main()
