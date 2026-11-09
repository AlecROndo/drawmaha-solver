"""Train Deep CFR on mini-drawmaha on this machine, resumably.

    caffeinate -i uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/deep-smoke --target 3 --walks 1000
    caffeinate -i uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/line --target 200 --boost line --boost-budget 500

Re-running the same command resumes from `<out>/resume`, the exact average
(`deepcfr.average`) included. Ctrl-C (or SIGTERM) finishes the iteration and
saves. See `deepcfr.train` for what is kept and which flags a resume may
change (`--workers` and `--device` yes; the seed, the walks, the capacity,
the net's size, `--pot-power` and every `--boost…` flag no). `--device mps`
is checked against the machine before the run is loaded, so a missing GPU
costs no walks. The refit's targets are each regret sample over √pot at its
spot by default (`deepcfr.train.pot_scaled`); a run saved before that was
the default fitted raw chips, and resumes only with `--pot-power 0`.

`--boost` names the targeted-sampling arm (`deepcfr.targeting`: random,
infoset, line or line_class; none by default, which is today's run bit for
bit). The `--boost-*` flags are its budget B, refresh R, α, whose leak it
follows, its pool and noise cap, and what one boosted row samples below its
row (`boost_target.Sampling`: the play-outs, and for each random source a
count or `all`). A sampling that draws nothing is refused here: the boost is
extra sampling, never the exact sum. With an arm named, this script builds
the grader-backed refresher (`targeting_refresh.GraderRefresher`, the one
place the trainer's side runs the meter) around the run's exact average and
hands the loop a `deepcfr.boost.Boost`.
"""

import argparse
import importlib.util
from collections.abc import Sequence
from pathlib import Path

from drawmaha_solver.deepcfr.average import ExactAverage
from drawmaha_solver.deepcfr.boost import Boost
from drawmaha_solver.deepcfr.targeting import GRANULARITIES, LEAK_SOURCES, TargetSpec
from drawmaha_solver.deepcfr.train import DEVICES, DeepConfig, run_deep
from drawmaha_solver.minidrawmaha.solver.boost_target import Sampling

def _sibling(name: str):
    """A module from this directory, loaded by path: scripts are run from the repo root and are not a package."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

GraderRefresher = _sibling("targeting_refresh").GraderRefresher

# The flags that describe the aim (`TargetSpec`); each is refused without `--boost <arm>`.
# The `--boost-*` sampling flags are not among them: they describe `DeepConfig.boost`, which every run carries.
AIM_FLAGS = (
    "--boost-budget", "--boost-refresh", "--boost-alpha", "--boost-leak", "--boost-pool", "--boost-noise-cap", "--boost-round-two-only",
)

def _dest(flag: str) -> str:
    """argparse's attribute name for a flag: `--boost-noise-cap` → `boost_noise_cap`."""
    return flag.removeprefix("--").replace("-", "_")

def int_or_all(text: str) -> int | None:
    """A `Sampling` count from the command line: a positive int, or `all` for every branch."""
    if text == "all":
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive int or 'all', got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive int or 'all', got {text!r}")
    return value

def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line: `--out` and `--target`, then one flag per run parameter."""
    defaults = DeepConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2_000, help="iterations to reach")
    parser.add_argument(
        "--walks", type=int, default=defaults.walks, help="K walks per seat per iteration"
    )
    parser.add_argument(
        "--workers", type=int, default=defaults.workers, help="W walk processes; 0 walks in-process"
    )
    parser.add_argument("--capacity", type=int, default=defaults.capacity, help="C rows per memory")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Adam steps per refit")
    parser.add_argument("--batch", type=int, default=defaults.batch, help="rows per step")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Adam's rate for every refit step")
    parser.add_argument(
        "--threads", type=int, default=defaults.threads, help="torch threads for the refit"
    )
    parser.add_argument(
        "--device", choices=DEVICES, default=defaults.device, help="where the refit runs"
    )
    parser.add_argument(
        "--rule", choices=("linear", "dcfr", "cfr+"), default=defaults.rule,
        help="regret rule of the refit's targets",
    )
    parser.add_argument(
        "--pot-power", type=float, default=defaults.pot_power,
        help="divide each regret sample by its spot's pot to this power before the refit; "
        "0 fits raw chips (the spec's), 0.5 weighs each spot by its chip cost (the default); "
        "a run saved before this flag existed resumes only under 0",
    )
    parser.add_argument(
        "--save-every-s", type=float, default=defaults.save_every_s,
        help="seconds between resume saves; a resume may change it",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    aim = TargetSpec.__dataclass_fields__
    boost = parser.add_argument_group("the targeted-sampling boost (deepcfr.targeting, deepcfr.boost)")
    boost.add_argument(
        "--boost", choices=("none", *GRANULARITIES), default="none",
        help="the arm: which groups the boost aims at; none runs without the boost",
    )
    boost.add_argument("--boost-budget", type=int, default=aim["budget"].default, help="B, rows boosted per seat per iteration")
    boost.add_argument("--boost-refresh", type=int, default=aim["refresh"].default, help="R, iterations between refreshes")
    boost.add_argument(
        "--boost-alpha", type=float, default=0.0,
        help="weight left on the walk's own rows at a targeted infoset; the boosted row takes (1 − α) · K · arrival",
    )
    boost.add_argument(
        "--boost-leak", choices=LEAK_SOURCES, default=aim["leak"].default,
        help="whose leak the aim follows: the frozen exact average (graded) or the table's current iterate",
    )
    boost.add_argument("--boost-pool", type=int, default=aim["pool"].default, help="line arms pool groups until pool × B reached rows")
    boost.add_argument("--boost-noise-cap", type=float, default=aim["noise_cap"].default, help="cap on a group's noise ratio")
    boost.add_argument("--boost-round-two-only", action="store_true", help="leave round-1 and draw rows out of the arm")
    sampling = Sampling()
    boost.add_argument("--boost-playouts", type=int, default=sampling.playouts, help="independent play-outs per boosted row, averaged")
    for source, text in (
        ("opponent_range", "opponent hands per physical entry of the row; all faces the whole range in one sum"),
        ("board_cards", "of the 14 second board cards"),
        ("opponent_draws", "of the opponent's 12 replacements per throw"),
        ("hero_draws", "of the traverser's 12 replacements per throw"),
        ("opponent_actions", "opponent actions drawn from σ per state; all splits by σ"),
    ):
        default = getattr(sampling, source)
        boost.add_argument(
            f"--boost-{source.replace('_', '-')}", type=int_or_all, default=default,
            help=f"{text} (default {'all' if default is None else default})",
        )
    args = parser.parse_args(argv)
    if args.device == "mps" and not mps_available():
        parser.error("--device mps needs an Apple GPU that torch can use; this machine has none")
    # An aim flag with no arm named would be today's run with nothing in the log.
    if args.boost == "none":
        aimed = [flag for flag in AIM_FLAGS if getattr(args, _dest(flag)) != parser.get_default(_dest(flag))]
        if aimed:
            parser.error(f"{', '.join(aimed)} describe an aim, but --boost is none: name the arm")
    # A bad count or an exact sampling is refused at the command line, before a run is loaded.
    try:
        sampling = boost_sampling(args)
    except ValueError as why:
        parser.error(str(why))
    if sampling.is_exact:
        parser.error(
            "the --boost-* sampling draws nothing: the boost is extra sampling at high-variance nodes, "
            "never the exact sum below a row; leave at least one source sampled"
        )
    return args

def mps_available() -> bool:
    """Whether torch can see an Apple GPU here; imported only when `--device mps` asks."""
    import torch

    return torch.backends.mps.is_available()

def boost_sampling(args: argparse.Namespace) -> Sampling:
    """What one boosted row samples, as the `--boost-*` flags say."""
    return Sampling(
        playouts=args.boost_playouts,
        opponent_range=args.boost_opponent_range,
        board_cards=args.boost_board_cards,
        opponent_draws=args.boost_opponent_draws,
        hero_draws=args.boost_hero_draws,
        opponent_actions=args.boost_opponent_actions,
    )

def target_spec(args: argparse.Namespace) -> TargetSpec | None:
    """The aim the flags describe, or None for `--boost none`."""
    if args.boost == "none":
        return None
    return TargetSpec(
        granularity=args.boost,
        alpha=args.boost_alpha,
        budget=args.boost_budget,
        refresh=args.boost_refresh,
        noise_cap=args.boost_noise_cap,
        pool=args.boost_pool,
        round_two_only=args.boost_round_two_only,
        leak=args.boost_leak,
    )

def config_from_args(args: argparse.Namespace) -> DeepConfig:
    """The run the flags describe; refused by name if a value could not be used."""
    return DeepConfig(
        seed=args.seed,
        walks=args.walks,
        workers=args.workers,
        capacity=args.capacity,
        width=args.width,
        depth=args.depth,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        threads=args.threads,
        device=args.device,
        rule=args.rule,
        pot_power=args.pot_power,
        save_every_s=args.save_every_s,
        target=target_spec(args),
        boost=boost_sampling(args),
    )

def main() -> None:
    args = parse()
    config = config_from_args(args)
    # The exact average rides along from iteration 1: it cannot be rebuilt
    # later without replaying every kept net, and its file joins the resume.
    # The boost's refresher measures its leak on that same average.
    average = ExactAverage()
    boost = None
    if config.target is not None:
        refresher = GraderRefresher(config.target, walks=config.walks, average=average, seed=config.seed)
        boost = Boost(refresher)
    run_deep(args.out, config, target=args.target, observers=(average,), boost=boost)

if __name__ == "__main__":
    main()
