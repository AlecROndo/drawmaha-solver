---
title: Training runs on a Mac and on Modal
description: Every trainer is resumable from its output folder; the Modal launchers deploy first, claim each run with a heartbeat, and continue themselves across the 24-hour cap. Only a person launches them.
---

Two pieces of training need more than a laptop afternoon. One is the mini-drawmaha race: four regret rules, each run to 2M lockstep iterations. The other is Deep CFR's runs: 4 × 256 nets, up to 2,000 iterations of 10,000 walks per seat. Each has a local script for a Mac and a Modal launcher for the same work in the cloud.

:::warning
**Modal costs money.** Every `modal run` in this repo starts paid containers. They are launched by a person, never by an agent on its own initiative (`AGENTS.md`, `scripts/AGENTS.md`).
:::

## The local trainers resume from their output folder

```bash
caffeinate -i uv run python scripts/minidrawmaha/run_lockstep.py lcfr --out runs/lockstep-lcfr
caffeinate -i uv run --extra deep python scripts/deepcfr/run_deep_cfr.py --out runs/deep-smoke --target 3 --walks 1000
```

- **`--out` is the run.** Re-running the same command resumes from what is in the folder: `resume.npz` for lockstep, `<out>/resume/` for Deep CFR, the exact average included.
- **Ctrl-C or SIGTERM is safe.** The trainer finishes the current iteration and saves.
- **`caffeinate -i`** keeps the Mac awake. Mac sleep still counts toward the training clock, so a run's wall-clock figures include any sleep it suffered.
- **What a resume may change.** For Deep CFR, `--workers` and `--device` may change on resume. The seed, walks, capacity, net size and `--pot-power` may not. `--device mps` is checked against the machine before the run loads, so a missing GPU costs no walks.

The rows a Deep CFR run collects do not depend on the worker count, because `collect.py` draws every walk from `(seed, t, seat, walk)`. A Mac run and a Modal run of the same config see identical data. Only the refit's arithmetic order can differ.

## The Modal launchers share one shape

```bash
uv run --with modal modal deploy scripts/deepcfr/modal_rung4.py          # once, and after any code change
uv run --with modal modal run scripts/deepcfr/modal_rung4.py::smoke      # seconds per step, priced
uv run --with modal modal run scripts/deepcfr/modal_rung4.py::train      # the run, resumable
uv run --with modal modal run scripts/deepcfr/modal_rung4.py::status     # where every run stands
uv run --with modal modal run scripts/deepcfr/modal_rung4.py::grade      # every kept mark → figures/rung4/grades.json
```

`scripts/minidrawmaha/modal_rung3.py` has the same verbs for the race (`smoke`, `main --target … --rules …`, `status`). Every launcher's docstring opens with its exact commands. The design behind both:

1. **Deploy before you run.** `modal deploy` first, then `modal run …`. The run spawns on the *deployed* app, so it outlives your terminal and can spawn its own continuation.
2. **One container trains one run.** The race gives each rule a container of 11 cores (10 workers and the parent) and writes to the Volume `drawmaha-rung3` under `<rule>/` (`cfr+` is stored as `cfrplus/`). Deep CFR gives a run one container of W + 1 cores (32 walk workers and the parent) and writes to `drawmaha-rung4` under the run's name, in exactly the layout a Mac run writes: `nets/`, `marks/t00010/` …, `resume/` and `progress.jsonl`.
3. **A heartbeat claims the run.** Each trainer writes `owner.json` at every save. A newcomer refuses to start while another heartbeat is under 15 minutes old, unless that trainer's last progress line says it stopped and saved (a preemption), in which case the newcomer takes over. Two newcomers in the same second are separated by a 30-second settle and a re-read.
4. **Runs cross Modal's 24-hour cap by themselves.** A container stops after 23 hours, saves, and spawns its own continuation. Saves are committed at once, so a container that dies loses at most the time since its last save.

Deep CFR's claim, heartbeat and continuation are lifted from the race's launcher rather than imported, because that script builds its own app and Volume at import time.

## Grading fans out one container per mark

`modal_rung4.py::grade` starts one container per kept mark. Each loads the mark's memories, nets and sums and grades **three objects** with the untouched exact meter:

| Object | What it is |
| --- | --- |
| **exact average** | the true average strategy over every iteration's nets, carried along by an observer from iteration 1. It cannot be rebuilt later without replaying every kept net |
| **Π** | a fresh policy net fit on the policy memory as it stands at the mark: what a deployed Deep CFR would actually play |
| **current iterate** | the strategy this iteration's advantage nets imply, which cycles and is graded to show it |

Beside each grade go `hands_per_seat` (t · K, which lines a mark up with the tabular race), `wall_clock_s` and the run's config. The local entrypoint writes everything into `figures/rung4/grades.json`, keyed `deep/<t>/<exact|pi|current>`. A grade is a pure function of the mark, so regrading is idempotent. Then `uv run deep-analysis` draws the figures from the grades file. The three graded strategies are also kept on the Volume under `frozen/`, which is where a [release](wiki:ops/releases) comes from.

## Every arm and its marks go in the run log

Every Deep CFR arm trained so far, with where it ran, its net and refit settings, and its graded marks, is in [`docs/notes/rung4-run-log.md`](repo:docs/notes/rung4-run-log.md). Add a row when an arm is graded. Every number there is read from a file under `figures/rung4/`, and `tests/deepcfr/test_rung4_quotes.py` checks the scoreboard against those files. See [the run log page](wiki:research/run-log) for what the arms showed.

:::details Grading a checkpoint by hand
```bash
uv run python scripts/minidrawmaha/grade_checkpoint.py runs/mccfr-seed0/solve.npz
uv run python scripts/minidrawmaha/grade_checkpoint.py runs/dcfr/solve.npz --column quadratic
```
This grades one of a lockstep checkpoint's averages (linear by default, or any extra column the run banked) and prints both seats' best-response values, the exploitability and the expected value, with timings. Budget about 2 GB of memory: the packed table is about 0.3 GB, and the grader holds 1.5 GB of showdown tables.
:::
