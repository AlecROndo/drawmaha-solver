"""The released strategies: which frozen files the site ships, and where each one's grade is recorded.

Two exporters put a frozen strategy in front of a browser. `range_export`
cuts one by public decision point for `/solver` (and `/rung3`'s Learn tab,
which reads the same exports); `deal_pack` deals a few hundred hands in
advance for `/rung3`'s Play tab with the strategy's mix at every node. Both
need the same three things said about the strategy they ship, and this
module is the one place they are said:

* **where the file is published** — a GitHub release asset, fetched by `tag`
  and `asset` and refused unless it hashes to `sha256`, so every export is
  reproducible from a clean checkout and a local copy is known by its digest;
* **where its grade is recorded** — the committed file under `figures/` and
  the object in it that holds the exact best response's verdict, copied into
  the export's manifest at export time (`read_grade`) and held to the file by
  the tests, so a page quotes a number with a committed source and never one
  typed into TypeScript;
* **how the manifest names it** — `provenance_json`, the `strategy` block
  both `index.json` files open with, byte for byte the same shape, so the
  two pages can share the words they say about a strategy.

The catalogue (`RELEASES`) holds every released strategy, whichever page
ships it. A release with a `directory` is exported to `/solver` under
`web/solver-viz/public/<directory>/`; one without is shipped another way —
today the deal pack — and `minidraw-ranges` refuses to export it without an
explicit `--out`, so the switch on `/solver` never grows a strategy by
accident. The three `/solver` exports share their key tables and point walk
byte for byte and differ only in the chunks; that is what lets the switch
change the strategy under a spot, and it holds for any strategy of the same
rules, exported or packed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from drawmaha_solver.minidrawmaha.answer.strategy import (
    RELEASE_TAG,
    STRATEGY_FILE,
    STRATEGY_SHA256,
    StrategyInfo,
)

# The repo root, for the grades files a release names. The package is
# installed editable, so this file sits under `src/` in the checkout.
REPO = Path(__file__).resolve().parents[4]

# Where `/solver`'s exports land, relative to the repo root the CLI is run from.
PUBLIC = Path("web/solver-viz/public")

GITHUB_RELEASES = "https://github.com/AlecROndo/drawmaha-solver/releases/download"


@dataclass(frozen=True, slots=True)
class Release:
    """One strategy the site ships: where its file is published, and where its grade is recorded.

    `name` is what `--release` takes on either exporter. `directory` is where
    `minidraw-ranges` lands its export under `web/solver-viz/public/`, or
    `None` for a strategy the site ships only in the deal pack. The asset is
    fetched by `tag` and `asset` and refused unless it hashes to `sha256`, so
    the export is reproducible from a clean checkout. `grade_file` is the
    committed record the page quotes and `grade_key` the object in it that
    holds the strategy's `exploitability` and `value_p0`; `read_grade` copies
    them into the manifest, and the tests hold the committed copy to the file.
    """

    name: str
    directory: str | None
    tag: str
    asset: str
    sha256: str
    grade_file: str
    grade_key: str

    @property
    def url(self) -> str:
        return f"{GITHUB_RELEASES}/{self.tag}/{self.asset}"

    @property
    def exported(self) -> bool:
        """Whether `/solver` carries this strategy behind its switch."""
        return self.directory is not None

    @property
    def out(self) -> Path:
        if self.directory is None:
            raise ValueError(f"release {self.name} is not exported to /solver; it has no directory")
        return PUBLIC / self.directory


# The pot-normalised arm (PR #68): the 4 × 256 net with a 16,000-step refit at
# lr 5e-4 and its regret targets divided by √pot at each spot, frozen at
# iteration 200 (2,000,000 hands per seat). Its exact average is the best
# graded Deep CFR object so far, and what `/rung3`'s Play tab deals against.
POT_TAG = "rung4-deep-pot0.5-t200"
POT_GRADES = "figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json"

RELEASES: dict[str, Release] = {
    release.name: release
    for release in (
        Release(
            name="tabular",
            directory="mini",
            tag=RELEASE_TAG,
            asset=STRATEGY_FILE,
            sha256=STRATEGY_SHA256,
            grade_file="figures/rung3/answer_sheet.json",
            grade_key="frozen_strategy",
        ),
        Release(
            name="deep-pi",
            directory="mini-deep-pi",
            tag="rung4-deep-t500",
            asset="deep-256x4-s16000-lr5e-4-t500-pi.npz",
            sha256="7aa08383a7f6389bc60633904b050793ed7a3d42cb017c877c3d26ca5a2058a5",
            grade_file="figures/rung4/sweep_256x4-s16000-lr5e-4.json",
            grade_key="deep/00500/pi",
        ),
        Release(
            name="deep-exact",
            directory="mini-deep-exact",
            tag="rung4-deep-t500",
            asset="deep-256x4-s16000-lr5e-4-t500-exact.npz",
            sha256="117c936030b0271c73db516ec4bcedaf83162952ee2a9740e8fd4832424115aa",
            grade_file="figures/rung4/sweep_256x4-s16000-lr5e-4.json",
            grade_key="deep/00500/exact",
        ),
        Release(
            name="deep-pot-exact",
            directory=None,
            tag=POT_TAG,
            asset="deep-pot0.5-256x4-s16000-lr5e-4-t200-exact.npz",
            sha256="44e042f0a448331073acb5d726788877041035dba7d324c762af9095c95c0e75",
            grade_file=POT_GRADES,
            grade_key="deep/00200/exact",
        ),
        Release(
            name="deep-pot-pi",
            directory=None,
            tag=POT_TAG,
            asset="deep-pot0.5-256x4-s16000-lr5e-4-t200-pi.npz",
            sha256="374c7552958c9b24dc1e134800ad53ee20aa67e52559a6d18de078eb05255c82",
            grade_file=POT_GRADES,
            grade_key="deep/00200/pi",
        ),
    )
}


def release_of(sha256: str, releases: Mapping[str, Release] = RELEASES) -> Release | None:
    """The release whose asset hashes to `sha256`, so a local copy of one is treated as it."""
    return next((release for release in releases.values() if release.sha256 == sha256), None)


def provenance_json(info: StrategyInfo, *, sha256: str, asset: str, release: Release | None) -> dict[str, object]:
    """The `strategy` block of a manifest: the digest, the release tag (or null), the file name, and the run it came from."""
    return {
        "sha256": sha256,
        "release": release.tag if release else None,
        "asset": asset,
        "rule": info.rule,
        "column": info.column,
        "iteration": info.iteration,
        "workers": info.workers,
        "seed": info.seed,
    }


@dataclass(frozen=True, slots=True)
class Grade:
    """The strategy's exact verdict as the page quotes it, and where the numbers were read from.

    `exploitability` is what a perfect adversary wins per hand; `value_p0`
    is P0's expectation when the strategy plays itself, which is what each
    seat is worth to a player who matches the bot.
    """

    exploitability: float
    value_p0: float
    hands_per_seat: int
    file: str
    key: str

    def as_json(self) -> dict[str, object]:
        return {
            "exploitability": self.exploitability,
            "valueP0": self.value_p0,
            "handsPerSeat": self.hands_per_seat,
            "file": self.file,
            "key": self.key,
        }


# A rung-4 grades key: the object (`deep`), the zero-padded mark, the column graded.
DEEP_KEY = re.compile(r"^deep/(?P<iteration>\d+)/(?P<column>exact|pi|current)$")


def entry_mismatch(release: Release, entry: Mapping[str, object], info: StrategyInfo) -> str | None:
    """Why `entry`, found under `release.grade_key`, is not the grade of the strategy `info` describes — or `None`.

    A rung-4 key names the mark and the column, so both are read off the key.
    Any entry that records the run's own `rule`, `column` or `iteration`
    (rung 3's answer sheet does) is held to those too. An entry that says
    nothing about the run cannot be checked and is not refused.
    """
    claimed: dict[str, object] = {}
    if (key := DEEP_KEY.match(release.grade_key)) is not None:
        claimed.update(rule="deep", iteration=int(key["iteration"]), column=key["column"])
    for field in ("rule", "column", "iteration"):
        if field in entry:
            claimed[field] = entry[field]
    for field, value in claimed.items():
        if getattr(info, field) != value:
            return f"it grades {field} = {value!r} and the strategy's is {getattr(info, field)!r}"
    return None


def read_grade(release: Release, info: StrategyInfo, *, root: Path = REPO) -> Grade:
    """`release`'s grade from its committed file under `root`.

    The entry's `hands_per_seat` is kept when it is there (rung 4's grades
    carry it). Rung 3's answer sheet predates the field, and lockstep plays
    one hand per seat per worker per iteration, so there it is
    iteration × workers. The fallback is taken only when the field is absent
    — a recorded 0 is a recorded 0, not a missing field — and neither route
    may come to zero: a deep strategy records `workers=0` (its hands are
    counted by the run, not by lockstep), so a deep entry without the field
    would fall through to 0 × iteration, and the page would print "0 hands".
    A missing file, a file without the key, an entry without `value_p0`, a
    count of zero and a count that is not an integer are each refused by
    name rather than read as a number the page would then quote.

    The entry must also be the strategy's own. The digest pins the file and
    the key names the grade, and nothing else ties the two: a catalogue entry
    whose `sha256` and `grade_key` drifted apart would quote one strategy's
    grade under another's digest. So the key is read back against the
    strategy's `info` — a rung-4 key spells the mark and the column
    (`deep/00200/exact`), and rung 3's answer sheet records the run's
    `rule`, `column` and `iteration` in the entry itself — and a mismatch is
    refused by name (`entry_mismatch`).
    """
    file = root / release.grade_file
    if not file.exists():
        raise FileNotFoundError(
            f"{release.name}'s grades file {release.grade_file} is not under {root}; the exporter "
            "reads it from the checkout (the package installed editable), not from site-packages"
        )
    recorded = json.loads(file.read_text())
    if release.grade_key not in recorded:
        raise KeyError(f"{release.grade_file} has no {release.grade_key}; it holds {sorted(recorded)}")
    entry = recorded[release.grade_key]
    if (why := entry_mismatch(release, entry, info)) is not None:
        raise ValueError(f"{release.grade_file}'s {release.grade_key} is not {release.name}'s grade: {why}")
    if "value_p0" not in entry:
        raise KeyError(
            f"{release.grade_file}'s {release.grade_key} records no value_p0; the Play tab "
            "quotes what each seat is worth in self-play, so a grade it reads must carry it"
        )
    recorded_hands = "hands_per_seat" in entry
    # `type(...) is int`, not `isinstance`: a bool is an int to isinstance, and
    # a JSON `true` would otherwise be read as one hand.
    if recorded_hands and type(entry["hands_per_seat"]) is not int:
        raise ValueError(
            f"{release.grade_file}'s {release.grade_key} records hands_per_seat = "
            f"{entry['hands_per_seat']!r}, not an integer"
        )
    hands = entry["hands_per_seat"] if recorded_hands else info.iteration * info.workers
    if hands <= 0:
        counted = (
            f"records hands_per_seat = {hands}"
            if recorded_hands
            else f"records no hands_per_seat and the strategy's iteration × workers is {info.iteration} × {info.workers}"
        )
        raise ValueError(
            f"{release.grade_file}'s {release.grade_key} {counted}; a grade the page quotes "
            "must say how many hands it was trained on"
        )
    return Grade(
        exploitability=float(entry["exploitability"]),
        value_p0=float(entry["value_p0"]),
        hands_per_seat=hands,
        file=release.grade_file,
        key=release.grade_key,
    )
