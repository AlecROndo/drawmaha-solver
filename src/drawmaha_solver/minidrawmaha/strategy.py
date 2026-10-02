"""The frozen strategy: rung 3's answer, as one file anyone can load and play.

A training checkpoint is the wrong thing to hand a player. The LCFR run that
won the four-rule race left a 555 MB checkpoint at 2,000,000 iterations, and
almost all of it — the regrets, the two extra averaging columns, the DCFR
stamps — is the learner's working state. What the game needs from it is the
one number per legal action that the run's average strategy assigns, and
that is 14,253,840 float32s: 57 MB, and much less compressed, because most of
the rows are nearly pure.

**The layout.** Two arrays, in the packed table's order: `probabilities`,
every infoset's average strategy laid end to end in `all_infosets()` order,
and `widths`, each infoset's number of legal actions. The widths are the
fingerprint of the key layout. A file written against a different enumeration
— the old unordered-board key had 3,142,290 rows — has a different width
array, and is refused on load rather than read as somebody else's strategy.

**Which average.** The run's primary sum is linear (weight t), which is what
the race graded, but a checkpoint may carry uniform or quadratic columns
beside it and any of them can be frozen. An infoset the run never reached has
an all-zero sum and plays uniformly, exactly as `RegretMatcher.average_strategy`
reads it; 8,036 of the LCFR run's 6,220,050 rows are such.

**Float32 is enough.** The grader accepts rows within 1e-6 of summing to one,
and float32 normalisation lands within about 1e-7. Graded, the frozen LCFR
strategy matches the checkpoint's own number to the sixth decimal.

**Where it lives.** Too big to commit, so it is a GitHub release asset, and
`fetch_strategy` downloads it once into a cache directory and checks its
SHA-256 before trusting it. The digest is pinned here, so a re-export
cannot silently swap the strategy a player meets.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drawmaha_solver.minidrawmaha.game import InfoSet
from drawmaha_solver.minidrawmaha.lockstep import read_lockstep
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import (
    PRIMARY_AVERAGE,
    Average,
    validate_averages,
)

RELEASE_TAG = "rung3-lcfr-2m"
STRATEGY_FILE = "strategy-lcfr-2m.npz"
RELEASE_URL = (
    "https://github.com/AlecROndo/drawmaha-solver/releases/download/"
    f"{RELEASE_TAG}/{STRATEGY_FILE}"
)
# `fetch_strategy` refuses any download that does not hash to this.
STRATEGY_SHA256 = "d49f0f3792627cd5b0279956c7b67690af1507b5beec264cbf5e4606f437b6b8"


@dataclass(frozen=True)
class StrategyInfo:
    """Where a frozen strategy came from: enough to say what a player is facing."""

    rule: str
    column: str
    iteration: int
    seed: int
    workers: int


class FrozenStrategy(Mapping[InfoSet, np.ndarray]):
    """Every infoset's average strategy, read from two flat arrays.

    A `Mapping` from `InfoSet` to its probability row, which is the profile the
    exact grader and the play loop both take. A row is a read-only view into
    `probabilities`, never a copy. The key → row arithmetic is the packed
    table's whole-game index, borrowed from a table whose arrays are never
    touched and so cost nothing.
    """

    def __init__(
        self, probabilities: np.ndarray, widths: np.ndarray, info: StrategyInfo
    ) -> None:
        self._table = PackedTable.whole_game()
        expected = self._table.widths()
        widths = np.asarray(widths, dtype=np.int64)
        if widths.shape != expected.shape or not np.array_equal(widths, expected):
            raise ValueError(
                f"this strategy has {widths.size:,} rows whose widths do not match "
                f"the game's {expected.size:,}: it was frozen against a different "
                "infoset layout"
            )
        if probabilities.shape != (int(expected.sum()),):
            raise ValueError(
                f"{probabilities.shape} probabilities for {int(expected.sum()):,} columns"
            )
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.probabilities.setflags(write=False)
        self.info = info

    @property
    def widths(self) -> np.ndarray:
        return self._table.widths()

    def __getitem__(self, key: InfoSet) -> np.ndarray:
        _, start, width = self._table.slot_of(key)
        return self.probabilities[start : start + width]

    def __iter__(self) -> Iterator[InfoSet]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


def average_rows(sums: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Normalise each row of a flat strategy sum; a row that sums to zero plays uniformly."""
    widths = np.asarray(widths, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(widths)[:-1]))
    totals = np.add.reduceat(sums, starts) if sums.size else np.zeros(0)
    per_column = np.repeat(totals, widths)
    uniform = np.repeat(1.0 / widths, widths)
    with np.errstate(invalid="ignore", divide="ignore"):
        rows = np.where(per_column > 0.0, sums / per_column, uniform)
    return rows.astype(np.float32)


def from_checkpoint(path: Path, column: Average = PRIMARY_AVERAGE) -> FrozenStrategy:
    """Freeze one averaging column of a lockstep checkpoint.

    Reads only the sum it needs straight out of the checkpoint, never building
    the 0.3 GB table or touching the regrets.
    """
    run = read_lockstep(path)
    averages = validate_averages(run["averages"])
    with np.load(path) as saved:
        if column is PRIMARY_AVERAGE:
            sums = saved["strategy_sum"]
        elif column in averages:
            sums = saved["extra_sums"][averages.index(column)]
        else:
            raise ValueError(
                f"this run banks no {column.value!r} average; its columns are "
                f"{[a.value for a in (PRIMARY_AVERAGE, *averages)]}"
            )
        widths = saved["widths"]
    info = StrategyInfo(
        rule=str(run["rule"]),
        column=column.value,
        iteration=int(run["iteration"]),
        seed=int(run["seed"]),
        workers=int(run["workers"]),
    )
    return FrozenStrategy(average_rows(sums, widths), widths, info)


def save_strategy(strategy: FrozenStrategy, path: Path) -> None:
    """Write a frozen strategy as a compressed `.npz`."""
    info = strategy.info
    np.savez_compressed(
        path,
        probabilities=strategy.probabilities,
        widths=strategy.widths.astype(np.int8),
        rule=info.rule,
        column=info.column,
        iteration=np.int64(info.iteration),
        seed=np.int64(info.seed),
        workers=np.int64(info.workers),
    )


def load_strategy(path: Path) -> FrozenStrategy:
    """Read a frozen strategy back; refused if its rows are not this game's."""
    with np.load(path) as saved:
        info = StrategyInfo(
            rule=str(saved["rule"]),
            column=str(saved["column"]),
            iteration=int(saved["iteration"]),
            seed=int(saved["seed"]),
            workers=int(saved["workers"]),
        )
        return FrozenStrategy(saved["probabilities"], saved["widths"], info)


def default_cache() -> Path:
    """`$XDG_CACHE_HOME/drawmaha-solver`, else `~/.cache/drawmaha-solver`."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "drawmaha-solver"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_strategy(
    cache: Path | None = None,
    *,
    url: str = RELEASE_URL,
    sha256: str = STRATEGY_SHA256,
) -> Path:
    """The release's strategy file, downloaded once and checked against its digest.

    A cached copy is re-hashed on every call (a fraction of a second) and
    replaced if it does not match, so a truncated download never sticks. The
    download lands in a temporary file first and is moved into place only
    after it hashes correctly.
    """
    if not sha256:
        raise RuntimeError("no strategy digest is pinned yet; pass --strategy PATH")
    cache = cache or default_cache()
    target = cache / Path(url).name
    if target.exists() and sha256_of(target) == sha256:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as partial:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, partial)
    found = sha256_of(Path(partial.name))
    if found != sha256:
        Path(partial.name).unlink()
        raise ValueError(f"{url} hashed to {found}, expected {sha256}")
    Path(partial.name).replace(target)
    return target
