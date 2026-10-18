"""The memories: Algorithm R over (row, t, target) rows, with an exact resume.

Deep CFR trains each iteration's net on a MEMORY of what the walks saw: for the
advantage memory of seat p, every traverser visit's regret sample r = u − ⟨σ, u⟩
at some table row; for the policy memory, every opponent visit's σ. A row of
either memory is 24 bytes — `row: int32` (which table row), `t: int32` (which
iteration offered it, the refit's weight) and `target: float32 × 4` (r or σ at
the row's legal columns, zero-padded to the widest ledger, 4). Over 2,000
iterations the walks offer about 280 million advantage rows per seat; the
memory keeps a capacity C of them, 50 million in the run.

**Which C of the n rows ever offered?** A uniform sample without replacement —
every offer resident with probability C/n — so that the refit sees the
iterations in proportion to what they offered and no iteration is favoured by
when it happened. Algorithm R (Waterman's, in Knuth's naming; Vitter 1985 keeps
the letter) gives exactly that in one pass and O(1) memory beyond the sample:
the first C offers fill the slots; the offer with global index g ≥ C (0-based)
draws one integer j uniformly from [0, g] and, if j < C, overwrites slot j, else
is dropped. Induction: before offer g arrives every earlier offer is resident
with probability C/g; offer g lands with probability C/(g+1); each resident is
evicted with probability 1/(g+1), so it stays with probability
C/g · g/(g+1) = C/(g+1). `merge` is that algorithm
vectorised over a phase's rows: one draw per late offer, and when two offers
of the same phase draw the same slot the LATER one wins, which is what the
sequential algorithm does.

**Why the stream is counter-based.** A run resumed from a checkpoint must merge
the next phase exactly as the uninterrupted run would, which needs two things:
the count n of offers ever made (saved with the memory, since it is what the
next offer's draw is conditioned on) and the same random draws. The draws come
from `reservoir_stream(seed, t, seat, memory)`, a fresh `default_rng` per phase
seeded by five words, so a resumed run asks for the same stream by name and no
RNG state is ever checkpointed — the rule rung 3's `lockstep.stream` set. The
fifth word, the ASCII constant "RESV", keeps a reservoir stream from ever
equalling a walk's four-word stream.

The three arrays are `np.zeros` at construction, which the kernel pages in
lazily: a 50M-row memory costs 1.2 GB only as it fills, nothing when three of
them are built on a laptop. `contents()` is the refit's input — read-only views
of the resident rows, the shape `fit_from_scratch(rows, t, targets, …)` takes.
"""

from __future__ import annotations

import os
from enum import IntEnum
from pathlib import Path

import numpy as np

# The widest ledger in mini-drawmaha (the draw's four throws); every target is
# padded to it. Equal to `features.MAX_WIDTH`, pinned by a test rather than
# imported, so the memory does not depend on the encoder.
TARGET_WIDTH = 4

# The fifth word of every reservoir stream: "RESV" in ASCII. A walk's stream
# has four words, so no reservoir stream can equal one.
_STREAM_TAG = 0x52455356

# What a reservoir file holds. `offered` marks a file as a reservoir: no other
# checkpoint in the repo records a count of offers.
_FILE_KEYS = ("rows", "t", "targets", "offered", "capacity")

class Memory(IntEnum):
    """Which memory a stream feeds; the word that tells the two apart in the seed."""

    ADVANTAGE = 0
    POLICY = 1

def reservoir_stream(*, seed: int, t: int, seat: int, memory: Memory) -> np.random.Generator:
    """The random stream that merges `memory`'s rows from seat `seat`'s phase at iteration t."""
    return np.random.default_rng([int(seed), int(t), int(seat), int(Memory(memory)), _STREAM_TAG])

# ---------------------------------------------------------------------------
# The reservoir
# ---------------------------------------------------------------------------

class Reservoir:
    """A uniform sample of at most `capacity` of the rows ever offered, Algorithm R.

    `merge` offers one phase's rows; `contents` is what is resident; `save` and
    `load` round-trip the resident rows and the count of offers, so the next
    merge on the loaded memory equals the next merge on the original.
    """

    def __init__(self, capacity: int) -> None:
        # A capacity under 1 could hold no row.
        if not _is_int(capacity) or capacity < 1:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        self._capacity = int(capacity)
        self._offered = 0
        self._rows = np.zeros(self._capacity, dtype=np.int32)
        self._t = np.zeros(self._capacity, dtype=np.int32)
        self._targets = np.zeros((self._capacity, TARGET_WIDTH), dtype=np.float32)

    @property
    def capacity(self) -> int:
        """The most rows the memory holds."""
        return self._capacity

    @property
    def offered(self) -> int:
        """n: how many rows have ever been offered, resident or not."""
        return self._offered

    @property
    def size(self) -> int:
        """How many rows are resident: min(n, capacity)."""
        return min(self._offered, self._capacity)

    def merge(self, rows: np.ndarray, targets: np.ndarray, *, t: int, rng) -> None:
        """Offer one phase's rows, all stamped `t`, drawing from `rng`.

        `rng` needs `Generator.integers(low, high)` with an array `high`: one
        draw per late offer. Rows below capacity are kept outright and draw
        nothing, so a memory still filling consumes no randomness.
        """
        rows, targets = _validate_offer(rows, targets, t)
        count = rows.shape[0]
        capacity = self._capacity
        # Every offer's global index among all offers ever made.
        index = self._offered + np.arange(count, dtype=np.int64)
        slot = np.where(index < capacity, index, -1)
        late = index >= capacity
        if late.any():
            # Inclusive of the offer's own index: j = g means "keep the reservoir".
            draw = np.asarray(rng.integers(0, index[late] + 1), dtype=np.int64)
            slot[late] = np.where(draw < capacity, draw, -1)
        # Sequential Algorithm R would write the offers in order, so of two
        # offers that drew one slot the later overwrites the earlier. Repeated
        # fancy-index assignment leaves which one lands unspecified, so the
        # last occurrence of each slot is picked out explicitly.
        landing = np.flatnonzero(slot >= 0)
        _, last_of_each = np.unique(slot[landing][::-1], return_index=True)
        winners = landing[::-1][last_of_each]
        self._rows[slot[winners]] = rows[winners]
        self._t[slot[winners]] = t
        self._targets[slot[winners]] = targets[winners]
        self._offered += count

    def contents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The resident rows as read-only views: `(rows, t, targets)`, the refit's input.

        Live views, not a snapshot: a later `merge` shows through them. The
        loop takes them after the phase's merge and hands them to the refit.
        """
        size = self.size
        views = (self._rows[:size], self._t[:size], self._targets[:size])
        for view in views:
            view.setflags(write=False)
        return views

    # -- the file ------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the resident rows and the count of offers to one `.npz`, atomically.

        A temp file renamed into place, as `lockstep.save_lockstep` does, so a
        crash mid-write leaves the previous file whole.
        """
        path = Path(path)
        rows, t, targets = self.contents()
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez(
                file,
                rows=rows,
                t=t,
                targets=targets,
                offered=np.int64(self._offered),
                capacity=np.int64(self._capacity),
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> Reservoir:
        """A memory as `save` left it, ready for the next `merge`.

        Refuses a file that is not a reservoir (no `offered`) and one whose
        resident rows do not match its counts, naming the file either way.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as saved:
            missing = [key for key in _FILE_KEYS if key not in saved.files]
            if missing:
                raise ValueError(f"{path} is not a reservoir checkpoint (no {missing[0]!r})")
            capacity, offered = int(saved["capacity"]), int(saved["offered"])
            rows, t, targets = saved["rows"], saved["t"], saved["targets"]
        memory = cls(capacity)
        size = min(offered, capacity)
        if (
            offered < 0
            or rows.shape != (size,)
            or t.shape != (size,)
            or targets.shape != (size, TARGET_WIDTH)
        ):
            raise ValueError(
                f"{path} holds {rows.shape[0]} rows, but {offered} offers into a capacity "
                f"of {capacity} means {size} resident"
            )
        # The arrays are poured in as they are: a float `rows` would be
        # truncated into the int32 slots without a word.
        expected = (("rows", rows, np.int32), ("t", t, np.int32), ("targets", targets, np.float32))
        for name, array, dtype in expected:
            if array.dtype != dtype:
                raise ValueError(f"{path}: {name} must be {dtype.__name__}, got {array.dtype}")
        memory._rows[:size] = rows
        memory._t[:size] = t
        memory._targets[:size] = targets
        memory._offered = offered
        return memory

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_int(value: object) -> bool:
    # A bool is an int to Python and never what a caller meant.
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

def _validate_offer(rows: np.ndarray, targets: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """The phase's rows as int32 and its targets as float32, or a ValueError naming the fault."""
    rows = np.asarray(rows)
    targets = np.asarray(targets)
    # A row is a table row number: a float or a negative is a caller handing
    # the wrong array.
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"rows must be a 1-D integer array, got {rows.dtype} {rows.shape}")
    if rows.size and rows.min() < 0:
        raise ValueError(f"rows must not be negative, got {int(rows.min())}")
    # Rows are stored as int32; a bigger one would wrap silently in `astype`.
    if rows.size and rows.max() >= 2**31:
        raise ValueError(f"rows must fit int32, got {int(rows.max())}")
    if targets.ndim != 2 or targets.shape[1] != TARGET_WIDTH:
        raise ValueError(
            f"targets must be shaped (rows, {TARGET_WIDTH}), got {targets.shape}"
        )
    if targets.shape[0] != rows.shape[0]:
        raise ValueError(
            "rows and targets must have the same length, "
            f"got {rows.shape[0]} and {targets.shape[0]}"
        )
    # Iterations start at 1; the refit weights a row by its t, so 0 would
    # erase it and a negative would pull against the others.
    if not _is_int(t) or t < 1:
        raise ValueError(f"t must be a positive int, got {t!r}")
    return rows.astype(np.int32, copy=False), targets.astype(np.float32, copy=False)
