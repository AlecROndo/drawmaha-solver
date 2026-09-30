"""The packed table: every mini-drawmaha ledger as a window into four flat arrays.

Rung 3's first table was a `dict[InfoSet, RegretMatcher]` — 6,220,050 keys,
6,220,050 ledger objects, four small NumPy arrays each. The trained numbers in
it are 228 MB (the two float64 accumulators; 278 MB once the contract's int64
stamp sits beside them); the objects around them were 3.4 GB before the ledger
contract added its two slots and 5.4 GB after, and the dict took half a minute
to build. Ten such tables cannot share a Modal machine, and a solve cannot
hand one to a worker. This module keeps the numbers and drops the objects.

**The layout.** Four arrays, all of them public attributes because the next
step — shared memory, a flat checkpoint — needs to reach them by name:

- `cumulative_regret`, `strategy_sum`: float64, every ledger's columns laid end
  to end in `all_infosets()` order — ledger 0's two or three or four entries,
  then ledger 1's, and so on. That is the checkpoint's format already
  (`mccfr.save_solve` concatenates any other table's ledgers in the same
  order), so a checkpoint's arrays ARE this table's arrays: `save_solve`
  writes all four as they stand and `load_solve` pours them straight back,
  without building a window.
- `extra_sums`: float64, `(k, total width)`: the k further averages of the
  contract, one row per weighting, columns aligned with the two above.
- `stamp`: int64, one per LEDGER, not per column: the discount stamp.

**The window.** `table[key]` returns a `RegretMatcher` built with
`RegretMatcher.over` on slices of those arrays: no copy, so an `update` through
it lands in the table and the ledger object can be dropped after the visit.
The walk gets a fresh window per visit, which costs about a microsecond
against the ~25 µs a state costs it. The one rule the contract asks of every
writer follows from this: `ledger.strategy_sum[:] = ...` writes into the table,
`ledger.strategy_sum = ...` replaces the window and writes into nothing.

**The index, and why it holds no keys.** A dict from `InfoSet` to row would
mean holding 6.2 million `InfoSet`s, which is most of the gigabytes the object
table spent. Instead a key's row is COMPUTED from the two halves the enumerator
builds it from: its public half (`draws`, `betting` and how many board cards
are out) names one of the 141 public points, and its private half is a
position in that point's shape of `private_keys`, which is already cached and
sorted. So the row is a point's offset plus a position, and the index is one
141-entry dict plus one dict per private-key shape — three shapes, 111,540
entries in all, borrowing their tuples from the cache. Two lookups, measured
at 0.12 µs a key against the 0.10 µs of the object table's one dict lookup.
A key whose public half no point has, or whose private half
its shape does not hold, raises `KeyError` — a walk that reached a position
the census says does not exist.

A table over a **listed** slice of keys — what a readout or a test allocates —
keeps a small dict from key to row instead, because a slice can be any keys in
any order and is small by construction. Both indexes answer the same three
questions (row, first column, width) and the store does not know which it has.

Nothing here reads a strategy or walks the tree: `infoset_table.py` owns the
readouts and the allocator's public face, `mccfr.py` the walk and the
checkpoint. This file is only where the numbers live.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, Mapping, ValuesView
from typing import Protocol

import numpy as np

from drawmaha_solver.minidrawmaha.enumeration import (
    PrivateKey,
    all_infosets,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.game import InfoSet
from drawmaha_solver.regret_matching import RegretMatcher

# Where one ledger lives: its row, the first of its columns, and how many.
Slot = tuple[int, int, int]

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class PackedTable(Mapping[InfoSet, RegretMatcher]):
    """One ledger per infoset, all of them stored in four flat arrays.

    A read-only `Mapping`: `len`, iteration in row order, `KeyError` on a key
    it does not hold, and `table[key]` a `RegretMatcher` whose four slots are
    views into the arrays (see the module docstring). Assigning `table[key]`
    is not offered: every key's row is fixed at allocation, which is what makes
    the flat layout a checkpoint format.

    Built by `whole_game` or `listed`, never directly: the store is the same
    either way and only the index differs.
    """

    def __init__(self, index: _RowIndex, *, extra_averages: int):
        # Refused here by name rather than left to NumPy, whose "negative
        # dimensions are not allowed" points at an array shape and not at the
        # argument the caller got wrong.
        if not isinstance(extra_averages, (int, np.integer)) or extra_averages < 0:
            raise ValueError(
                f"extra_averages must be a non-negative int, got {extra_averages!r}"
            )
        self._index = index
        total = int(index.widths().sum())
        self.cumulative_regret = np.zeros(total)
        self.strategy_sum = np.zeros(total)
        self.extra_sums = np.zeros((extra_averages, total))
        self.stamp = np.zeros(len(index), dtype=np.int64)

    @classmethod
    def whole_game(cls, *, extra_averages: int = 0) -> PackedTable:
        """Every infoset in mini-drawmaha, rows in `all_infosets()` order.

        Cheap: the arrays are zeros the kernel hands out lazily, and the index
        is built from the 141 public points and the three cached private-key
        shapes without producing a single `InfoSet`. About 280 MB once every
        row has been touched, a fraction of a second to allocate.
        """
        return cls(_WholeGameIndex(), extra_averages=extra_averages)

    @classmethod
    def listed(cls, keys: Iterable[InfoSet], *, extra_averages: int = 0) -> PackedTable:
        """Exactly these keys, rows in the order given.

        A slice of `all_infosets()` and the whole table agree row for row on
        the keys they share. A key arriving twice raises rather than being
        quietly given one row for two spots.
        """
        return cls(_ListedIndex(keys), extra_averages=extra_averages)

    def over(
        self,
        *,
        cumulative_regret: np.ndarray,
        strategy_sum: np.ndarray,
        extra_sums: np.ndarray,
        stamp: np.ndarray,
    ) -> PackedTable:
        """The same keys and rows over four arrays somebody else owns — no copy.

        What a shared-memory run needs: every process builds its own index (a
        second of arithmetic) and lays it over the one set of buffers. Checked
        rather than trusted, for the reason `RegretMatcher.over` checks: a
        buffer of the wrong length or dtype would not crash, it would put
        ledgers in the wrong columns.
        """
        for name, array, like in (
            ("cumulative_regret", cumulative_regret, self.cumulative_regret),
            ("strategy_sum", strategy_sum, self.strategy_sum),
            ("extra_sums", extra_sums, self.extra_sums),
            ("stamp", stamp, self.stamp),
        ):
            if array.shape != like.shape or array.dtype != like.dtype:
                raise ValueError(
                    f"{name} must be {like.dtype.name} {like.shape}, "
                    f"got {array.dtype.name} {array.shape}"
                )
        table = object.__new__(PackedTable)
        table._index = self._index
        table.cumulative_regret = cumulative_regret
        table.strategy_sum = strategy_sum
        table.extra_sums = extra_sums
        table.stamp = stamp
        return table

    def __getitem__(self, key: InfoSet) -> RegretMatcher:
        return self._window(*self._locate(key))

    def __iter__(self) -> Iterator[InfoSet]:
        return self._index.keys()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        # `Mapping` answers this by building `self[key]` and catching the miss,
        # which here is a four-view ledger built to be thrown away. The index
        # alone knows.
        try:
            self._locate(key)
        except KeyError:
            return False
        return True

    def __eq__(self, other: object) -> bool:
        # `Mapping.__eq__` materialises `dict(self.items())` on both sides —
        # 6.2 million windows each for the whole game — and then compares
        # `RegretMatcher`s, which compare by identity, so two walks of the
        # same table never came out equal anyway. A table is equal to itself
        # and to nothing else; compare the arrays by name to compare numbers.
        return self is other

    __hash__ = None  # a Mapping is unhashable; saying so keeps it that way

    def row_of(self, key: InfoSet) -> int:
        """Which row `key` occupies: its position in the table's iteration order."""
        return self._locate(key)[0]

    def slot_of(self, key: InfoSet) -> Slot:
        """Where `key`'s ledger lives: its row, its first column, its width."""
        return self._locate(key)

    def widths(self) -> np.ndarray:
        """Every ledger's width, int64, one per row — the checkpoint's fingerprint.

        The same read-only array on every call, built once with the index: for
        the whole game it is 50 MB, and a fingerprint a caller could write into
        would not be one.
        """
        return self._index.widths()

    def items(self) -> ItemsView[InfoSet, RegretMatcher]:
        return _Items(self)

    def values(self) -> ValuesView[RegretMatcher]:
        return _Values(self)

    def _locate(self, key: object) -> Slot:
        # A non-`InfoSet` (a Leduc key handed to the wrong table) is a miss,
        # as it would be in a dict, not an `AttributeError` from the index.
        if not isinstance(key, InfoSet):
            raise KeyError(key)
        return self._index.locate(key)

    def _window(self, row: int, start: int, width: int) -> RegretMatcher:
        stop = start + width
        return RegretMatcher.over(
            cumulative_regret=self.cumulative_regret[start:stop],
            strategy_sum=self.strategy_sum[start:stop],
            extra_sums=self.extra_sums[:, start:stop],
            stamp=self.stamp[row : row + 1],
        )

    def _windows(self) -> Iterator[tuple[InfoSet, RegretMatcher]]:
        """Every (key, ledger) pair in row order, with a running cursor.

        What `items()` and `values()` walk. The default `Mapping` versions look
        each key up again, which over the whole game is 6.2 million index
        lookups to rediscover rows that arrive in order anyway. The cursor is
        a running sum rather than a starts array, which over the whole game
        would be 6.2 million Python ints held at once.
        """
        start = 0
        for row, (key, width) in enumerate(
            zip(self._index.keys(), self._index.widths(), strict=True)
        ):
            width = int(width)
            yield key, self._window(row, start, width)
            start += width

class _Items(ItemsView[InfoSet, RegretMatcher]):
    def __iter__(self) -> Iterator[tuple[InfoSet, RegretMatcher]]:
        return self._mapping._windows()

class _Values(ValuesView[RegretMatcher]):
    def __iter__(self) -> Iterator[RegretMatcher]:
        return (ledger for _, ledger in self._mapping._windows())

# ---------------------------------------------------------------------------
# The two indexes: where a key's row is
# ---------------------------------------------------------------------------

class _RowIndex(Protocol):
    """What the store asks of an index: a row per key, and the keys in row order."""

    def __len__(self) -> int: ...
    def locate(self, key: InfoSet) -> Slot: ...
    def keys(self) -> Iterator[InfoSet]: ...
    def widths(self) -> np.ndarray: ...

class _WholeGameIndex:
    """Rows computed from the enumerator's two halves; no `InfoSet` is held.

    `all_infosets()` yields, for each public point in walk order, that point's
    shape of `private_keys` in sorted order. So a key's row is the number of
    keys standing on earlier points plus its position in its shape, and its
    first column is the earlier points' columns plus that position times the
    point's width — every key on one point has one width. Both offsets are
    141-entry lists, built once from the census arithmetic.
    """

    def __init__(self) -> None:
        points = public_decision_points()
        # A decision point is named by its public signature — how many board
        # cards are out, both draw counts, and the betting. `player` is left
        # out because `InfoSet` already refuses a key whose player is not the
        # one the betting says acts, and `discards` because it is fixed by
        # `draws`. Every key on a point has the same three, so one dict of 141.
        self._point = {
            (point.board_cards, point.draws, point.betting): index
            for index, point in enumerate(points)
        }
        self._shape = [(point.board_cards, point.discards) for point in points]
        self._position: dict[tuple[int, int], dict[PrivateKey, int]] = {
            shape: {
                key: position
                for position, key in enumerate(
                    private_keys(board_cards=shape[0], discards=shape[1])
                )
            }
            for shape in set(self._shape)
        }
        self._width = [point.width for point in points]
        sizes = [len(self._position[shape]) for shape in self._shape]
        self._row_offset = _offsets(sizes)
        self._column_offset = _offsets(
            [size * width for size, width in zip(sizes, self._width, strict=True)]
        )
        self._rows = sum(sizes)
        self._widths = np.repeat(np.asarray(self._width, dtype=np.int64), sizes)
        self._widths.setflags(write=False)

    def __len__(self) -> int:
        return self._rows

    def locate(self, key: InfoSet) -> Slot:
        try:
            point = self._point[(len(key.board), key.draws, key.betting)]
            position = self._position[self._shape[point]][
                (key.hole, key.discarded, key.board)
            ]
        except KeyError:
            # Re-raised with the whole key: the inner dicts would name only the
            # half that missed, and the walk's caller wants the position.
            raise KeyError(key) from None
        width = self._width[point]
        return (
            self._row_offset[point] + position,
            self._column_offset[point] + width * position,
            width,
        )

    def keys(self) -> Iterator[InfoSet]:
        return all_infosets()

    def widths(self) -> np.ndarray:
        return self._widths

class _ListedIndex:
    """Rows for an explicit sequence of keys, in the order given.

    A dict from key to row, which is fine at the sizes a slice has — a readout
    is a few dozen keys, the largest public point 100,400 — and would be the
    gigabytes the whole game must not spend.
    """

    def __init__(self, keys: Iterable[InfoSet]) -> None:
        self._keys: list[InfoSet] = []
        self._row: dict[InfoSet, int] = {}
        widths: list[int] = []
        for key in keys:
            # The object table counted its keys instead, to avoid holding a set
            # of 6.2 million; this index holds the dict regardless, so the
            # plain check is free and names the key at the moment it repeats.
            if key in self._row:
                raise ValueError(f"{key} was handed to the table twice")
            self._row[key] = len(self._keys)
            self._keys.append(key)
            widths.append(len(key.legal_actions()))
        self._widths = np.asarray(widths, dtype=np.int64)
        self._widths.setflags(write=False)
        self._starts = _offsets(widths)

    def __len__(self) -> int:
        return len(self._keys)

    def locate(self, key: InfoSet) -> Slot:
        row = self._row[key]
        return row, self._starts[row], int(self._widths[row])

    def keys(self) -> Iterator[InfoSet]:
        return iter(self._keys)

    def widths(self) -> np.ndarray:
        return self._widths

def _offsets(sizes: list[int]) -> list[int]:
    """Where each block starts when blocks of these sizes are laid end to end."""
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return offsets
