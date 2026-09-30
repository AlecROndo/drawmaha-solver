"""The packed table: every mini-drawmaha ledger as a window into four flat arrays.

Rung 3's first table was a `dict[InfoSet, RegretMatcher]` — 6,220,050 keys,
6,220,050 ledger objects, four small NumPy arrays each. The numbers in it are
228 MB; the objects around them were 3.4 GB before the ledger contract added
its two slots and 5.2 GB after, and the dict took half a minute to build. Ten
such tables cannot share a Modal machine, and a solve cannot hand one to a
worker. This module keeps the numbers and drops the objects.

**The layout.** Four arrays, all of them public attributes because the next
step — shared memory, a flat checkpoint — needs to reach them by name:

- `cumulative_regret`, `strategy_sum`: float64, every ledger's columns laid end
  to end in `all_infosets()` order — ledger 0's two or three or four entries,
  then ledger 1's, and so on. That is the checkpoint's format already
  (`mccfr.save_solve` concatenates the object table's ledgers in the same
  order), so a checkpoint's arrays ARE this table's arrays.
- `extra_sums`: float64, `(k, total width)`: the k further averages of the
  contract, one row per weighting, columns aligned with the two above.
- `stamp`: int64, one per LEDGER, not per column: the discount stamp.

**The window.** `table[key]` returns a `RegretMatcher` built with
`RegretMatcher.over` on slices of those arrays: no copy, so an `update` through
it lands in the table and the ledger object can be dropped after the visit.
The walk gets a fresh window per visit, which costs under a microsecond
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
141-entry dict plus one dict per private-key shape, ~120k entries that borrow
their tuples from the cache. Two lookups, the same cost as hashing the whole
`InfoSet` once. A key whose public half no point has, or whose private half
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
        is built from the 141 public points and the four cached private-key
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

    def __getitem__(self, key: InfoSet) -> RegretMatcher:
        return self._window(*self._index.locate(key))

    def __iter__(self) -> Iterator[InfoSet]:
        return self._index.keys()

    def __len__(self) -> int:
        return len(self._index)

    def row_of(self, key: InfoSet) -> int:
        """Which row `key` occupies: its position in the table's iteration order."""
        return self._index.locate(key)[0]

    def widths(self) -> np.ndarray:
        """Every ledger's width, int64, one per row — the checkpoint's fingerprint."""
        return self._index.widths()

    def items(self) -> ItemsView[InfoSet, RegretMatcher]:
        return _Items(self)

    def values(self) -> ValuesView[RegretMatcher]:
        return _Values(self)

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
        lookups to rediscover rows that arrive in order anyway.
        """
        widths = self._index.widths()
        starts = np.cumsum(widths) - widths
        for row, (key, start, width) in enumerate(
            zip(self._index.keys(), starts.tolist(), widths.tolist(), strict=True)
        ):
            yield key, self._window(row, start, width)

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
        self._sizes = sizes

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
        return np.repeat(np.asarray(self._width, dtype=np.int64), self._sizes)

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
            # Counted rather than checked with `in`: the same one-integer guard
            # the object table used, and it catches the duplicate at the key
            # that repeats rather than at the end of the list.
            self._row[key] = len(self._keys)
            self._keys.append(key)
            if len(self._row) != len(self._keys):
                raise ValueError(f"{key} was handed to the table twice")
            widths.append(len(key.legal_actions()))
        self._widths = widths
        self._starts = _offsets(widths)

    def __len__(self) -> int:
        return len(self._keys)

    def locate(self, key: InfoSet) -> Slot:
        row = self._row[key]
        return row, self._starts[row], self._widths[row]

    def keys(self) -> Iterator[InfoSet]:
        return iter(self._keys)

    def widths(self) -> np.ndarray:
        return np.asarray(self._widths, dtype=np.int64)

def _offsets(sizes: list[int]) -> list[int]:
    """Where each block starts when blocks of these sizes are laid end to end."""
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return offsets
