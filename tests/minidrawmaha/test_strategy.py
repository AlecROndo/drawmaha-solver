"""The frozen strategy: a checkpoint's average, as a file a player can load.

What the module has to make good on:

* **it is the checkpoint's average, row for row** — freezing a trained run and
  reading any key back gives what the live table's `column_average` gives, to
  float32 precision, for every averaging column the run banked;
* **a file is refused if its rows are not this game's** — the width array is
  the fingerprint;
* **a download is trusted only once it hashes to the pinned digest**.
"""

from itertools import islice

import numpy as np
import pytest

from drawmaha_solver.minidrawmaha.lockstep import new_lockstep, save_lockstep, train_lockstep
from drawmaha_solver.minidrawmaha.packed_table import PackedTable
from drawmaha_solver.minidrawmaha.regret_rules import Average, column_average
from drawmaha_solver.minidrawmaha.strategy import (
    FrozenStrategy,
    StrategyInfo,
    average_rows,
    fetch_strategy,
    from_checkpoint,
    load_strategy,
    save_strategy,
    sha256_of,
)

INFO = StrategyInfo(rule="lcfr", column="linear", iteration=7, seed=0, workers=2)


def test_rows_are_normalised_and_an_empty_row_plays_uniformly():
    sums = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 0.0])
    rows = average_rows(sums, np.array([2, 3, 4]))
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows[:2], [0.25, 0.75])
    np.testing.assert_allclose(rows[2:5], [1 / 3] * 3)
    np.testing.assert_allclose(rows[5:], [0.25, 0.25, 0.5, 0.0])


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A short whole-game lockstep run with both extra columns, saved."""
    solve = new_lockstep(
        0, workers=2, rule="lcfr", averages=(Average.UNIFORM, Average.QUADRATIC)
    )
    train_lockstep(solve, 5)
    path = tmp_path_factory.mktemp("strategy") / "run.npz"
    save_lockstep(solve, path)
    return solve, path


@pytest.mark.parametrize("column", list(Average))
def test_freezing_reads_back_the_live_average(trained, column):
    solve, path = trained
    frozen = from_checkpoint(path, column)
    assert frozen.info.iteration == 5 and frozen.info.column == column.value
    touched = list(
        islice((key for key, ledger in solve.table.items() if ledger.strategy_sum.any()), 200)
    )
    assert touched, "five iterations should reach some spots"
    for key in touched:
        expected = column_average(solve.table[key], column, solve.averages)
        np.testing.assert_allclose(frozen[key], expected, rtol=1e-6, atol=1e-7)


def test_an_unreached_infoset_plays_uniformly(trained):
    solve, path = trained
    frozen = from_checkpoint(path)
    key = next(key for key, ledger in solve.table.items() if not ledger.strategy_sum.any())
    width = len(key.legal_actions())
    np.testing.assert_allclose(frozen[key], np.full(width, 1 / width), rtol=1e-6)


def test_a_saved_strategy_loads_back_unchanged(trained, tmp_path):
    _, path = trained
    frozen = from_checkpoint(path)
    saved = tmp_path / "frozen.npz"
    save_strategy(frozen, saved)
    loaded = load_strategy(saved)
    assert loaded.info == frozen.info
    np.testing.assert_array_equal(loaded.probabilities, frozen.probabilities)
    assert len(loaded) == len(PackedTable.whole_game())


def test_a_strategy_for_another_layout_is_refused():
    widths = PackedTable.whole_game().widths().copy()
    widths[-1] += 1
    with pytest.raises(ValueError, match="different infoset layout"):
        FrozenStrategy(np.zeros(int(widths.sum()), dtype=np.float32), widths, INFO)


def test_a_download_is_kept_only_when_it_hashes_right(tmp_path):
    source = tmp_path / "source" / "strategy.npz"
    source.parent.mkdir()
    source.write_bytes(b"not really a strategy")
    digest = sha256_of(source)
    cache = tmp_path / "cache"
    url = source.as_uri()

    with pytest.raises(ValueError, match="expected"):
        fetch_strategy(cache, url=url, sha256="0" * 64)
    assert not any(cache.iterdir()), "a bad download must not be left behind"

    fetched = fetch_strategy(cache, url=url, sha256=digest)
    assert fetched == cache / "strategy.npz" and sha256_of(fetched) == digest
    source.unlink()
    assert fetch_strategy(cache, url=url, sha256=digest) == fetched, "served from cache"
