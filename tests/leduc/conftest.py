"""The referee's testimony, loaded once for every rung-2 test that needs it.

`referee.json` is what OpenSpiel says about Leduc — the node census, the
infoset collapse, the payoff ladder, the exact value of the game, and its own
solver's convergence pace. `scripts/generate_leduc_referee.py` writes it;
nothing here imports OpenSpiel, because its wheels are large and the Vercel
deployment build must not pull them.

It lives in a conftest rather than a shared module because the suite runs with
`--import-mode=importlib` (each rung has its own `test_game.py`, and without
that mode the identical basenames collide), which leaves sibling test modules
un-importable. A pytest fixture is the one sharing seam that always works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drawmaha_solver.leduc.game import LP_VALUE_P0

_FIXTURE = Path(__file__).parent / "referee.json"


@pytest.fixture(scope="session")
def referee() -> dict:
    """Every fact the outside referee established about Leduc."""
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="session")
def lp_value(referee) -> float:
    """The exact value of Leduc to the first player, from the sequence-form LP.

    Returns `game.LP_VALUE_P0` — the copy callers outside the test suite use —
    after checking it against the referee's, so a test asserting on it is
    asserting on the same number `analysis.py` and `play.py` report. The
    equality check lives in `test_referee.py`; this only guarantees a test
    never silently reads a stale constant.
    """
    assert LP_VALUE_P0 == pytest.approx(
        referee["lp_value_to_p0"], abs=referee["lp_value_precision"]
    )
    return LP_VALUE_P0
