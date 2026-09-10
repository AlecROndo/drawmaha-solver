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

_FIXTURE = Path(__file__).parent / "referee.json"


@pytest.fixture(scope="session")
def referee() -> dict:
    """Every fact the outside referee established about Leduc."""
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="session")
def lp_value(referee) -> float:
    """The exact value of Leduc to the first player, from the sequence-form LP.

    The number rung 2 exists to reach. Solved numerically, so its last digits
    move between solver versions — assert against it with a band well outside
    `lp_value_precision`, never for equality.
    """
    return referee["lp_value_to_p0"]
