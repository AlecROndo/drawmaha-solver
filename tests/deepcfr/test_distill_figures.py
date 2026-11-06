"""`scripts/deepcfr/distill_figures.py` draws its four figures from the committed results file.

The figures under `figures/rung4/distill/` are regenerated, never edited;
this runs the script on the committed results into a scratch directory and
checks that each figure is written and non-trivial, so a renamed key in the
results file or a broken plotting call fails here and not on the next
regeneration. Torch-free: the script reads JSON and draws.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "deepcfr" / "distill_figures.py"
RESULTS = ROOT / "figures" / "rung4" / "distill" / "results.json"


def test_the_four_figures_are_drawn_from_the_committed_results(tmp_path):
    spec = importlib.util.spec_from_file_location("distill_figures_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([str(RESULTS), str(tmp_path)])
    for name in ("by_size.png", "gaps.png", "curves.png", "fraction.png"):
        written = tmp_path / name
        assert written.exists(), name
        assert written.stat().st_size > 20_000, name
        assert (ROOT / "figures" / "rung4" / "distill" / name).exists(), name
