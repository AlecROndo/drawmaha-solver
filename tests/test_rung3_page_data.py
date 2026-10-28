"""`/rung3` reads copies of committed data files; the copies cannot drift from the originals.

`web/rung3-viz/src/data/` holds a copy of every file the page quotes —
rung 3's grades and answer sheet, every rung-4 grade file and the policy-net
sweep — refreshed by
hand after a regeneration (`web/rung3-viz/AGENTS.md`). Each is held byte
for byte to its `figures/` original here, so a regenerated figure file
cannot leave the page quoting a stale number. The scripts that wrote the
outside data (`pi_loss_sweep.py` needs torch; the other does not) are at
least compiled, so a renamed import in the package breaks the suite and
not a reader's first run.
"""

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COPIES = ROOT / "web" / "rung3-viz" / "src" / "data"

# copy under src/data -> original under figures/
PAIRS = {
    "grades.json": "figures/rung3/grades.json",
    "answer_sheet.json": "figures/rung3/answer_sheet.json",
    "exploit_by_reach.json": "figures/rung3/exploit_by_reach.json",
    "rung4/grades.json": "figures/rung4/grades.json",
    "rung4/sweep.json": "figures/rung4/sweep.json",
    "rung4/sweep_128x3_t200.json": "figures/rung4/sweep_128x3_t200.json",
    "rung4/sweep_128x3-s8000.json": "figures/rung4/sweep_128x3-s8000.json",
    "rung4/sweep_128x4.json": "figures/rung4/sweep_128x4.json",
    "rung4/sweep_256x4.json": "figures/rung4/sweep_256x4.json",
    "rung4/sweep_256x4-s8000.json": "figures/rung4/sweep_256x4-s8000.json",
    "rung4/sweep_256x4-s16000-lr5e-4.json": "figures/rung4/sweep_256x4-s16000-lr5e-4.json",
    "rung4/sweep_256x4-s16000.json": "figures/rung4/sweep_256x4-s16000.json",
    "rung4/pot_0.5-256x4-s16000-lr5e-4.json": "figures/rung4/pot_0.5-256x4-s16000-lr5e-4.json",
    "rung4/pot_1-256x4-s16000-lr5e-4.json": "figures/rung4/pot_1-256x4-s16000-lr5e-4.json",
    "rung4/rule_dcfr-256x4.json": "figures/rung4/rule_dcfr-256x4.json",
    "rung4/rule_cfrplus-256x4.json": "figures/rung4/rule_cfrplus-256x4.json",
    "rung4/rule_sign_flips.json": "figures/rung4/rule_sign_flips.json",
    "rung4/pi_sweep_results.json": "figures/rung4/pi_sweep/results.json",
    "rung4/pi_sweep_reach_weighted.json": "figures/rung4/pi_sweep/reach_weighted.json",
    "rung4/exploit_by_reach_deep-exact.json": "figures/rung4/exploit_by_reach_deep-exact.json",
    "rung4/exploit_by_reach_deep-pi.json": "figures/rung4/exploit_by_reach_deep-pi.json",
}

SCRIPTS = (
    "scripts/deepcfr/pi_loss_sweep.py",
    "scripts/deepcfr/pi_reach_weighted.py",
)


@pytest.mark.parametrize("copy", sorted(PAIRS))
def test_every_page_copy_equals_its_figures_original(copy: str) -> None:
    original = ROOT / PAIRS[copy]
    assert original.exists(), original
    assert (COPIES / copy).read_bytes() == original.read_bytes(), f"{copy} drifted from {PAIRS[copy]}"


def test_no_page_copy_is_unaccounted_for() -> None:
    found = {str(p.relative_to(COPIES)) for p in COPIES.rglob("*.json")}
    assert found == set(PAIRS), found ^ set(PAIRS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_data_scripts_compile(script: str) -> None:
    py_compile.compile(str(ROOT / script), doraise=True)


@pytest.mark.parametrize("script", SCRIPTS[1:])
def test_the_torch_free_scripts_import_without_torch_and_run_nothing_at_import(script: str) -> None:
    """Each script is executed in a fresh interpreter with `torch` blocked, under a
    name that is not `__main__`: it imports every package name it uses, pulls in no
    torch (directly or through the package), and does its work only in `main()`.
    The sweep script imports torch at the top by design and is only compiled."""
    code = (
        "import runpy, sys\n"
        "sys.modules['torch'] = None  # any `import torch` now raises ImportError\n"
        f"g = runpy.run_path({str(ROOT / script)!r}, run_name='imported')\n"
        "assert callable(g['main']), 'no main()'\n"
        "assert 'torch' not in sys.modules or sys.modules['torch'] is None\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert proc.returncode == 0, f"{script} failed to import without torch:\n{proc.stderr}"
