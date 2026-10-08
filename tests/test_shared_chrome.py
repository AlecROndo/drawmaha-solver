"""The visualizers each carry a copy of the site's chrome; keep them one.

`web/rung{0,1,2,3}-viz` and `web/solver-viz` each hold the same set of files,
`SHARED` below: the theme, the entry point, the shared chrome (top bar, hero,
figure windows, footer), the turning chip and the rung objects. They are copied rather than
extracted to a package on purpose (see the note at the top of
`src/ui/site.tsx`: the apps deploy independently, and a workspace would buy
~300 lines at the cost of a build step). The cost of copying is that the next
edit to one copy drifts the other two silently: nothing in a build or a vitest
run compares apps. This does.

Rung 0's copy is the reference only because it is the first; a fix belongs in
all five, and the failure message says which one was missed.
"""

from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
APPS = ("rung0-viz", "rung1-viz", "rung2-viz", "rung3-viz", "solver-viz")
SHARED = (
    "src/theme.css",
    "src/main.tsx",
    "src/ui/site.tsx",
    "src/ui/mark.ts",
    "src/ui/ascii.ts",
)


@pytest.mark.parametrize("path", SHARED)
def test_the_shared_file_is_byte_identical_in_every_app(path: str) -> None:
    reference = WEB / APPS[0] / path
    assert reference.is_file(), (
        f"{APPS[0]} has no {path} — the shared set has moved"
    )
    expected = reference.read_bytes()
    for app in APPS[1:]:
        copy = WEB / app / path
        assert copy.is_file(), (
            f"{app} has no {path}; every app carries all {len(SHARED)}"
        )
        assert copy.read_bytes() == expected, (
            f"web/{app}/{path} differs from web/{APPS[0]}/{path}: a change to "
            "a shared file goes into all three apps (cp it across, then cmp)"
        )
