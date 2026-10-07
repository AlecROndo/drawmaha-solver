"""The site moves the way web/DESIGN.md says it does, on every stylesheet.

The standard has a mechanical half — which curve, how long, what a press
does, what reduced motion keeps — and this is where it is enforced, because
the next edit to a minified `<style>` block or a copied theme is exactly the
kind of thing a reviewer reads past. What is checked, per stylesheet:

- the motion tokens exist in every `:root` and agree across the four copies;
- no `transition: all`, no `ease-in`, and no hard-coded curve outside the
  tokens except in a figure that is allowlisted with its reason;
- nothing in the chrome runs over 300 ms, and only `transform`/`opacity`
  move, with the same allowlist for the figures that are the exception;
- every pressable has a press (`scale(0.97)` on `:active`, with `transform`
  in its transition), and a press is always between 0.95 and 0.98;
- a hover that moves something is gated behind `(hover: hover)`;
- the reduced-motion block exists and is not the nuclear `* { none }`.

Adding motion that breaks a rule on purpose means adding its selector to
`EXPLANATORY` with a one-line reason, the way the existing ones are.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

APPS = ("rung0-viz", "rung1-viz", "rung2-viz")
PAGES = ("index.html", "rules.html", "cover.html")

TOKENS = ("--ease-out", "--ease-in-out", "--t-press")
TOKEN_VALUES = {
    "--ease-out": "cubic-bezier(0.23,1,0.32,1)",
    "--ease-in-out": "cubic-bezier(0.77,0,0.175,1)",
    "--t-press": "160ms",
}

MAX_UI_MS = 300
PRESS_RANGE = (0.95, 0.98)

# Selectors whose motion is explanatory rather than chrome: a figure drawing
# itself, a section arriving on scroll, a status pulse. Each is allowed to
# exceed 300 ms, to own a curve, or to animate a layout property — and each
# says why. Matched as a substring of the rule's selector.
EXPLANATORY = {
    ".reveal": "a section arriving on scroll, seen once per visit; 550 ms travel",
    ".beat": "the Rules hand's current beat lights as it reaches mid-screen",
    ".fig ": "the ladder's figures draw themselves once as they scroll in",
    ".meter .track": "the Leduc meter fills once; its fill is a gradient scaleX would squash",
    ".sizes .bar": "the Rules tab's game-size bars fill once on scroll",
    ".act .bar": "the homepage's action-mix bars, a reading not a control",
    ".bars .b i": "the homepage's side-panel strategy bars, a reading not a control",
    ".console .prog .bar": "a progress bar: linear, 120 ms, tracks a live counter",
    ".live i": "the live dot's pulse, 2 s, constant",
    ".st i": "the in-progress status dot's pulse",
    ".st.prog i": "the ladder's in-progress badge pulse",
    ".st::before": "the Cover ladder's in-progress dot pulse",
    ".sweep": "rung 2's chips crossing 58 px of felt to the pot, one-shot, 400 ms",
    ".rungs-menu": "the Rungs menu animates width on purpose: it pushes Rules aside",
    ".card": "the Rules tab's worked-hand cards, highlighted once by the reading",
}

# Every pressable thing, per stylesheet. A pressable has `.x:active` with a
# scale in range and `transform` in its base transition list.
PRESSABLES = {
    "theme": (".btn", ".navbtn", ".view", ".rmain", ".racts a"),
    "rung0-viz": (".mode-tabs button",),
    "rung1-viz": (".locks button.unlock",),
    "rung2-viz": (".tl button.stn", ".next button", ".lockbtn"),
    "index.html": (
        ".btn",
        ".navbtn",
        ".rmain",
        ".racts a",
        ".console button",
        ".titlebar .tabs a",
        ".seg button",
        ".act",
        ".playbtns button",
    ),
    "rules.html": (".btn", ".navbtn", ".rmain", ".racts a"),
    "cover.html": (".btn", ".navbtn", ".rmain", ".racts a"),
}


# ---------------------------------------------------------------- the CSS


def _sheets() -> dict[str, str]:
    sheets = {"theme": (WEB / APPS[0] / "src" / "theme.css").read_text()}
    for app in APPS:
        sheets[app] = (WEB / app / "src" / "index.css").read_text()
    for page in PAGES:
        html = (WEB / "cover" / page).read_text()
        styles = re.findall(r"<style>(.*?)</style>", html, re.DOTALL)
        assert styles, f"{page} has no <style> block — the regex stopped matching"
        sheets[page] = "\n".join(styles)
    return sheets


SHEETS = _sheets()


class Rule:
    def __init__(self, media: str | None, selector: str, body: str) -> None:
        self.media = media
        self.selector = selector
        self.body = body

    def __repr__(self) -> str:
        return f"{self.media + ' ' if self.media else ''}{self.selector}"

    def declarations(self, prop: str) -> list[str]:
        """Values of `prop` (exact name, not a `prop-*` longhand)."""
        return [
            m.group(1).strip()
            for m in re.finditer(rf"(?:^|[;{{\s]){re.escape(prop)}\s*:\s*([^;}}]+)", self.body)
        ]

    @property
    def explanatory(self) -> bool:
        return any(key in self.selector for key in EXPLANATORY)


def _rules(css: str) -> list[Rule]:
    """Every `selector { body }` with the @media it sits in, comments stripped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    out: list[Rule] = []

    def walk(s: str, media: str | None) -> None:
        i = 0
        while True:
            j = s.find("{", i)
            if j < 0:
                return
            head = s[i:j].strip()
            depth, k = 1, j + 1
            while depth:
                depth += (s[k] == "{") - (s[k] == "}")
                k += 1
            body = s[j + 1 : k - 1]
            if head.startswith(("@media", "@supports")):
                walk(body, head)
            else:
                out.append(Rule(media, head, body))
            i = k

    walk(css, None)
    assert out, "no rules parsed — the tokenizer stopped matching"
    return out


RULES = {name: _rules(css) for name, css in SHEETS.items()}


def _durations_ms(value: str) -> list[float]:
    """Every `<n>s` / `<n>ms` in a transition or animation value, in ms."""
    out = []
    for num, unit in re.findall(r"(\d*\.?\d+)(ms|s)\b", value):
        out.append(float(num) * (1 if unit == "ms" else 1000))
    return out


def _motion_values(rule: Rule) -> list[str]:
    values = []
    for prop in ("transition", "animation", "transition-duration", "animation-duration"):
        values.extend(rule.declarations(prop))
    return values


# -------------------------------------------------------------- the tokens


@pytest.mark.parametrize("name", ["theme", *PAGES])
def test_the_motion_tokens_are_declared_and_agree(name: str) -> None:
    roots = [r for r in RULES[name] if r.selector == ":root"]
    assert roots, f"{name}: no :root block"
    body = " ".join(r.body for r in roots)
    for token in TOKENS:
        m = re.search(rf"{re.escape(token)}\s*:\s*([^;]+);", body)
        assert m, f"{name}: {token} is not declared in :root (see web/DESIGN.md)"
        got = re.sub(r"\s+", "", m.group(1))
        got = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", got)  # .23 → 0.23
        assert got == TOKEN_VALUES[token], (
            f"{name}: {token} is {got}, the standard says {TOKEN_VALUES[token]}"
        )


# ------------------------------------------------------------ the curves


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_transition_all_and_no_ease_in(name: str) -> None:
    for rule in RULES[name]:
        for value in _motion_values(rule):
            assert not re.match(r"all\b", value), (
                f"{name}: {rule!r} transitions `all`; name the property"
            )
            assert not re.search(r"\bease-in\b(?!-out)", value), (
                f"{name}: {rule!r} uses ease-in; entrances use var(--ease-out)"
            )


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_chrome_reads_the_easing_tokens_rather_than_writing_a_curve(name: str) -> None:
    for rule in RULES[name]:
        if rule.selector == ":root" or rule.explanatory:
            continue
        for value in _motion_values(rule):
            assert "cubic-bezier(" not in value, (
                f"{name}: {rule!r} hard-codes a curve; use var(--ease-out) / "
                "var(--ease-in-out), or allowlist the figure in EXPLANATORY"
            )


# ---------------------------------------------------------- the durations


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_chrome_motion_stays_under_300ms(name: str) -> None:
    for rule in RULES[name]:
        if rule.explanatory:
            continue
        for value in _motion_values(rule):
            for ms in _durations_ms(value):
                assert ms <= MAX_UI_MS, (
                    f"{name}: {rule!r} runs {ms:g} ms (`{value}`); the chrome stays "
                    f"under {MAX_UI_MS} ms, or the figure goes in EXPLANATORY with a reason"
                )


LAYOUT_PROPS = ("width", "height", "padding", "margin", "top", "left", "right", "bottom")


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_only_transform_and_opacity_move_in_the_chrome(name: str) -> None:
    for rule in RULES[name]:
        if rule.explanatory:
            continue
        for value in rule.declarations("transition"):
            for part in value.split(","):
                prop = part.strip().split()[0] if part.strip() else ""
                assert prop not in LAYOUT_PROPS, (
                    f"{name}: {rule!r} animates `{prop}`, which lays out and paints; "
                    "move it with transform, or allowlist it in EXPLANATORY"
                )


# --------------------------------------------------------------- the press


def _press_scale(body: str) -> float | None:
    m = re.search(r"transform\s*:\s*scale\(\s*(\d*\.?\d+)\s*\)", body)
    return float(m.group(1)) if m else None


@pytest.mark.parametrize("name", sorted(PRESSABLES))
def test_every_pressable_answers_a_press(name: str) -> None:
    rules = RULES[name]
    for sel in PRESSABLES[name]:
        actives = [
            r
            for r in rules
            if r.media is None
            and any(
                re.fullmatch(rf"{re.escape(sel)}:active(:not\([^)]*\))?", s.strip())
                for s in r.selector.split(",")
            )
        ]
        assert actives, f"{name}: {sel} has no :active rule; a press has to be heard"
        scales = [_press_scale(r.body) for r in actives]
        assert any(s is not None for s in scales), (
            f"{name}: {sel}:active does not scale; the standard is scale(0.97)"
        )

        bases = [r for r in rules if r.media is None and sel in [s.strip() for s in r.selector.split(",")]]
        assert bases, f"{name}: no base rule for {sel}"
        transitions = " ".join(v for r in bases for v in r.declarations("transition"))
        assert re.search(r"\btransform\s+var\(--t-press\)\s+var\(--ease-out\)", transitions), (
            f"{name}: {sel}'s transition list does not carry "
            "`transform var(--t-press) var(--ease-out)`, so the press would snap"
        )


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_a_press_is_subtle(name: str) -> None:
    lo, hi = PRESS_RANGE
    for rule in RULES[name]:
        if ":active" not in rule.selector or rule.media is not None:
            continue
        scale = _press_scale(rule.body)
        if scale is None:
            continue
        assert lo <= scale <= hi, (
            f"{name}: {rule!r} presses to scale({scale:g}); the range is {lo}–{hi}"
        )


# --------------------------------------------------------------- the hover


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_a_hover_that_moves_something_is_gated_on_a_real_pointer(name: str) -> None:
    for rule in RULES[name]:
        if ":hover" not in rule.selector:
            continue
        moves = [v for v in rule.declarations("transform") if v.strip() != "none"]
        if not moves:
            continue
        assert rule.media and "hover: hover" in re.sub(r"\s*:\s*", ": ", rule.media), (
            f"{name}: {rule!r} moves on hover outside `@media (hover: hover) and "
            "(pointer: fine)`; a tap would leave it stuck"
        )


# ------------------------------------------------------- reduced motion


@pytest.mark.parametrize("name", ["theme", *PAGES])
def test_reduced_motion_is_fewer_and_gentler_not_none(name: str) -> None:
    reduced = [
        r for r in RULES[name] if r.media and "prefers-reduced-motion" in r.media
    ]
    assert reduced, f"{name}: no @media (prefers-reduced-motion: reduce) block"
    for rule in reduced:
        if rule.selector.strip() == "*":
            assert "!important" not in rule.body, (
                f"{name}: reduced motion is `* {{ none !important }}`; remove the "
                "travel and the pulse by name and keep colour and opacity"
            )
    # the press is one of the things that goes quiet
    assert any(":active" in r.selector and "none" in r.body for r in reduced), (
        f"{name}: the press scale is not turned off under reduced motion"
    )
